"""Frozen-codec reader fitting with RoPE-pair-constrained Q/K and functional V.

No original model/codec parameters are trained. Selecting/scaling complete native
pairs commutes exactly with native RoPE on those retained pairs; omitted pairs
still cause approximation error. The fitted maps are folded into codec-native readers.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reader_training import ReaderFit

def native(case):
    return [case[k].float().cuda()[0] for k in ('q_native','k_native','v')]

def pair_features(q,k):
    q=q.reshape(40,-1,64,2); k=k.reshape(40,-1,64,2)
    return torch.einsum('hqpc,hkpc->hqkp',q,k)/math.sqrt(128)

def mass(case, logits):
    if 'current_lse' not in case:
        return torch.ones_like(logits[:,:,0])
    lh=torch.logsumexp(logits,-1)+math.log(case['history_token_count']/logits.shape[-1])
    return torch.sigmoid(lh-case['current_lse'].float().cuda()[0])

def select_pairs(g,count,gain_cap=None):
    # Forward greedy least-squares feature selection in the Fisher quadratic.
    # Target is the full 64-pair logit (sum of all feature columns).
    b=g.sum(-1); h=torch.arange(40,device=g.device)
    indices=torch.empty(40,0,dtype=torch.long,device=g.device)
    weight=torch.empty(40,0,device=g.device)
    used=torch.zeros(40,64,dtype=torch.bool,device=g.device)
    diagonal=g.diagonal(dim1=-2,dim2=-1).clamp_min(1e-12)
    for _ in range(count):
        residual=b.clone()
        if indices.shape[-1]:
            cols=g.gather(2,indices[:,None,:].expand(-1,64,-1))
            residual-=torch.einsum('hpi,hi->hp',cols,weight)
        score=residual.square()/diagonal
        score.masked_fill_(used,-torch.inf)
        chosen=score.argmax(-1); used[h,chosen]=True
        indices=torch.cat((indices,chosen[:,None]),-1)
        sub=g[h[:,None,None],indices[:,:,None],indices[:,None,:]]
        ridge=diagonal.mean(-1)[:,None,None]*1e-4
        weight=torch.linalg.solve(sub+ridge*torch.eye(indices.shape[1],device=g.device)[None], b.gather(1,indices).unsqueeze(-1)).squeeze(-1)
        if gain_cap is not None:
            # Convex box-constrained quadratic refinement, initialized from
            # clipped unconstrained LS. Step <= 1/L preserves descent.
            matrix=sub+ridge*torch.eye(indices.shape[1],device=g.device)[None]
            rhs=b.gather(1,indices)
            step=torch.linalg.eigvalsh(matrix).amax(-1).clamp_min(1e-12).reciprocal()
            weight=weight.clamp(0,gain_cap)
            for _ in range(96):
                gradient=torch.einsum('hij,hj->hi',matrix,weight)-rhs
                weight=(weight-step[:,None]*gradient).clamp(0,gain_cap)
    order=indices.argsort(-1)
    return indices.gather(1,order),weight.gather(1,order)

def fit(cases, width, initial, operator, layer, rrr=False, center=False,gain_cap=None):
    g=torch.zeros(40,64,64,device='cuda')
    output_cov=torch.zeros(40,128,128,device='cuda')
    with torch.no_grad():
        for case in cases:
            factor=case.get('_fit_weight',1.0)
            q,k,v=native(case)
            features=pair_features(q,k)
            logits=features.sum(-1)
            prob=logits.softmax(-1)
            m=mass(case,logits)
            mu=torch.einsum('hqk,hqkp->hqp',prob,features)
            centered=features-mu[:,:,None,:]
            weighted=centered*(prob*m[:,:,None]).sqrt()[...,None]
            flat=weighted.flatten(1,2)
            g+=factor*(flat.transpose(-1,-2)@flat)
            # Retain absolute logit-level information for branch probability mass.
            mm=mu*(m*(1-m)+0.05).sqrt()[...,None]
            g+=factor*(mm.transpose(-1,-2)@mm)
            teacher=prob@v
            if center:
                teacher=teacher-v.mean(1,keepdim=True)
            weighted_out=teacher*m.sqrt()[...,None]
            output_cov+=factor*(weighted_out.transpose(-1,-2)@weighted_out)
        pairs,scales=select_pairs(g,width//2,gain_cap)
        indices=torch.stack((2*pairs,2*pairs+1),-1).flatten(-2)
        basis=torch.zeros(40,128,width,device='cuda')
        basis.scatter_(1,indices[:,None,:],1)
        qbasis=basis*scales.repeat_interleave(2,-1)[:,None,:]
        _,vectors=torch.linalg.eigh(output_cov)
        vbasis=vectors[:,:,-width:]
        vlift=vbasis.transpose(-1,-2)
        if rrr:
            xx=torch.zeros(40,128,128,device='cuda',dtype=torch.float64)
            xy=torch.zeros_like(xx)
            for case in cases:
                q,k,v=native(case)
                logits=(q@k.transpose(-1,-2))/math.sqrt(128)
                qp=q.gather(2,indices[:,None,:].expand(-1,q.shape[1],-1))*scales.repeat_interleave(2,-1)[:,None,:]
                kp=k.gather(2,indices[:,None,:].expand(-1,k.shape[1],-1))
                lowlogits=(qp@kp.transpose(-1,-2))/math.sqrt(128)
                mt=mass(case,logits);ms=mass(case,lowlogits)
                x=(lowlogits.softmax(-1)@v)*ms[...,None]
                y=(logits.softmax(-1)@v)*mt[...,None]
                if 'current_out' in case:
                    y+=(ms-mt)[...,None]*case['current_out'].float().cuda()[0]
                if center:
                    # Deployment adds the exact-history first moment outside
                    # the low-rank map: ms*((hs-mu)F+mu)+(1-ms)*current.
                    # Therefore regress x=ms*(hs-mu) against
                    # y=mt*ht+(ms-mt)*current-ms*mu, not uncentered outputs.
                    shift=ms[...,None]*v.mean(1,keepdim=True)
                    x=x-shift;y=y-shift
                x=x.double();y=y.double()
                factor=case.get('_fit_weight',1.0)
                xx+=factor*(x.transpose(-1,-2)@x);xy+=factor*(x.transpose(-1,-2)@y)
            # Reduced-rank ridge regression: optimal rank-d map under the
            # sampled quadratic output loss, not a guarantee outside calibration.
            ridge=xx.diagonal(dim1=-2,dim2=-1).mean(-1)*1e-3
            eig,vec=torch.linalg.eigh(xx+ridge[:,None,None]*torch.eye(128,device='cuda')[None])
            invsqrt=(vec*eig.clamp_min(1e-12).rsqrt()[:,None,:])@vec.transpose(-1,-2)
            u,s,vh=torch.linalg.svd(invsqrt@xy,full_matrices=False)
            root=s[:,:width].sqrt()
            vbasis=((invsqrt@u[:,:,:width])*root[:,None,:]).float()
            vlift=(root[:,:,None]*vh[:,:width,:]).float()
        group={
            'heads':torch.arange(40),'indices':indices.cpu(),
            'q_reduce':qbasis.cpu(),'k_affine':basis.cpu(),
            'v_affine':vbasis.cpu(),'lift':vlift.cpu().contiguous(),
        }
        for target in ['k','v']:
            item=operator['entries'][f'rank00:{target}:{layer:02d}']['decoder']
            dec=(item['qweight'].bfloat16()*item['scale'].bfloat16()).float().reshape(40,128,-1)
            group[target+'_read']=torch.einsum('hod,hor->hdr',group[target+'_affine'],dec)
        result=dict(initial)
        result['groups']=[group];result['width']=width;result['calibrated']=True
        result['initialization']='fisher_rope_pair_qk_attention_output_v'+('_rrr' if rrr else '')
        if center:
            result['centered_v_basis_full']=vectors.cpu()
            result['requires_value_mean_residual']=True
        return result

def assess(state,cases):
    model=ReaderFit(state['groups'][0]).cuda()
    accum=torch.zeros(4,device='cuda')
    with torch.no_grad():
        for case in cases: accum+=torch.stack(model(case))
    return dict(zip(['logit_rel_mse','lse_rel_mse','output_rel_mse','value_rel_mse'],(accum/len(cases)).tolist()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--initial-root',type=Path,required=True);p.add_argument('--operator',type=Path,required=True);p.add_argument('--samples',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',default=','.join(map(str,range(40))));p.add_argument('--width',type=int,default=64);p.add_argument('--rrr',action='store_true');p.add_argument('--center-output',action='store_true');p.add_argument('--sink-frames',type=int,default=0);p.add_argument('--balance-images',action='store_true');p.add_argument('--analytic-only',action='store_true');p.add_argument('--require-train-split',action='store_true');p.add_argument('--gain-cap',type=float);a=p.parse_args()
    if a.gain_cap is not None:assert a.gain_cap>0
    source=a.initial_root
    operator=torch.load(a.operator,map_location='cpu',weights_only=True)
    manifest=json.loads((source/'analytic_d64/manifest.json').read_text())
    manifest['layers']={};manifest['policy']=dict(sink_native_frames=0,recent_native_frames=0,anchor_stride=0);manifest['layer_policy']={}
    report=[];a.output.mkdir(parents=True,exist_ok=True)
    for layer in map(int,a.layers.split(',')):
        paths=[f for root in a.samples for f in sorted(Path(root).glob(f'**/layer_{layer:02d}.pt'))]
        if a.require_train_split:
            import re
            assert a.analytic_only, 'train-only fitting requires data-free initialization'
            assert paths and all(re.search(r'/train_\d+_',str(f)) and not re.search(r'/(test|dev)_\d+',str(f)) for f in paths), 'non-training sample detected'
        cases=[torch.load(f,map_location='cpu',weights_only=True) for f in paths]
        if a.sink_frames:
            from reader_calibration import protect_sink
            cases=[protect_sink(c,a.sink_frames) for c in cases]
        if a.balance_images:
            import re,collections
            ids=[re.search(r'(?:train|dev|test)_\d+',str(f)).group() for f in paths]
            counts=collections.Counter(i for i,c in zip(ids,cases) if int(c['chunk'])!=8)
            for i,c in zip(ids,cases):c['_fit_weight']=1/counts[i]
        train=[c for c in cases if int(c['chunk'])!=8];valid=[c for c in cases if int(c['chunk'])==8]
        if not train or not valid: raise RuntimeError(f'Missing train/held-out cases for {layer}')
        old_path=source/f'trained_d{a.width}_multiaction_lse/layer_{layer:02d}.pt'
        if a.analytic_only or not old_path.exists():
            old_path=source/f'analytic_d{a.width}/readers/rank_00/layer_{layer:02d}.pt'
        old=torch.load(old_path,map_location='cpu',weights_only=True)
        new=fit(train,a.width,old,operator,layer,a.rrr,a.center_output,a.gain_cap)
        row=dict(layer=layer,train_samples=len(train),valid_samples=len(valid),old=assess(old,valid),pair_fisher=assess(new,valid))
        if a.center_output:
            from reader_calibration import risks
            row['with_mean_residual_head_mse']=sum(risks(new['groups'][0],valid,mean_residual=True))/40
        folder=a.output/'readers/rank_00';folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'layer_{layer:02d}.pt';torch.save(new,target)
        manifest['layers'][f'0:{layer}']=dict(path=str(target.relative_to(a.output)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),calibrated=True,widths=[a.width])
        report.append(row);print(json.dumps(row),flush=True)
        (a.output/'local_report.json').write_text(json.dumps(report,indent=2))
    manifest['train_split_only']=a.require_train_split
    manifest['analytic_initialization_only']=a.analytic_only
    manifest['policy']['sink_native_frames']=a.sink_frames
    if a.gain_cap is not None:manifest['query_gain_bound']=[0,a.gain_cap];manifest['gain_bound_fitted_not_postclipped']=True
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
