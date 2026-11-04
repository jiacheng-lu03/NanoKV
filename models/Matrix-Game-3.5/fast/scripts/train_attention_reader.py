"""Shared head projections, full visible keys, frozen codec teacher.

Only projection matrices are optimized; decoder times projection is folded at
export. Last captured call of each original train trajectory is held out.
"""
import argparse,copy,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from codec_native.reader_training import decoder,device_tree,indices_for
from codec_native.core import ReaderGroup,rope_selected,camera_selected
from codec_native.reader_tools import initialize_reader_state
from codec_native.artifact_utils import OP,sha

def heads(x):return x.reshape(x.shape[0],-1,24,128).permute(0,2,1,3).float()

def prepare(raw,op,l):
    s=device_tree(raw,'cuda');s['q']=heads(s['q_pre'])
    for seg in s['segments']:
        for kind in ('k','v'):
            d=seg[kind];w=decoder(op,kind,l).cuda().bfloat16()
            x=F.linear(d['latent'].bfloat16(),w,d['mean'].bfloat16())
            if d['centers'] is not None:
                x=(x.float()+d['centers'].gather(1,d['ids'][...,None].expand(-1,-1,3072)).float()).bfloat16()
            d['decoded']=x
    s['segments'].append(dict(k=dict(decoded=s['native_k']),v=dict(decoded=s['native_v']),freqs=s['native_freqs'],camera=s.get('native_camera'),keep=s['native_keep']))
    nh=sum(x['k']['decoded'].shape[1] for x in s['segments'][:-1])
    s['keep']=torch.cat((torch.ones(s['native_keep'].shape[0],nh,device='cuda',dtype=torch.bool),s['native_keep']),-1)[None,None]
    with torch.no_grad():s['teacher_logits'],s['teacher_prob'],s['teacher_out']=predict(None,s)
    return s

def predict(g,s):
    idx=torch.arange(128,device='cuda').expand(24,-1) if g is None else g.indices
    q=s['q'] if g is None else g.query(s['q'])
    q=rope_selected(q,s['q_freqs'],idx)
    if s.get('q_camera') is not None:q=camera_selected(q,s['q_camera'][1],idx)
    kk=[];vv=[];qq=[];dims=[]
    for number,seg in enumerate(s['segments']):
        native=g is not None and s.get('history_only',False) and number==len(s['segments'])-1
        si=torch.arange(128,device='cuda').expand(24,-1) if native else idx
        sq=q
        if native:
            sq=rope_selected(s['q'],s['q_freqs'],si)
            if s.get('q_camera') is not None:sq=camera_selected(sq,s['q_camera'][1],si)
        vi=si if g is None or native else g.v_indices
        qq.append(sq);dims.append(vi)
        for kind,target in [('k',kk),('v',vv)]:
            x=heads(seg[kind]['decoded'])
            if g is not None and not native:x=torch.einsum('bhnd,hde->bhne',x,getattr(g,kind+'_affine'))
            if kind=='k':x=rope_selected(x,seg['freqs'],si)
            if seg.get('camera') is not None:x=camera_selected(x,seg['camera'][2],si if kind=='k' else vi)
            target.append(x)
    logits=(torch.cat([qk@k.transpose(-1,-2) for qk,k in zip(qq,kk)],-1)/(128**.5)).masked_fill(~s['keep'],-torch.inf)
    p=logits.softmax(-1);offset=0;out=0
    for v,si in zip(vv,dims):
        part=p[...,offset:offset+v.shape[-2]]@v;offset+=v.shape[-2]
        if s.get('q_camera') is not None:part=camera_selected(part,s['q_camera'][0],si)
        if g is not None and si.shape[-1]!=128:part=g.lift_output(part)
        out=out+part
    return logits,p,out

def loss(g,s):
    a,p,o=predict(g,s);b=s['teacher_logits'];t=s['teacher_out']
    rel=(o-t).square().mean((-1,-2))/t.square().mean((-1,-2)).clamp_min(1e-8)
    kl=(s['teacher_prob']*(s['teacher_prob'].clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(-1).mean()
    lse=(a.logsumexp(-1)-b.logsumexp(-1)).square().mean()
    return rel.mean()+.1*kl+.02*lse,rel.sqrt().detach().cpu().tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dim',type=int,default=96);p.add_argument('--steps',type=int,default=400);p.add_argument('--layers',default='0,15,29');p.add_argument('--history-only',action='store_true');p.add_argument('--exact-k',action='store_true');p.add_argument('--exact-v',action='store_true');a=p.parse_args()
    assert not (a.exact_k and a.exact_v)
    torch.set_num_threads(8);torch.manual_seed(20260912);a.out.mkdir(parents=True,exist_ok=True)
    op=torch.load(OP,map_location='cpu',weights_only=True)
    manifest=dict(format='codec_native_manifest_v1',model='matrix-game-3.5',codec_operator_sha256=sha(OP),policy=dict(recent_native_frames=0,anchor_stride=0,sink_native_frames=0,low_native='all'),layers={});reports={}
    if (a.out/'manifest.json').exists():
        manifest=json.loads((a.out/'manifest.json').read_text());reports=json.loads((a.out/'fit_report.json').read_text())
    manifest['policy']['low_native']='none' if a.history_only else 'all'
    for l in map(int,a.layers.split(',')):
        if str(l) in reports:continue
        files=sorted(a.samples.glob(f'**/full_layer{l:02d}_call*.pt'));assert len(files)>=3
        samples=[prepare(torch.load(f,map_location='cpu',weights_only=True),op,l) for f in files]
        for s in samples:s['history_only']=a.history_only
        last={f.parent:max(x.name for x in files if x.parent==f.parent) for f in files}
        train=[s for f,s in zip(files,samples) if f.name!=last[f.parent]];held=[s for f,s in zip(files,samples) if f.name==last[f.parent]]
        idx=indices_for(train,a.dim).cpu();payload=initialize_reader_state(decoder(op,'k',l),decoder(op,'v',l),[(torch.arange(24),idx)])
        if a.exact_k:
            full=initialize_reader_state(decoder(op,'k',l),decoder(op,'v',l),[(torch.arange(24),torch.arange(128).expand(24,-1))])['groups'][0]
            payload['groups'][0]['v_indices']=idx
            for key in ('indices','q_reduce','k_read','k_affine'):payload['groups'][0][key]=full[key]
        if a.exact_v:
            full=initialize_reader_state(decoder(op,'k',l),decoder(op,'v',l),[(torch.arange(24),torch.arange(128).expand(24,-1))])['groups'][0]
            payload['groups'][0]['v_indices']=full['indices']
            for key in ('v_read','v_affine','lift'):payload['groups'][0][key]=full[key]
        g=ReaderGroup(payload['groups'][0]).cuda();g.k_read.requires_grad_(False);g.v_read.requires_grad_(False)
        if a.exact_k:g.q_reduce.requires_grad_(False);g.k_affine.requires_grad_(False)
        if a.exact_v:g.v_affine.requires_grad_(False);g.lift.requires_grad_(False)
        optim=torch.optim.Adam([x for x in g.parameters() if x.requires_grad],lr=1e-4)
        best=float('inf');state=copy.deepcopy(g.state_dict());start=time.time()
        with torch.no_grad():initial=[loss(g,s)[1] for s in held]
        for step in range(a.steps):
            optim.zero_grad();v,_=loss(g,train[step%len(train)]);v.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1.);optim.step()
            if step==0 or (step+1)%50==0:
                with torch.no_grad():val=sum(float(loss(g,s)[0]) for s in held)/len(held)
                if val<best:best=val;state=copy.deepcopy(g.state_dict())
                print(json.dumps(dict(layer=l,step=step+1,loss=float(v),held=val,seconds=time.time()-start)),flush=True)
        g.load_state_dict(state)
        with torch.no_grad():
            final=[loss(g,s)[1] for s in held]
            for kind in ('k','v'):
                w=decoder(op,kind,l).cuda().reshape(24,128,-1)
                getattr(g,kind+'_read').copy_(torch.einsum('hnr,hnd->hdr',w,getattr(g,kind+'_affine')))
        payload['groups']=[{k:v.cpu() for k,v in g.state_dict().items()}];dest=a.out/f'layer{l:02d}.pt';torch.save(payload,dest)
        manifest['layers'][f'0:{l}']=dict(path=dest.name,sha256=sha(dest),calibrated=False)
        reports[str(l)]=dict(initial=initial,final=final,files=list(map(str,files)),seconds=time.time()-start,shared_projection=True,steps=a.steps)
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.out/'fit_report.json').write_text(json.dumps(reports,indent=2))
        del samples,train,held,g,optim;torch.cuda.empty_cache()
if __name__=='__main__':main()
