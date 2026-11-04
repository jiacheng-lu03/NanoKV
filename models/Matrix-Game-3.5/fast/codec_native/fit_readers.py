"""Local codec-teacher distillation on original train samples; held-out calls reported.

Only static reader/query/lift matrices are learned. Native DiT and codec are read-only.
This is a development fitter, not evidence of closed-loop quality by itself.
"""
import argparse, copy, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from codec_native.core import ReaderGroup, rope_selected, camera_selected
from codec_native.reader_tools import initialize_reader_state
from codec_native.build_initial_readers import OP, sha

def device_tree(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:device_tree(v,device) for k,v in x.items()}
    if isinstance(x,list):return [device_tree(v,device) for v in x]
    return x

def decoder(op,kind,l):
    p=op['entries'][f'{kind}:{l:02d}']['decoder'];q=p['qweight']
    return (q.reshape(q.shape[0],-1,p['group_size']).bfloat16()*p['scale'][:,:,None]).reshape_as(q).float()

def prepare(raw,op,l,device):
    s=device_tree(raw,device);q=s['q_pre'].reshape(s['q_pre'].shape[0],-1,24,128).permute(0,2,1,3)
    allidx=torch.arange(128,device=device).expand(24,-1)
    qt=rope_selected(q,s['q_freqs'],allidx)
    if 'q_camera' in s:qt=camera_selected(qt,s['q_camera'][1],allidx)
    keys=[];values=[]
    for seg in s['segments']:
        for kind,target in [('k',keys),('v',values)]:
            data=seg[kind]
            if 'decoded' not in data:
                w=decoder(op,kind,l).to(device=device,dtype=torch.bfloat16)
                mean=op['entries'][f'{kind}:{l:02d}']['mean'].to(device)
                decoded=F.linear(data['latent'],w,mean.bfloat16())
                data['decoded']=(decoded.float()+data['bias'].float()-mean.float()).bfloat16()
            x=data['decoded'].reshape(data['decoded'].shape[0],-1,24,128).permute(0,2,1,3)
            if kind=='k':x=rope_selected(x,seg['freqs'],allidx)
            if 'camera' in seg:x=camera_selected(x,seg['camera'][2],allidx)
            target.append(x)
    k=torch.cat(keys,2);v=torch.cat(values,2)
    logits=qt.float()@k.float().transpose(-1,-2)/(128**.5)
    prob=logits.softmax(-1);out=prob@v.float()
    if 'q_camera' in s:out=camera_selected(out,s['q_camera'][0],allidx)
    s.update(q=q,teacher_logits=logits,teacher_prob=prob,teacher_out=out)
    return s

def predict(g,s):
    q=g.query(s['q'].float());q=rope_selected(q,s['q_freqs'],g.indices)
    if 'q_camera' in s:q=camera_selected(q,s['q_camera'][1],g.indices)
    keys=[];values=[]
    for seg in s['segments']:
        for kind,target in [('k',keys),('v',values)]:
            data=seg[kind];w=getattr(g,kind+'_read');aff=getattr(g,kind+'_affine')
            bias=data['bias'].float().reshape(data['bias'].shape[0],-1,24,128).permute(0,2,1,3)
            x=torch.einsum('bnr,hdr->bhnd',data['latent'].float(),w)+torch.einsum('bhnd,hde->bhne',bias,aff)
            if kind=='k':x=rope_selected(x,seg['freqs'],g.indices)
            if 'camera' in seg:x=camera_selected(x,seg['camera'][2],g.indices)
            target.append(x)
    k=torch.cat(keys,2);v=torch.cat(values,2)
    logits=q@k.transpose(-1,-2)/(128**.5);prob=logits.softmax(-1)
    out=prob@v
    if 'q_camera' in s:out=camera_selected(out,s['q_camera'][0],g.indices)
    return logits,prob,g.lift_output(out)

def losses(g,s,lse_weight=0.02):
    logits,p,out=predict(g,s)
    target=s['teacher_out'];rel=(out-target).square().mean((-1,-2))/target.square().mean((-1,-2)).clamp_min(1e-8)
    kl=(s['teacher_prob']*(s['teacher_prob'].clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(-1).mean()
    lse=(logits.logsumexp(-1)-s['teacher_logits'].logsumexp(-1)).square().mean()
    lse=lse/(1+s['teacher_logits'].var(-1).mean())
    total=rel.mean()+.1*kl+lse_weight*lse
    return (total,rel.sqrt().detach().flatten(),
            float(rel.mean().detach()),float(kl.detach()),float(lse.detach()))

def indices_for(samples,d):
    # Whole (j,r) blocks preserve native camera groups and original RoPE pairs.
    score=0
    for s in samples:
        q=s['q'].float().square().mean((0,2))
        k=torch.cat([v['k']['decoded'] for v in s['segments']],1).float().reshape(-1,24,128).square().mean(0)
        v=torch.cat([v['v']['decoded'] for v in s['segments']],1).float().reshape(-1,24,128).var(0)
        term=(q*k).reshape(24,16,8).sum(-1); val=v.reshape(24,16,8).sum(-1)
        score=score+term/term.mean(-1,keepdim=True).clamp_min(1e-8)+val/val.mean(-1,keepdim=True).clamp_min(1e-8)
    blocks=score.topk(d//8,-1).indices.sort(-1).values
    return (blocks[...,None]*8+torch.arange(8,device=blocks.device)).flatten(1)

