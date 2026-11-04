"""Attention-output calibration helpers for codec-native readers."""
import math
import torch
from train_reader import rope_low

def risks(group,cases,gram=None,mean_residual=False):
    t={k:v.cuda().float() if k!='indices' else v.cuda() for k,v in group.items() if k!='heads'}
    numerator=torch.zeros(40,device='cuda');denominator=torch.zeros_like(numerator)
    with torch.no_grad():
        for c in cases:
            q=c['q_pre'].cuda().float();k=c['k_pre'].cuda().float();v=c['v'].cuda().float()
            ql=torch.einsum('bhqd,hde->bhqe',q,t['q_reduce']);kl=torch.einsum('bhkd,hde->bhke',k,t['k_affine'])
            qp=rope_low(ql,c['q_positions'].cuda(),c['grid_sizes'].cuda(),c['freqs'].cuda(),c['frame_seqlen'],t['indices'])
            kp=rope_low(kl,c['k_positions'].cuda(),c['grid_sizes'].cuda(),c['freqs'].cuda(),c['frame_seqlen'],t['indices'])
            logits=qp@kp.transpose(-1,-2)/math.sqrt(128)
            teacher=c['q_native'].cuda().float()@c['k_native'].cuda().float().transpose(-1,-2)/math.sqrt(128)
            vo=torch.einsum('bhkd,hde->bhke',v,t['v_affine'])
            out=torch.einsum('bhqe,hed->bhqd',logits.softmax(-1)@vo,t['lift'])
            if mean_residual:
                out+=v.mean(2,keepdim=True)-torch.einsum('bhqe,hed->bhqd',vo.mean(2,keepdim=True),t['lift'])
            target=teacher.softmax(-1)@v
            if 'current_lse' in c:
                offset=math.log(c['history_token_count']/logits.shape[-1])
                ms=torch.sigmoid(torch.logsumexp(logits,-1)+offset-c['current_lse'].cuda())
                mt=torch.sigmoid(torch.logsumexp(teacher,-1)+offset-c['current_lse'].cuda())
                current=c['current_out'].cuda().float()
                out=out*ms[...,None]+current*(1-ms[...,None])
                target=target*mt[...,None]+current*(1-mt[...,None])
            error=out-target
            if gram is None:
                numerator+=error.square().sum((0,2,3));denominator+=target.square().sum((0,2,3))
            else:
                numerator+=torch.einsum('bhqd,hde,bhqe->h',error,gram,error)
                denominator+=torch.einsum('bhqd,hde,bhqe->h',target,gram,target)
    denom=denominator if gram is None else denominator.sum()
    return (numerator/denom.clamp_min(1e-12)).cpu().tolist()

def protect_sink(case,frames):
    if not frames or 'current_lse' not in case:return case
    mask=case['k_positions'] < frames*int(case['frame_seqlen'])
    if not bool(mask.any()):return case
    if bool(mask.all()):raise ValueError('No low history after calibration protection')
    c=dict(case)
    q=case['q_native'].float().cuda()
    k=case['k_native'][:,:,mask].float().cuda();v=case['v'][:,:,mask].float().cuda()
    logits=q@k.transpose(-1,-2)/math.sqrt(128)
    # Uniformly sampled protected keys estimate their full branch mass.
    n=min(frames*int(case['frame_seqlen']),case['history_token_count'])
    lse=torch.logsumexp(logits,-1)+math.log(n/k.shape[2])
    out=logits.softmax(-1)@v
    current_lse=case['current_lse'].cuda().float()
    total=torch.logaddexp(lse,current_lse)
    c['current_out']=(out*(lse-total).exp()[...,None]+case['current_out'].cuda().float()*(current_lse-total).exp()[...,None]).cpu()
    c['current_lse']=total.cpu()
    for field in ('k_pre','k_native','v'):c[field]=case[field][:,:,~mask]
    c['k_positions']=case['k_positions'][~mask]
    c['history_token_count']=case['history_token_count']-n
    c['calibration_sink_frames']=frames
    return c
