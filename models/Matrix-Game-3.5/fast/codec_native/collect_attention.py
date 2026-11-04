"""Bounded train/dev-only codec teacher samples; never changes model/cache state."""
import os
from pathlib import Path
import torch
_COUNTS={}

@torch.no_grad()
def collect(q, freqs, camera_info, cache, cache_freqs, cache_frames, cur_frames,
            mosaic_tokens, layer_idx, cache_read_chunk_id, **native):
    dest=os.environ.get('MG35_READER_CAPTURE_DIR')
    if not dest:return
    if os.environ.get('MG35_READER_CAPTURE_FULL')=='1':
        from .attention_samples import collect_full
        return collect_full(q,freqs,camera_info,cache,cache_freqs,cache_frames,cur_frames,mosaic_tokens,layer_idx,cache_read_chunk_id,**native)
    if os.environ.get('MG35_READER_CAPTURE_SPLIT') not in ('train','dev'):
        raise RuntimeError('Reader calibration must not collect formal test cases')
    l=int(layer_idx);_COUNTS[l]=_COUNTS.get(l,0)+1;call=_COUNTS[l]
    if call not in (12,24,36,48) or not cache or cache.get('k') is None:return
    from diffsynth.models.nanokv_codec import value_shape,direct_latent_pair_segments
    n=value_shape(cache['k'])[1];frames=list(cache_frames or [])
    if not frames or n%len(frames):return
    device=q.device;ordinals=torch.arange(len(frames),device=device).repeat_interleave(n//len(frames))
    eligible=torch.ones(n,device=device,dtype=torch.bool)
    chunks=cache.get('chunk_ids',[])
    if cache_read_chunk_id is not None and len(chunks)==len(frames):
        allowed=torch.tensor([v<0 or v==int(cache_read_chunk_id) for v in chunks],device=device)
        eligible=allowed[ordinals]
    ids=eligible.nonzero().flatten()
    if not len(ids):return
    ids=ids[torch.linspace(0,len(ids)-1,min(256,len(ids)),device=device).long()]
    keep=torch.zeros_like(eligible);keep[ids]=True
    qids=torch.linspace(mosaic_tokens,q.shape[1]-1,min(64,q.shape[1]-mosaic_tokens),device=device).long()
    qframes=torch.tensor(cur_frames,device=device).repeat_interleave((q.shape[1]-mosaic_tokens)//len(cur_frames))
    qframes=qframes[qids-mosaic_tokens]
    result=dict(layer=l,call=call,q_pre=q[:,qids].cpu(),q_freqs=freqs[qids].cpu(),q_frames=qframes.cpu(),segments=[])
    if camera_info is not None:result['q_camera']=[m.index_select(1,qframes).cpu() for m in camera_info[1]]
    for s in direct_latent_pair_segments(cache['k'],cache['v'],device=device,token_keep=keep):
        logical=s['logical_indices'];f=torch.tensor(frames,device=device)[ordinals[logical]]
        row=dict(freqs=cache_freqs[logical].cpu(),frames=f.cpu())
        if camera_info is not None:row['camera']=[m.index_select(1,f).cpu() for m in camera_info[1]]
        for target in ('k','v'):
            z,(centers,assigned),mean=s[target]
            bias=mean[None,None].expand(z.shape[0],z.shape[1],-1)
            if centers is not None:bias=bias+centers.gather(1,assigned[...,None].expand(-1,-1,centers.shape[-1])).float()
            from diffsynth.models.nanokv_codec import _operator, _operator_linear
            operator = _operator(target, l, device)
            decoded = _operator_linear(z.reshape(-1,z.shape[-1]),operator['decoder'],operator['mean']).reshape(z.shape[0],z.shape[1],-1)
            if centers is not None:
                decoded = (decoded.float()+centers.gather(1,assigned[...,None].expand(-1,-1,centers.shape[-1])).float()).to(q.dtype)
            row[target]=dict(latent=z.cpu(),bias=bias.cpu(),decoded=decoded.cpu())
        result['segments'].append(row)
    out=Path(dest);out.mkdir(parents=True,exist_ok=True)
    torch.save(result,out/f'layer{l:02d}_call{call:03d}.pt')
