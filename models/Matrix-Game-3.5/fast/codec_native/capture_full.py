"""Original train rollout, sampled queries but ALL visible native/history keys."""
import os
from pathlib import Path
import torch
_COUNTS={}
@torch.no_grad()
def collect_full(q,freqs,camera_info,cache,cache_freqs,cache_frames,cur_frames,mosaic_tokens,layer_idx,read_id,*,all_k_pre,all_v_raw,mosaic_frames,cur_cache_chunk_ids,hole_keep):
    if os.environ.get('MG35_READER_CAPTURE_SPLIT') not in ('train','dev'):raise RuntimeError('Training data only')
    l=int(layer_idx);_COUNTS[l]=_COUNTS.get(l,0)+1;call=_COUNTS[l]
    calls=tuple(map(int,os.environ.get('MG35_READER_CAPTURE_CALLS','12,24,48').split(',')))
    if call not in calls or not cache or cache.get('k') is None:return
    from .mg35 import _expand_frames
    from diffsynth.models.worldplay_kv_codec import value_shape,direct_latent_pair_segments
    device=q.device;n=value_shape(cache['k'])[1];frames=list(cache_frames or []);chunks=cache.get('chunk_ids',[])
    eligible=torch.ones(n,dtype=torch.bool,device=device)
    if read_id is not None and len(chunks)==len(frames):
        eligible=torch.tensor([int(c)<0 or int(c)==int(read_id) for c in chunks],device=device).repeat_interleave(n//len(frames))
    qids=torch.linspace(mosaic_tokens,q.shape[1]-1,min(64,q.shape[1]-mosaic_tokens),device=device).long()
    native_frames=_expand_frames(mosaic_frames,mosaic_tokens)+_expand_frames(cur_frames,q.shape[1]-mosaic_tokens)
    nf=torch.tensor(native_frames,device=device);qf=nf[qids]
    keep=torch.ones(len(qids),q.shape[1],device=device,dtype=torch.bool)
    if hole_keep is not None:keep[:,:mosaic_tokens]=hole_keep[None]
    if cur_cache_chunk_ids is not None:
        gids=torch.tensor(cur_cache_chunk_ids,device=device).repeat_interleave((q.shape[1]-mosaic_tokens)//len(cur_frames))
        qg=gids[qids-mosaic_tokens];anchor=gids<0
        keep[:,mosaic_tokens:]&=torch.where(qg[:,None]<0,anchor[None],(qg[:,None]==gids[None])|anchor[None])
    result=dict(format='mg35_full_keys_v1',layer=l,call=call,q_pre=q[:,qids].cpu(),q_freqs=freqs[qids].cpu(),q_frames=qf.cpu(),native_k=all_k_pre.cpu(),native_v=all_v_raw.cpu(),native_freqs=freqs.cpu(),native_frames=nf.cpu(),native_keep=keep.cpu(),mosaic_tokens=mosaic_tokens,segments=[])
    if camera_info is not None:
        result['q_camera']=[m.index_select(1,qf).cpu() for m in camera_info[1]]
        result['native_camera']=[m.index_select(1,nf).cpu() for m in camera_info[1]]
    frame_table=torch.tensor(frames,device=device).repeat_interleave(n//len(frames))
    for s in direct_latent_pair_segments(cache['k'],cache['v'],device=device,token_keep=eligible):
        logical=s['logical_indices'];sf=frame_table[logical]
        row=dict(freqs=cache_freqs[logical].cpu(),frames=sf.cpu(),logical_indices=logical.cpu())
        if camera_info is not None:row['camera']=[m.index_select(1,sf).cpu() for m in camera_info[1]]
        for kind in ('k','v'):
            z,(centers,ids),mean=s[kind]
            row[kind]=dict(latent=z.cpu(),centers=None if centers is None else centers.cpu().bfloat16(),ids=None if ids is None else ids.cpu(),mean=mean.cpu())
        result['segments'].append(row)
    out=Path(os.environ['MG35_READER_CAPTURE_DIR']);out.mkdir(parents=True,exist_ok=True)
    torch.save(result,out/f'full_layer{l:02d}_call{call:03d}.pt')
