"""MG3.5 bridge from the frozen INT4 codec to low-dimensional history attention."""
from __future__ import annotations

import os
from itertools import groupby
import torch

from .core import (
    attention_lse, attention_grouped_mask, camera_selected, get_layer, merge, protected_tokens,
    rope_selected,
)


def _frame_token_ids(frames, tokens: int, device):
    if not frames:
        return torch.empty(0, device=device, dtype=torch.long)
    if tokens % len(frames):
        raise ValueError("cache tokens must contain complete equal-sized frames")
    return torch.arange(len(frames), device=device).repeat_interleave(tokens // len(frames))


def _frame_values(frames, token_ordinals, device):
    table = torch.as_tensor(list(frames), device=device, dtype=torch.long)
    return table.index_select(0, token_ordinals)


def _expand_frames(frames, tokens: int):
    if not frames:
        return []
    if tokens % len(frames):
        raise ValueError("tokens must contain complete equal-sized frames")
    if os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5':
        from .metadata_cache import expanded
        return expanded(tuple(frames),tokens)
    return [int(frame) for frame in frames for _ in range(tokens // len(frames))]


def _viewmats(camera_info, frames):
    if os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5':
        from .metadata_cache import views
        cached=views(camera_info,frames)
        if cached is not None:return cached
    P, P_T, P_inv = camera_info[1]
    # Complete frames share one camera matrix across spatial tokens. Preserve
    # irregular runs, but avoid materializing token-repeated matrices when all
    # frame runs have the same length (the regular inference path).
    runs=[(frame,sum(1 for _ in values)) for frame,values in groupby(frames)]
    if runs and len({count for _,count in runs})==1:
        frames=[frame for frame,_ in runs]
    index = torch.as_tensor(frames, device=P.device, dtype=torch.long)
    return P.index_select(1, index), P_T.index_select(1, index), P_inv.index_select(1, index)


def _native_prope(q, k, v, q_frames, k_frames, camera_info, layout):
    if camera_info is None or k.shape[-2]==0:
        return q, k, v, lambda value: value
    if os.environ.get('MG35_CODEC_NATIVE_FUSED_CAMERA')=='1' and layout=='full':
        from .camera_kernel import camera
        qmats,kmats=_viewmats(camera_info,q_frames),_viewmats(camera_info,k_frames)
        cams=(torch.arange(q.shape[-1]//8,device=q.device)%4).expand(q.shape[1],-1)
        return (camera(q,qmats[1],cams,mode='dot'),camera(k,kmats[2],cams,mode='dot'),
                camera(v,kmats[2],cams,mode='dot'),lambda value:camera(value,qmats[0],cams,mode='dot'))
    from diffsynth.models.prope_attention import _prepare_apply_fns
    q_apply, _, out_apply = _prepare_apply_fns(
        q.shape[-1], _viewmats(camera_info, q_frames), camera_layout=layout
    )
    _, kv_apply, _ = _prepare_apply_fns(
        k.shape[-1], _viewmats(camera_info, k_frames), camera_layout=layout
    )
    return q_apply(q), kv_apply(k), kv_apply(v), out_apply


def _commit(
    cache, cur_k_pre, cur_v_raw, *, layer_idx, cur_frames,
    cur_positions, cur_cache_chunk_ids,
):
    from diffsynth.models.causal_kv_codec import process_cache_write
    from diffsynth.models.nanokv_codec import concat_values, encode_tensor, value_shape
    cur_k_pre, cur_v_raw = process_cache_write(int(layer_idx), cur_k_pre.detach(), cur_v_raw.detach())
    k_value = encode_tensor("k", int(layer_idx), cur_k_pre)
    v_value = encode_tensor("v", int(layer_idx), cur_v_raw)
    positions = list(cur_frames) if cur_positions is None else [int(v) for v in cur_positions]
    chunk_ids = ([-1] * len(cur_frames) if cur_cache_chunk_ids is None
                 else [int(v) for v in cur_cache_chunk_ids])
    if len(positions) != len(cur_frames) or len(chunk_ids) != len(cur_frames):
        raise ValueError("cache frame metadata length mismatch")
    if value_shape(cache.get("k")) is None:
        cache["k"], cache["v"] = k_value, v_value
        cache["positions"], cache["frames"], cache["chunk_ids"] = positions, list(cur_frames), chunk_ids
    else:
        old_frames = list(cache.get("frames", []))
        old_chunks = list(cache.get("chunk_ids", [-1] * len(old_frames)))
        cache["k"] = concat_values(cache["k"], k_value)
        cache["v"] = concat_values(cache["v"], v_value)
        cache["positions"] = list(cache.get("positions", cache.get("frames", []))) + positions
        cache["frames"] = old_frames + list(cur_frames)
        cache["chunk_ids"] = old_chunks + chunk_ids


@torch.no_grad()
def maybe_attention(
    self_attn, q_pre, q_native, k_native, v_raw, cur_k_pre, freqs_cur, camera_info, *,
    layer_idx, num_heads, mosaic_tokens, cur_frames, cur_positions,
    mosaic_frames, cache, cache_freqs, cache_frames, cache_read_chunk_id,
    cur_cache_chunk_ids, write_cache, hole_keep, a1_pass, all_k_pre=None,
):
    """Return projected block attention, or None for the untouched native path."""
    if os.environ.get("MG35_CODEC_NATIVE_ENABLE", "0").lower() not in {"1", "true", "yes"}:
        return None
    if os.environ.get("MG35_KVCODEC_MODE", "").lower() != "worldplay_method":
        raise RuntimeError("MG codec-native attention requires the frozen worldplay_method cache")
    if a1_pass is not None:
        raise RuntimeError("codec-native deployment and legacy A1 training cannot run together")
    operator_path = os.environ["MG35_KVCODEC_PREPACKED_OPERATOR_PATH"]
    reader, manifest = get_layer(
        "MG35_CODEC_NATIVE", "matrix-game-3.5", int(layer_idx),
        device=q_pre.device, heads=num_heads, native_dim=q_pre.shape[-1] // num_heads,
        operator_path=operator_path,
    )
    from .accounting import record
    if reader is None:
        from diffsynth.models.nanokv_codec import value_shape
        shape = None if cache is None else value_shape(cache.get('k'))
        n = 0 if shape is None else int(shape[1])
        frames = list(cache_frames or [])
        chunks = [] if cache is None else cache.get('chunk_ids', [])
        if n and cache_read_chunk_id is not None and len(chunks)==len(frames) and frames:
            n = n//len(frames)*sum(int(c)<0 or int(c)==int(cache_read_chunk_id) for c in chunks)
        record(layer_idx,q_pre.shape[1]-mosaic_tokens,n+k_native.shape[1],0,num_heads,q_pre.shape[-1]//num_heads,0,write_cache,batch=q_pre.shape[0],history=n)
        return None
    if self_attn.prope_camera_layout != "full" and camera_info is not None:
        raise RuntimeError("calibrated MG low PRoPE currently supports only the formal full layout")

    policy = dict(manifest.get("policy", {}))
    policy.update(manifest.get("layer_policy", {}).get(str(layer_idx), {}))
    recent = int(policy.get("recent_native_frames", 0))
    stride = int(policy.get("anchor_stride", 0))
    sink = int(policy.get("sink_native_frames", 0))
    cache_shape = None
    if cache is not None and cache.get("k") is not None:
        from diffsynth.models.nanokv_codec import value_shape
        cache_shape = value_shape(cache["k"])
    cache_tokens = 0 if cache_shape is None else int(cache_shape[1])
    frames = [] if cache_frames is None else list(cache_frames)
    chunk_ids = [] if cache is None else list(cache.get("chunk_ids", []))
    routing = None
    if (os.environ.get('MG35_CODEC_NATIVE_ROUTING_CACHE')=='1'
            and os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5'
            and policy.get('low_native','none')=='none'):
        from .metadata_cache import route
        routing=route(frames,chunk_ids,cache_read_chunk_id,cache_tokens,recent,stride,sink,
                      policy.get('native_history',False),q_pre.device)
        protect,low_keep=routing['protect_cpu'],routing['low_cpu']
    else:
        token_frame_ord = _frame_token_ids(frames, cache_tokens, q_pre.device)
        active = torch.ones(cache_tokens, device=q_pre.device, dtype=torch.bool)
        if cache_read_chunk_id is not None and len(chunk_ids) == len(frames):
            frame_active = torch.tensor(
                [int(value) < 0 or int(value) == int(cache_read_chunk_id) for value in chunk_ids],
                device=q_pre.device, dtype=torch.bool,
            )
            active &= frame_active.index_select(0, token_frame_ord)
        protect = torch.zeros_like(active)
        if cache_tokens:
            frame_protect = protected_tokens(
                torch.arange(len(frames), device=q_pre.device), recent, stride, sink
            )
            protect = active & frame_protect.index_select(0, token_frame_ord)
        if policy.get('native_history',False):protect=active.clone()
        low_keep = active & ~protect
    native_low=torch.zeros(k_native.shape[1],device=q_pre.device,dtype=torch.bool)
    low_native=policy.get('low_native','none')
    if cache_tokens and low_native!='none':
        if low_native in ('all','mosaic'):native_low[:mosaic_tokens]=True
        if low_native in ('all','current'):native_low[mosaic_tokens:]=True
    fast_counts=os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5' and low_native=='none'
    if routing is not None:
        active_count,protected_count=routing['active_count'],routing['protected_count']
        low_count=active_count-protected_count
    elif fast_counts:
        from .metadata_cache import history_counts
        active_count,protected_count=history_counts(frames,chunk_ids,cache_read_chunk_id,cache_tokens,recent,stride,sink,policy.get('native_history',False))
        low_count=active_count-protected_count
    else:
        active_count=int(active.sum());low_count=int(low_keep.sum()+native_low.sum())
    record(layer_idx,q_pre.shape[1]-mosaic_tokens,active_count+k_native.shape[1],low_count,num_heads,reader.native_dim,sum(len(g.heads)*(g.indices.shape[1]+g.v_indices.shape[1])/2 for g in reader.groups),write_cache,batch=q_pre.shape[0],history=active_count)
    # No old low-dimensional history means the frozen implementation is both
    # cheaper and numerically authoritative (notably for the seed chunk).
    if not (low_count>0 if fast_counts else bool(low_keep.any() or native_low.any())):
        return None

    protected_k, protected_v = [], []
    if (protected_count>0 if fast_counts else bool(protect.any())):
        from diffsynth.models.nanokv_codec import decode_pair_selected
        pk, pv = decode_pair_selected(
            cache["k"], cache["v"], protect,
            dtype=q_pre.dtype, device=q_pre.device,
        )
        protected_k.append(pk); protected_v.append(pv)
    if routing is not None:
        protected_frames=routing['protected_frames']
        protected_index=routing['protected_index']
    else:
        protected_ord = token_frame_ord[protect]
        protected_frames = _frame_values(frames, protected_ord, q_pre.device).tolist()
        protected_index=torch.nonzero(protect,as_tuple=False).flatten().to(q_pre.device)
    protected_freqs = None if cache_freqs is None else cache_freqs.index_select(0, protected_index.to(cache_freqs.device))
    if protected_k:
        if not (self_attn.use_prope and camera_info is not None and self_attn.prope_disable_native_rope):
            from diffsynth.models.wan_video_dit import rope_apply
            rope_freqs = protected_freqs
            if self_attn.use_prope and camera_info is not None and self_attn.prope_disable_t_rope:
                rope_freqs = rope_freqs.clone(); rope_freqs[..., :self_attn.rope_t_pairs] = 1
            protected_k[0] = rope_apply(protected_k[0], rope_freqs, num_heads)

    low_segments = []
    if (low_count>0 if fast_counts else bool(low_keep.any())):
        from diffsynth.models.nanokv_codec import direct_latent_pair_segments
        low_segments = direct_latent_pair_segments(
            cache["k"], cache["v"], device=q_pre.device, token_keep=low_keep
        )
        for segment in low_segments:
            logical = segment["logical_indices"]
            ordinal = None if routing is not None else token_frame_ord.index_select(0, logical)
            segment["frames"] = ([frames[i//(cache_tokens//len(frames))] for i in segment['logical_indices_cpu']]
                if fast_counts else _frame_values(frames, ordinal, q_pre.device).tolist())
            segment["freqs"] = cache_freqs.index_select(0, logical.to(cache_freqs.device))

    # Default keeps current/mosaic native. Explicit low_native policy is the
    # separately calibrated extension; cache writes remain ORIGINAL full KV.
    native_k = protected_k + [k_native[:, :mosaic_tokens], k_native[:, mosaic_tokens:]]
    native_v = protected_v + [v_raw[:, :mosaic_tokens], v_raw[:, mosaic_tokens:]]
    native_frames = (
        protected_frames
        + _expand_frames(mosaic_frames, mosaic_tokens)
        + _expand_frames(cur_frames, int(k_native.shape[1]-mosaic_tokens))
    )
    native_k = torch.cat([value for value in native_k if value.shape[1]], 1)
    native_v = torch.cat([value for value in native_v if value.shape[1]], 1)
    prefix_tokens = sum(value.shape[1] for value in protected_k)
    retained=None;native_segment=None
    if not fast_counts and bool(native_low.any()):
        low_index=native_low.nonzero().flatten()
        retained=torch.cat((torch.arange(prefix_tokens,device=q_pre.device),prefix_tokens+(~native_low).nonzero().flatten()))
        all_current_frames=_expand_frames(mosaic_frames,mosaic_tokens)+_expand_frames(cur_frames,k_native.shape[1]-mosaic_tokens)
        native_segment=dict(native=True,k_pre=all_k_pre.index_select(1,low_index),v_pre=v_raw.index_select(1,low_index),freqs=freqs_cur.index_select(0,low_index),frames=[all_current_frames[i] for i in low_index.tolist()])
        low_segments.append(native_segment)
        native_k=native_k.index_select(1,retained);native_v=native_v.index_select(1,retained)
        native_frames=[native_frames[i] for i in retained.tolist()]
    qh = q_native[:, mosaic_tokens:].reshape(q_native.shape[0], -1, num_heads, reader.native_dim).permute(0, 2, 1, 3)
    kh = native_k.reshape(native_k.shape[0], -1, num_heads, reader.native_dim).permute(0, 2, 1, 3)
    vh = native_v.reshape(native_v.shape[0], -1, num_heads, reader.native_dim).permute(0, 2, 1, 3)
    qh, kh, vh, output_native = _native_prope(
        qh, kh, vh,
        _expand_frames(cur_frames, int(q_native.shape[1]-mosaic_tokens)),
        native_frames, camera_info, self_attn.prope_camera_layout
    )
    keep = torch.cat((
        torch.ones(prefix_tokens, device=q_pre.device, dtype=torch.bool),
        (hole_keep.to(q_pre.device) if hole_keep is not None else torch.ones(mosaic_tokens, device=q_pre.device, dtype=torch.bool)),
        torch.ones(k_native.shape[1]-mosaic_tokens, device=q_pre.device, dtype=torch.bool),
    ))
    uniform_fast = (os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT') in ('fast3','fast4','fast5')
                    and retained is None
                    and (cur_cache_chunk_ids is None or len(set(map(int,cur_cache_chunk_ids))) == 1))
    # A single chronological query group has identical visibility for every
    # query. Keep the same visible-key gather but avoid SxN mask materialization.
    mask = keep[None,None,None] if uniform_fast else keep[None, None, None].expand(1, 1, qh.shape[-2], -1).clone()
    query_groups=torch.zeros(qh.shape[-2],device=q_pre.device,dtype=torch.long)
    if cur_cache_chunk_ids is not None and not uniform_fast:
        ids = torch.as_tensor(cur_cache_chunk_ids, device=q_pre.device, dtype=torch.long)
        q_per_frame = qh.shape[-2] // len(cur_frames)
        k_per_frame = (k_native.shape[1]-mosaic_tokens) // len(cur_frames)
        q_ids = ids.repeat_interleave(q_per_frame)
        query_groups=q_ids
        k_ids = ids.repeat_interleave(k_per_frame)
        key_anchor = k_ids < 0; query_anchor = q_ids < 0
        current_keep = torch.where(
            query_anchor[:, None], key_anchor[None],
            (q_ids[:, None] == k_ids[None]) | key_anchor[None],
        )
        current_start = prefix_tokens + mosaic_tokens
        mask[0, 0, :, current_start:current_start+len(k_ids)] &= current_keep
    if retained is not None:
        lowmask=mask.index_select(-1,prefix_tokens+low_index)
        native_segment['mask']=None if bool(lowmask.all()) else lowmask
        mask=mask.index_select(-1,retained)
    cached_visibility = routing is not None and uniform_fast
    if cached_visibility:
        from .metadata_cache import uniform_visible
        visible=uniform_visible(hole_keep,prefix_tokens,mosaic_tokens,
                                k_native.shape[1]-mosaic_tokens,q_pre.device)
        if visible is None:mask=None
    elif bool(mask.all()):
        mask = None
    if uniform_fast and mask is not None:
        if not cached_visibility:visible=keep.nonzero(as_tuple=False).flatten()
        native_out,native_lse=attention_lse(qh.contiguous(),kh.index_select(2,visible),vh.index_select(2,visible),scale=reader.native_dim**-.5)
    elif mask is not None and os.environ.get('MG35_CODEC_NATIVE_GROUPED_MASK','1')=='1':
        native_out,native_lse=attention_grouped_mask(qh,kh,vh,mask,query_groups,scale=reader.native_dim**-.5)
    else:
        native_out, native_lse = attention_lse(qh, kh, vh, mask, scale=reader.native_dim ** -0.5)
    native_out = output_native(native_out)

    q_pre_h = q_pre[:, mosaic_tokens:].reshape(q_pre.shape[0], -1, num_heads, reader.native_dim).permute(0, 2, 1, 3)
    q_freqs = freqs_cur[mosaic_tokens:]
    if self_attn.use_prope and camera_info is not None and self_attn.prope_disable_t_rope:
        q_freqs = q_freqs.clone(); q_freqs[..., :self_attn.rope_t_pairs] = 1
    P = P_T = P_inv = None
    if camera_info is not None:
        P, P_T, P_inv = _viewmats(
            camera_info, _expand_frames(cur_frames, int(q_pre.shape[1]-mosaic_tokens))
        )
    result = torch.zeros_like(qh)
    for group in reader.groups:
        heads = group.heads
        query = group.query(q_pre_h.index_select(1, heads))
        if not (self_attn.use_prope and camera_info is not None and self_attn.prope_disable_native_rope):
            query = rope_selected(query, q_freqs, group.indices)
        if P_T is not None:
            query = camera_selected(query, P_T, group.indices)
        out = native_out.index_select(1, heads); lse = native_lse.index_select(1, heads)
        for segment in low_segments:
            if segment.get('native',False):
                kp=segment['k_pre'].reshape(q_pre.shape[0],-1,num_heads,reader.native_dim).permute(0,2,1,3).index_select(1,heads)
                vp=segment['v_pre'].reshape(q_pre.shape[0],-1,num_heads,reader.native_dim).permute(0,2,1,3).index_select(1,heads)
                kk=torch.einsum('bhnd,hde->bhne',kp,group.k_affine.to(q_pre.dtype))
                vv=torch.einsum('bhnd,hde->bhne',vp,group.v_affine.to(q_pre.dtype))
            else:
                kk = group.read(*segment["k"], target="k", dtype=q_pre.dtype)
                vv = group.read(*segment["v"], target="v", dtype=q_pre.dtype)
            if not (self_attn.use_prope and camera_info is not None and self_attn.prope_disable_native_rope):
                low_freqs = segment["freqs"]
                if self_attn.use_prope and camera_info is not None and self_attn.prope_disable_t_rope:
                    low_freqs = low_freqs.clone(); low_freqs[..., :self_attn.rope_t_pairs] = 1
                kk = rope_selected(kk, low_freqs, group.indices)
            if camera_info is not None:
                _, _, low_inv = _viewmats(camera_info, segment["frames"])
                kk = camera_selected(kk, low_inv, group.indices)
                vv = camera_selected(vv, low_inv, group.v_indices)
            if segment.get('mask') is not None:
                oo,ll=attention_grouped_mask(query,kk,vv,segment['mask'],query_groups,scale=reader.native_dim**-.5)
            else:
                oo, ll = attention_lse(query, kk, vv, scale=reader.native_dim ** -0.5)
            if P is not None:
                oo = camera_selected(oo, P, group.v_indices)
            out, lse = merge(out, lse, group.lift_output(oo), ll)
        result.index_copy_(1, heads, out)
    cur_out = result.permute(0, 2, 1, 3).flatten(2)
    out = torch.zeros_like(q_native); out[:, mosaic_tokens:] = cur_out
    if write_cache and cache is not None:
        _commit(
            cache, cur_k_pre, v_raw[:, mosaic_tokens:],
            layer_idx=layer_idx, cur_frames=cur_frames, cur_positions=cur_positions,
            cur_cache_chunk_ids=cur_cache_chunk_ids,
        )
    return self_attn.o(out)
