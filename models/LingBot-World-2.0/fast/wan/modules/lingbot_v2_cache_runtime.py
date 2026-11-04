"""Cache lifecycle shared by LingBot-World 2.0 codec and dump modes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .lingbot_v2_kv_codec import (
    decode_segments, decode_tensor, dump_paired_source, encode_pair,
)


_FULL_ROPE_INDICES: dict[tuple[str, int, int], torch.Tensor] = {}


@torch.amp.autocast("cuda", enabled=False)
def rope_pre_key_by_positions(
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    """Apply native 3-D RoPE to arbitrary absolute cache-token positions.

    Rolling attention orders the cache as ``sink + recent`` rather than one
    contiguous interval.  Absolute token positions therefore travel with the
    decoded pre-RoPE K instead of being reconstructed from a single start.
    """
    if key.ndim != 4 or key.shape[0] != 1:
        raise ValueError(f"expected [1,S,H,D], got {tuple(key.shape)}")
    if positions.ndim != 1 or positions.numel() != key.shape[1]:
        raise ValueError("RoPE position/K length mismatch")
    if os.environ.get("LINGBOT_PRE_ROPE_TRITON", "0") == "1":
        # The native cache keeps [B, S, H, D], while the existing exact
        # Triton RoPE kernel consumes [B, H, S, D].  It performs the same
        # FP64 pair rotations as the reference path but avoids materializing
        # the full complex-double history and multiplier tensors.
        from .codec_native_rope_triton import rope_selected

        cache_key = (str(key.device), int(key.shape[2]), int(key.shape[3]))
        indices = _FULL_ROPE_INDICES.get(cache_key)
        if indices is None:
            indices = torch.arange(
                key.shape[3], device=key.device, dtype=torch.long
            )[None].expand(key.shape[2], -1)
            _FULL_ROPE_INDICES[cache_key] = indices
        return rope_selected(
            key.permute(0, 2, 1, 3), positions, grid_sizes, freqs,
            frame_seqlen, indices,
        ).permute(0, 2, 1, 3)
    _, height, width = map(int, grid_sizes[0].tolist())
    if height * width != int(frame_seqlen):
        raise ValueError(
            f"frame_seqlen mismatch: {height}x{width} != {frame_seqlen}"
        )
    half = key.shape[-1] // 2
    split = [half - 2 * (half // 3), half // 3, half // 3]
    temporal, vertical, horizontal = freqs.split(split, dim=1)
    positions = positions.to(device=key.device, dtype=torch.long)
    frame = torch.div(positions, frame_seqlen, rounding_mode="floor")
    spatial = positions.remainder(frame_seqlen)
    row = torch.div(spatial, width, rounding_mode="floor")
    column = spatial.remainder(width)
    multiplier = torch.cat(
        (temporal.index_select(0, frame), vertical.index_select(0, row),
         horizontal.index_select(0, column)), dim=1,
    )[:, None]
    complex_key = torch.view_as_complex(
        key[0].to(torch.float64).reshape(key.shape[1], key.shape[2], -1, 2)
    )
    restored = torch.view_as_real(complex_key * multiplier).flatten(2)
    return restored[None].to(dtype=key.dtype)


def _selected_dump_chunk(cache: dict[str, Any], chunk_id: int) -> bool:
    spec = str(cache.get("codec_dump_chunks", "")).strip().lower()
    if not spec:
        return False
    if spec == "all":
        return True
    return chunk_id in {int(x) for x in spec.split(",") if x.strip()}


@torch.no_grad()
def maybe_dump_committed_source(
    cache: dict[str, Any], key: torch.Tensor, value: torch.Tensor
) -> None:
    if not cache.get("codec_commit", False) or not cache.get("codec_dump_root"):
        return
    chunk_id = int(cache.get("codec_chunk_id", 0))
    if not _selected_dump_chunk(cache, chunk_id):
        return
    root = Path(cache["codec_dump_root"])
    case = str(cache.get("codec_case_id", "unknown"))
    layer = int(cache["codec_layer"])
    sp_rank = int(cache.get("codec_sp_rank", 0))
    for target, tensor in (("k", key), ("v", value)):
        dump_paired_source(
            tensor,
            path=(root / str(cache.get("codec_dump_split", "train")) / case
                  / f"chunk_{chunk_id:02d}" / f"rank_{sp_rank:02d}"
                  / f"layer_{layer:02d}_{target}.pt"),
            target=target, layer=layer, sp_rank=sp_rank, case_id=case,
            chunk_id=chunk_id,
            sample_tokens=int(cache.get("codec_dump_tokens", 1024)),
            clusters=int(cache.get("codec_clusters", 64)),
            iters=int(cache.get("codec_iters", 2)),
        )


def _intersections(start: int, end: int, intervals: list[tuple[int, int]]):
    for left, right in intervals:
        begin, stop = max(start, left), min(end, right)
        if begin < stop:
            yield begin, stop


def _decode_active_history(
    segments: list[dict[str, Any]],
    *,
    committed_end: int,
    capacity: int,
    sink_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not segments or committed_end <= 0:
        return None
    sink_stop = min(sink_tokens, committed_end)
    recent_start = max(sink_stop, committed_end - max(capacity - sink_stop, 0))
    intervals = [(0, sink_stop)]
    if recent_start < committed_end:
        intervals.append((recent_start, committed_end))
    k_parts, v_parts, position_parts = [], [], []
    active = []
    for segment in segments:
        start, end = int(segment["global_start"]), int(segment["global_end"])
        pieces = list(_intersections(start, end, intervals))
        if pieces:
            active.append((segment, pieces))
    if not active:
        return None
    a1 = None if cache is None else cache.get("a1_controller")
    # Frozen inference has one operator per layer/target.  Decode all active
    # immutable segments with one K and one V GEMM, then select the rolling
    # intervals.  A1 training retains its explicit Teacher/Student path.
    if a1 is None:
        decoded = decode_segments(
            [item[0] for item in active], dtype=dtype, device=device
        )
        offset = 0
        for segment, pieces in active:
            start, end = int(segment["global_start"]), int(segment["global_end"])
            for begin, stop in pieces:
                local = slice(offset + begin - start, offset + stop - start)
                k_parts.append(decoded[0][:, local])
                v_parts.append(decoded[1][:, local])
                position_parts.append(torch.arange(begin, stop, device=device))
            offset += end - start
        return (
            torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1),
            torch.cat(position_parts),
        )
    for segment, pieces in active:
        start, end = int(segment["global_start"]), int(segment["global_end"])
        a1 = None if cache is None else cache.get("a1_controller")
        layer = -1 if cache is None else int(cache.get("codec_layer", -1))
        if a1 is not None and a1.wants_clean_k(layer):
            key = segment.get("_a1_clean_k")
            if not torch.is_tensor(key):
                raise RuntimeError(f"A1 Teacher missing clean committed K at L{layer}")
            key = key.to(device=device, dtype=dtype)
        elif a1 is not None and a1.wants_trainable_k(layer):
            key = a1.decode_student(segment["k"], dtype=dtype, device=device)
        else:
            key = decode_tensor(segment["k"], dtype=dtype, device=device)
        value = decode_tensor(segment["v"], dtype=dtype, device=device)
        for begin, stop in pieces:
            local = slice(begin - start, stop - start)
            k_parts.append(key[:, local])
            v_parts.append(value[:, local])
            position_parts.append(torch.arange(begin, stop, device=device))
    if not k_parts:
        return None
    return (torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1),
            torch.cat(position_parts))


def _prune_segments(
    segments: list[dict[str, Any]], *, committed_end: int, capacity: int, sink_tokens: int
) -> list[dict[str, Any]]:
    sink_stop = min(sink_tokens, committed_end)
    recent_start = max(sink_stop, committed_end - max(capacity - sink_stop, 0))
    keep = []
    for segment in segments:
        start, end = int(segment["global_start"]), int(segment["global_end"])
        if start < sink_stop or end > recent_start:
            keep.append(segment)
    return keep


@torch.no_grad()
def pre_rope_clean_attention_inputs(
    cache: dict[str, Any], key: torch.Tensor, value: torch.Tensor, *,
    current_start: int, capacity: int, sink_tokens: int,
    max_attention_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native BF16 rolling cache with K stored before positional rotation."""
    current_end = current_start + int(key.shape[1])
    global_end = int(cache["global_end_index"].item())
    local_end = int(cache["local_end_index"].item())
    num_new = int(key.shape[1])
    if current_end > global_end and num_new + local_end > capacity:
        evicted = num_new + local_end - capacity
        rolled = local_end - evicted - sink_tokens
        if rolled > 0:
            cache["k"][:, sink_tokens : sink_tokens + rolled] = cache["k"][
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
            cache["v"][:, sink_tokens : sink_tokens + rolled] = cache["v"][
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
        local_end = local_end + current_end - global_end - evicted
    else:
        local_end = local_end + current_end - global_end
    local_start = local_end - num_new
    cache["k"][:, local_start:local_end] = key
    cache["v"][:, local_start:local_end] = value
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(local_end)

    if current_end <= capacity:
        positions = torch.arange(local_end, device=key.device)
    else:
        protected = min(sink_tokens, local_end)
        positions = torch.cat((
            torch.arange(protected, device=key.device),
            torch.arange(
                current_end - (local_end - protected), current_end,
                device=key.device,
            ),
        ))
    key_cache, value_cache = cache["k"][:, :local_end], cache["v"][:, :local_end]
    if local_end > max_attention_size:
        protected = min(sink_tokens, max_attention_size)
        tail = max_attention_size - protected
        if tail:
            key_cache = torch.cat((key_cache[:, :protected], key_cache[:, -tail:]), 1)
            value_cache = torch.cat((value_cache[:, :protected], value_cache[:, -tail:]), 1)
            positions = torch.cat((positions[:protected], positions[-tail:]))
        else:
            key_cache = key_cache[:, :protected]
            value_cache = value_cache[:, :protected]
            positions = positions[:protected]
    return key_cache, value_cache, positions


def codec_cache_attention_inputs(
    cache: dict[str, Any],
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    current_start: int,
    capacity: int,
    sink_tokens: int,
    max_attention_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return history+current K/V and commit final x0 when requested.

    ``key`` is pre-RoPE and ``value`` is raw projected V.  The function is
    called after Ulysses all-to-all, so its head dimension is either all 40
    heads (single rank) or the local shard.
    """
    maybe_dump_committed_source(cache, key, value)
    segments = cache.setdefault("codec_segments", [])
    committed_end = int(cache.get("codec_committed_end", 0))
    history = _decode_active_history(
        segments, committed_end=committed_end, capacity=capacity,
        sink_tokens=sink_tokens, dtype=value.dtype, device=value.device,
        cache=cache,
    )
    if history is None:
        key_cache, value_cache = key, value
        positions = torch.arange(
            current_start, current_start + key.shape[1], device=key.device
        )
    else:
        key_cache = torch.cat((history[0], key), dim=1)
        value_cache = torch.cat((history[1], value), dim=1)
        positions = torch.cat((
            history[2],
            torch.arange(current_start, current_start + key.shape[1], device=key.device),
        ))
    if key_cache.shape[1] > max_attention_size:
        # Match the native rolling cache: retain the attention sink and take
        # the newest non-sink rows, rather than blindly dropping the prefix.
        protected = min(sink_tokens, committed_end, max_attention_size)
        tail = max_attention_size - protected
        if tail:
            key_cache = torch.cat((key_cache[:, :protected], key_cache[:, -tail:]), dim=1)
            value_cache = torch.cat((value_cache[:, :protected], value_cache[:, -tail:]), dim=1)
            positions = torch.cat((positions[:protected], positions[-tail:]))
        else:
            key_cache = key_cache[:, :protected]
            value_cache = value_cache[:, :protected]
            positions = positions[:protected]

    current_end = current_start + int(key.shape[1])
    if cache.get("codec_commit", False):
        if current_start != committed_end:
            raise RuntimeError(
                f"non-causal codec commit: start={current_start}, committed={committed_end}"
            )
        cache["codec_commit_start"] = current_start
        encoded = encode_pair(key, value, cache=cache)
        a1 = cache.get("a1_controller")
        if a1 is not None and int(cache.get("codec_layer", -1)) in a1.layers:
            # Training-only Teacher source.  It is excluded from payload byte
            # accounting and never emitted by the inference artifact.
            encoded["_a1_clean_k"] = key.detach().clone()
        segments.append(encoded)
        cache["codec_committed_end"] = current_end
        cache["codec_segments"] = _prune_segments(
            segments, committed_end=current_end, capacity=capacity,
            sink_tokens=sink_tokens,
        )
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(min(current_end, capacity))
    return key_cache, value_cache, positions


def configure_cache(
    cache: dict[str, Any],
    *,
    layer: int,
    sp_rank: int,
    options: dict[str, Any],
) -> None:
    """Attach immutable runtime options without allocating a full BF16 cache."""
    cache.update(options)
    cache.update({
        "codec_layer": int(layer), "codec_sp_rank": int(sp_rank),
        "codec_segments": [], "codec_committed_end": 0,
        "baseline_segments": [], "baseline_committed_end": 0,
        "codec_commit": False, "codec_chunk_id": 0,
    })
