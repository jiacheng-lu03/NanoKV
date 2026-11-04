"""Optional full-history teacher capture for fitting codec-native attention later.

This records uncompressed inputs once, before the codec checkpoint exists. It
does not modify attention or the cache. The codec can be applied offline to
the saved history after fitting, avoiding a second model rollout.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


_COUNTS: dict[int, int] = {}


@torch.no_grad()
def capture(
    q_pre, k_pre, v_raw, freqs, camera_info, cache, cache_freqs,
    cache_frames, cur_frames, mosaic_frames, mosaic_tokens, layer_idx,
    cache_read_chunk_id, cur_cache_chunk_ids, hole_keep,
):
    destination = os.environ.get("MG35_READER_RAW_CAPTURE_DIR")
    if not destination:
        return
    if os.environ.get("MG35_READER_RAW_CAPTURE_SPLIT") not in {"train", "dev"}:
        raise RuntimeError("reader raw capture is restricted to train/dev")
    layer = int(layer_idx)
    _COUNTS[layer] = _COUNTS.get(layer, 0) + 1
    call = _COUNTS[layer]
    selected_calls = {
        int(item) for item in os.environ.get(
            "MG35_READER_RAW_CAPTURE_CALLS", "12,24,48"
        ).split(",") if item.strip()
    }
    if call not in selected_calls or not cache or cache.get("k") is None:
        return
    history_k = cache["k"]
    history_v = cache["v"]
    if not torch.is_tensor(history_k) or not torch.is_tensor(history_v):
        raise RuntimeError("raw reader capture requires uncompressed historical KV")
    frames = list(cache_frames or [])
    if not frames or history_k.shape[1] % len(frames):
        raise RuntimeError("historical token/frame alignment is invalid")
    device = q_pre.device
    tokens_per_frame = history_k.shape[1] // len(frames)
    chunk_ids = list(cache.get("chunk_ids", [-1] * len(frames)))
    if len(chunk_ids) != len(frames):
        raise RuntimeError("historical chunk/frame alignment is invalid")
    eligible_frames = [
        chunk < 0 or chunk == int(cache_read_chunk_id)
        for chunk in chunk_ids
    ] if cache_read_chunk_id is not None else [True] * len(frames)
    eligible = torch.tensor(eligible_frames, device=device).repeat_interleave(tokens_per_frame)
    historical_indices = eligible.nonzero().flatten()
    query_ids = torch.linspace(
        mosaic_tokens, q_pre.shape[1] - 1,
        min(64, q_pre.shape[1] - mosaic_tokens), device=device,
    ).long()

    def expand(frame_ids, count):
        frame_ids = list(frame_ids or [])
        if not frame_ids:
            return []
        if count % len(frame_ids):
            raise RuntimeError("current token/frame alignment is invalid")
        return [int(frame) for frame in frame_ids for _ in range(count // len(frame_ids))]

    native_frames = expand(mosaic_frames, mosaic_tokens) + expand(
        cur_frames, q_pre.shape[1] - mosaic_tokens
    )
    native_frames_tensor = torch.tensor(native_frames, device=device)
    historical_frames_tensor = torch.tensor(frames, device=device).repeat_interleave(tokens_per_frame)
    native_keep = torch.ones(len(query_ids), q_pre.shape[1], device=device, dtype=torch.bool)
    if hole_keep is not None:
        native_keep[:, :mosaic_tokens] = hole_keep[None]
    if cur_cache_chunk_ids is not None:
        current_chunks = torch.tensor(cur_cache_chunk_ids, device=device).repeat_interleave(
            (q_pre.shape[1] - mosaic_tokens) // len(cur_frames)
        )
        query_chunks = current_chunks[query_ids - mosaic_tokens]
        anchors = current_chunks < 0
        native_keep[:, mosaic_tokens:] &= torch.where(
            query_chunks[:, None] < 0,
            anchors[None],
            (query_chunks[:, None] == current_chunks[None]) | anchors[None],
        )
    result = {
        "format": "mg35_raw_reader_teacher_v1", "layer": layer, "call": call,
        "q_pre": q_pre[:, query_ids].cpu(),
        "q_freqs": freqs[query_ids].cpu(),
        "q_frames": native_frames_tensor[query_ids].cpu(),
        "history_k": history_k[:, historical_indices].cpu(),
        "history_v": history_v[:, historical_indices].cpu(),
        "history_freqs": cache_freqs[historical_indices].cpu(),
        "history_frames": historical_frames_tensor[historical_indices].cpu(),
        "native_k": k_pre.cpu(), "native_v": v_raw.cpu(),
        "native_freqs": freqs.cpu(),
        "native_frames": native_frames_tensor.cpu(),
        "native_keep": native_keep.cpu(),
        "mosaic_tokens": int(mosaic_tokens),
        "source_chunk_ids": chunk_ids,
    }
    if camera_info is not None:
        result["q_camera"] = [matrix.index_select(1, native_frames_tensor[query_ids]).cpu()
                              for matrix in camera_info[1]]
        result["history_camera"] = [matrix.index_select(1, historical_frames_tensor[historical_indices]).cpu()
                                    for matrix in camera_info[1]]
        result["native_camera"] = [matrix.index_select(1, native_frames_tensor).cpu()
                                   for matrix in camera_info[1]]
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, output / f"raw_layer{layer:02d}_call{call:03d}.pt")
