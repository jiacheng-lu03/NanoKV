"""Diagnostic dump for codec-native mixed history/current attention training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .kv_codecs import _joint_semantic_smooth, unpack_quantized_codes
from .kv_codec_training import (
    _sample_indices,
    _select_hidden,
    _select_sequence_metadata,
    _select_tokens,
    maybe_decompress,
)


_DUMP_KEYS = set()


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _layers() -> set[int]:
    raw = os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_LAYERS", "13,40,53").strip()
    if raw.lower() == "all":
        return set(range(54))
    return {
        int(item) for item in os.environ.get(
            "HY_CODEC_NATIVE_MIXED_DUMP_LAYERS", "13,40,53"
        ).split(",") if item.strip()
    }


def _priority_sample(primary: torch.Tensor, length: int, count: int) -> torch.Tensor:
    """Keep attention-relevant indices, then fill the budget uniformly."""
    count = min(int(count), int(length))
    primary = torch.unique(primary[(primary >= 0) & (primary < length)], sorted=True)
    if primary.numel() >= count:
        take = torch.linspace(
            0, primary.numel() - 1, steps=count, device=primary.device
        ).round().long()
        return primary.index_select(0, take)
    uniform = _sample_indices(length, count, primary.device)
    fill = uniform[~torch.isin(uniform, primary)]
    return torch.cat((primary, fill[: count - primary.numel()])).sort().values


def _aligned_kv_indices(q_idx, history_thw, current_thw, history_length, current_length):
    """Include the same spatial sites as sampled queries across history time.

    WorldPlay's main RoPE attention is strongly local.  Pure linspace sampling
    often drops every spatially corresponding history key for a query, making
    the sampled training softmax unlike the production attention.
    """
    ht, hh, hw = (int(value) for value in history_thw)
    _, ch, cw = (int(value) for value in current_thw)
    if (hh, hw) != (ch, cw):
        return q_idx.new_empty(0), q_idx
    spatial = q_idx.remainder(ch * cw)
    times = torch.arange(ht, device=q_idx.device, dtype=q_idx.dtype)
    history = (times[:, None] * (hh * hw) + spatial[None, :]).reshape(-1)
    current = q_idx.clamp_max(current_length - 1)
    return history[history < history_length], current


def _select_formal_codec_payload(payload: dict, indices: torch.Tensor):
    """Select the actual first-pass formal INT4+C64 cache representation."""
    if not isinstance(payload, dict) or payload.get("codec") != "global_pca":
        return None
    if payload.get("coeff_quant") != "int4" or not payload.get("semantic_smooth"):
        return None
    codes = unpack_quantized_codes(
        payload["latent_int8"], 4, int(payload["latent_numel"]),
        tuple(payload["latent_shape"]),
    )
    selected_codes = codes.index_select(1, indices)
    selected_scale = payload["latent_scale"].index_select(1, indices)
    selected_ids = payload["semantic_ids"].index_select(1, indices)
    return {
        "dequant": (selected_codes.float() * selected_scale.float()).cpu().half(),
        "semantic_centers": payload["semantic_centers"].detach().cpu().half(),
        "semantic_ids": selected_ids.detach().cpu().to(torch.uint8),
    }


def _select_full_semantic_payload(x: torch.Tensor, layer_idx: int, indices: torch.Tensor):
    """Run production C64 on the full history, then select training tokens."""
    residual, centers, ids = _joint_semantic_smooth(x, layer_idx)
    return {
        "residual": residual.index_select(2, indices).detach().cpu().half(),
        "semantic_centers": centers.detach().cpu().half(),
        "semantic_ids": ids.index_select(1, indices).detach().cpu().to(torch.uint8),
    }


@torch.no_grad()
def maybe_dump_mixed_history_sample(
    *,
    layer_idx: int,
    layer_cache: dict,
    current_q_pre: torch.Tensor,
    current_k_pre: torch.Tensor,
    current_v_pre: torch.Tensor,
    current_q_main: torch.Tensor,
    current_k_main: torch.Tensor,
    current_v_main: torch.Tensor,
    current_q_prope: torch.Tensor,
    current_k_prope: torch.Tensor,
    current_v_prope: torch.Tensor,
    attention_main: torch.Tensor,
    attention_prope_restored: torch.Tensor,
    layer_input: torch.Tensor,
    layer_output: torch.Tensor,
    vec: torch.Tensor,
    current_freqs_cis,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor,
    current_thw,
) -> None:
    if not _enabled("HY_CODEC_NATIVE_MIXED_DUMP_ENABLE"):
        return
    if int(layer_idx) not in _layers():
        return
    if layer_cache.get("vision_kv_position_mode") != "pre":
        return
    case_id = os.environ.get("HY_KV_AE3D_CASE_ID", "unknown")
    split = os.environ.get("HY_KV_AE3D_SPLIT", "unknown")
    chunk_id = int(os.environ.get("HY_KV_AE3D_AR_CHUNK_ID", "-1"))
    requested = os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_AR_CHUNKS", "2").strip()
    if requested and chunk_id not in {int(item) for item in requested.split(",") if item.strip()}:
        return
    key = (case_id, chunk_id, int(layer_idx))
    if key in _DUMP_KEYS:
        return
    _DUMP_KEYS.add(key)

    restore_device = current_q_pre.device
    restore_dtype = current_q_pre.dtype
    formal_k_payload = layer_cache["k_vision"]
    formal_v_payload = layer_cache["v_vision"]
    history_k = maybe_decompress(formal_k_payload, restore_dtype, restore_device)
    history_v = maybe_decompress(formal_v_payload, restore_dtype, restore_device)
    # Resident pre-position tensors use [B,H,S,D]; dump helpers consume [B,S,H,D].
    history_k_bshd = history_k.transpose(1, 2).contiguous()
    history_v_bshd = history_v.transpose(1, 2).contiguous()
    q_idx = _sample_indices(
        current_q_pre.shape[1],
        int(os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_Q_TOKENS", "256")),
        restore_device,
    )
    history_idx = _sample_indices(
        history_k_bshd.shape[1],
        int(os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_HISTORY_TOKENS", "1024")),
        restore_device,
    )
    current_idx = _sample_indices(
        current_k_pre.shape[1],
        int(os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_CURRENT_TOKENS", "256")),
        restore_device,
    )
    if _enabled("HY_CODEC_NATIVE_MIXED_DUMP_ALIGNED_TOKENS"):
        aligned_history, aligned_current = _aligned_kv_indices(
            q_idx, layer_cache["vision_thw"], current_thw,
            history_k_bshd.shape[1], current_k_pre.shape[1],
        )
        history_idx = _priority_sample(
            aligned_history, history_k_bshd.shape[1],
            int(os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_HISTORY_TOKENS", "1024")),
        )
        current_idx = _priority_sample(
            aligned_current, current_k_pre.shape[1],
            int(os.environ.get("HY_CODEC_NATIVE_MIXED_DUMP_CURRENT_TOKENS", "256")),
        )
    history_freqs = layer_cache.get("vision_freqs_cis")
    history_cos, history_sin = history_freqs if isinstance(history_freqs, tuple) else (None, None)
    current_cos, current_sin = current_freqs_cis if isinstance(current_freqs_cis, tuple) else (None, None)
    out_dir = Path(os.environ.get(
        "HY_CODEC_NATIVE_MIXED_DUMP_DIR", "kv_ae_outputs/codec_native_mixed_samples"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "dump_kind": "codec_native_mixed_history_current",
        "case_id": case_id,
        "split": split,
        "ar_chunk_id": chunk_id,
        "layer_idx": int(layer_idx),
        "history_thw": tuple(int(v) for v in layer_cache["vision_thw"]),
        "current_thw": tuple(int(v) for v in current_thw),
        "q_indices": q_idx.cpu(),
        "history_indices": history_idx.cpu(),
        "current_indices": current_idx.cpu(),
        "history_k_pre": _select_tokens(history_k_bshd, history_idx).cpu().half(),
        "history_v_pre": _select_tokens(history_v_bshd, history_idx).cpu().half(),
        "current_q_pre": _select_tokens(current_q_pre, q_idx).cpu().half(),
        "current_k_pre": _select_tokens(current_k_pre, current_idx).cpu().half(),
        "current_v_pre": _select_tokens(current_v_pre, current_idx).cpu().half(),
        "current_q_main": _select_tokens(current_q_main, q_idx).cpu().half(),
        "current_k_main": _select_tokens(current_k_main, current_idx).cpu().half(),
        "current_v_main": _select_tokens(current_v_main, current_idx).cpu().half(),
        "current_q_prope": _select_tokens(current_q_prope, q_idx).cpu().half(),
        "current_k_prope": _select_tokens(current_k_prope, current_idx).cpu().half(),
        "current_v_prope": _select_tokens(current_v_prope, current_idx).cpu().half(),
        "teacher_attention_main": _select_hidden(attention_main, q_idx),
        "teacher_attention_prope": _select_hidden(attention_prope_restored, q_idx),
        "layer_input": _select_hidden(layer_input, q_idx),
        "layer_output": _select_hidden(layer_output, q_idx),
        "vec": _select_hidden(vec, q_idx),
        "text_k": layer_cache["k_txt"].detach().cpu().half(),
        "text_v": layer_cache["v_txt"].detach().cpu().half(),
        "history_freqs_cos": _select_sequence_metadata(history_cos, history_idx),
        "history_freqs_sin": _select_sequence_metadata(history_sin, history_idx),
        "current_freqs_cos_q": _select_sequence_metadata(current_cos, q_idx),
        "current_freqs_sin_q": _select_sequence_metadata(current_sin, q_idx),
        "history_viewmats": _select_sequence_metadata(layer_cache.get("vision_viewmats"), history_idx),
        "history_Ks": _select_sequence_metadata(layer_cache.get("vision_Ks"), history_idx),
        "current_viewmats_q": _select_sequence_metadata(current_viewmats, q_idx),
        "current_Ks_q": _select_sequence_metadata(current_Ks, q_idx),
    }
    selected_formal_k = _select_formal_codec_payload(formal_k_payload, history_idx)
    selected_formal_v = _select_formal_codec_payload(formal_v_payload, history_idx)
    if selected_formal_k is not None and selected_formal_v is not None:
        payload["history_k_formal_payload"] = selected_formal_k
        payload["history_v_formal_payload"] = selected_formal_v
    elif _enabled("HY_CODEC_NATIVE_MIXED_DUMP_FULL_C64"):
        payload["history_k_semantic_payload"] = _select_full_semantic_payload(
            history_k, layer_idx, history_idx
        )
        payload["history_v_semantic_payload"] = _select_full_semantic_payload(
            history_v, layer_idx, history_idx
        )
    path = out_dir / f"{case_id}_chunk{chunk_id:02d}_layer{layer_idx:02d}_mixed.pt"
    torch.save(payload, path)
    print(
        f"[CodecNativeMixedDump] saved={path} q={q_idx.numel()} "
        f"history={history_idx.numel()} current={current_idx.numel()}",
        flush=True,
    )
