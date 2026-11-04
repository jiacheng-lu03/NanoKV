"""Paper-faithful INT2 KV cache for Taylor/Jensen score correction.

This adapter implements the RTN configuration from "Quantized Keys Steal
Attention" on HY-WorldPlay's shared *pre-position* visual KV cache:

* per-token, per-head, asymmetric INT2;
* groups of 32 channels;
* FP8-E4M3 step sizes and BF16 zero points;
* four INT2 values physically packed into one uint8;
* current-chunk and text K/V are never quantized by this adapter.

The key step sizes remain resident because the attention kernel consumes them
to compute the Taylor correction.  No learned state or calibration data is
used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


_CODEC = "jensen_rtn_int2_v1"
_BITS = 2
_GROUP_SIZE = 32
_STATS = {
    "calls": 0,
    "original_by_layer": {},
    "compressed_by_layer": {},
    "fallback_count": 0,
}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on",
    }


def enabled() -> bool:
    return _flag("HY_JENSEN_KV_ENABLE")


def quarot_enabled() -> bool:
    return _flag("HY_JENSEN_QUAROT")


def _rotation_signs(
    channels: int, layer_idx: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Regenerate a fixed randomized-Hadamard sign vector without cache state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x514B0000 + int(layer_idx))
    signs = torch.randint(0, 2, (channels,), generator=generator, dtype=torch.int8)
    return signs.to(device=device, dtype=dtype).mul_(2).sub_(1)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last dimension."""
    channels = int(x.shape[-1])
    if channels <= 0 or channels & (channels - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {channels}")
    work = x.float()
    width = 1
    while width < channels:
        paired = work.reshape(*work.shape[:-1], -1, 2 * width)
        left, right = paired[..., :width], paired[..., width:]
        work = torch.cat((left + right, left - right), dim=-1).reshape_as(work)
        width *= 2
    return work.mul_(channels ** -0.5)


def randomized_hadamard(
    x: torch.Tensor, layer_idx: int, *, inverse: bool = False
) -> torch.Tensor:
    """Apply H D (forward) or D H (inverse), preserving the input dtype."""
    signs = _rotation_signs(x.shape[-1], layer_idx, x.device, torch.float32)
    if inverse:
        result = _fwht(x).mul_(signs)
    else:
        result = _fwht(x.float() * signs)
    return result.to(dtype=x.dtype)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _pack_int2(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] % 4:
        raise ValueError("INT2 packing requires a last dimension divisible by four")
    q = q.to(torch.uint8)
    return (
        q[..., 0::4]
        | (q[..., 1::4] << 2)
        | (q[..., 2::4] << 4)
        | (q[..., 3::4] << 6)
    ).contiguous()


def _unpack_int2(packed: torch.Tensor, channels: int) -> torch.Tensor:
    parts = tuple((packed >> shift) & 0x3 for shift in (0, 2, 4, 6))
    return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], channels)


@torch.no_grad()
def quantize_tensor(
    x: torch.Tensor,
    *,
    target: str,
    layer_idx: int | None = None,
    rotate_key: bool = False,
) -> dict[str, Any]:
    if x.ndim != 4:
        raise ValueError(f"expected [B,H,N,D], got {tuple(x.shape)}")
    channels = int(x.shape[-1])
    if rotate_key:
        if target != "k" or layer_idx is None:
            raise ValueError("QuaRot is defined only for K and requires layer_idx")
        x = randomized_hadamard(x, int(layer_idx))
    if channels % _GROUP_SIZE:
        raise ValueError(
            f"head dimension {channels} is not divisible by group size {_GROUP_SIZE}"
        )
    groups = channels // _GROUP_SIZE
    grouped = x.float().reshape(*x.shape[:-1], groups, _GROUP_SIZE)
    lower = grouped.amin(dim=-1)
    upper = grouped.amax(dim=-1)
    step = ((upper - lower) / 3.0).clamp_min(torch.finfo(torch.float32).eps)

    # The stored FP8 step is the actual grid used for RTN, rather than merely a
    # post-hoc accounting copy of a higher precision grid.
    step_fp8 = step.to(torch.float8_e4m3fn)
    step_runtime = step_fp8.to(torch.float32).clamp_min(
        torch.finfo(torch.float32).eps
    )
    zero = torch.round(-lower / step_runtime).clamp_(0, 3)
    zero_bf16 = zero.to(torch.bfloat16)
    q = torch.round(grouped / step_runtime[..., None] + zero[..., None])
    q = q.clamp_(0, 3).to(torch.uint8).reshape_as(x)
    return {
        "codec": _CODEC,
        "target": str(target),
        "packed": _pack_int2(q),
        "step": step_fp8,
        "zero_point": zero_bf16,
        "shape": tuple(int(v) for v in x.shape),
        "bits": _BITS,
        "group_size": _GROUP_SIZE,
        "scale_dtype": "float8_e4m3fn",
        "zero_point_dtype": "bfloat16",
        "rotation": "randomized_hadamard" if rotate_key else "none",
        "layer_idx": int(layer_idx) if layer_idx is not None else None,
    }


def is_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") == _CODEC


@torch.no_grad()
def dequantize_tensor(
    payload: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if not is_payload(payload):
        raise TypeError("not a Jensen INT2 payload")
    shape = tuple(int(v) for v in payload["shape"])
    channels = shape[-1]
    q = _unpack_int2(payload["packed"].to(device), channels)
    groups = channels // int(payload["group_size"])
    q = q.reshape(*shape[:-1], groups, int(payload["group_size"]))
    step = payload["step"].to(device=device, dtype=torch.float32)
    zero = payload["zero_point"].to(device=device, dtype=torch.float32)
    restored = (q.float() - zero[..., None]) * step[..., None]
    restored = restored.reshape(shape).to(dtype=dtype)
    if payload.get("rotation") == "randomized_hadamard":
        restored = randomized_hadamard(
            restored, int(payload["layer_idx"]), inverse=True
        )
    return restored


def decompress_pair(
    key: Any, value: Any, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (is_payload(key) and is_payload(value)):
        return None
    return (
        dequantize_tensor(key, dtype, device),
        dequantize_tensor(value, dtype, device),
    )


@torch.no_grad()
def compress_layer_cache(layer_cache: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    if not enabled():
        return layer_cache
    cache_mode = layer_cache.get("vision_kv_position_mode")
    expected_mode = "post" if quarot_enabled() else "pre"
    if cache_mode != expected_mode:
        raise RuntimeError(
            f"Jensen {'QuaRot' if quarot_enabled() else 'RTN'} requires "
            f"{expected_mode}-position cache, got {cache_mode!r}"
        )
    if any(
        _flag(name)
        for name in (
            "HY_KV_CODEC_ENABLE",
            "HY_KV_AE3D_ENABLE",
            "HY_QVG_ENABLE",
            "HY_KVTC_ENABLE",
        )
    ):
        raise RuntimeError("Jensen baseline must be isolated from every other KV codec")
    key = layer_cache.get("k_vision")
    value = layer_cache.get("v_vision")
    if not (torch.is_tensor(key) and torch.is_tensor(value)):
        raise TypeError(
            f"Jensen baseline expected raw K/V at layer {layer_idx}, got "
            f"{type(key).__name__}/{type(value).__name__}"
        )
    original_bytes = _tensor_bytes(key) + _tensor_bytes(value)
    key_payload = quantize_tensor(
        key,
        target="k",
        layer_idx=layer_idx,
        rotate_key=quarot_enabled(),
    )
    value_payload = quantize_tensor(value, target="v", layer_idx=layer_idx)
    compressed_bytes = _tensor_bytes(key_payload) + _tensor_bytes(value_payload)
    layer_cache["k_vision"] = key_payload
    layer_cache["v_vision"] = value_payload
    layer_cache["jensen_quantized"] = True
    _STATS["calls"] += 1
    _STATS["original_by_layer"][int(layer_idx)] = int(original_bytes)
    _STATS["compressed_by_layer"][int(layer_idx)] = int(compressed_bytes)
    return layer_cache


def key_steps(layer_cache: dict[str, Any], device: torch.device) -> torch.Tensor:
    payload = layer_cache.get("k_vision")
    if not is_payload(payload) or payload.get("target") != "k":
        raise RuntimeError("Taylor correction requires a packed Jensen K payload")
    return payload["step"].to(device=device)


def write_summary() -> None:
    if not enabled():
        return
    path_value = os.environ.get("HY_JENSEN_KV_SUMMARY", "").strip()
    if not path_value:
        return
    original = sum(_STATS["original_by_layer"].values())
    compressed = sum(_STATS["compressed_by_layer"].values())
    output = {
        "enabled": True,
        "method": "Quantized Keys Steal Attention",
        "variant": "QuaRot_RTN_INT2_Taylor" if quarot_enabled() else "RTN_INT2_Taylor",
        "cache_position": "post" if quarot_enabled() else "pre",
        "bits": _BITS,
        "group_size": _GROUP_SIZE,
        "scale_dtype": "float8_e4m3fn",
        "zero_point_dtype": "bfloat16",
        "paper_effective_bits": _BITS + 24.0 / _GROUP_SIZE,
        "calls": int(_STATS["calls"]),
        "layers": len(_STATS["original_by_layer"]),
        "original_cache_bytes": int(original),
        "compressed_cache_bytes": int(compressed),
        "actual_cache_compression_ratio": (
            float(original / compressed) if compressed else 0.0
        ),
        "fallback_count": int(_STATS["fallback_count"]),
        "includes": [
            "packed_int2",
            "fp8_step",
            "bf16_zero_point",
            *(["static_randomized_hadamard"] if quarot_enabled() else []),
        ],
        "correction_storage_bytes": 0,
    }
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
