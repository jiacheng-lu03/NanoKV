"""Paper-faithful TurboQuant baseline for HY-WorldPlay pre-position KV.

This adapter implements Algorithms 1 and 2 from
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate".
It deliberately does not import the third-party 0xSero implementation.

The single reported configuration is uniform 3-bit TurboQuant:

* K uses TurboQuant_prod: 2-bit MSE indices plus a 1-bit QJL residual.
* V uses TurboQuant_mse with exact 3-bit index packing.
* Every historical visual token is compressed.  Text and current-chunk KV
  never pass through this adapter.
* Canonical pre-position K/V are reconstructed before the model's native
  Main-RoPE and ProPE transforms.

The paper's printed mixed-precision "2.5-bit" example is arithmetically
ambiguous, so it is not used here.  Uniform 3-bit makes the baseline fully
specified and reproducible from Algorithms 1 and 2 alone.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CODEC = "turboquant_uniform3_pre_v1"
_BITS = 3
_MSE_BITS_K = _BITS - 1
_MATRIX_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_DEVICE_CODEBOOK_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_STATS = {
    "calls": 0,
    "fallback_count": 0,
    "original_by_layer": {},
    "compressed_by_layer": {},
    "layers_seen": set(),
}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on",
    }


def enabled() -> bool:
    return _flag("HY_TURBOQUANT_ENABLE")


def triton_enabled(value: torch.Tensor | None = None) -> bool:
    if not _flag("HY_TURBOQUANT_TRITON", "0"):
        return False
    return value is None or bool(value.is_cuda)


def _triton_stage_enabled(stage: str, value: torch.Tensor) -> bool:
    name = f"HY_TURBOQUANT_TRITON_{stage.upper()}"
    if name in os.environ:
        return _flag(name)
    return triton_enabled(value)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


@lru_cache(maxsize=None)
def _lloyd_max_codebook(dim: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Numerically solve the paper's one-dimensional Beta Lloyd-Max problem."""
    if dim < 3 or bits < 1 or bits > 4:
        raise ValueError(f"unsupported TurboQuant codebook d={dim}, bits={bits}")

    # A dense symmetric grid is sufficient for d=128 and b<=3, while avoiding
    # a runtime dependency on the third-party implementation or its artifacts.
    grid = np.linspace(-1.0 + 1e-7, 1.0 - 1e-7, 200_001, dtype=np.float64)
    exponent = (dim - 3.0) / 2.0
    log_pdf = exponent * np.log1p(-(grid * grid))
    log_pdf -= log_pdf.max()
    weights = np.exp(log_pdf)
    weights /= weights.sum()
    cdf = np.cumsum(weights)

    clusters = 1 << bits
    quantiles = (np.arange(clusters, dtype=np.float64) + 0.5) / clusters
    centroids = grid[np.searchsorted(cdf, quantiles)]
    for _ in range(256):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        assignment = np.searchsorted(boundaries, grid)
        updated = centroids.copy()
        for index in range(clusters):
            mask = assignment == index
            mass = weights[mask].sum()
            if mass > 0:
                updated[index] = np.sum(grid[mask] * weights[mask]) / mass
        if np.max(np.abs(updated - centroids)) < 1e-13:
            centroids = updated
            break
        centroids = updated
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.astype(np.float32), boundaries.astype(np.float32)


def _codebook(
    dim: int, bits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(dim), int(bits), str(device))
    cached = _DEVICE_CODEBOOK_CACHE.get(key)
    if cached is None:
        centroids, boundaries = _lloyd_max_codebook(int(dim), int(bits))
        cached = (
            torch.from_numpy(centroids).to(device=device),
            torch.from_numpy(boundaries).to(device=device),
        )
        _DEVICE_CODEBOOK_CACHE[key] = cached
    return cached


def _random_matrices(
    layer_idx: int, dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded Haar rotation Pi and iid Gaussian QJL matrix S."""
    key = (int(layer_idx), int(dim), str(device))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x54510000 + int(layer_idx))
    gaussian = torch.randn((dim, dim), generator=generator, dtype=torch.float64)
    rotation, triangular = torch.linalg.qr(gaussian)
    diagonal_sign = torch.sign(torch.diagonal(triangular))
    diagonal_sign[diagonal_sign == 0] = 1
    rotation = (rotation * diagonal_sign[None, :]).float().contiguous()

    qjl_generator = torch.Generator(device="cpu")
    qjl_generator.manual_seed(0x54518000 + int(layer_idx))
    projection = torch.randn(
        (dim, dim), generator=qjl_generator, dtype=torch.float32
    ).contiguous()
    cached = (rotation.to(device=device), projection.to(device=device))
    _MATRIX_CACHE[key] = cached
    return cached


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack exactly ``bits`` per coordinate, including true cross-byte INT3."""
    if indices.dtype != torch.uint8:
        indices = indices.to(torch.uint8)
    channels = int(indices.shape[-1])
    groups = (channels + 7) // 8
    padded_channels = groups * 8
    if padded_channels != channels:
        indices = torch.nn.functional.pad(indices, (0, padded_channels - channels))
    values = indices.reshape(*indices.shape[:-1], groups, 8).to(torch.int32)
    shifts = torch.arange(8, device=indices.device, dtype=torch.int32) * int(bits)
    words = torch.sum(values << shifts, dim=-1, dtype=torch.int32)
    packed = torch.stack(
        tuple(((words >> (8 * offset)) & 0xFF).to(torch.uint8) for offset in range(bits)),
        dim=-1,
    )
    return packed.reshape(*indices.shape[:-1], groups * bits).contiguous()


def _unpack_indices(packed: torch.Tensor, bits: int, channels: int) -> torch.Tensor:
    groups = (int(channels) + 7) // 8
    expected = groups * int(bits)
    if int(packed.shape[-1]) != expected:
        raise RuntimeError(
            f"bad TurboQuant packed width {packed.shape[-1]}, expected {expected}"
        )
    byte_groups = packed.reshape(*packed.shape[:-1], groups, bits).to(torch.int32)
    words = torch.zeros(byte_groups.shape[:-1], device=packed.device, dtype=torch.int32)
    for offset in range(bits):
        words |= byte_groups[..., offset] << (8 * offset)
    mask = (1 << int(bits)) - 1
    values = torch.stack(
        tuple((words >> (int(bits) * offset)) & mask for offset in range(8)),
        dim=-1,
    )
    return values.reshape(*packed.shape[:-1], groups * 8)[..., :channels].to(torch.uint8)


def _nearest_indices(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    indices = torch.zeros(values.shape, device=values.device, dtype=torch.uint8)
    for boundary in boundaries:
        indices.add_(values > boundary)
    return indices


def _lookup_centroids(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # Avoid materializing an int64 index tensor the size of the full KV cache.
    output = torch.empty(indices.shape, device=indices.device, dtype=torch.float32)
    for index, centroid in enumerate(centroids):
        output.masked_fill_(indices == index, centroid)
    return output


def is_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") == _CODEC


@torch.no_grad()
def quantize_tensor(
    tensor: torch.Tensor, *, target: str, layer_idx: int
) -> dict[str, Any]:
    if tensor.ndim != 4:
        raise ValueError(f"TurboQuant expects [B,H,N,D], got {tuple(tensor.shape)}")
    if target not in {"k", "v"}:
        raise ValueError(f"invalid TurboQuant target {target!r}")
    dim = int(tensor.shape[-1])
    configured_dim = int(os.environ.get("HY_TURBOQUANT_HEAD_DIM", str(dim)))
    if dim != configured_dim:
        raise ValueError(f"runtime head_dim={dim} but configured={configured_dim}")

    rotation, projection = _random_matrices(int(layer_idx), dim, tensor.device)
    norm = tensor.float().norm(dim=-1).clamp_min(1e-12)
    unit = tensor.float() / norm[..., None]
    mse_bits = _MSE_BITS_K if target == "k" else _BITS
    centroids, boundaries = _codebook(dim, mse_bits, tensor.device)
    rotated = torch.matmul(unit, rotation.t())
    reconstructed_unit = None
    if _triton_stage_enabled("encode", rotated):
        from hyvideo.models.transformers.modules.turboquant_triton import (
            quantize_pack_lookup,
        )

        packed, reconstructed_unit = quantize_pack_lookup(
            rotated,
            centroids,
            boundaries,
            mse_bits,
            return_reconstructed=target == "k",
        )
    else:
        indices = _nearest_indices(rotated, boundaries)
        packed = _pack_indices(indices, mse_bits)

    payload: dict[str, Any] = {
        "codec": _CODEC,
        "target": target,
        "layer_idx": int(layer_idx),
        "shape": tuple(int(value) for value in tensor.shape),
        "bits": _BITS,
        "mse_bits": mse_bits,
        "indices": packed,
        "norms": norm.to(torch.bfloat16),
        "rotation_seed": 0x54510000 + int(layer_idx),
    }
    if target == "k":
        if reconstructed_unit is None:
            reconstructed_unit = _lookup_centroids(indices, centroids)
        reconstructed_unit = torch.matmul(reconstructed_unit, rotation)
        residual = tensor.float() - reconstructed_unit * norm[..., None]
        residual_norm = residual.norm(dim=-1).clamp_min(1e-12)
        projected_residual = torch.matmul(residual, projection.t())
        if _triton_stage_enabled("encode", projected_residual):
            from hyvideo.models.transformers.modules.turboquant_triton import (
                pack_signs,
            )

            packed_signs = pack_signs(projected_residual)
        else:
            packed_signs = _pack_indices(
                (projected_residual >= 0).to(torch.uint8), 1
            )
        payload.update(
            {
                "qjl_signs": packed_signs,
                "residual_norms": residual_norm.to(torch.bfloat16),
                "qjl_seed": 0x54518000 + int(layer_idx),
            }
        )
    return payload


@torch.no_grad()
def dequantize_tensor(
    payload: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if not is_payload(payload):
        raise TypeError("not a TurboQuant payload")
    shape = tuple(int(value) for value in payload["shape"])
    dim = shape[-1]
    layer_idx = int(payload["layer_idx"])
    rotation, projection = _random_matrices(layer_idx, dim, device)
    mse_bits = int(payload["mse_bits"])
    centroids, _ = _codebook(dim, mse_bits, device)
    packed_indices = payload["indices"].to(device)
    if _triton_stage_enabled("decode", packed_indices):
        from hyvideo.models.transformers.modules.turboquant_triton import (
            unpack_lookup,
        )

        rotated = unpack_lookup(packed_indices.contiguous(), centroids, mse_bits, shape)
    else:
        indices = _unpack_indices(packed_indices, mse_bits, dim)
        rotated = _lookup_centroids(indices, centroids)
    norm = payload["norms"].to(device=device, dtype=torch.float32)
    restored = torch.matmul(rotated, rotation) * norm[..., None]

    if payload["target"] == "k":
        packed_signs = payload["qjl_signs"].to(device)
        if _triton_stage_enabled("decode", packed_signs):
            from hyvideo.models.transformers.modules.turboquant_triton import (
                unpack_signs,
            )

            signs = unpack_signs(packed_signs.contiguous(), shape)
        else:
            signs = _unpack_indices(packed_signs, 1, dim).float().mul_(2).sub_(1)
        residual_norm = payload["residual_norms"].to(
            device=device, dtype=torch.float32
        )
        scale = math.sqrt(math.pi / 2.0) / float(dim)
        restored.add_(
            torch.matmul(signs, projection)
            * (scale * residual_norm)[..., None]
        )
    return restored.reshape(shape).to(dtype=dtype)


def decompress_pair(
    key: Any, value: Any, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (is_payload(key) and is_payload(value)):
        return None
    if key.get("target") != "k" or value.get("target") != "v":
        raise RuntimeError("TurboQuant payload targets are inconsistent")
    return (
        dequantize_tensor(key, dtype, device),
        dequantize_tensor(value, dtype, device),
    )


@torch.no_grad()
def compress_layer_cache(layer_cache: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    if not enabled():
        return layer_cache
    if layer_cache.get("vision_kv_position_mode") != "pre":
        raise RuntimeError("TurboQuant baseline requires canonical pre-position KV")
    incompatible = (
        "HY_KV_CODEC_ENABLE",
        "HY_KV_AE3D_ENABLE",
        "HY_PREKV_AE_ENABLE",
        "HY_QVG_ENABLE",
        "HY_KVTC_ENABLE",
        "HY_JENSEN_KV_ENABLE",
    )
    active = [name for name in incompatible if _flag(name)]
    if active:
        raise RuntimeError(f"TurboQuant is mutually exclusive with {active}")
    key = layer_cache.get("k_vision")
    value = layer_cache.get("v_vision")
    if not (torch.is_tensor(key) and torch.is_tensor(value)):
        raise TypeError(
            f"TurboQuant expected raw tensor K/V at layer {layer_idx}, got "
            f"{type(key).__name__}/{type(value).__name__}"
        )

    original_bytes = _tensor_bytes(key) + _tensor_bytes(value)
    key_payload = quantize_tensor(key, target="k", layer_idx=int(layer_idx))
    value_payload = quantize_tensor(value, target="v", layer_idx=int(layer_idx))
    compressed_bytes = _tensor_bytes(key_payload) + _tensor_bytes(value_payload)
    layer_cache["k_vision"] = key_payload
    layer_cache["v_vision"] = value_payload
    layer_cache["turboquant_quantized"] = True
    _STATS["calls"] += 1
    _STATS["original_by_layer"][int(layer_idx)] = int(original_bytes)
    _STATS["compressed_by_layer"][int(layer_idx)] = int(compressed_bytes)
    _STATS["layers_seen"].add(int(layer_idx))
    return layer_cache


def write_summary() -> None:
    if not enabled():
        return
    path_value = os.environ.get("HY_TURBOQUANT_SUMMARY", "").strip()
    if not path_value:
        return
    original = sum(_STATS["original_by_layer"].values())
    compressed = sum(_STATS["compressed_by_layer"].values())
    layers = len(_STATS["layers_seen"])
    dim = int(os.environ.get("HY_TURBOQUANT_HEAD_DIM", "128"))
    # One FP32 Haar rotation and one FP32 Gaussian QJL matrix per layer.
    resident_matrices = layers * 2 * dim * dim * 4
    codebook_bytes = ((1 << _BITS) + (1 << _MSE_BITS_K)) * 4
    resident_bytes = resident_matrices + codebook_bytes
    total = compressed + resident_bytes
    output = {
        "enabled": True,
        "backend": "paper_algorithms_1_2_cleanroom",
        "cache_position": "pre",
        "configuration": "uniform_3bit",
        "key_quantizer": "TurboQuant_prod",
        "value_quantizer": "TurboQuant_mse",
        "bits_k": _BITS,
        "bits_v": _BITS,
        "exact_int3_packing": True,
        "historical_fp_buffer_tokens": 0,
        "operator_backend": (
            "triton_decode_only"
            if _flag("HY_TURBOQUANT_TRITON_DECODE")
            and not _flag("HY_TURBOQUANT_TRITON_ENCODE")
            else "pytorch_reference"
        ),
        "calls": int(_STATS["calls"]),
        "fallback_count": int(_STATS["fallback_count"]),
        "layers": layers,
        "original_cache_bytes": int(original),
        "compressed_token_payload_bytes": int(compressed),
        "resident_codec_bytes": int(resident_bytes),
        "compressed_including_resident_bytes": int(total),
        "token_payload_compression_ratio": (
            float(original / compressed) if compressed else 0.0
        ),
        "actual_compression_ratio_including_resident": (
            float(original / total) if total else 0.0
        ),
        "includes": [
            "exact_packed_indices",
            "bf16_vector_norms",
            "bf16_qjl_residual_norms",
            "qjl_sign_bits",
            "fp32_rotation_and_projection_matrices",
        ],
    }
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
