"""Frozen QVG and TurboQuant baselines for MG3.5 pre-RoPE KV history.

QVG calls the official Quant-VideoGen implementation. TurboQuant is the same
uniform three-bit Algorithm-1/2 clean-room implementation used by WorldPlay
and LingBot. Payloads, rather than reconstructed BF16 tensors, are resident.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

QVG_CODEC = "mg35_qvg_official_pre_v1"
TURBO_CODEC = "mg35_turboquant_uniform3_pre_v1"
BITS = 3
K_MSE_BITS = 2
_OFFICIAL_API = None
_MATRICES = {}
_CODEBOOKS = {}


def enabled_mode() -> str:
    return os.environ.get("MG35_KVCODEC_MODE", "").strip().lower()


def is_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") in {QVG_CODEC, TURBO_CODEC}


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def value_shape(value: dict) -> tuple[int, int, int]:
    return tuple(int(v) for v in value["shape"])


def _heads() -> int:
    return int(os.environ.get("MG35_BASELINE_NUM_HEADS", "24"))


def _to_heads(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"MG3.5 baseline expects [B,S,C], got {tuple(value.shape)}")
    b, s, c = value.shape
    heads = _heads()
    if c % heads:
        raise ValueError(f"cache width {c} is not divisible by {heads} heads")
    return value.reshape(b, s, heads, c // heads)


def _from_heads(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], value.shape[1], -1).contiguous()


def _qvg_config() -> SimpleNamespace:
    config = SimpleNamespace(
        quant_type=os.environ.get("MG35_QVG_QUANT_TYPE", "triton-nstages-kmeans-int2"),
        quant_block_size=int(os.environ.get("MG35_QVG_BLOCK_SIZE", "64")),
        cache_num_k_centroids=int(os.environ.get("MG35_QVG_K_CENTROIDS", "256")),
        cache_num_v_centroids=int(os.environ.get("MG35_QVG_V_CENTROIDS", "256")),
        kmeans_max_iters=int(os.environ.get("MG35_QVG_KMEANS_ITERS", "2")),
        num_prq_stages=int(os.environ.get("MG35_QVG_PRQ_STAGES", "1")),
    )
    if config.quant_type != "triton-nstages-kmeans-int2":
        raise ValueError("formal MG3.5 QVG baseline is frozen to official INT2")
    if max(config.cache_num_k_centroids, config.cache_num_v_centroids) > 256:
        raise ValueError("official QVG assignment storage is uint8")
    return config


def _official_api():
    global _OFFICIAL_API
    if _OFFICIAL_API is not None:
        return _OFFICIAL_API
    checkout = os.environ.get("MG35_QVG_ROOT", "external/Quant-VideoGen")
    if checkout and checkout not in sys.path:
        sys.path.insert(0, checkout)
    import triton
    original = triton.autotune

    def compatible_autotune(*args, **kwargs):
        if args:
            return original(*args, **kwargs)
        configs, keys = kwargs.pop("configs"), list(kwargs.pop("key"))
        def decorate(fn):
            return original(
                configs=configs, key=[key for key in keys if key in fn.arg_names], **kwargs
            )(fn)
        return decorate

    triton.autotune = compatible_autotune
    try:
        if "loguru" not in sys.modules:
            try:
                __import__("loguru")
            except ModuleNotFoundError:
                shim = types.ModuleType("loguru")
                shim.logger = logging.getLogger("quant_videogen")
                sys.modules["loguru"] = shim
        from quant_videogen.compress import compress_kv_cache, get_quantize_fn
        from quant_videogen.uncompress import uncompress_kv_cache
    finally:
        triton.autotune = original
    _OFFICIAL_API = compress_kv_cache, get_quantize_fn, uncompress_kv_cache
    return _OFFICIAL_API


@torch.no_grad()
def encode_qvg_pair(key: torch.Tensor, value: torch.Tensor, *, layer_idx: int):
    if key.shape != value.shape:
        raise ValueError("QVG K/V shapes differ")
    config = _qvg_config()
    compress, get_quantize_fn, _ = _official_api()
    key_h = _to_heads(key).permute(0, 2, 1, 3).contiguous()
    value_h = _to_heads(value).permute(0, 2, 1, 3).contiguous()
    key_q, value_q = compress(
        key_h, value_h, config.quant_type, config,
        get_quantize_fn(config.quant_type, config),
    )
    if not isinstance(key_q, dict) or not isinstance(value_q, dict):
        raise RuntimeError("official QVG did not return packed dictionaries")
    key_q["info"] = {"output_dtype": key.dtype, "quant_config": config}
    value_q["info"] = {"output_dtype": value.dtype, "quant_config": config}
    base = {"layer_idx": int(layer_idx), "shape": tuple(int(v) for v in key.shape)}
    return (
        {"codec": QVG_CODEC, "target": "k", "payload": key_q, **base},
        {"codec": QVG_CODEC, "target": "v", "payload": value_q, **base},
    )


@torch.no_grad()
def _decode_qvg(value: dict, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    _, _, uncompress = _official_api()
    # The official API accepts a pair. Supplying the same packed target twice
    # avoids retaining or duplicating its counterpart in each cache entry.
    decoded, _ = uncompress(value["payload"], value["payload"])
    return _from_heads(decoded.permute(0, 2, 1, 3)).to(device=device, dtype=dtype)


@torch.no_grad()
def _decode_qvg_pair(
    key: dict, value: dict, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the official paired API once instead of decoding K and V twice."""
    _, _, uncompress = _official_api()
    decoded_k, decoded_v = uncompress(key["payload"], value["payload"])
    return (
        _from_heads(decoded_k.permute(0, 2, 1, 3)).to(device=device, dtype=dtype),
        _from_heads(decoded_v.permute(0, 2, 1, 3)).to(device=device, dtype=dtype),
    )


@lru_cache(maxsize=None)
def _lloyd_max(dim: int, bits: int):
    grid = np.linspace(-1.0 + 1e-7, 1.0 - 1e-7, 200_001, dtype=np.float64)
    log_pdf = ((dim - 3.0) / 2.0) * np.log1p(-(grid * grid))
    log_pdf -= log_pdf.max()
    weights = np.exp(log_pdf); weights /= weights.sum()
    cdf = np.cumsum(weights)
    count = 1 << bits
    centers = grid[np.searchsorted(cdf, (np.arange(count) + 0.5) / count)]
    for _ in range(256):
        boundaries = 0.5 * (centers[:-1] + centers[1:])
        labels = np.searchsorted(boundaries, grid)
        updated = centers.copy()
        for index in range(count):
            mask = labels == index
            if weights[mask].sum() > 0:
                updated[index] = np.sum(grid[mask] * weights[mask]) / weights[mask].sum()
        if np.max(np.abs(updated - centers)) < 1e-13:
            centers = updated; break
        centers = updated
    return centers.astype(np.float32), (0.5 * (centers[:-1] + centers[1:])).astype(np.float32)


def _codebook(dim: int, bits: int, device: torch.device):
    key = dim, bits, str(device)
    if key not in _CODEBOOKS:
        c, b = _lloyd_max(dim, bits)
        _CODEBOOKS[key] = torch.from_numpy(c).to(device), torch.from_numpy(b).to(device)
    return _CODEBOOKS[key]


def _matrices(layer_idx: int, dim: int, device: torch.device):
    key = layer_idx, dim, str(device)
    if key not in _MATRICES:
        gen = torch.Generator(device="cpu"); gen.manual_seed(0x54510000 + layer_idx)
        q, r = torch.linalg.qr(torch.randn((dim, dim), generator=gen, dtype=torch.float64))
        sign = torch.sign(torch.diagonal(r)); sign[sign == 0] = 1
        q = (q * sign[None, :]).float()
        gen = torch.Generator(device="cpu"); gen.manual_seed(0x54518000 + layer_idx)
        proj = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
        _MATRICES[key] = q.to(device), proj.to(device)
    return _MATRICES[key]


def _pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    channels = indices.shape[-1]; groups = (channels + 7) // 8
    values = indices.to(torch.uint8)
    if groups * 8 != channels:
        values = torch.nn.functional.pad(values, (0, groups * 8 - channels))
    values = values.reshape(*values.shape[:-1], groups, 8).to(torch.int32)
    shifts = torch.arange(8, device=values.device, dtype=torch.int32) * bits
    words = torch.sum(values << shifts, dim=-1, dtype=torch.int32)
    return torch.stack([((words >> (8*i)) & 255).byte() for i in range(bits)], -1).reshape(
        *indices.shape[:-1], groups * bits
    ).contiguous()


def _nearest(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values, dtype=torch.uint8)
    for boundary in boundaries:
        result.add_(values > boundary)
    return result


def _lookup(indices: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(indices, dtype=torch.float32)
    for index, center in enumerate(centers):
        result.masked_fill_(indices == index, center)
    return result


@torch.no_grad()
def encode_turbo_tensor(tensor: torch.Tensor, *, target: str, layer_idx: int):
    headed = _to_heads(tensor)
    dim = headed.shape[-1]
    rotation, projection = _matrices(layer_idx, dim, tensor.device)
    norm = headed.float().norm(dim=-1).clamp_min(1e-12)
    unit = headed.float() / norm[..., None]
    mse_bits = K_MSE_BITS if target == "k" else BITS
    centers, boundaries = _codebook(dim, mse_bits, tensor.device)
    rotated = torch.matmul(unit, rotation.t())
    indices = _nearest(rotated, boundaries)
    payload = {
        "codec": TURBO_CODEC, "target": target, "layer_idx": int(layer_idx),
        "shape": tuple(int(v) for v in tensor.shape), "head_shape": tuple(headed.shape),
        "bits": BITS, "mse_bits": mse_bits, "indices": _pack(indices, mse_bits),
        "norms": norm.to(torch.bfloat16),
    }
    if target == "k":
        reconstructed = torch.matmul(_lookup(indices, centers), rotation)
        residual = headed.float() - reconstructed * norm[..., None]
        residual_norm = residual.norm(dim=-1).clamp_min(1e-12)
        payload["qjl_signs"] = _pack((torch.matmul(residual, projection.t()) >= 0).byte(), 1)
        payload["residual_norms"] = residual_norm.to(torch.bfloat16)
    return payload


@torch.no_grad()
def encode_turbo_pair(key: torch.Tensor, value: torch.Tensor, *, layer_idx: int):
    return encode_turbo_tensor(key, target="k", layer_idx=layer_idx), encode_turbo_tensor(
        value, target="v", layer_idx=layer_idx
    )


@torch.no_grad()
def _decode_turbo(value: dict, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from .training_free_kv_baselines_triton import unpack_lookup, unpack_signs
    shape = tuple(int(v) for v in value["head_shape"])
    dim, layer_idx = shape[-1], int(value["layer_idx"])
    rotation, projection = _matrices(layer_idx, dim, device)
    centers, _ = _codebook(dim, int(value["mse_bits"]), device)
    rotated = unpack_lookup(value["indices"].to(device), centers, int(value["mse_bits"]), shape)
    norm = value["norms"].to(device=device, dtype=torch.float32)
    restored = torch.matmul(rotated, rotation) * norm[..., None]
    if value["target"] == "k":
        signs = unpack_signs(value["qjl_signs"].to(device), shape)
        residual_norm = value["residual_norms"].to(device=device, dtype=torch.float32)
        restored.add_(torch.matmul(signs, projection) * (
            (math.sqrt(math.pi / 2.0) / dim) * residual_norm
        )[..., None])
    return _from_heads(restored).to(dtype=dtype)


@torch.no_grad()
def _decode_turbo_segments(
    parts: list[dict], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode immutable TurboQuant chunks with one rotation/projection GEMM."""
    first = parts[0]
    target, layer_idx = str(first["target"]), int(first["layer_idx"])
    batch, _, heads, dim = (int(v) for v in first["head_shape"])
    rotation, projection = _matrices(layer_idx, dim, device)
    centers, _ = _codebook(dim, int(first["mse_bits"]), device)
    rotated_parts = []
    norm_parts = []
    sign_parts = []
    residual_norm_parts = []
    for part in parts:
        shape = tuple(int(v) for v in part["head_shape"])
        if (
            part.get("codec") != TURBO_CODEC
            or str(part["target"]) != target
            or int(part["layer_idx"]) != layer_idx
            or (shape[0], shape[2], shape[3]) != (batch, heads, dim)
            or int(part["mse_bits"]) != int(first["mse_bits"])
        ):
            raise ValueError("incompatible TurboQuant stream segments")
        from .training_free_kv_baselines_triton import unpack_lookup, unpack_signs

        rotated_parts.append(
            unpack_lookup(
                part["indices"].to(device), centers, int(part["mse_bits"]), shape
            )
        )
        norm_parts.append(part["norms"].to(device=device, dtype=torch.float32))
        if target == "k":
            sign_parts.append(unpack_signs(part["qjl_signs"].to(device), shape))
            residual_norm_parts.append(
                part["residual_norms"].to(device=device, dtype=torch.float32)
            )
    rotated = torch.cat(rotated_parts, dim=1)
    norm = torch.cat(norm_parts, dim=1)
    restored = torch.matmul(rotated, rotation) * norm[..., None]
    if target == "k":
        signs = torch.cat(sign_parts, dim=1)
        residual_norm = torch.cat(residual_norm_parts, dim=1)
        restored.add_(
            torch.matmul(signs, projection)
            * ((math.sqrt(math.pi / 2.0) / dim) * residual_norm)[..., None]
        )
    return _from_heads(restored).to(dtype=dtype)


def decode_value(value: dict, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if value.get("codec") == QVG_CODEC:
        return _decode_qvg(value, dtype, device)
    if value.get("codec") == TURBO_CODEC:
        return _decode_turbo(value, dtype, device)
    raise ValueError(f"unknown training-free baseline payload {value.get('codec')}")


def decode_pair(
    key: dict, value: dict, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if key.get("codec") != value.get("codec"):
        raise ValueError("training-free K/V codecs differ")
    if key.get("codec") == QVG_CODEC:
        return _decode_qvg_pair(key, value, dtype, device)
    if key.get("codec") == TURBO_CODEC:
        return _decode_turbo(key, dtype, device), _decode_turbo(value, dtype, device)
    raise ValueError(f"unknown training-free baseline payload {key.get('codec')}")


def decode_segment_pairs(
    keys: list[dict],
    values: list[dict],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(keys) != len(values) or not keys:
        raise ValueError("invalid training-free segment pair")
    codec = keys[0].get("codec")
    if any(k.get("codec") != codec or v.get("codec") != codec for k, v in zip(keys, values)):
        raise ValueError("training-free segment codecs differ")
    if codec == QVG_CODEC:
        pairs = [
            _decode_qvg_pair(k, v, dtype, device) for k, v in zip(keys, values)
        ]
        return (
            torch.cat([pair[0] for pair in pairs], dim=1),
            torch.cat([pair[1] for pair in pairs], dim=1),
        )
    if codec == TURBO_CODEC:
        return (
            _decode_turbo_segments(keys, dtype, device),
            _decode_turbo_segments(values, dtype, device),
        )
    raise ValueError(f"unknown training-free baseline payload {codec}")


@torch.no_grad()
def encode_pair(mode: str, key: torch.Tensor, value: torch.Tensor, *, layer_idx: int):
    if mode == "qvg":
        return encode_qvg_pair(key, value, layer_idx=layer_idx)
    if mode == "turboquant":
        return encode_turbo_pair(key, value, layer_idx=layer_idx)
    raise ValueError(mode)


@torch.no_grad()
def encode_single(mode: str, tensor: torch.Tensor, *, target: str, layer_idx: int):
    if mode == "qvg":
        pair = encode_qvg_pair(tensor, tensor, layer_idx=layer_idx)
        return pair[0 if target == "k" else 1]
    if mode == "turboquant":
        return encode_turbo_tensor(tensor, target=target, layer_idx=layer_idx)
    raise ValueError(mode)


def resident_codec_bytes(mode: str, layers: int, head_dim: int = 128) -> int:
    if mode != "turboquant":
        return 0
    return layers * 2 * head_dim * head_dim * 4 + ((1 << BITS) + (1 << K_MSE_BITS)) * 4


def method_metadata(mode: str) -> dict[str, Any] | None:
    if mode == "qvg":
        config = _qvg_config()
        return {
            "backend": "official_quant_videogen",
            "cache_position": "pre_rope_k_raw_v",
            "quant_type": config.quant_type,
            "quant_block_size": config.quant_block_size,
            "k_centroids": config.cache_num_k_centroids,
            "v_centroids": config.cache_num_v_centroids,
            "kmeans_max_iters": config.kmeans_max_iters,
            "prq_stages": config.num_prq_stages,
            # The upstream default is FP8 E4M3, but A100 cannot compile the
            # corresponding Triton fp8e4nv kernel.  All three paper ports use
            # this same BF16 scale compatibility setting and account for its
            # two bytes per scale in the resident payload.
            "scale_dtype": "bf16_a100_compat",
            "upstream_commit": "d3219e2ef6f8d196c2634df0a27953399860fb28",
            "history_lifecycle": "immutable_incremental_segments",
            "boundary_compaction": "only_cut_segment_reencoded_once",
            "fallback_count": 0,
        }
    if mode == "turboquant":
        return {
            "backend": "paper_algorithms_1_2_cleanroom",
            "cache_position": "pre_rope_k_raw_v",
            "configuration": "uniform_3bit",
            "key_quantizer": "TurboQuant_prod",
            "value_quantizer": "TurboQuant_mse",
            "bits_k": BITS,
            "bits_v": BITS,
            "exact_int3_packing": True,
            "history_lifecycle": "immutable_incremental_segments",
            "operator_backend": "triton_decode_only",
            "historical_fp_buffer_tokens": 0,
            "fallback_count": 0,
        }
    return None
