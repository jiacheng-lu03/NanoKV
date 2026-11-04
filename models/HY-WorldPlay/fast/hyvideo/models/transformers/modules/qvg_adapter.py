"""Optional QuantVideoGen adapter for the shared pre-position KV backend.

The quantization kernels are imported from the official Quant-VideoGen checkout.
This file only owns the HY-WorldPlay cache lifecycle and accounting glue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


_STATS = {
    "calls": 0,
    "fallback_count": 0,
    "original_by_layer": {},
    "compressed_by_layer": {},
}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on",
    }


def enabled() -> bool:
    return _flag("HY_QVG_ENABLE")


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _config() -> SimpleNamespace:
    # QVG paper/default open-source configuration: INT2, S=1, B=64, K=256.
    return SimpleNamespace(
        quant_type=os.environ.get(
            "HY_QVG_QUANT_TYPE", "triton-nstages-kmeans-int2"
        ),
        quant_block_size=int(os.environ.get("HY_QVG_BLOCK_SIZE", "64")),
        cache_num_k_centroids=int(os.environ.get("HY_QVG_K_CENTROIDS", "256")),
        cache_num_v_centroids=int(os.environ.get("HY_QVG_V_CENTROIDS", "256")),
        kmeans_max_iters=int(os.environ.get("HY_QVG_KMEANS_ITERS", "2")),
        num_prq_stages=int(os.environ.get("HY_QVG_PRQ_STAGES", "1")),
    )


def _validate_config(config: SimpleNamespace) -> None:
    if config.quant_type not in {
        "triton-nstages-kmeans-int2",
        "triton-nstages-kmeans-int4",
    }:
        raise ValueError(f"unsupported real QVG type: {config.quant_type}")
    if config.cache_num_k_centroids > 256 or config.cache_num_v_centroids > 256:
        raise ValueError("official QVG uint8 assignment storage requires K <= 256")


def compress_layer_cache(layer_cache: dict, layer_idx: int) -> dict:
    """Compress one freshly generated pre-position K/V layer in place."""
    if not enabled():
        return layer_cache
    if layer_cache.get("vision_kv_position_mode") != "pre":
        raise RuntimeError("QVG comparison requires the shared pre-position cache")
    incompatible = (
        "HY_KV_CODEC_ENABLE",
        "HY_KV_AE3D_ENABLE",
        "HY_PREKV_AE_ENABLE",
        "HY_KVTC_ENABLE",
        "HY_JENSEN_KV_ENABLE",
        "HY_TURBOQUANT_ENABLE",
    )
    active = [name for name in incompatible if _flag(name)]
    if active:
        raise RuntimeError(f"QVG is mutually exclusive with {active}")

    key = layer_cache.get("k_vision")
    value = layer_cache.get("v_vision")
    if not torch.is_tensor(key) or not torch.is_tensor(value):
        raise TypeError(
            f"QVG expected raw tensor K/V at layer {layer_idx}, got "
            f"{type(key).__name__}/{type(value).__name__}"
        )
    config = _config()
    _validate_config(config)

    from quant_videogen.compress import compress_kv_cache, get_quantize_fn

    quantize_fn = get_quantize_fn(config.quant_type, config)
    key_quant, value_quant = compress_kv_cache(
        key, value, config.quant_type, config, quantize_fn
    )
    if not isinstance(key_quant, dict) or not isinstance(value_quant, dict):
        raise RuntimeError("QVG real packed path did not return packed dictionaries")
    # Match the official ChunkedKVCache metadata contract used by uncompress.py.
    key_quant["info"] = {"output_dtype": key.dtype, "quant_config": config}
    value_quant["info"] = {"output_dtype": value.dtype, "quant_config": config}

    original_bytes = _tensor_bytes(key) + _tensor_bytes(value)
    compressed_bytes = _tensor_bytes(key_quant) + _tensor_bytes(value_quant)
    _STATS["calls"] += 1
    _STATS["original_by_layer"][int(layer_idx)] = original_bytes
    _STATS["compressed_by_layer"][int(layer_idx)] = compressed_bytes
    layer_cache["k_vision"] = key_quant
    layer_cache["v_vision"] = value_quant
    layer_cache["qvg_quantized"] = True
    return layer_cache


def maybe_decompress(key: Any, value: Any) -> tuple[Any, Any]:
    if not (isinstance(key, dict) and isinstance(value, dict)):
        return key, value
    if "info" not in key or "info" not in value:
        return key, value
    from quant_videogen.uncompress import uncompress_kv_cache

    return uncompress_kv_cache(key, value)


def write_summary() -> None:
    if not enabled():
        return
    path_value = os.environ.get("HY_QVG_SUMMARY", "").strip()
    if not path_value:
        return
    config = _config()
    original = sum(_STATS["original_by_layer"].values())
    compressed = sum(_STATS["compressed_by_layer"].values())
    output = {
        "enabled": True,
        "backend": "official_quant_videogen",
        "cache_position": "pre",
        "quant_type": config.quant_type,
        "quant_block_size": config.quant_block_size,
        "k_centroids": config.cache_num_k_centroids,
        "v_centroids": config.cache_num_v_centroids,
        "kmeans_max_iters": config.kmeans_max_iters,
        "prq_stages": config.num_prq_stages,
        "scale_dtype": "bf16_a100_compat",
        "calls": int(_STATS["calls"]),
        "fallback_count": int(_STATS["fallback_count"]),
        "layers": len(_STATS["original_by_layer"]),
        "original_cache_bytes": int(original),
        "compressed_cache_bytes": int(compressed),
        "actual_cache_compression_ratio": (
            float(original / compressed) if compressed else 0.0
        ),
        "includes": ["packed_values", "scales", "uint8_indices", "centroids"],
    }
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
