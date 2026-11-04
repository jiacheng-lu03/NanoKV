#!/usr/bin/env python3
"""INT8 storage helpers for pre-KV dataset tensors.

The raw pre-KV dumps store a BF16 tensor under ``payload["tensor"]``.  These
helpers provide a lossless metadata wrapper plus lossy per-token symmetric INT8
storage:

    x_bf16 -> tensor_int8 + int8_scale -> dequantized BF16/FP32 tensor

This is intended for offline training datasets, not online resident cache
packing.  The default granularity is per-token, i.e. one scale for each
128-dimensional token vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch


SUPPORTED_GRANULARITIES = {"per_token", "per_channel", "per_tensor"}


def _normalize_granularity(granularity: str) -> str:
    value = granularity.strip().lower().replace("-", "_")
    if value not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported INT8 granularity={granularity}")
    return value


def quantize_int8_tensor(
    tensor: torch.Tensor,
    granularity: str = "per_token",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a KV tensor with symmetric signed INT8 levels.

    Args:
        tensor: Usually ``[B,H,S,D]``.
        granularity: ``per_token`` uses one scale per final-dim vector.
        scale_dtype: Storage dtype for scales. BF16 is the default because it
            matches the source dataset and keeps scale overhead small.
    """
    granularity = _normalize_granularity(granularity)
    x = tensor.float()
    eps = torch.finfo(torch.float32).eps
    if granularity == "per_tensor":
        scale = x.abs().amax().clamp_min(eps) / 127.0
    elif granularity == "per_channel":
        dims = tuple(range(x.ndim - 1))
        scale = x.abs().amax(dim=dims, keepdim=True).clamp_min(eps) / 127.0
    else:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    codes = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scale.to(scale_dtype).contiguous()


def dequantize_int8_tensor(
    codes: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Restore an INT8 dataset tensor to the requested floating dtype."""
    return (codes.float() * scale.float()).to(dtype)


def load_kv_dataset_payload(path: Path, dtype: torch.dtype | None = None) -> Tuple[torch.Tensor, Dict]:
    """Load either a raw BF16 dataset file or an INT8-compressed dataset file.

    Returns:
        tensor: Floating tensor, dequantized if needed.
        meta: All non-tensor metadata.
    """
    item = torch.load(path, map_location="cpu")
    if not isinstance(item, dict):
        tensor = item
        meta = {}
    elif "tensor_int8" in item and "int8_scale" in item:
        target_dtype = dtype or _dtype_from_name(item.get("original_dtype", item.get("dtype", "bf16")))
        tensor = dequantize_int8_tensor(item["tensor_int8"], item["int8_scale"], target_dtype)
        meta = {
            k: v
            for k, v in item.items()
            if k not in {"tensor", "tensor_int8", "int8_scale"}
        }
    else:
        tensor = item.get("tensor", item.get("v", item.get("data")))
        meta = {k: v for k, v in item.items() if k not in {"tensor", "v", "data"}}
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor payload")
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected [B,H,S,D], got {tuple(tensor.shape)} in {path}")
    return tensor, meta


def make_int8_payload(
    source_payload: Dict,
    tensor: torch.Tensor,
    granularity: str = "per_token",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    """Build a metadata-preserving INT8 payload from a raw dataset payload."""
    codes, scale = quantize_int8_tensor(tensor, granularity, scale_dtype)
    out = {k: v for k, v in source_payload.items() if k not in {"tensor", "v", "data"}}
    out.update(
        {
            "format_version": max(int(out.get("format_version", 1)), 1),
            "storage_format": "int8_symmetric",
            "quantization": {
                "bits": 8,
                "scheme": "symmetric_signed",
                "granularity": _normalize_granularity(granularity),
                "qmin": -127,
                "qmax": 127,
                "scale_dtype": _name_from_dtype(scale_dtype),
            },
            "original_dtype": _name_from_dtype(tensor.dtype),
            "shape": tuple(tensor.shape),
            "tensor_int8": codes,
            "int8_scale": scale,
        }
    )
    return out


def estimate_int8_payload_bytes(payload: Dict) -> int:
    """Estimate tensor storage bytes for an INT8 payload, excluding pickle overhead."""
    codes = payload["tensor_int8"]
    scale = payload["int8_scale"]
    return codes.numel() * codes.element_size() + scale.numel() * scale.element_size()


def _dtype_from_name(name) -> torch.dtype:
    value = str(name).replace("torch.", "").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return torch.bfloat16


def _name_from_dtype(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")
