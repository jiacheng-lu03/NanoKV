# NanoKV public workflow entrypoint.
"""Strict loaders for LingBot-World 2.0 paired KV calibration dumps."""
from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "lingbot_v2_paired_raw_c64_v1"


def discover(root: Path, split: str, sp_rank: int, layer: int, target: str) -> list[Path]:
    paths = sorted(
        (root / split).glob(
            f"*/chunk_*/rank_{sp_rank:02d}/layer_{layer:02d}_{target}.pt"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"no dumps: root={root}, split={split}, rank={sp_rank}, "
            f"layer={layer}, target={target}"
        )
    return paths


def load_rows(paths: list[Path], mode: str) -> tuple[torch.Tensor, list[dict]]:
    if mode not in {"raw", "residual"}:
        raise ValueError(mode)
    rows, audit, width = [], [], None
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != FORMAT:
            raise RuntimeError(f"bad dump format: {path}: {item.get('format')}")
        value = item[f"{mode}_tensor_int8"].float() * item[f"{mode}_int8_scale"].float()
        value = value.flatten(1).contiguous()
        width = value.shape[1] if width is None else width
        if value.shape[1] != width:
            raise RuntimeError(f"mixed feature widths: {path}")
        rows.append(value)
        audit.append({
            "path": str(path.resolve()), "rows": value.shape[0],
            "case_id": item["case_id"], "chunk_id": item["chunk_id"],
            "sp_rank": item["sp_rank"], "cache_stage": item["cache_stage"],
        })
    return torch.cat(rows), audit


def load_residual_offset(paths: list[Path]):
    residual, audit = load_rows(paths, "residual")
    raw, _ = load_rows(paths, "raw")
    if raw.shape != residual.shape:
        raise RuntimeError("paired raw/residual mismatch")
    return residual, raw - residual, audit
