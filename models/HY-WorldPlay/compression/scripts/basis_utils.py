"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import pca_utils as oracle
from kv_dataset_int8_utils import load_kv_dataset_payload


def parse_layers(value: str):
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def sample_global_rows(path: Path, count: int, seed: int):
    tensor, metadata = load_kv_dataset_payload(path, dtype=torch.bfloat16)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"{path}: expected [1,H,S,D], got {tuple(tensor.shape)}")
    _, heads, sequence, dim = tensor.shape
    if heads * dim != 2048:
        raise ValueError(f"{path}: expected H*D=2048, got {heads}*{dim}")
    indices = oracle.sampled_linear_indices(sequence, count, seed)
    # Preserve all heads/channels of a sampled token as one PCA observation.
    rows = tensor[0, :, indices, :].permute(1, 0, 2).reshape(-1, heads * dim)
    return rows.float().contiguous(), {
        "path": str(path),
        "rows": int(rows.shape[0]),
        "storage_format": metadata.get("storage_format", "raw"),
    }


def target_atom_path(root, split, atom, layer, target):
    case_id = str(atom["case_id"])
    position = int(atom["position"])
    return (
        root
        / split
        / case_id
        / f"position_{position:02d}"
        / "kv"
        / (
            f"case-{case_id}_position{position:02d}_layer{layer:02d}"
            f"_target-{target}_branch-main_rope_kvstage-pre_{target}_vision.pt"
        )
    )


def collect_rows(config, root, split, atoms, layer, count, seed, target):
    parts = []
    audit = []
    for index, atom in enumerate(atoms):
        path = target_atom_path(root, split, atom, layer, target)
        atom_seed = oracle.stable_seed(
            seed,
            "all_layer_global_pca",
            target,
            split,
            index,
            atom["case_id"],
            atom["position"],
            layer,
        )
        rows, item = sample_global_rows(path, count, atom_seed)
        parts.append(rows)
        audit.append(
            item
            | {
                "case_id": atom["case_id"],
                "position": int(atom["position"]),
            }
        )
    return torch.cat(parts), audit


@torch.no_grad()
def evaluate(rows, mean, components, device, batch_size, cast_bf16):
    numerator = 0.0
    denominator = 0.0
    cosine_sum = 0.0
    count = 0
    for start in range(0, rows.shape[0], batch_size):
        target = rows[start : start + batch_size].to(device)
        z = F.linear(target - mean, components)
        if cast_bf16:
            z = z.to(torch.bfloat16).float()
        pred = F.linear(z, components.T, mean)
        numerator += float((pred - target).square().sum().cpu())
        denominator += float(target.square().sum().cpu())
        cosine_sum += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
        count += int(target.shape[0])
    return {
        "rel_l2": (numerator / max(denominator, 1e-20)) ** 0.5,
        "cosine": cosine_sum / max(1, count),
        "rows": count,
    }


def load_bf16_reference_rows(config, root, layer, count, seed):
    if not root.exists():
        return None
    try:
        parts = []
        for index, atom in enumerate(config["val_atoms"]):
            path = oracle.atom_path(root, "val", atom, layer)
            tensor, thw = oracle.load_raw_bf16(path)
            atom_seed = oracle.stable_seed(
                seed,
                "all_layer_global_pca",
                "val",
                index,
                atom["case_id"],
                atom["position"],
                layer,
            )
            rows, _, _ = oracle.pack_samples(
                tensor, thw, "global_head", count, atom_seed
            )
            parts.append(rows)
        return torch.cat(parts)
    except FileNotFoundError:
        return None




