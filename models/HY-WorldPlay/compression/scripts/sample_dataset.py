"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import basis_utils as old_fit
import pca_utils as oracle


def parse_layers(value):
    return old_fit.parse_layers(value)


def discover_atoms(root: Path, split: str):
    atoms = []
    split_root = root / split
    if not split_root.is_dir():
        return atoms
    for case_dir in sorted(split_root.iterdir()):
        if not case_dir.is_dir():
            continue
        for pos_dir in sorted(case_dir.glob("position_*")):
            atoms.append(
                {
                    "case_id": case_dir.name,
                    "position": int(pos_dir.name.removeprefix("position_")),
                }
            )
    return atoms


def sampled_rows_fast(path: Path, count: int, seed: int):
    item = torch.load(path, map_location="cpu")
    if "tensor_int8" in item and "int8_scale" in item:
        codes = item["tensor_int8"]
        scale = item["int8_scale"]
        sequence = int(codes.shape[2])
        indices = oracle.sampled_linear_indices(sequence, count, seed)
        selected = codes[0, :, indices, :].float() * scale[0, :, indices, :].float()
        rows = selected.permute(1, 0, 2).reshape(-1, 2048)
        storage = "int8_symmetric"
    else:
        tensor = item.get("tensor", item.get("v", item.get("data")))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        sequence = int(tensor.shape[2])
        indices = oracle.sampled_linear_indices(sequence, count, seed)
        rows = tensor[0, :, indices, :].permute(1, 0, 2).reshape(-1, 2048).float()
        storage = "raw"
    return rows.contiguous(), {
        "path": str(path),
        "rows": int(rows.shape[0]),
        "storage": storage,
    }


def collect(root, target, split, atoms, layer, count, seed):
    rows = []
    audit = []
    for index, atom in enumerate(atoms):
        path = old_fit.target_atom_path(root, split, atom, layer, target)
        atom_seed = oracle.stable_seed(
            seed, "full96_dual", target, split, index,
            atom["case_id"], atom["position"], layer
        )
        part, info = sampled_rows_fast(path, count, atom_seed)
        rows.append(part)
        audit.append(
            info
            | {
                "case_id": atom["case_id"],
                "position": int(atom["position"]),
            }
        )
    return torch.cat(rows), audit


def equal_subset(rows, audit, per_atom):
    parts = []
    offset = 0
    for item in audit:
        count = int(item["rows"])
        if count < per_atom:
            raise ValueError(f"atom has only {count} rows, need {per_atom}: {item}")
        parts.append(rows[offset : offset + per_atom])
        offset += count
    return torch.cat(parts)


@torch.no_grad()
def evaluate_production(rows, mean, components, device, batch_size):
    numerator = denominator = cosine_sum = 0.0
    count = 0
    basis = components[-512:].to(device)
    mean = mean.to(device)
    for start in range(0, rows.shape[0], batch_size):
        target = rows[start : start + batch_size].to(device)
        z = F.linear(target - mean, basis)
        scale = z.abs().amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 7.0
        codes = (z / scale).round().clamp(-7, 7)
        zq = codes * scale.to(torch.bfloat16).float()
        pred = F.linear(zq, basis.T, mean)
        numerator += float((pred - target).square().sum().cpu())
        denominator += float(target.square().sum().cpu())
        cosine_sum += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
        count += int(target.shape[0])
    return {
        "rank": 512,
        "coeff_quant": "int4",
        "rel_l2": (numerator / max(denominator, 1e-20)) ** 0.5,
        "cosine": cosine_sum / max(1, count),
        "rows": count,
    }


def artifact(target, layer, mean, components, fit_name, atoms, rows):
    return {
        "format_version": 2,
        "candidate": "global_head",
        "tensor_name": target,
        "layer": layer,
        "mean": mean.detach().cpu().float(),
        "components": components.detach().cpu().float(),
        "input_dim": 2048,
        "latent_dim": 1024,
        "fit_storage": "dequantized_int8",
        "fit_split": "train",
        "fit_name": fit_name,
        "fit_atoms": atoms,
        "fit_rows": rows,
    }




