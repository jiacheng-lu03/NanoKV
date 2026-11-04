"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sample_dataset as full96
import pca_utils as oracle


def load_heads(path: Path, count: int, seed: int):
    item = torch.load(path, map_location="cpu")
    codes = item["tensor_int8"]
    scales = item["int8_scale"]
    indices = oracle.sampled_linear_indices(int(codes.shape[2]), count, seed)
    # [H,N,D], matching QVG's independent (B,H) clustering.
    x = codes[0, :, indices, :].float() * scales[0, :, indices, :].float()
    return x.contiguous(), int(codes.shape[2])


@torch.no_grad()
def head_kmeans(x: torch.Tensor, clusters: int, iters: int, seed: int):
    """Deterministic batched Euclidean K-Means for [H,N,D]."""
    heads, count, dim = x.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(count, generator=gen)[:clusters].to(x.device)
    centers = x.index_select(1, initial).clone()
    labels = torch.zeros((heads, count), dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = (
            x.square().sum(-1, keepdim=True)
            + centers.square().sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(x, centers.transpose(1, 2))
        )
        labels = dist.argmin(-1)
        updated = torch.zeros_like(centers)
        counts = torch.zeros((heads, clusters), dtype=torch.float32, device=x.device)
        ones = torch.ones((count,), dtype=torch.float32, device=x.device)
        for head in range(heads):
            updated[head].scatter_add_(
                0, labels[head, :, None].expand(-1, dim), x[head]
            )
            counts[head].scatter_add_(0, labels[head], ones)
        nonempty = counts > 0
        updated = updated / counts.clamp_min(1.0).unsqueeze(-1)
        centers = torch.where(nonempty.unsqueeze(-1), updated, centers)
    assigned = torch.gather(
        centers, 1, labels[..., None].expand(-1, -1, dim)
    )
    return labels, centers, assigned


def target_atom_path(root: Path, split: str, atom, layer: int, target: str) -> Path:
    case_id = str(atom["case_id"])
    position = int(atom["position"])
    return (root / split / case_id / f"position_{position:02d}" / "kv" /
            (f"case-{case_id}_position{position:02d}_layer{layer:02d}"
             f"_target-{target}_branch-main_rope_kvstage-pre_{target}_vision.pt"))


def collect_smoothed(root: Path, split: str, atoms, layer: int, rows_per_atom: int,
                     clusters: int, iters: int, seed: int, device, cluster_mode: str,
                     target: str = "v"):
    raw_parts, residual_parts, assigned_parts, audit = [], [], [], []
    for index, atom in enumerate(atoms):
        path = target_atom_path(root, split, atom, layer, target)
        atom_seed = oracle.stable_seed(
            seed, "full96_dual", target, split, index,
            atom["case_id"], atom["position"], layer,
        )
        heads, sequence = load_heads(path, rows_per_atom, atom_seed)
        heads = heads.to(device)
        to_rows = lambda value: value.permute(1, 0, 2).reshape(-1, 2048).cpu()
        if cluster_mode == "per_head":
            labels, centers, assigned = head_kmeans(
                heads, clusters, iters, atom_seed + 991
            )
            raw_rows = to_rows(heads)
            residual_rows = to_rows(heads - assigned)
            assigned_rows = to_rows(assigned)
        elif cluster_mode == "joint":
            joint = heads.permute(1, 0, 2).reshape(1, -1, 2048)
            labels, centers, assigned = head_kmeans(
                joint, clusters, iters, atom_seed + 991
            )
            raw_rows = joint[0].cpu()
            residual_rows = (joint - assigned)[0].cpu()
            assigned_rows = assigned[0].cpu()
        else:
            raise ValueError(cluster_mode)
        raw_parts.append(raw_rows)
        residual_parts.append(residual_rows)
        assigned_parts.append(assigned_rows.to(torch.float16))
        counts = torch.bincount(labels.reshape(-1), minlength=clusters)
        audit.append({
            "case_id": atom["case_id"], "position": int(atom["position"]),
            "path": str(path), "rows": int(heads.shape[1]),
            "sequence_tokens": sequence, "nonempty_global_ids": int((counts > 0).sum()),
        })
        print(f"[{split}] {index + 1}/{len(atoms)} {atom['case_id']}@{atom['position']}", flush=True)
    return (
        torch.cat(raw_parts), torch.cat(residual_parts),
        torch.cat(assigned_parts), audit,
    )


def exact_int4(z: torch.Tensor):
    scale = z.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / 7.0
    return (z / scale).round().clamp(-7, 7) * scale.to(torch.bfloat16).float()


@torch.no_grad()
def reconstruct(rows, mean, encoder, decoder, device, batch_size, quant, offsets=None):
    num = den = cos = 0.0
    count = 0
    for start in range(0, rows.shape[0], batch_size):
        target = rows[start:start + batch_size].to(device).float()
        z = F.linear(target - mean, encoder)
        if quant == "bf16":
            z = z.to(torch.bfloat16).float()
        elif quant == "int4":
            z = exact_int4(z)
        pred = F.linear(z, decoder, mean)
        if offsets is not None:
            off = offsets[start:start + batch_size].to(device).float()
            pred = pred + off
            target = target + off
        num += float((pred - target).square().sum().cpu())
        den += float(target.square().sum().cpu())
        cos += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
        count += target.shape[0]
    return {"rel_l2": math.sqrt(num / max(den, 1e-20)), "cosine": cos / count, "rows": count}




