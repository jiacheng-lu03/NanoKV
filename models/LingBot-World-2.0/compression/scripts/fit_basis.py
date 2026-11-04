#!/usr/bin/env python3
"""Exact-covariance PCA for one LingBot-World 2.0 rank/layer/KV target."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from codec_data import discover, load_rows


@torch.no_grad()
def evaluate(rows, mean, components, device, batch=1024):
    num = den = cosine = 0.0
    for start in range(0, rows.shape[0], batch):
        target = rows[start:start + batch].to(device)
        latent = F.linear(target - mean, components)
        pred = F.linear(latent, components.T, mean)
        num += float((pred - target).square().sum().cpu())
        den += float(target.square().sum().cpu())
        cosine += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
    return {"rel_l2": (num / max(den, 1e-20)) ** 0.5,
            "cosine": cosine / rows.shape[0], "rows": rows.shape[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=("k", "v"), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sp-rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=640)
    parser.add_argument("--mode", choices=("residual", "raw"), default="residual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-atoms", type=int, default=192)
    parser.add_argument("--expected-dev-atoms", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    train_paths = discover(args.data_root, "train", args.sp_rank, args.layer, args.target)
    dev_paths = discover(args.data_root, "dev", args.sp_rank, args.layer, args.target)
    if len(train_paths) != args.expected_train_atoms or len(dev_paths) != args.expected_dev_atoms:
        raise RuntimeError(
            f"atom gate failed: train/dev={len(train_paths)}/{len(dev_paths)}, "
            f"expected={args.expected_train_atoms}/{args.expected_dev_atoms}"
        )
    train, train_audit = load_rows(train_paths, args.mode)
    dev, dev_audit = load_rows(dev_paths, args.mode)
    if not 0 < args.rank <= train.shape[1]:
        raise ValueError(f"rank={args.rank}, width={train.shape[1]}")
    device = torch.device(args.device)
    source = train.to(device=device, dtype=torch.float32)
    mean = source.mean(0)
    centered = source - mean
    covariance = centered.T @ centered / max(source.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues.clamp_min_(0)
    components = eigenvectors[:, -args.rank:].T.contiguous()
    out = (args.output_root / f"rank_{args.sp_rank:02d}"
           / f"layer_{args.layer:02d}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "lingbot_v2_exact_pca_v1", "target": args.target,
        "layer": args.layer, "sp_rank": args.sp_rank, "rank": args.rank,
        "joint_dim": train.shape[1], "mean": mean.cpu(),
        "components": components.cpu(), "clusters": 64 if args.mode == "residual" else 0,
        "input_mode": args.mode, "pca_solver": "exact_covariance_eigh",
    }, out / "global_head_pca.pt")
    report = {
        "target": args.target, "layer": args.layer, "sp_rank": args.sp_rank,
        "rank": args.rank, "joint_dim": train.shape[1],
        "explained_variance": float(eigenvalues[-args.rank:].sum() /
                                    eigenvalues.sum().clamp_min(1e-20)),
        "train": evaluate(train, mean, components, device),
        "dev": evaluate(dev, mean, components, device),
        "train_atoms": len(train_paths), "dev_atoms": len(dev_paths),
        "elapsed_sec": time.time() - started,
        "train_audit": train_audit, "dev_audit": dev_audit,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
