#!/usr/bin/env python3
"""WorldPlay-aligned V INT4-QAT for one LingBot-World 2.0 layer/shard."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec_data import discover, load_residual_offset


class LinearInt4Codec(nn.Module):
    def __init__(self, mean, reference, rank):
        super().__init__()
        width = reference.shape[1]
        basis = torch.linalg.qr(torch.randn(width, rank), mode="reduced").Q.T.contiguous()
        self.register_buffer("mean", mean.float())
        self.register_buffer("basis", basis)
        self.encoder_delta = nn.Parameter(torch.zeros_like(basis))
        self.decoder_delta = nn.Parameter(torch.zeros_like(basis.T))

    def forward(self, value):
        latent = F.linear(value.float() - self.mean, self.basis + self.encoder_delta)
        scale = latent.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
        exact = (latent / scale).round().clamp(-7, 7) * scale
        quant = latent + (exact - latent).detach() if self.training else exact
        return F.linear(quant, self.basis.T + self.decoder_delta, self.mean)


def loss_fn(pred, target):
    l1 = (pred - target).abs().mean() / target.abs().mean().clamp_min(1e-12)
    l2 = (pred - target).square().sum() / target.square().sum().clamp_min(1e-20)
    cosine = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return l1 + 0.2 * l2 + cosine


@torch.no_grad()
def evaluate(model, rows, offsets, device, batch):
    model.eval()
    num = den = cosine = 0.0
    for start in range(0, rows.shape[0], batch):
        residual = rows[start:start + batch].to(device)
        offset = offsets[start:start + batch].to(device)
        target, pred = residual + offset, model(residual) + offset
        num += float((pred - target).square().sum().cpu())
        den += float(target.square().sum().cpu())
        cosine += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
    return {"full_v_rel_l2": (num / max(den, 1e-20)) ** 0.5,
            "full_v_cosine": cosine / rows.shape[0], "rows": rows.shape[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pca-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sp-rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=640)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-atoms", type=int, default=192)
    parser.add_argument("--expected-dev-atoms", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(args.seed + args.layer + 1009 * args.sp_rank)
    train_paths = discover(args.data_root, "train", args.sp_rank, args.layer, "v")
    dev_paths = discover(args.data_root, "dev", args.sp_rank, args.layer, "v")
    if (len(train_paths), len(dev_paths)) != (args.expected_train_atoms, args.expected_dev_atoms):
        raise RuntimeError(f"atom gate failed: {len(train_paths)}/{len(dev_paths)}")
    train, train_offset, train_audit = load_residual_offset(train_paths)
    dev, dev_offset, dev_audit = load_residual_offset(dev_paths)
    pca_path = (args.pca_root / f"rank_{args.sp_rank:02d}"
                / f"layer_{args.layer:02d}" / "global_head_pca.pt")
    pca = torch.load(pca_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = LinearInt4Codec(pca["mean"], pca["components"], args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    generator = torch.Generator(device="cpu").manual_seed(
        args.seed + args.layer + 1009 * args.sp_rank
    )
    out = (args.output_root / f"rank_{args.sp_rank:02d}"
           / f"layer_{args.layer:02d}")
    out.mkdir(parents=True, exist_ok=True)
    initial = evaluate(model, dev, dev_offset, device, args.batch_size)
    best, best_step = initial["full_v_rel_l2"], 0

    def save(step):
        torch.save({
            "format": "lingbot_v2_c64_int4_qat_v1", "target": "v",
            "layer": args.layer, "sp_rank": args.sp_rank, "rank": args.rank,
            "step": step, "model": model.state_dict(),
        }, out / "checkpoint_best.pt")

    save(0)
    history = [{"step": 0, "dev": initial}]
    for step in range(1, args.steps + 1):
        model.train()
        index = torch.randint(train.shape[0], (args.batch_size,), generator=generator)
        target = train.index_select(0, index).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(target), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metric = evaluate(model, dev, dev_offset, device, args.batch_size)
            history.append({"step": step, "loss": float(loss.detach().cpu()), "dev": metric})
            if metric["full_v_rel_l2"] < best:
                best, best_step = metric["full_v_rel_l2"], step
                save(step)
    report = {
        "target": "v", "layer": args.layer, "sp_rank": args.sp_rank,
        "rank": args.rank, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "initial_dev": initial,
        "best_dev_rel_l2": best, "best_step": best_step, "history": history,
        "train_audit": train_audit, "dev_audit": dev_audit,
        "elapsed_sec": time.time() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
