#!/usr/bin/env python3
"""INT4 QAT for the joint-QVG-smoothed V residual."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sample_dataset as full96
from residual_data import collect_smoothed
from quantized_linear import PCALinearInt4QAT, reconstruction_loss


@torch.no_grad()
def evaluate(model, residual, offsets, device, batch_size):
    model.eval()
    num_res = den_res = num_full = den_full = cos_full = 0.0
    count = 0
    for start in range(0, residual.shape[0], batch_size):
        target_r = residual[start:start + batch_size].to(device).float()
        off = offsets[start:start + batch_size].to(device).float()
        pred_r = model(target_r)[0]
        pred = pred_r + off
        target = target_r + off
        num_res += float((pred_r - target_r).square().sum().cpu())
        den_res += float(target_r.square().sum().cpu())
        num_full += float((pred - target).square().sum().cpu())
        den_full += float(target.square().sum().cpu())
        cos_full += float(F.cosine_similarity(pred, target, dim=-1).sum().cpu())
        count += target.shape[0]
    return {
        "residual_rel_l2": (num_res / max(den_res, 1e-20)) ** 0.5,
        "full_v_rel_l2": (num_full / max(den_full, 1e-20)) ** 0.5,
        "full_v_cosine": cos_full / count,
        "rows": count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument(
        "--val-data-root", type=Path, default=None,
        help="Optional independent dataset root whose val split is used only for checkpoint selection.",
    )
    ap.add_argument("--residual-basis", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--kmeans-iters", type=int, default=2)
    ap.add_argument("--rows-per-atom", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--initialization", choices=("pca", "random_orthogonal"), required=True)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", choices=("k", "v"), default="v")
    ap.add_argument("--expected-train-atoms", type=int, default=None)
    ap.add_argument("--expected-val-atoms", type=int, default=None)
    ap.add_argument(
        "--internal-val-images", type=int, default=0,
        help="When the data root has no val split, hold out this many train image IDs for QAT checkpoint selection.",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.layer)
    started = time.time()

    train_atoms = full96.discover_atoms(args.data_root, "train")
    val_data_root = args.val_data_root if args.val_data_root is not None else args.data_root
    val_atoms = full96.discover_atoms(val_data_root, "val")
    if args.expected_train_atoms is not None and len(train_atoms) != args.expected_train_atoms:
        raise RuntimeError(
            f"expected {args.expected_train_atoms} train atoms, got {len(train_atoms)}"
        )
    if args.expected_val_atoms is not None and len(val_atoms) != args.expected_val_atoms:
        raise RuntimeError(
            f"expected {args.expected_val_atoms} val atoms, got {len(val_atoms)}"
        )
    val_storage_split = "val"
    internal_val_image_ids = []
    if not val_atoms and args.internal_val_images:
        image_ids = sorted({"_".join(str(atom["case_id"]).split("_")[:2]) for atom in train_atoms})
        if args.internal_val_images >= len(image_ids):
            raise RuntimeError(
                f"internal val images={args.internal_val_images} must be smaller than train images={len(image_ids)}"
            )
        chooser = random.Random(args.seed)
        internal_val_image_ids = sorted(chooser.sample(image_ids, args.internal_val_images))
        heldout = set(internal_val_image_ids)
        val_atoms = [
            atom for atom in train_atoms
            if "_".join(str(atom["case_id"]).split("_")[:2]) in heldout
        ]
        train_atoms = [
            atom for atom in train_atoms
            if "_".join(str(atom["case_id"]).split("_")[:2]) not in heldout
        ]
        val_storage_split = "train"
    if not train_atoms or not val_atoms:
        raise RuntimeError(
            f"QAT requires non-empty train/val atoms, got {len(train_atoms)}/{len(val_atoms)}"
        )
    train_raw, train_res, train_off, train_audit = collect_smoothed(
        args.data_root, "train", train_atoms, args.layer, args.rows_per_atom,
        args.clusters, args.kmeans_iters, args.seed, device, "joint", args.target,
    )
    val_raw, val_res, val_off, val_audit = collect_smoothed(
        val_data_root if val_storage_split == "val" else args.data_root,
        val_storage_split, val_atoms, args.layer, args.rows_per_atom,
        args.clusters, args.kmeans_iters, args.seed, device, "joint", args.target,
    )
    del train_raw, val_raw, train_off
    basis = torch.load(args.residual_basis, map_location="cpu")
    if args.rank <= 0 or args.rank > int(basis["components"].shape[0]):
        raise RuntimeError(
            f"rank must be in [1,{basis['components'].shape[0]}], got {args.rank}"
        )
    model = PCALinearInt4QAT(
        basis["mean"], basis["components"], args.rank,
        "encoder_decoder", args.initialization,
    ).to(device)
    initial_val = evaluate(model, val_res, val_off, device, args.batch_size)
    initial_train = evaluate(model, train_res, torch.zeros_like(train_res, dtype=torch.float16), device, args.batch_size)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + args.layer)
    best = initial_val["full_v_rel_l2"]
    best_step = 0
    history = [{"step": 0, "val": initial_val}]
    common = {
        "format_version": 1, "model_kind": "joint_qvg_smooth_pca_int4_qat",
        "target": args.target, "layer": args.layer, "rank": args.rank,
        "clusters": args.clusters, "kmeans_iters": args.kmeans_iters,
        "mode": "encoder_decoder", "initialization": args.initialization,
    }
    torch.save(common | {"model": model.state_dict(), "step": 0}, args.output_dir / "checkpoint_best.pt")
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(train_res.shape[0], (args.batch_size,), generator=generator)
        target = train_res.index_select(0, idx).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(target)[0]
        loss = reconstruction_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_metric = evaluate(model, val_res, val_off, device, args.batch_size)
            history.append({"step": step, "loss": float(loss.detach().cpu()), "val": val_metric})
            print(f"[smooth-QAT] init={args.initialization} step={step} val_full={val_metric['full_v_rel_l2']:.6f}", flush=True)
            if val_metric["full_v_rel_l2"] < best:
                best = val_metric["full_v_rel_l2"]
                best_step = step
                torch.save(common | {"model": model.state_dict(), "step": step}, args.output_dir / "checkpoint_best.pt")
    payload = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(payload["model"])
    final_train_res = evaluate(
        model, train_res, torch.zeros_like(train_res, dtype=torch.float16),
        device, args.batch_size,
    )
    final_val = evaluate(model, val_res, val_off, device, args.batch_size)
    report = common | {
        "best_step": best_step, "initial_train_residual_frame": initial_train,
        "initial_val": initial_val, "best_train_residual_frame": final_train_res,
        "best_val": final_val, "history": history,
        "train_atoms": len(train_atoms), "val_atoms": len(val_atoms),
        "val_storage_split": val_storage_split,
        "val_data_root": str(val_data_root),
        "internal_val_image_ids": internal_val_image_ids,
        "rows_per_atom": args.rows_per_atom, "steps": args.steps, "lr": args.lr,
        "elapsed_sec": time.time() - started,
        "train_audit": train_audit, "val_audit": val_audit,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_step": best_step, "best_val": final_val,
                      "elapsed_sec": report["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
