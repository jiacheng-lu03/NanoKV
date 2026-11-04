#!/usr/bin/env python3
"""Source-faithful HY-WorldPlay V C64 INT4-QAT for MG3.5.

Only tensor layout, layer count, feature dimension, and dataset discovery are
adapted.  The frozen HY recipe is retained: random-orthogonal initialization,
jointly trainable encoder/decoder, exact BF16-scale symmetric INT4 STE,
12,000 steps at 1e-4, and checkpoint selection by held-out full-V rel-L2 after
adding the paired C64 center offset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_sample_rows(payload: dict) -> torch.Tensor:
    if payload.get("format") != "mg35_worldplay_pca_samples_v3":
        raise RuntimeError("formal V-QAT requires v3 INT8+BF16 paired samples")
    rows = payload["tensor_int8"].float() * payload["int8_scale"].float()
    return rows.reshape(rows.shape[0], -1)


def load_paired_rows(
    roots: list[Path], layer: int, split: str, seed: int,
    clusters: int, kmeans_iters: int, center_storage: str, center_block_size: int,
    expected_roots: int, expected_rows_per_root: int, expected_total_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del split, seed, clusters, kmeans_iters
    resolved = [root.resolve() for root in roots]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("duplicate V-QAT calibration roots")
    if expected_roots and len(roots) != expected_roots:
        raise RuntimeError(
            f"formal V-QAT requires {expected_roots} roots, got {len(roots)}"
        )
    raw_parts, residual_parts = [], []
    for root in roots:
        raw_path = root / "v" / f"layer_{layer:02d}.pt"
        residual_path = root / "v_c64" / f"layer_{layer:02d}.pt"
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        residual = torch.load(residual_path, map_location="cpu", weights_only=True)
        if raw.get("target") != "v" or residual.get("target") != "v_c64":
            raise RuntimeError(f"bad paired V calibration payload: {root}")
        if int(raw.get("layer", -1)) != layer or int(residual.get("layer", -1)) != layer:
            raise RuntimeError(f"layer mismatch in paired V calibration payload: {root}")
        if raw.get("center_storage") != "none" or int(raw.get("center_block_size", -1)) != 0:
            raise RuntimeError(f"bad raw-V center provenance at {root}")
        if residual.get("center_storage") != center_storage:
            raise RuntimeError(f"V residual center storage mismatch at {root}")
        if int(residual.get("center_block_size", -1)) != int(center_block_size):
            raise RuntimeError(f"V residual center block-size mismatch at {root}")
        raw_ids = raw.get("sample_ids")
        residual_ids = residual.get("sample_ids")
        if raw_ids is None or residual_ids is None or not torch.equal(raw_ids, residual_ids):
            raise RuntimeError(f"unpaired raw/residual V sample IDs at {root}")
        raw_rows = decode_sample_rows(raw)
        residual_rows = decode_sample_rows(residual)
        if raw_rows.shape != residual_rows.shape:
            raise RuntimeError(f"raw/residual row mismatch at {root}")
        if expected_rows_per_root and raw_rows.shape[0] != expected_rows_per_root:
            raise RuntimeError(
                f"formal V-QAT row count {raw_rows.shape[0]} != "
                f"{expected_rows_per_root}: {root}"
            )
        if int(torch.unique(raw_ids, dim=0).shape[0]) != int(raw_rows.shape[0]):
            raise RuntimeError(f"duplicate V-QAT sample IDs at {root}")
        # v and v_c64 are sampled with the identical deterministic row IDs
        # immediately after full immutable-chunk C64.
        raw_parts.append(raw_rows)
        residual_parts.append(residual_rows)
    if not raw_parts:
        raise RuntimeError("no paired V calibration roots")
    raw_all, residual_all = torch.cat(raw_parts), torch.cat(residual_parts)
    if expected_total_rows and raw_all.shape[0] != expected_total_rows:
        raise RuntimeError(
            f"formal V-QAT total rows {raw_all.shape[0]} != {expected_total_rows}"
        )
    return raw_all, residual_all


def validate_collection_manifest(
    path: Path, split: str, expected_roots: int, expected_total_rows: int,
    center_storage: str, center_block_size: int, clusters: int, kmeans_iters: int,
) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("split") != split:
        raise RuntimeError(f"V-QAT collection manifest is not complete {split}: {path}")
    checks = {
        "trajectories": expected_roots,
        "total_rows_per_layer_kind": expected_total_rows,
        "center_block_size": center_block_size,
        "c64_clusters": clusters,
        "c64_iters": kmeans_iters,
    }
    for key, expected in checks.items():
        if int(manifest.get(key, -1)) != int(expected):
            raise RuntimeError(f"V-QAT manifest {key} mismatch: {path}")
    if manifest.get("center_storage") != center_storage:
        raise RuntimeError(f"V-QAT manifest center storage mismatch: {path}")
    if manifest.get("c64_seed_rule") != "20260804+layer":
        raise RuntimeError(f"V-QAT manifest C64 seed-rule mismatch: {path}")
    return manifest


class WorldPlayVQAT(nn.Module):
    def __init__(self, mean: torch.Tensor, reference: torch.Tensor, rank: int):
        super().__init__()
        # Frozen HY PCA artifacts keep selected eigenvectors in ascending
        # eigenvalue order; `[-rank:]` therefore means the highest-variance
        # subspace when the source artifact contains more than the deploy rank.
        reference = reference[-rank:].float().clone()
        # Exact frozen HY random-orthogonal initialization.
        basis = torch.linalg.qr(
            torch.randn(reference.shape[1], rank, dtype=torch.float32),
            mode="reduced",
        ).Q.T.contiguous()
        self.rank = int(rank)
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("basis", basis)
        self.register_buffer("reference_basis", reference)
        self.encoder_delta = nn.Parameter(torch.zeros_like(basis))
        self.decoder_delta = nn.Parameter(torch.zeros_like(basis.T))

    def exact_quantize(self, z: torch.Tensor) -> torch.Tensor:
        scale = z.float().abs().amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 7.0
        codes = (z.float() / scale).round().clamp(-7, 7)
        return codes * scale.to(torch.bfloat16).float()

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        encoder = self.basis + self.encoder_delta
        decoder = self.basis.T + self.decoder_delta
        z = F.linear(residual.float() - self.mean, encoder)
        exact = self.exact_quantize(z)
        zq = z + (exact - z).detach() if self.training else exact
        return F.linear(zq, decoder, self.mean)

    def deployment_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis + self.encoder_delta, self.basis.T + self.decoder_delta


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Exact frozen HY objective.
    pred, target = pred.float(), target.float()
    normalized_l1 = (pred - target).abs().mean() / target.abs().mean().clamp_min(1e-12)
    relative_l2_square = (pred - target).square().sum() / target.square().sum().clamp_min(1e-20)
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return normalized_l1 + 0.2 * relative_l2_square + cosine


@torch.no_grad()
def evaluate(
    model: WorldPlayVQAT,
    raw: torch.Tensor,
    residual: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    num_res = den_res = num_full = den_full = cos_full = 0.0
    count = 0
    for start in range(0, residual.shape[0], batch_size):
        target_r = residual[start : start + batch_size].to(device).float()
        target_full = raw[start : start + batch_size].to(device).float()
        center_offset = target_full - target_r
        pred_r = model(target_r)
        pred_full = pred_r + center_offset
        num_res += float((pred_r - target_r).square().sum().cpu())
        den_res += float(target_r.square().sum().cpu())
        num_full += float((pred_full - target_full).square().sum().cpu())
        den_full += float(target_full.square().sum().cpu())
        cos_full += float(F.cosine_similarity(pred_full, target_full, dim=-1).sum().cpu())
        count += int(target_r.shape[0])
    return {
        "residual_rel_l2": (num_res / max(den_res, 1e-20)) ** 0.5,
        "full_v_rel_l2": (num_full / max(den_full, 1e-20)) ** 0.5,
        "full_v_cosine": cos_full / max(count, 1),
        "rows": count,
    }


def save_checkpoint(
    path: Path, model: WorldPlayVQAT, layer: int, step: int, metrics: dict,
    center_storage: str, center_block_size: int,
) -> None:
    encoder, decoder = model.deployment_weights()
    torch.save(
        {
            "format": "mg35_worldplay_v_c64_int4_qat_v1",
            "target": "v_c64",
            "layer": int(layer),
            "rank": int(model.rank),
            "step": int(step),
            "mode": "encoder_decoder",
            "initialization": "random_orthogonal",
            "mean": model.mean.detach().cpu().float(),
            # Training/master artifacts remain FP32.  BF16 is reserved for
            # dynamic scales/centers; static operators are quantized once at
            # the formal INT8 export boundary.
            "encoder": encoder.detach().cpu().float(),
            "decoder": decoder.detach().cpu().float(),
            "metrics": dict(metrics),
            "center_storage": center_storage,
            "center_block_size": int(center_block_size),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--val", type=Path, action="append", required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--center-storage", choices=("bf16", "int8_block"), required=True)
    parser.add_argument("--center-block-size", type=int, default=64)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--expected-train-roots", type=int, default=48)
    parser.add_argument("--expected-val-roots", type=int, default=4)
    parser.add_argument("--expected-rows-per-root", type=int, default=4096)
    parser.add_argument("--expected-train-rows", type=int, default=196608)
    parser.add_argument("--expected-val-rows", type=int, default=16384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate_collection_manifest(
        args.train_manifest, "train", args.expected_train_roots,
        args.expected_train_rows, args.center_storage, args.center_block_size,
        args.clusters, args.kmeans_iters,
    )
    validate_collection_manifest(
        args.val_manifest, "dev", args.expected_val_roots,
        args.expected_val_rows, args.center_storage, args.center_block_size,
        args.clusters, args.kmeans_iters,
    )

    torch.manual_seed(args.seed + args.layer)
    train_raw, train_residual = load_paired_rows(
        args.train, args.layer, "train", args.seed, args.clusters, args.kmeans_iters,
        args.center_storage, args.center_block_size,
        args.expected_train_roots, args.expected_rows_per_root, args.expected_train_rows,
    )
    val_raw, val_residual = load_paired_rows(
        args.val, args.layer, "val", args.seed, args.clusters, args.kmeans_iters,
        args.center_storage, args.center_block_size,
        args.expected_val_roots, args.expected_rows_per_root, args.expected_val_rows,
    )
    artifact = torch.load(
        args.basis_dir / "v_c64" / f"layer_{args.layer:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact_sources = {
        str(Path(path).resolve().parent.parent) for path in artifact.get("sources", [])
    }
    train_sources = {str(path.resolve()) for path in args.train}
    if artifact_sources != train_sources:
        raise RuntimeError(
            "V-QAT train roots are not exactly the sources used to fit the V-C64 basis"
        )
    if int(artifact.get("rows", -1)) != args.expected_train_rows or int(
        artifact.get("source_count", -1)
    ) != args.expected_train_roots:
        raise RuntimeError("V-QAT basis formal-scale provenance mismatch")
    if artifact.get("center_storage") != args.center_storage or int(
        artifact.get("center_block_size", -1)
    ) != int(args.center_block_size):
        raise RuntimeError("V-QAT basis center provenance mismatch")
    if args.rank <= 0 or args.rank > int(artifact["components"].shape[0]):
        raise RuntimeError(f"invalid rank={args.rank}")
    device = torch.device(args.device)
    model = WorldPlayVQAT(artifact["mean"], artifact["components"], args.rank).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + args.layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"layer_{args.layer:02d}.pt"
    report_path = args.output_dir / f"layer_{args.layer:02d}.json"

    initial_val = evaluate(model, val_raw, val_residual, device, args.batch_size)
    best = float(initial_val["full_v_rel_l2"])
    best_step = 0
    history = [{"step": 0, "val": initial_val}]
    save_checkpoint(
        checkpoint, model, args.layer, 0, initial_val,
        args.center_storage, args.center_block_size,
    )
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(
            train_residual.shape[0], (args.batch_size,), generator=generator
        )
        target = train_residual.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(target)
        loss = reconstruction_loss(pred, target)
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError(f"nonfinite gradient at step={step}")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_raw, val_residual, device, args.batch_size)
            history.append({"step": step, "loss": float(loss.detach().cpu()), "val": val})
            print(
                f"[WorldPlay-V-QAT] L{args.layer:02d} step={step} "
                f"full_v={val['full_v_rel_l2']:.7f} best={best:.7f}",
                flush=True,
            )
            if float(val["full_v_rel_l2"]) < best:
                best = float(val["full_v_rel_l2"])
                best_step = step
                save_checkpoint(
                    checkpoint, model, args.layer, step, val,
                    args.center_storage, args.center_block_size,
                )

    report_path.write_text(
        json.dumps(
            {
                "format": "mg35_worldplay_v_c64_int4_qat_report_v1",
                "layer": args.layer,
                "rank": args.rank,
                "mode": "encoder_decoder",
                "initialization": "random_orthogonal",
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "eval_every": args.eval_every,
                "train_rows": int(train_residual.shape[0]),
                "val_rows": int(val_residual.shape[0]),
                "initial_val": initial_val,
                "best_val_full_v_rel_l2": best,
                "best_step": best_step,
                "center_storage": args.center_storage,
                "center_block_size": args.center_block_size,
                "train_manifest": str(args.train_manifest.resolve()),
                "val_manifest": str(args.val_manifest.resolve()),
                "history": history,
                "elapsed_sec": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[WorldPlay-V-QAT] COMPLETE L{args.layer:02d} best={best:.7f}@{best_step}")


if __name__ == "__main__":
    main()
