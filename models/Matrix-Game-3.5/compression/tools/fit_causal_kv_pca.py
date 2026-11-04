#!/usr/bin/env python3
"""Fit per-layer head-channel PCA bases from causal cache-write samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_rows(payload: dict) -> torch.Tensor:
    if payload.get("format") != "mg35_worldplay_pca_samples_v3":
        raise RuntimeError("formal PCA fitting requires v3 INT8+BF16 paired samples")
    rows = payload["tensor_int8"].float() * payload["int8_scale"].float()
    return rows.reshape(rows.shape[0], -1)


def fit_one(
    rows: torch.Tensor, target: str, layer: int, sources: list[Path],
    output: Path, rank: int, device: str, provenance: dict,
) -> dict:
    if rows.ndim != 2:
        raise ValueError(f"Expected [rows, channels] for {output}, got {tuple(rows.shape)}")
    if rank <= 0 or rank > min(rows.shape):
        raise ValueError(f"rank {rank} is invalid for rows {tuple(rows.shape)}")
    work = rows.to(device)
    mean = work.mean(0)
    centered = work - mean
    # Keep this identical to the frozen WorldPlay `fit_pca`: covariance in
    # FP32, ascending `eigh`, and the final columns transposed into encoder
    # rows.  Do not replace this with a thin-SVD shortcut: although the span is
    # mathematically equivalent, it is not the source-faithful numerical path.
    covariance = centered.t().matmul(centered) / max(1, int(work.shape[0]) - 1)
    eigenvalues_all, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues_all = eigenvalues_all.clamp_min(0)
    components = eigenvectors[:, -rank:].t().contiguous()
    kept_eigenvalues = eigenvalues_all[-rank:]
    retained = kept_eigenvalues.sum() / eigenvalues_all.sum().clamp_min(1e-20)
    result = {
        "format": "mg35_causal_kv_pca_v1",
        "target": target,
        "layer": int(layer),
        "rows": int(rows.shape[0]),
        "sources": [str(source) for source in sources],
        "channels": int(rows.shape[1]),
        "rank": int(rank),
        "mean": mean.cpu(),
        "components": components.cpu(),
        "eigenvalues": kept_eigenvalues.cpu(),
        "retained_variance": float(retained.item()),
        **provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return {k: v for k, v in result.items() if not torch.is_tensor(v)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=Path, action="append", required=True,
        help="Calibration root; repeat to fit one basis from multiple scenes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--targets", nargs="+", choices=("k", "v", "k_c64", "v_c64"),
        default=("k", "v", "k_c64", "v_c64"),
    )
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--center-storage", choices=("bf16", "int8_block"), required=True)
    parser.add_argument("--center-block-size", type=int, default=64)
    parser.add_argument("--expected-roots", type=int, default=0)
    parser.add_argument("--expected-rows-per-root", type=int, default=0)
    parser.add_argument("--expected-total-rows", type=int, default=0)
    parser.add_argument("--collection-manifest", type=Path)
    args = parser.parse_args()
    resolved_samples = [path.resolve() for path in args.samples]
    if len(set(resolved_samples)) != len(resolved_samples):
        raise RuntimeError("duplicate PCA calibration roots")
    if args.expected_roots and len(args.samples) != args.expected_roots:
        raise RuntimeError(
            f"formal PCA requires {args.expected_roots} roots, got {len(args.samples)}"
        )
    manifest = None
    if args.collection_manifest is not None:
        manifest = json.loads(args.collection_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("split") != "train":
            raise RuntimeError("PCA collection manifest is not a complete train split")
        if int(manifest.get("trajectories", -1)) != len(args.samples):
            raise RuntimeError("PCA collection manifest trajectory mismatch")
        if int(manifest.get("total_rows_per_layer_kind", -1)) != args.expected_total_rows:
            raise RuntimeError("PCA collection manifest row-count mismatch")
        if manifest.get("center_storage") != args.center_storage or int(
            manifest.get("center_block_size", -1)
        ) != args.center_block_size:
            raise RuntimeError("PCA collection manifest center provenance mismatch")
        if int(manifest.get("c64_clusters", -1)) != args.clusters:
            raise RuntimeError("PCA collection manifest C64 cluster mismatch")
        if int(manifest.get("c64_iters", -1)) != args.kmeans_iters:
            raise RuntimeError("PCA collection manifest C64 iteration mismatch")
        if manifest.get("c64_seed_rule") != "20260804+layer":
            raise RuntimeError("PCA collection manifest C64 seed-rule mismatch")
    report = []
    for target in args.targets:
        by_name = {}
        for root in args.samples:
            for source in sorted((root / target).glob("layer_*.pt")):
                by_name.setdefault(source.name, []).append(source)
        for name, sources in sorted(by_name.items()):
            layer = int(name.removeprefix("layer_").removesuffix(".pt"))
            if not args.layer_start <= layer < args.layer_end:
                continue
            if len(sources) != len(args.samples):
                raise RuntimeError(
                    f"Incomplete multi-scene PCA input {target}/{name}: "
                    f"{len(sources)}/{len(args.samples)} roots"
                )
            payloads = [
                torch.load(source, map_location="cpu", weights_only=True)
                for source in sources
            ]
            if any(
                payload.get("target") != target
                or int(payload.get("layer", -1)) != layer
                for payload in payloads
            ):
                raise RuntimeError(f"bad PCA calibration payload for {target} L{layer}")
            expected_storage = payloads[0].get("center_storage")
            expected_block = int(payloads[0].get("center_block_size", -1))
            if expected_storage not in {"none", "bf16", "int8_block"}:
                raise RuntimeError(f"missing center provenance for {target} L{layer}")
            required_storage = args.center_storage if target.endswith("_c64") else "none"
            required_block = args.center_block_size if target.endswith("_c64") else 0
            if expected_storage != required_storage or expected_block != required_block:
                raise RuntimeError(f"unexpected center provenance for {target} L{layer}")
            if any(
                payload.get("center_storage") != expected_storage
                or int(payload.get("center_block_size", -1)) != expected_block
                for payload in payloads
            ):
                raise RuntimeError(f"mixed center provenance for {target} L{layer}")
            for payload, source in zip(payloads, sources):
                payload_rows = int(payload["tensor_int8"].shape[0])
                if args.expected_rows_per_root and payload_rows != args.expected_rows_per_root:
                    raise RuntimeError(
                        f"formal PCA row count {payload_rows} != "
                        f"{args.expected_rows_per_root}: {source}"
                    )
                sample_ids = payload.get("sample_ids")
                if sample_ids is None or tuple(sample_ids.shape) != (payload_rows, 2):
                    raise RuntimeError(f"bad PCA sample IDs: {source}")
                if int(torch.unique(sample_ids, dim=0).shape[0]) != payload_rows:
                    raise RuntimeError(f"duplicate PCA sample IDs: {source}")
            rows = torch.cat([load_rows(payload) for payload in payloads])
            if args.expected_total_rows and int(rows.shape[0]) != args.expected_total_rows:
                raise RuntimeError(
                    f"formal PCA total rows {rows.shape[0]} != {args.expected_total_rows}"
                )
            report.append(fit_one(
                rows, target, layer, sources,
                args.output / target / name, args.rank, args.device,
                {
                    "center_storage": expected_storage,
                    "center_block_size": expected_block,
                    "source_count": len(sources),
                    "rows_per_source": args.expected_rows_per_root,
                    "c64_clusters": args.clusters if target.endswith("_c64") else 0,
                    "c64_iters": args.kmeans_iters if target.endswith("_c64") else 0,
                    "c64_seed_rule": "20260804+layer" if target.endswith("_c64") else "none",
                    "collection_manifest": (
                        str(args.collection_manifest.resolve())
                        if args.collection_manifest is not None else ""
                    ),
                },
            ))
    if not report:
        raise RuntimeError(f"No samples found under {args.samples}")
    report_name = (
        "FIT_REPORT.json"
        if tuple(args.targets) == ("k", "v", "k_c64", "v_c64")
        else "FIT_REPORT_" + "_".join(args.targets) + ".json"
    )
    (args.output / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"artifacts": len(report), "output": str(args.output)}))


if __name__ == "__main__":
    main()
