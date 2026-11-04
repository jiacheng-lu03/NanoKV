#!/usr/bin/env python3
"""Build the consolidated INT8 resident operator used by frozen HY deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def pack(weight: torch.Tensor, group_size: int) -> dict:
    weight = weight.detach().float().cpu().contiguous()
    actual_group = int(group_size) if int(group_size) > 0 else int(weight.shape[1])
    if weight.shape[1] % actual_group:
        raise RuntimeError(f"group_size={actual_group} does not divide {weight.shape[1]}")
    grouped = weight.reshape(weight.shape[0], -1, actual_group)
    scale = grouped.abs().amax(dim=2).clamp_min(1e-12) / 127.0
    qweight = (grouped / scale[:, :, None]).round().clamp(-127, 127).to(torch.int8)
    return {
        "qweight": qweight.reshape_as(weight),
        "scale": scale.to(torch.bfloat16),
        "group_size": actual_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--k-decoder-dir", type=Path)
    parser.add_argument("--v-qat-dir", type=Path)
    parser.add_argument(
        "--rung", choices=("pca", "c64", "c64_vqat", "full"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--rank", type=int, default=768)
    parser.add_argument("--k-rank", type=int)
    parser.add_argument("--v-rank", type=int)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--center-storage", choices=("bf16", "int8_block"), default="bf16")
    parser.add_argument("--center-block-size", type=int, default=64)
    args = parser.parse_args()
    k_rank = int(args.rank if args.k_rank is None else args.k_rank)
    v_rank = int(args.rank if args.v_rank is None else args.v_rank)
    if k_rank <= 0 or v_rank <= 0:
        raise ValueError("K/V ranks must be positive")
    entries = {}
    if args.rung == "full" and args.k_decoder_dir is None:
        raise RuntimeError("full rung requires --k-decoder-dir")
    if args.rung in {"c64_vqat", "full"} and args.v_qat_dir is None:
        raise RuntimeError(f"{args.rung} rung requires --v-qat-dir")
    for layer in range(args.num_layers):
        basis_target = "k" if args.rung == "pca" else "k_c64"
        k_basis = torch.load(
            args.basis_dir / basis_target / f"layer_{layer:02d}.pt",
            map_location="cpu", weights_only=True,
        )
        expected_storage = "none" if args.rung == "pca" else args.center_storage
        expected_block = 0 if args.rung == "pca" else args.center_block_size
        if k_basis.get("center_storage") != expected_storage or int(
            k_basis.get("center_block_size", -1)
        ) != expected_block:
            raise RuntimeError(f"K basis center provenance mismatch layer={layer}")
        k_encoder = k_basis["components"][-k_rank:].float()
        if args.rung == "full":
            k_decoder_payload = torch.load(
                args.k_decoder_dir / f"layer_{layer:02d}.pt",
                map_location="cpu", weights_only=True,
            )
            if k_decoder_payload.get("target") != "k_c64":
                raise RuntimeError(f"bad K decoder layer={layer}")
            if k_decoder_payload.get("center_storage") != args.center_storage or int(
                k_decoder_payload.get("center_block_size", -1)
            ) != args.center_block_size:
                raise RuntimeError(f"K decoder center provenance mismatch layer={layer}")
            k_decoder = k_decoder_payload["decoder"].float()
            if tuple(k_decoder.shape) != (int(k_encoder.shape[1]), k_rank):
                raise RuntimeError(
                    f"K decoder rank mismatch layer={layer}: "
                    f"{tuple(k_decoder.shape)} versus K rank {k_rank}"
                )
        else:
            k_decoder = k_encoder.T.contiguous()
        entries[f"k:{layer:02d}"] = {
            "mean": k_basis["mean"].float(),
            "encoder": pack(k_encoder, args.group_size),
            "decoder": pack(k_decoder, args.group_size),
            "basis_dir": str(args.basis_dir.resolve()),
            "decoder_dir": "" if args.rung != "full" else str(args.k_decoder_dir.resolve()),
        }
        if args.rung in {"c64_vqat", "full"}:
            v = torch.load(
                args.v_qat_dir / f"layer_{layer:02d}.pt",
                map_location="cpu", weights_only=True,
            )
            if v.get("format") != "mg35_worldplay_v_c64_int4_qat_v1":
                raise RuntimeError(f"bad V-QAT layer={layer}")
            if v.get("center_storage") != args.center_storage or int(
                v.get("center_block_size", -1)
            ) != args.center_block_size:
                raise RuntimeError(f"V-QAT center provenance mismatch layer={layer}")
            v_mean, v_encoder, v_decoder = v["mean"], v["encoder"], v["decoder"]
            if int(v_encoder.shape[0]) != v_rank or int(v_decoder.shape[1]) != v_rank:
                raise RuntimeError(
                    f"V-QAT rank mismatch layer={layer}: "
                    f"encoder={tuple(v_encoder.shape)} decoder={tuple(v_decoder.shape)} "
                    f"versus V rank {v_rank}"
                )
        else:
            v_target = "v" if args.rung == "pca" else "v_c64"
            v = torch.load(
                args.basis_dir / v_target / f"layer_{layer:02d}.pt",
                map_location="cpu", weights_only=True,
            )
            if v.get("center_storage") != expected_storage or int(
                v.get("center_block_size", -1)
            ) != expected_block:
                raise RuntimeError(f"V basis center provenance mismatch layer={layer}")
            v_mean = v["mean"]
            v_encoder = v["components"][-v_rank:].float()
            v_decoder = v_encoder.T.contiguous()
        entries[f"v:{layer:02d}"] = {
            "mean": v_mean.float(),
            "encoder": pack(v_encoder, args.group_size),
            "decoder": pack(v_decoder, args.group_size),
            "qat_dir": "" if args.rung not in {"c64_vqat", "full"} else str(args.v_qat_dir.resolve()),
            "basis_dir": str(args.basis_dir.resolve()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mg35_worldplay_operator_int8_v1",
            "rank": k_rank if k_rank == v_rank else None,
            "rank_by_kind": {
                "k": int(entries["k:00"]["encoder"]["qweight"].shape[0]),
                "v": int(entries["v:00"]["encoder"]["qweight"].shape[0]),
            },
            "group_size": args.group_size,
            "num_layers": args.num_layers,
            "rung": args.rung,
            "center_storage": expected_storage,
            "center_block_size": expected_block,
            "entries": entries,
        },
        args.output,
    )
    print({"output": str(args.output), "entries": len(entries)})


if __name__ == "__main__":
    main()
