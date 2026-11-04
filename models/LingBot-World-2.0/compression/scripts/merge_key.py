# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Overlay partial LingBot K-A1 checkpoints into one causal-stage artifact."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


FORMAT = "lingbot_v2_k_a1_decoder_delta_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    merged = {}
    basis_root = None
    rank = None
    sources = []
    for path in args.inputs:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != FORMAT:
            raise RuntimeError(f"bad K-A1 checkpoint: {path}")
        basis_root = basis_root or item["basis_root"]
        rank = rank or int(item["rank"])
        if basis_root != item["basis_root"] or rank != int(item["rank"]):
            raise RuntimeError("incompatible K-A1 checkpoints")
        merged.update({str(k): v.float().cpu() for k, v in item["decoder_delta"].items()})
        sources.append(str(path.resolve()))
    layers = sorted(map(int, merged))
    if args.require_complete and layers != list(range(40)):
        raise RuntimeError(f"final K-A1 artifact is incomplete: {layers}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({
        "format": FORMAT, "basis_root": basis_root, "rank": rank, "sp_rank": 0,
        "layers": layers, "decoder_delta": merged, "source_groups": sources,
        "created_at": time.time(),
    }, temporary)
    temporary.replace(args.output)
    print(f"wrote {args.output} with {len(layers)} layers")


if __name__ == "__main__":
    main()
