# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Offline-fuse and pack formal K/V codec operators for direct inference load."""

import argparse
import os
from pathlib import Path

import torch

from hyvideo.models.transformers.modules import kv_codecs


def pack_target(entries, target, basis_root, linear_root, rank, layers):
    for layer in layers:
        basis = kv_codecs._load_global_basis(basis_root, layer, target)
        components = basis["components"][-rank:].float()
        delta = kv_codecs._load_global_linear_delta(
            linear_root, layer, target, basis_root
        )["model"]
        encoder_delta = delta.get("encoder_delta")
        encoder = components if encoder_delta is None else components + encoder_delta.float()
        decoder = components.T.contiguous() + delta["decoder_delta.weight"].float()
        encoder_packed = kv_codecs._pack_operator_weight(encoder, "int8", "cpu")
        decoder_packed = kv_codecs._pack_operator_weight(decoder, "int8", "cpu")
        key = f"{target}:{layer:02d}"
        entries[key] = {
            "basis_root": str(basis_root.resolve()),
            "linear_root": str(linear_root.resolve()),
            "mean": basis["mean"].float().contiguous(),
            "encoder": {
                "qweight": encoder_packed["qweight"].cpu(),
                "scale": encoder_packed["scale"].cpu(),
                "group_size": int(encoder_packed["group_size"]),
            },
            "decoder": {
                "qweight": decoder_packed["qweight"].cpu(),
                "scale": decoder_packed["scale"].cpu(),
                "group_size": int(decoder_packed["group_size"]),
            },
        }
        print(f"packed {key}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-basis", type=Path, required=True)
    parser.add_argument("--k-linear", type=Path, required=True)
    parser.add_argument("--v-basis", type=Path, required=True)
    parser.add_argument("--v-linear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--layers", type=int, default=54)
    args = parser.parse_args()
    os.environ["HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE"] = "0"
    entries = {}
    layers = range(args.layers)
    pack_target(entries, "k", args.k_basis, args.k_linear, args.rank, layers)
    pack_target(entries, "v", args.v_basis, args.v_linear, args.rank, layers)
    payload = {
        "format": "hy_kv_codec_operator_int8_v1",
        "rank": args.rank,
        "group_size": 0,
        "layers": args.layers,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size = args.output.stat().st_size / (1024**2)
    print(f"saved {args.output} ({size:.2f} MiB, {len(entries)} entries)")


if __name__ == "__main__":
    main()
