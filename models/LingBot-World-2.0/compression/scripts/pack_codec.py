# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Pack all formal K-PCA and V-QAT operators as resident row-wise INT8."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def pack(weight):
    weight = weight.detach().float().cpu().contiguous()
    scale = weight.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return {"qweight": (weight / scale).round().clamp(-127, 127).to(torch.int8),
            "scale": scale.to(torch.bfloat16)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-basis", type=Path, required=True)
    parser.add_argument("--v-basis", type=Path, required=True)
    parser.add_argument("--v-qat", type=Path, required=True)
    parser.add_argument("--k-a1", type=Path, default=None)
    parser.add_argument(
        "--k-a1-scale", type=float, default=1.0,
        help="Multiply every K-A1 decoder delta before INT8 packing.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=640)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--sp-ranks", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.k_a1_scale <= 1.0:
        raise ValueError("--k-a1-scale must lie in [0, 1]")
    a1 = None
    if args.k_a1 is not None:
        a1 = torch.load(args.k_a1, map_location="cpu", weights_only=False)
        if a1.get("format") != "lingbot_v2_k_a1_decoder_delta_v1":
            raise RuntimeError(f"bad K-A1 artifact: {args.k_a1}")
        if str(Path(a1["basis_root"]).resolve()) != str(args.k_basis.resolve()):
            raise RuntimeError("K-A1/PCA basis provenance mismatch")
        if sorted(map(int, a1["decoder_delta"])) != list(range(args.layers)):
            raise RuntimeError("K-A1 artifact does not cover every packed layer")
    entries = {}
    for sp_rank in range(args.sp_ranks):
        for layer in range(args.layers):
            for target, root in (("k", args.k_basis), ("v", args.v_basis)):
                path = (root / f"rank_{sp_rank:02d}" / f"layer_{layer:02d}"
                        / "global_head_pca.pt")
                basis = torch.load(path, map_location="cpu", weights_only=False)
                encoder = basis["components"][-args.rank:].float()
                decoder = encoder.T.contiguous()
                mean = basis["mean"].float()
                if target == "v":
                    checkpoint = (args.v_qat / f"rank_{sp_rank:02d}"
                                  / f"layer_{layer:02d}" / "checkpoint_best.pt")
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
                    encoder = state["basis"].float() + state["encoder_delta"].float()
                    decoder = state["basis"].float().T + state["decoder_delta"].float()
                    mean = state["mean"].float()
                elif a1 is not None:
                    delta = a1["decoder_delta"].get(str(layer))
                    if delta is None:
                        raise KeyError(f"K-A1 artifact missing layer {layer}")
                    decoder = decoder + delta.float() * float(args.k_a1_scale)
                entries[f"rank{sp_rank:02d}:{target}:{layer:02d}"] = {
                    "mean": mean, "encoder": pack(encoder), "decoder": pack(decoder),
                    "basis_root": str(root.resolve()),
                    "linear_root": (
                        str(args.v_qat.resolve()) if target == "v"
                        else "" if args.k_a1 is None else str(args.k_a1.resolve())
                    ),
                }
    payload = {
        "format": "lingbot_v2_operator_int8_v1", "rank": args.rank,
        "layers": args.layers, "sp_ranks": args.sp_ranks,
        "k_a1_scale": float(args.k_a1_scale), "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
