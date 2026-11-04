# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Build analytic codec-native readers bound to a frozen LingBot A1 operator.

This is a correctness/initialization utility, not a learned calibration pass.
K/Q keep complete native RoPE pairs selected by decoder energy.  V uses the
leading left-singular subspace of the deployed decoder.  Width 128 is an exact
algebraic reader and is used as the first runtime-equivalence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


NATIVE_DIM = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_linear(item: dict[str, torch.Tensor]) -> torch.Tensor:
    return item["qweight"].float() * item["scale"].float()


def pair_indices(decoder: torch.Tensor, width: int) -> torch.Tensor:
    """Choose high-energy native complex pairs independently for every head."""
    heads = decoder.shape[0]
    if width == NATIVE_DIM:
        return torch.arange(NATIVE_DIM).expand(heads, -1).clone()
    pair_energy = decoder.square().sum(-1).reshape(heads, NATIVE_DIM // 2, 2).sum((-1,))
    pair_count = width // 2
    chosen = pair_energy.topk(pair_count, dim=-1, largest=True, sorted=False).indices
    chosen = chosen.sort(dim=-1).values
    return torch.stack((2 * chosen, 2 * chosen + 1), dim=-1).flatten(-2)


def leading_native_subspace(decoder: torch.Tensor, width: int) -> torch.Tensor:
    """Return [H,128,width] orthonormal bases without a large 128x640 SVD."""
    if width == NATIVE_DIM:
        return torch.eye(NATIVE_DIM).expand(decoder.shape[0], -1, -1).clone()
    covariance = decoder @ decoder.transpose(-1, -2)
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[..., -width:].contiguous()


def build_layer(k_entry: dict, v_entry: dict, width: int, heads: int) -> dict:
    k_decoder = unpack_linear(k_entry["decoder"]).reshape(heads, NATIVE_DIM, -1)
    v_decoder = unpack_linear(v_entry["decoder"]).reshape(heads, NATIVE_DIM, -1)
    rank = int(k_decoder.shape[-1])
    if v_decoder.shape != k_decoder.shape:
        raise ValueError("K/V decoder shapes differ")

    indices = pair_indices(k_decoder, width)
    k_basis = torch.zeros((heads, NATIVE_DIM, width), dtype=torch.float32)
    k_basis.scatter_(1, indices[:, None, :], 1.0)
    v_basis = leading_native_subspace(v_decoder, width)

    # With an untrained coordinate subset, normalize Q so the selected-dot
    # variance matches D128 under an isotropic model.  Width128 remains exact.
    q_reduce = k_basis * math.sqrt(NATIVE_DIM / width)
    k_read = torch.einsum("hod,hor->hdr", k_basis, k_decoder)
    v_read = torch.einsum("hod,hor->hdr", v_basis, v_decoder)
    return {
        "heads": heads,
        "native_dim": NATIVE_DIM,
        "rank": rank,
        "groups": [{
            "heads": torch.arange(heads),
            "indices": indices,
            "q_reduce": q_reduce,
            "k_read": k_read,
            "v_read": v_read,
            "k_affine": k_basis,
            "v_affine": v_basis,
            "lift": v_basis.transpose(-1, -2).contiguous(),
        }],
        "initialization": "exact_d128" if width == NATIVE_DIM else "k_rope_pair_energy_v_svd",
        "width": width,
        "calibrated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Keep widths aligned to complete RoPE complex pairs.  D112 is a useful
    # intermediate quality/compute point for models that are markedly more
    # sensitive than the WorldPlay baseline at D96.
    parser.add_argument("--width", type=int, choices=(32, 64, 96, 112, 128), required=True)
    parser.add_argument("--recent-native-frames", type=int, default=0)
    parser.add_argument("--anchor-stride", type=int, default=0)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument(
        "--mark-calibrated", action="store_true",
        help="Permit deployment of this analytic initialization for a gate.",
    )
    args = parser.parse_args()
    if args.width % 2:
        raise ValueError("RoPE-preserving widths must be even")

    operator = args.operator.resolve()
    packed = torch.load(operator, map_location="cpu", weights_only=True)
    if packed.get("format") != "lingbot_v2_operator_int8_v1":
        raise ValueError("not a frozen LingBot INT8 operator")
    if int(packed["rank"]) != 640:
        raise ValueError("the formal LingBot 32x point must have rank 640")

    output = args.output.resolve()
    reader_root = output / "readers" / "rank_00"
    reader_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "codec_native_manifest_v1",
        "model": "lingbot-world-v2",
        "codec_operator_sha256": sha256(operator),
        "operator": str(operator),
        "initialization": "analytic_v1",
        "policy": {
            "recent_native_frames": args.recent_native_frames,
            "anchor_stride": args.anchor_stride,
        },
        "layer_policy": {},
        "layers": {},
    }
    for layer in range(args.layers):
        state = build_layer(
            packed["entries"][f"rank00:k:{layer:02d}"],
            packed["entries"][f"rank00:v:{layer:02d}"],
            args.width,
            args.heads,
        )
        artifact = reader_root / f"layer_{layer:02d}.pt"
        torch.save(state, artifact)
        manifest["layers"][f"0:{layer}"] = {
            "path": str(artifact.relative_to(output)),
            "sha256": sha256(artifact),
            "calibrated": bool(args.mark_calibrated),
            "width": args.width,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(output / "manifest.json"),
        "operator_sha256": manifest["codec_operator_sha256"],
        "layers": args.layers,
        "heads": args.heads,
        "width": args.width,
        "calibrated": bool(args.mark_calibrated),
    }, indent=2))


if __name__ == "__main__":
    main()
