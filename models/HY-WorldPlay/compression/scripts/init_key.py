# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Convert K global-PCA artifacts into zero-delta A1 decoder checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from hyvideo.models.transformers.modules.kv_ae3d import build_pre_kv_ae


def parse_layers(value: str):
    if value.strip().lower() == "all":
        return list(range(54))
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--rank", type=int, default=512, choices=(128, 256, 512, 1024))
    parser.add_argument("--semantic-smooth", action="store_true")
    parser.add_argument("--semantic-clusters", type=int, default=64)
    parser.add_argument("--semantic-iters", type=int, default=2)
    args = parser.parse_args()

    rank = int(args.rank)
    config = {
        "architecture": "global_pca_linear_d512",
        "dim": 128,
        "hidden": 128,
        "latent_dim": rank,
    }
    manifest = []
    for layer in parse_layers(args.layers):
        basis_path = (
            args.basis_root
            / f"layer_{int(layer):02d}"
            / "global_head_pca.pt"
        )
        basis = torch.load(basis_path, map_location="cpu")
        if basis.get("tensor_name") != "k":
            raise ValueError(f"expected K basis: {basis_path}")
        mean = basis["mean"].float()
        components = basis["components"].float()[-rank:].contiguous()
        if tuple(mean.shape) != (2048,) or tuple(components.shape) != (rank, 2048):
            raise ValueError(f"invalid basis shapes in {basis_path}")

        model = build_pre_kv_ae(config)
        with torch.no_grad():
            model.mean.copy_(mean)
            model.encoder_basis.copy_(components)
            model.decoder_delta.weight.zero_()
        metadata = {
            "target": "k",
            "layer": int(layer),
            "rank": rank,
            "coeff_quant": "int4",
            "scale_dtype": "bf16",
            "basis_root": str(args.basis_root.resolve()),
            "basis_file": str(basis_path.resolve()),
            "basis_sha256": sha256(basis_path),
            "zero_delta": True,
            "semantic_smooth": bool(args.semantic_smooth),
            "semantic_clusters": int(args.semantic_clusters),
            "semantic_iters": int(args.semantic_iters),
        }
        payload = {
            "model_kind": "global_pca_int4_linear_a1",
            "model": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "model_config": config,
            "layer_id": int(layer),
            "target": "k",
            "source_checkpoint": str(basis_path.resolve()),
            "optimizer": None,
            "optimizer_type": "adamw",
            "codec_metadata": metadata,
        }
        path = (
            args.output_root
            / f"layer_{int(layer):02d}"
            / "checkpoints"
            / "checkpoint_best.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        # The online fused-operator loader uses the canonical functional name
        # for both trained A1 deltas and the zero-delta/no-A1 initialization.
        functional_path = path.parent / "checkpoint_functional_ft.pt"
        torch.save(payload, functional_path)
        manifest.append(metadata | {"checkpoint": str(path.resolve())})
        print(f"[PCA-A1-Init] K L{layer} -> {path}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
