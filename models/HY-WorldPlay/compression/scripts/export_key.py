# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Export the 54 K-C64 residual PCA artifacts for the resident codec."""
import argparse
from pathlib import Path
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("kv_ae_outputs/ae16x_v2/k_qvg_joint_smooth_pca_all54"))
parser.add_argument("--output-root", type=Path, default=Path("kv_ae_outputs/ae16x_v2/k_qvg_joint_smooth_pca_all54_resident"))
args = parser.parse_args()
SOURCE = args.source
OUT = args.output_root / "basis"
manifest = []

for layer in range(54):
    source = SOURCE / f"layer_{layer:02d}" / "residual_pca512.pt"
    payload = torch.load(source, map_location="cpu")
    destination = OUT / f"layer_{layer:02d}" / "global_head_pca.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = {
        "format_version": 3,
        "candidate": "global_head",
        "tensor_name": "k",
        "layer": layer,
        "mean": payload["mean"].float(),
        # Stored in ascending retained-eigenvalue order; runtime rank493 uses
        # the suffix, exactly matching a direct PCA493 fit.
        "components": payload["components"].float(),
        "input_dim": 2048,
        "latent_dim": 512,
        "fit_name": "joint_semantic64_residual_pca",
        "source": str(source.resolve()),
    }
    torch.save(exported, destination)
    manifest.append({"layer": layer, "source": str(source), "basis": str(destination)})

manifest_path = args.output_root / "manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"exported {len(manifest)} K semantic residual PCA layers")
