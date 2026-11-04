# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Export all 54 joint-semantic V-QAT layers into resident codec format."""
import argparse
from pathlib import Path
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()
SOURCE = args.source
OUT = args.output_root
BASIS_ROOT, LINEAR_ROOT = OUT / "basis", OUT / "linear"
manifest = []

for layer in range(54):
    source = SOURCE / f"layer_{layer:02d}" / "checkpoint_best.pt"
    payload = torch.load(source, map_location="cpu")
    state = payload["model"]
    rank = int(payload.get("rank", state["basis"].shape[0]))
    if tuple(state["basis"].shape) != (rank, 2048):
        raise RuntimeError(f"L{layer} rank/state mismatch: rank={rank} basis={tuple(state['basis'].shape)}")
    encoder = (state["basis"] + state["encoder_delta"]).float()
    decoder = (state["basis"].T + state["decoder_delta"]).float()
    basis_path = BASIS_ROOT / f"layer_{layer:02d}" / "global_head_pca.pt"
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 3, "candidate": "global_head", "tensor_name": "v",
        "layer": layer, "mean": state["mean"].float(), "components": encoder,
        "input_dim": 2048, "latent_dim": rank,
        "fit_name": "joint_semantic64_residual_qat", "qat_source": str(source.resolve()),
    }, basis_path)
    linear_path = LINEAR_ROOT / f"layer_{layer:02d}" / "checkpoints" / "checkpoint_functional_ft.pt"
    linear_path.parent.mkdir(parents=True, exist_ok=True)
    delta = decoder - encoder.T
    torch.save({
        "model_kind": "global_pca_int4_linear_a1",
        "model": {"decoder_delta.weight": delta},
        "model_config": {"architecture": f"global_pca_linear_d{rank}", "latent_dim": rank},
        "layer_id": layer, "target": "v", "source_checkpoint": str(source.resolve()),
        "optimizer": None, "optimizer_type": "semantic_qat_export",
        "codec_metadata": {
            "target": "v", "layer": layer, "rank": rank, "coeff_quant": "int4",
            "scale_dtype": "bf16", "basis_root": str(BASIS_ROOT.resolve()),
            "basis_file": str(basis_path.resolve()),
        },
    }, linear_path)
    probe = torch.randn(8, 2048)
    z = torch.nn.functional.linear(probe - state["mean"].float(), encoder)
    direct = torch.nn.functional.linear(z, decoder, state["mean"].float())
    resident = torch.nn.functional.linear(z, encoder.T, state["mean"].float()) + torch.nn.functional.linear(z, delta)
    gap = float((direct - resident).abs().max())
    if gap > 1e-4:
        raise RuntimeError(f"L{layer} export gap={gap}")
    manifest.append({"layer": layer, "rank": rank, "source": str(source), "basis": str(basis_path),
                     "linear": str(linear_path), "equivalence_max_abs": gap})

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"exported {len(manifest)} layers; max_gap={max(x['equivalence_max_abs'] for x in manifest):.3e}")
