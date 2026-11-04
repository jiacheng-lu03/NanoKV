"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import pca_utils as oracle
import sample_dataset as full96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rows(
    config, split: str, layer: int, rows_per_atom: int, seed: int, target: str
):
    parts = []
    audit = []
    root = Path(config["data_root"])
    for index, atom in enumerate(config[f"{split}_atoms"]):
        path = oracle.atom_path(root, split, atom, layer)
        tensor, thw = oracle.load_raw_bf16(path)
        atom_seed = oracle.stable_seed(
            seed, f"{target}_pca16_int4_qat", split, index,
            atom["case_id"], atom["position"], layer,
        )
        rows, _, _ = oracle.pack_samples(
            tensor, thw, "global_head", rows_per_atom, atom_seed
        )
        parts.append(rows.float())
        audit.append(
            {
                "case_id": atom["case_id"],
                "position": int(atom["position"]),
                "path": str(path),
                "rows": int(rows.shape[0]),
            }
        )
    return torch.cat(parts), audit


class PCALinearInt4QAT(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        rank: int,
        mode: str,
        initialization: str,
    ):
        super().__init__()
        reference_basis = components[-rank:].float().clone()
        if initialization == "pca":
            basis = reference_basis.clone()
        elif initialization == "random_orthogonal":
            # Orthonormal rows give a numerically stable from-scratch linear
            # autoencoder without leaking the PCA subspace into initialization.
            basis = torch.linalg.qr(
                torch.randn(reference_basis.shape[1], rank, dtype=torch.float32),
                mode="reduced",
            ).Q.T.contiguous()
        else:
            raise ValueError(initialization)
        self.rank = int(rank)
        self.mode = mode
        self.initialization = initialization
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("basis", basis)
        self.register_buffer("reference_basis", reference_basis)
        self.encoder_delta = nn.Parameter(torch.zeros_like(basis))
        self.decoder_delta = nn.Parameter(torch.zeros_like(basis.T))
        if mode == "decoder":
            self.encoder_delta.requires_grad_(False)
        elif mode != "encoder_decoder":
            raise ValueError(mode)

    def exact_quantize(self, z: torch.Tensor):
        scale = z.float().abs().amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 7.0
        codes = (z.float() / scale).round().clamp(-7, 7)
        # Match resident runtime: per-token scale is stored in BF16.
        stored_scale = scale.to(torch.bfloat16).float()
        return codes * stored_scale, codes, stored_scale

    def forward(self, x: torch.Tensor):
        encoder = self.basis + self.encoder_delta
        decoder = self.basis.T + self.decoder_delta
        z = F.linear(x.float() - self.mean, encoder)
        z_exact, codes, scale = self.exact_quantize(z)
        zq = z + (z_exact - z).detach() if self.training else z_exact
        pred = F.linear(zq, decoder, self.mean)
        return pred, z, z_exact, codes, scale

    @torch.no_grad()
    def base_int4(self, x: torch.Tensor):
        z = F.linear(x.float() - self.mean, self.reference_basis)
        zq, _, _ = self.exact_quantize(z)
        return F.linear(zq, self.reference_basis.T, self.mean)

    @torch.no_grad()
    def base_bf16(self, x: torch.Tensor):
        z = F.linear(x.float() - self.mean, self.reference_basis)
        return F.linear(
            z.to(torch.bfloat16).float(), self.reference_basis.T, self.mean
        )


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.float()
    target = target.float()
    l1 = (pred - target).abs().mean() / target.abs().mean().clamp_min(1e-12)
    rel_l2_sq = (pred - target).square().sum() / target.square().sum().clamp_min(1e-20)
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return l1 + 0.2 * rel_l2_sq + cosine


@torch.no_grad()
def evaluate(model, rows, audit, device, batch_size):
    model.eval()
    totals = {
        key: {"num": 0.0, "den": 0.0, "cos": 0.0}
        for key in ("pca_bf16", "pca_int4", "qat")
    }
    per_atom = []
    offset = 0
    quant_num = quant_den = 0.0
    for item in audit:
        count = int(item["rows"])
        source = rows[offset : offset + count]
        offset += count
        atom = {
            key: {"num": 0.0, "den": 0.0, "cos": 0.0}
            for key in totals
        }
        for start in range(0, count, batch_size):
            target = source[start : start + batch_size].to(device)
            qat, z, zq, _, _ = model(target)
            preds = {
                "pca_bf16": model.base_bf16(target),
                "pca_int4": model.base_int4(target),
                "qat": qat,
            }
            quant_num += float((zq - z).square().sum().cpu())
            quant_den += float(z.square().sum().cpu())
            for key, pred in preds.items():
                num = float((pred.float() - target.float()).square().sum().cpu())
                den = float(target.float().square().sum().cpu())
                cos = float(
                    F.cosine_similarity(pred.float(), target.float(), dim=-1).sum().cpu()
                )
                atom[key]["num"] += num
                atom[key]["den"] += den
                atom[key]["cos"] += cos
                totals[key]["num"] += num
                totals[key]["den"] += den
                totals[key]["cos"] += cos
        metrics = {}
        for key, value in atom.items():
            metrics[key] = {
                "rel_l2": (value["num"] / max(value["den"], 1e-20)) ** 0.5,
                "cosine": value["cos"] / count,
            }
        metrics["qat_improvement_vs_int4_percent"] = 100.0 * (
            metrics["pca_int4"]["rel_l2"] - metrics["qat"]["rel_l2"]
        ) / metrics["pca_int4"]["rel_l2"]
        per_atom.append({**item, **metrics})
    result = {}
    for key, value in totals.items():
        result[key] = {
            "rel_l2": (value["num"] / max(value["den"], 1e-20)) ** 0.5,
            "cosine": value["cos"] / max(1, rows.shape[0]),
        }
    result["qat_improvement_vs_int4_percent"] = 100.0 * (
        result["pca_int4"]["rel_l2"] - result["qat"]["rel_l2"]
    ) / result["pca_int4"]["rel_l2"]
    result["quant_rel_l2"] = (quant_num / max(quant_den, 1e-20)) ** 0.5
    result["positive_atoms"] = sum(
        item["qat_improvement_vs_int4_percent"] > 0 for item in per_atom
    )
    result["atom_count"] = len(per_atom)
    result["atoms"] = per_atom
    return result




