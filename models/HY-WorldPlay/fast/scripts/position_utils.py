"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from hyvideo.models.transformers.modules.posemb_layers import (
    apply_rotary_emb_one,
    rotate_half,
)
from hyvideo.prope.camera_rope import _prepare_apply_fns_all_dim


RANKS = (16, 24, 32, 48, 64, 96, 128)
FUNCTIONAL_LAYERS = (13, 40, 53)


def load_final_operator(basis_root: Path, linear_root: Path, layer: int, target: str):
    artifact_path = basis_root / f"layer_{layer:02d}" / "global_head_pca.pt"
    artifact = torch.load(artifact_path, map_location="cpu")
    mean = artifact["mean"].float()
    components = artifact["components"].float()[-512:]
    decoder = components.T.contiguous()
    delta_path = (
        linear_root
        / f"layer_{layer:02d}"
        / "checkpoints"
        / "checkpoint_functional_ft.pt"
    )
    checkpoint = torch.load(delta_path, map_location="cpu")
    delta = checkpoint["model"]["decoder_delta.weight"].float()
    if delta.shape != decoder.shape:
        raise ValueError(f"bad {target} decoder delta at {delta_path}: {delta.shape}")
    return {
        "mean": mean,
        "encoder": components,
        "decoder": decoder + delta,
        "basis_path": str(artifact_path),
        "linear_path": str(delta_path),
    }


def numerical_rank_rows(operator, thresholds=(0.99, 0.999)):
    rows = []
    for head in range(16):
        matrix = operator[head * 128 : (head + 1) * 128]
        singular = torch.linalg.svdvals(matrix.double()).square()
        cumulative = singular.cumsum(0) / singular.sum().clamp_min(1e-30)
        row = {"head": head}
        for threshold in thresholds:
            rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
            row[f"rank_{str(threshold).replace('.', 'p')}"] = rank
        row["condition"] = float((singular[0] / singular[-1].clamp_min(1e-30)).sqrt())
        rows.append(row)
    return rows


def normalize_bhsd(x):
    if x.ndim != 4:
        raise ValueError(tuple(x.shape))
    if x.shape[1] == 16:
        return x.contiguous()
    if x.shape[2] == 16:
        return x.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"cannot normalize shape {tuple(x.shape)}")


def inverse_rope(k_rope, cos, sin):
    cos = cos.to(k_rope.device, torch.float32)[None, None]
    sin = sin.to(k_rope.device, torch.float32)[None, None]
    x = k_rope.float()
    # The model pairs adjacent real/imag channels, so use its exact rotation
    # convention. R(theta)^-1 x = cos(theta)x - sin(theta)Jx.
    return x * cos - rotate_half(x) * sin


def encode_int4(pre_bhsd, operator):
    rows = pre_bhsd.permute(0, 2, 1, 3).reshape(-1, 2048).float()
    mean = operator["mean"].to(rows.device)
    encoder = operator["encoder"].to(rows.device)
    latent = F.linear(rows - mean, encoder)
    scale = latent.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 7.0
    scale = scale.to(torch.bfloat16).float()
    latent = (latent / scale).round().clamp(-7, 7) * scale
    return latent.reshape(pre_bhsd.shape[0], pre_bhsd.shape[2], 512)


def decode_latent(latent, operator, decoder=None):
    decoder = operator["decoder"].to(latent.device) if decoder is None else decoder
    mean = operator["mean"].to(latent.device)
    rows = F.linear(latent.reshape(-1, 512).float(), decoder, mean)
    return rows.reshape(latent.shape[0], latent.shape[1], 16, 128).permute(0, 2, 1, 3)


def rope_selected(pre, payload, suffix="k"):
    freqs = (
        payload[f"freqs_cos_{suffix}"].to(pre.device, pre.dtype),
        payload[f"freqs_sin_{suffix}"].to(pre.device, pre.dtype),
    )
    return apply_rotary_emb_one(pre.transpose(1, 2), freqs, head_first=False).transpose(1, 2)


def prope_functions(payload, suffix, head_dim, device):
    viewmats = payload[f"viewmats_{suffix}"].to(device=device, dtype=torch.float32)
    Ks = payload.get(f"Ks_{suffix}")
    Ks = Ks.to(device=device, dtype=torch.float32) if Ks is not None else None
    return _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=None,
        patches_y=None,
        image_width=None,
        image_height=None,
    )


def position_kv(pre_k, pre_v, payload):
    k_main = rope_selected(pre_k, payload, "k")
    _, apply_kv, _ = prope_functions(payload, "k", pre_k.shape[-1], pre_k.device)
    return k_main, pre_v, apply_kv(pre_k.float()), apply_kv(pre_v.float())


def restore_prope_output(raw, payload):
    _, _, apply_o = prope_functions(payload, "q", raw.shape[-1], raw.device)
    restored = apply_o(raw.float())
    return restored.permute(0, 2, 1, 3).reshape(restored.shape[0], restored.shape[2], -1)


def attention_state(q, k, v, text_k, text_v):
    q = q.float()
    k_all = torch.cat([text_k.float(), k.float()], dim=2)
    v_all = torch.cat([text_v.float(), v.float()], dim=2)
    logits = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    probs = logits.softmax(dim=-1)
    output = torch.matmul(probs, v_all)
    return {"logits": logits, "probs": probs, "output": output, "text_len": text_k.shape[2]}


def second_moment(rows, dimension):
    if not rows:
        return torch.eye(dimension, dtype=torch.float64)
    work = torch.cat(rows, dim=0).double()
    return work.T @ work / max(work.shape[0], 1)


def symmetric_sqrt(matrix, floor_ratio=1e-6):
    values, vectors = torch.linalg.eigh(matrix.double())
    floor = values.max().clamp_min(1e-20) * floor_ratio
    values = values.clamp_min(floor)
    sqrt = (vectors * values.sqrt()[None]) @ vectors.T
    invsqrt = (vectors * values.rsqrt()[None]) @ vectors.T
    return sqrt, invsqrt


def weighted_factors(decoder_head, left_moment, right_moment):
    left_sqrt, left_inv = symmetric_sqrt(left_moment)
    right_sqrt, right_inv = symmetric_sqrt(right_moment)
    weighted = left_sqrt @ decoder_head.double() @ right_sqrt
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    factors = {}
    for rank in RANKS:
        root_s = singular[:rank].sqrt()
        left = left_inv @ (u[:, :rank] * root_s[None])
        right = (root_s[:, None] * vh[:rank]) @ right_inv
        factors[rank] = (left.float(), right.float())
    energy = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-30)
    return factors, singular.float(), energy.float()


def compose_decoder(factors_by_head, rank, device):
    heads = []
    for head in range(16):
        left, right = factors_by_head[head][rank]
        heads.append((left @ right).to(device))
    return torch.cat(heads, dim=0)


def rel_l2(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))


def distribution_metrics(candidate, teacher):
    p = teacher["probs"].float()
    q = candidate["probs"].float()
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
        + (q * (q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
    ).mean()
    topk = min(32, p.shape[-1])
    a = teacher["logits"].topk(topk, dim=-1).indices
    b = candidate["logits"].topk(topk, dim=-1).indices
    recall = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()
    return float(js), float(recall)


def load_block_helpers():
    from block_replay import load_block_weights, replay_block
    return load_block_weights, replay_block


def sample_paths(root: Path, layers):
    paths = sorted(root.rglob("*_attn_sample.pt"))
    return [path for path in paths if any(f"layer{layer}_" in path.name for layer in layers)]


def extract_pre_and_latent(payload, k_operator, v_operator, device):
    k_rope = normalize_bhsd(payload["k_rope"]).to(device)
    pre_k = inverse_rope(k_rope, payload["freqs_cos_k"], payload["freqs_sin_k"])
    rope_closure = rel_l2(rope_selected(pre_k, payload, "k"), k_rope)
    if rope_closure >= 1e-3:
        raise RuntimeError(f"inverse RoPE closure failed: rel-L2={rope_closure:.6g}")
    pre_v = normalize_bhsd(payload["v_main"]).to(device).float()
    z_k = encode_int4(pre_k, k_operator)
    z_v = encode_int4(pre_v, v_operator)
    return pre_k, pre_v, z_k, z_v


def fit_layer(layer, paths, k_operator, v_operator, device):
    z_k_rows, q_by_head = [], [[] for _ in range(16)]
    bar_v_by_head = [[] for _ in range(16)]
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        _, _, z_k, z_v = extract_pre_and_latent(payload, k_operator, v_operator, device)
        z_k_rows.append(z_k.reshape(-1, 512).cpu())
        full_k = decode_latent(z_k, k_operator)
        full_v = decode_latent(z_v, v_operator)
        k_main, v_main, k_prope, v_prope = position_kv(full_k, full_v, payload)
        text_k = payload["text_k"].to(device)
        text_v = payload["text_v"].to(device)
        q_main = payload["q_rope"].to(device)
        q_prope = payload["q_prope"].to(device)
        main = attention_state(q_main, k_main, v_main, text_k, text_v)
        prope = attention_state(q_prope, k_prope, v_prope, text_k, text_v)
        for head in range(16):
            q_by_head[head].append(q_main[0, head].cpu())
            q_by_head[head].append(q_prope[0, head].cpu())
            for state in (main, prope):
                vision_prob = state["probs"][0, head, :, state["text_len"] :]
                bar = vision_prob @ z_v[0]
                bar_v_by_head[head].append(bar.cpu())
        del payload, full_k, full_v, main, prope
    cz_k = second_moment(z_k_rows, 512)
    k_factors, v_factors = {}, {}
    spectral = {"k": [], "v": []}
    identity = torch.eye(128, dtype=torch.float64)
    for head in range(16):
        cq = second_moment(q_by_head[head], 128)
        cbar = second_moment(bar_v_by_head[head], 512)
        d_k = k_operator["decoder"][head * 128 : (head + 1) * 128]
        d_v = v_operator["decoder"][head * 128 : (head + 1) * 128]
        k_factors[head], k_s, k_energy = weighted_factors(d_k, cq, cz_k)
        v_factors[head], v_s, v_energy = weighted_factors(d_v, identity, cbar)
        spectral["k"].append({"singular": k_s, "energy": k_energy})
        spectral["v"].append({"singular": v_s, "energy": v_energy})
    return k_factors, v_factors, spectral


def evaluate_layer(
    layer,
    paths,
    k_operator,
    v_operator,
    k_factors,
    v_factors,
    block_weights,
    replay_block,
    device,
):
    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        _, _, z_k, z_v = extract_pre_and_latent(payload, k_operator, v_operator, device)
        full_k = decode_latent(z_k, k_operator)
        full_v = decode_latent(z_v, v_operator)
        teacher_positioned = position_kv(full_k, full_v, payload)
        text_k = payload["text_k"].to(device)
        text_v = payload["text_v"].to(device)
        q_main = payload["q_rope"].to(device)
        q_prope = payload["q_prope"].to(device)
        teacher_main = attention_state(q_main, teacher_positioned[0], teacher_positioned[1], text_k, text_v)
        teacher_prope = attention_state(q_prope, teacher_positioned[2], teacher_positioned[3], text_k, text_v)
        teacher_prope_restored = restore_prope_output(teacher_prope["output"], payload)
        baseline_block = replay_block(
            payload["layer_input"].to(device),
            payload["vec"].to(device),
            teacher_main["output"],
            teacher_prope_restored,
            block_weights,
        )
        for rank in RANKS:
            d_k = compose_decoder(k_factors, rank, device)
            d_v = compose_decoder(v_factors, rank, device)
            candidate_k = decode_latent(z_k, k_operator, d_k)
            candidate_v = decode_latent(z_v, v_operator, d_v)
            positioned = position_kv(candidate_k, candidate_v, payload)
            # K-only states use teacher V; V-only readout uses teacher probabilities.
            k_main_state = attention_state(q_main, positioned[0], teacher_positioned[1], text_k, text_v)
            k_prope_state = attention_state(q_prope, positioned[2], teacher_positioned[3], text_k, text_v)
            v_main_all = torch.cat([text_v.float(), positioned[1].float()], dim=2)
            v_prope_all = torch.cat([text_v.float(), positioned[3].float()], dim=2)
            v_main_output = teacher_main["probs"] @ v_main_all
            v_prope_output = teacher_prope["probs"] @ v_prope_all
            kv_main = attention_state(q_main, positioned[0], positioned[1], text_k, text_v)
            kv_prope = attention_state(q_prope, positioned[2], positioned[3], text_k, text_v)
            candidate_prope_restored = restore_prope_output(kv_prope["output"], payload)
            candidate_block = replay_block(
                payload["layer_input"].to(device),
                payload["vec"].to(device),
                kv_main["output"],
                candidate_prope_restored,
                block_weights,
            )
            js_main, top_main = distribution_metrics(k_main_state, teacher_main)
            js_prope, top_prope = distribution_metrics(k_prope_state, teacher_prope)
            rows.append(
                {
                    "case_id": payload.get("case_id", path.parent.parent.name),
                    "layer": layer,
                    "rank": rank,
                    "k_logit_rel_l2_main": rel_l2(k_main_state["logits"], teacher_main["logits"]),
                    "k_logit_rel_l2_prope": rel_l2(k_prope_state["logits"], teacher_prope["logits"]),
                    "softmax_js_main": js_main,
                    "softmax_js_prope": js_prope,
                    "top32_recall_main": top_main,
                    "top32_recall_prope": top_prope,
                    "v_readout_rel_l2_main": rel_l2(v_main_output, teacher_main["output"]),
                    "v_readout_rel_l2_prope": rel_l2(v_prope_output, teacher_prope["output"]),
                    "kv_readout_rel_l2_main": rel_l2(kv_main["output"], teacher_main["output"]),
                    "kv_readout_rel_l2_prope": rel_l2(kv_prope["output"], teacher_prope["output"]),
                    "block_output_rel_l2": rel_l2(candidate_block, baseline_block),
                }
            )
        del payload
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["rank"])].append(row)
    output = []
    metrics = [key for key in rows[0] if key not in {"case_id", "layer", "rank"}]
    for (layer, rank), items in sorted(grouped.items()):
        result = {"layer": layer, "rank": rank, "samples": len(items)}
        for metric in metrics:
            values = torch.tensor([item[metric] for item in items])
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_max"] = float(values.max())
        output.append(result)
    return output


def roofline_rows(rank):
    # Real first-recache shape from the audit dumps.
    nq, nk, heads, dim, latent = 6360, 6360, 16, 128, 512
    decode_flops = 2 * nk * 2048 * latent * 2  # K and V
    dense_attention = 4 * nq * nk * heads * dim  # QK and PV
    latent_attention = 4 * nq * nk * heads * rank
    project_cache = 2 * nk * heads * rank * latent * 2
    expand_output = 2 * nq * heads * dim * rank
    return {
        "rank": rank,
        "cache_elements_vs_z512": heads * rank / latent,
        "attention_flops_ratio": rank / dim,
        "materialized_head_latent_flops_ratio_vs_decode_plus_attention": (
            project_cache + latent_attention + expand_output
        ) / (decode_flops + dense_attention),
        "exact_z512_attention_flops_ratio_vs_decode_plus_attention": (
            4 * nq * nk * heads * latent
        ) / (decode_flops + dense_attention),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, aggregate_rows, roofline):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for layer in FUNCTIONAL_LAYERS:
        data = [row for row in aggregate_rows if row["layer"] == layer]
        ranks = [row["rank"] for row in data]
        axes[0].plot(ranks, [row["block_output_rel_l2_mean"] for row in data], marker="o", label=f"L{layer}")
        axes[1].plot(ranks, [0.5 * (row["kv_readout_rel_l2_main_mean"] + row["kv_readout_rel_l2_prope_mean"]) for row in data], marker="o", label=f"L{layer}")
    axes[2].plot([row["rank"] for row in roofline], [row["materialized_head_latent_flops_ratio_vs_decode_plus_attention"] for row in roofline], marker="o", color="black")
    axes[0].set_title("Same-state block error")
    axes[1].set_title("KV readout error")
    axes[2].set_title("Theoretical FLOPs ratio")
    for axis in axes:
        axis.set_xlabel("per-head latent rank")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("relative L2")
    axes[1].set_ylabel("relative L2")
    axes[2].set_ylabel("candidate / decode+attention")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)




