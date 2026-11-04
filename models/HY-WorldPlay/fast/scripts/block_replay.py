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
from safetensors import safe_open


MODES = (
    "global_head",
    "grouped_head_repr_cluster",
    "grouped_head_contiguous_per_group",
    "grouped_head_contiguous",
    "spatiotemporal",
    "spatial",
)


def normalize_full(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected [B,S,H,D] or [B,H,S,D], got {tuple(x.shape)}")
    # Dumps made by dump_attention_sample keep the live [B,S,H,D] layout.
    if x.shape[2] <= 64 and x.shape[1] > x.shape[2]:
        return x.contiguous()
    if x.shape[1] <= 64 and x.shape[2] > x.shape[1]:
        return x.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"cannot infer sequence/head dimensions for {tuple(x.shape)}")


def project_rows(
    rows: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    out = torch.empty_like(rows, dtype=torch.float32)
    for start in range(0, rows.shape[0], chunk_rows):
        part = rows[start : start + chunk_rows].float()
        centered = part - mean
        out[start : start + chunk_rows] = (
            centered @ components.T
        ) @ components + mean
    return out


def reconstruct_v(
    value: torch.Tensor,
    thw: tuple[int, int, int],
    mode: str,
    artifact: dict,
    chunk_rows: int,
) -> torch.Tensor:
    """Reconstruct [B,S,H,D] with exactly the Stage-1 packing convention."""
    b, sequence, heads, dim = value.shape
    t, height, width = thw
    if sequence != t * height * width:
        raise ValueError(f"sequence/thw mismatch: {sequence} vs {thw}")
    grid = value.float().reshape(b, t, height, width, heads, dim)
    mean = artifact["mean"].to(value.device, torch.float32)
    components = artifact["components"].to(value.device, torch.float32)

    if mode == "global_head":
        rows = grid.reshape(-1, heads * dim)
        rec = project_rows(rows, mean, components, chunk_rows)
        return rec.reshape(b, sequence, heads, dim)

    if mode in {
        "grouped_head_contiguous",
        "grouped_head_contiguous_per_group",
        "grouped_head_repr_cluster",
    }:
        group_size = 4
        groups = heads // group_size
        head_groups = artifact.get("head_groups")
        if head_groups is None:
            head_groups = [
                list(range(start, start + group_size))
                for start in range(0, heads, group_size)
            ]
        head_order = torch.tensor(
            [head for group in head_groups for head in group],
            device=grid.device,
            dtype=torch.long,
        )
        ordered = grid.index_select(4, head_order)
        rows = ordered.reshape(b, t, height, width, groups, group_size, dim)
        rows = rows.reshape(-1, groups, group_size * dim)
        if mode in {
            "grouped_head_contiguous_per_group",
            "grouped_head_repr_cluster",
        }:
            rec = torch.empty_like(rows, dtype=torch.float32)
            for group in range(groups):
                rec[:, group] = project_rows(
                    rows[:, group],
                    mean[group],
                    components[group],
                    chunk_rows,
                )
        else:
            rec = project_rows(
                rows.reshape(-1, group_size * dim),
                mean,
                components,
                chunk_rows,
            ).reshape(-1, groups, group_size * dim)
        rec = rec.reshape(b, t, height, width, groups, group_size, dim).reshape(
            b, t, height, width, heads, dim
        )
        inverse_order = torch.argsort(head_order)
        return rec.index_select(4, inverse_order).reshape(b, sequence, heads, dim)

    if mode == "spatial":
        padded_height = height + (height % 2)
        padded_width = width + (width % 2)
        work = grid
        if padded_width != width:
            work = torch.cat([work, work[:, :, :, -1:, :, :]], dim=3)
        if padded_height != height:
            work = torch.cat([work, work[:, :, -1:, :, :, :]], dim=2)
        rows = work.reshape(
            b, t, padded_height // 2, 2, padded_width // 2, 2, heads, dim
        ).permute(0, 1, 2, 4, 6, 3, 5, 7)
        shape = rows.shape
        rec = project_rows(rows.reshape(-1, 4 * dim), mean, components, chunk_rows)
        rec = rec.reshape(shape).permute(0, 1, 2, 5, 3, 6, 4, 7)
        rec = rec.reshape(b, t, padded_height, padded_width, heads, dim)
        return rec[:, :, :height, :width].reshape(b, sequence, heads, dim)

    if mode == "spatiotemporal":
        padded_t = t + (t % 2)
        padded_height = height + (height % 2)
        padded_width = width + (width % 2)
        work = grid
        if padded_width != width:
            work = torch.cat([work, work[:, :, :, -1:, :, :]], dim=3)
        if padded_height != height:
            work = torch.cat([work, work[:, :, -1:, :, :, :]], dim=2)
        if padded_t != t:
            work = torch.cat([work, work[:, -1:, :, :, :, :]], dim=1)
        rows = work.reshape(
            b,
            padded_t // 2,
            2,
            padded_height // 2,
            2,
            padded_width // 2,
            2,
            heads,
            dim,
        ).permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        shape = rows.shape
        rec = project_rows(rows.reshape(-1, 8 * dim), mean, components, chunk_rows)
        rec = rec.reshape(shape).permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
        rec = rec.reshape(
            b, padded_t, padded_height, padded_width, heads, dim
        )
        return rec[:, :t, :height, :width].reshape(b, sequence, heads, dim)

    raise ValueError(mode)


def attention(q, vision_k, vision_v, text_k, text_v):
    # Inputs to SDPA are [B,H,S,D].
    q = q.float()
    vision_k = vision_k.permute(0, 2, 1, 3).float()
    vision_v = vision_v.permute(0, 2, 1, 3).float()
    if text_k is not None:
        key = torch.cat([text_k.float(), vision_k], dim=2)
        value = torch.cat([text_v.float(), vision_v], dim=2)
    else:
        key, value = vision_k, vision_v
    return F.scaled_dot_product_attention(
        q, key, value, dropout_p=0.0, is_causal=False
    )


def metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict:
    ref = reference.float()
    est = estimate.float()
    diff = est - ref
    return {
        "rel_l2": float(diff.norm().div(ref.norm().clamp_min(1e-12)).item()),
        "mse": float(diff.square().mean().item()),
        "cosine": float(
            F.cosine_similarity(
                ref.reshape(-1, ref.shape[-1]),
                est.reshape(-1, est.shape[-1]),
                dim=-1,
                eps=1e-8,
            ).mean().item()
        ),
    }


def load_block_weights(path: str, layer: int, device: torch.device) -> dict:
    prefix = f"double_blocks.{int(layer)}."
    names = (
        "img_mod.linear.weight",
        "img_mod.linear.bias",
        "img_attn_proj.weight",
        "img_attn_proj.bias",
        "img_attn_prope_proj.weight",
        "img_attn_prope_proj.bias",
        "img_mlp.fc1.weight",
        "img_mlp.fc1.bias",
        "img_mlp.fc2.weight",
        "img_mlp.fc2.bias",
    )
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name: handle.get_tensor(prefix + name).to(device=device, dtype=torch.float32)
            for name in names
        }


def replay_block(
    layer_input: torch.Tensor,
    vec: torch.Tensor,
    main_readout: torch.Tensor,
    prope_readout: torch.Tensor,
    weights: dict,
) -> torch.Tensor:
    """Replay the image residual/MLP portion for sampled same-state tokens."""
    x = layer_input.float()
    vec = vec.float()
    modulation = F.linear(
        F.silu(vec),
        weights["img_mod.linear.weight"],
        weights["img_mod.linear.bias"],
    )
    (
        _mod1_shift,
        _mod1_scale,
        mod1_gate,
        mod2_shift,
        mod2_scale,
        mod2_gate,
    ) = modulation.chunk(6, dim=-1)
    main_flat = main_readout.permute(0, 2, 1, 3).reshape(
        main_readout.shape[0], main_readout.shape[2], -1
    ).float()
    attention_update = F.linear(
        main_flat,
        weights["img_attn_proj.weight"],
        weights["img_attn_proj.bias"],
    ) + F.linear(
        prope_readout.float(),
        weights["img_attn_prope_proj.weight"],
        weights["img_attn_prope_proj.bias"],
    )
    x = x + attention_update * mod1_gate.unsqueeze(0)
    mlp_input = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    mlp_input = (
        mlp_input * (1.0 + mod2_scale.unsqueeze(0))
        + mod2_shift.unsqueeze(0)
    )
    mlp = F.linear(
        F.gelu(
            F.linear(
                mlp_input,
                weights["img_mlp.fc1.weight"],
                weights["img_mlp.fc1.bias"],
            ),
            approximate="tanh",
        ),
        weights["img_mlp.fc2.weight"],
        weights["img_mlp.fc2.bias"],
    )
    return x + mlp * mod2_gate.unsqueeze(0)




