# NanoKV public workflow entrypoint.
#!/usr/bin/env python3
"""Calibrate one LingBot codec-native reader on sampled formal A1-codec Q/K/V."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def rel_mse(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (value - target).square().mean() / target.square().mean().clamp_min(1e-8)


def centered_rel_mse(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rel_mse(value - value.mean(-1, keepdim=True), target - target.mean(-1, keepdim=True))


def selected_multiplier(positions, grid_sizes, freqs, frame_seqlen, indices):
    _, _, width = map(int, grid_sizes[0].tolist())
    original_half = freqs.shape[1]
    split = [original_half - 2 * (original_half // 3), original_half // 3, original_half // 3]
    temporal, vertical, horizontal = freqs.split(split, dim=1)
    frame = torch.div(positions, frame_seqlen, rounding_mode="floor")
    spatial = positions.remainder(frame_seqlen)
    row = torch.div(spatial, width, rounding_mode="floor")
    column = spatial.remainder(width)
    multiplier = torch.cat((temporal[frame], vertical[row], horizontal[column]), 1)
    return multiplier[:, indices[:, ::2] // 2].permute(1, 0, 2)


def rope_low(value, positions, grid_sizes, freqs, frame_seqlen, indices):
    multiplier = selected_multiplier(positions, grid_sizes, freqs, frame_seqlen, indices)
    pair = torch.view_as_complex(value.double().reshape(*value.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(pair * multiplier[None]).flatten(-2).to(value.dtype)


class ReaderFit(nn.Module):
    def __init__(self, group: dict[str, torch.Tensor]):
        super().__init__()
        self.register_buffer("indices", group["indices"].long())
        self.q_reduce = nn.Parameter(group["q_reduce"].float())
        self.k_affine = nn.Parameter(group["k_affine"].float())
        self.v_affine = nn.Parameter(group["v_affine"].float())
        self.lift = nn.Parameter(group["lift"].float())

    def forward(self, case: dict[str, torch.Tensor]):
        q = case["q_pre"].float().cuda(non_blocking=True)
        k = case["k_pre"].float().cuda(non_blocking=True)
        v = case["v"].float().cuda(non_blocking=True)
        q_low = torch.einsum("bhqd,hde->bhqe", q, self.q_reduce)
        k_low = torch.einsum("bhkd,hde->bhke", k, self.k_affine)
        q_low = rope_low(
            q_low, case["q_positions"].cuda(), case["grid_sizes"].cuda(),
            case["freqs"].cuda(), int(case["frame_seqlen"]), self.indices,
        )
        k_low = rope_low(
            k_low, case["k_positions"].cuda(), case["grid_sizes"].cuda(),
            case["freqs"].cuda(), int(case["frame_seqlen"]), self.indices,
        )
        teacher_q = case["q_native"].float().cuda(non_blocking=True)
        teacher_k = case["k_native"].float().cuda(non_blocking=True)
        teacher_logits = teacher_q @ teacher_k.transpose(-1, -2) / math.sqrt(128)
        logits = q_low @ k_low.transpose(-1, -2) / math.sqrt(128)
        lse = rel_mse(torch.logsumexp(logits, -1), torch.logsumexp(teacher_logits, -1))
        teacher_output = teacher_logits.softmax(-1) @ v
        value_low = torch.einsum("bhkd,hde->bhke", v, self.v_affine)
        output = torch.einsum("bhqe,hed->bhqd", logits.softmax(-1) @ value_low, self.lift)
        reconstruction = torch.einsum("bhke,hed->bhkd", value_low, self.lift)
        return (
            centered_rel_mse(logits, teacher_logits),
            lse,
            rel_mse(output, teacher_output),
            rel_mse(reconstruction, v),
        )


def unpack_decoder(entry: dict) -> torch.Tensor:
    packed = entry["decoder"]
    return packed["qweight"].float() * packed["scale"].float()




