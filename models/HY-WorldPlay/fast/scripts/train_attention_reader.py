#!/usr/bin/env python3
"""Train codec-native dictionary + residual low-dimensional Attention.

This is the first training gate that follows the formal codec decomposition
exactly::

    x_j = (semantic_center[id_j] + pca_mean) + decoder @ z_j

The 128-D C64/mean anchor is preserved exactly.  Only the decoder residual is
read from the existing rank-256 or rank-512 INT4 latent into a smaller
Attention space.
Current-chunk and text K/V remain native 128-D and all three segments share a
single softmax.  The prototype deliberately materializes assigned anchors to
validate quality; ``audit_c64_anchor_factorization.py`` proves that the same
anchor terms can be evaluated by low-cardinality tables in a fused kernel.

Main 3D-RoPE K keeps complete native frequency pairs.  PRoPE uses maps of the
form ``A x I4`` so that dimensionality reduction commutes with every tiled
4x4 camera transform.  No full 2048-D residual K/V reconstruction is consumed
by the student.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from position_utils import (
    prope_functions,
    rel_l2,
    restore_prope_output,
)
from attention_utils import load_operator, normalized_mse
from attention_samples import load_case as load_formal_case
from block_replay import load_block_weights, replay_block


HEADS = 16
HEAD_DIM = 128
LATENT_DIM = 512
AXIS_PAIR_RANGES = ((0, 8), (8, 36), (36, 64))  # T16, H56, W56.


def _gather_last_per_head(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a distinct ordered channel list for each head."""
    if x.shape[1] != indices.shape[0]:
        raise ValueError((tuple(x.shape), tuple(indices.shape)))
    view = indices[None, :, None].expand(x.shape[0], -1, x.shape[2], -1)
    return torch.gather(x, -1, view)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape_as(x)


def _apply_selected_rope(
    pre: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    # ``pre`` may already be the low decoder readout.  ``indices`` always
    # refers to its represented native RoPE channels and is therefore still
    # needed to gather the matching frequencies.
    selected = pre if pre.shape[-1] == indices.shape[-1] else _gather_last_per_head(pre, indices)
    # Position metadata is shared by heads: [S,128] -> [H,S,D].
    per_head = indices[:, None, :].expand(-1, cos.shape[0], -1)
    selected_cos = torch.gather(cos.float()[None].expand(HEADS, -1, -1), -1, per_head)
    selected_sin = torch.gather(sin.float()[None].expand(HEADS, -1, -1), -1, per_head)
    return selected.float() * selected_cos[None] + _rotate_pairs(selected.float()) * selected_sin[None]


def _decode_low(latent: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[B,N,512] x [H,D,512] -> [B,H,N,D]."""
    return torch.einsum("bnr,hdr->bhnd", latent.float(), weight.float())


def _assigned(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[B,H,C,D] + [B,N] IDs -> [B,H,N,D]."""
    index = ids[:, None, :, None].expand(-1, table.shape[1], -1, table.shape[-1])
    return torch.gather(table, 2, index)


def _anchor_table(case: dict, operator: dict, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    raw = case["raw"]
    payload = case.get(f"{target}_payload")
    if payload is None or "semantic_centers" not in payload:
        payload = raw.get(f"history_{target}_semantic_payload")
    if payload is None:
        payload = raw[f"history_{target}_formal_payload"]
    centers = payload["semantic_centers"].float()
    ids = payload["semantic_ids"].long()
    # Formal decoder uses the PCA mean as a BF16 GEMM bias.  Match that stored
    # numerical value instead of silently upgrading this one term to FP32.
    formal_mean = operator["mean"].to(torch.bfloat16).float()
    table = centers + formal_mean[None, None, :]
    table = table.reshape(table.shape[0], table.shape[1], HEADS, HEAD_DIM)
    return table.permute(0, 2, 1, 3).contiguous(), ids


def _balanced_pair_quota(pair_count: int) -> tuple[int, int, int]:
    widths = [end - start for start, end in AXIS_PAIR_RANGES]
    exact = [pair_count * width / sum(widths) for width in widths]
    quota = [int(value) for value in exact]
    for axis in sorted(range(3), key=lambda i: exact[i] - quota[i], reverse=True):
        if sum(quota) == pair_count:
            break
        quota[axis] += 1
    return tuple(quota)


@torch.no_grad()
def _main_pair_indices(
    decoder: torch.Tensor,
    train_cases: list[dict],
    dim: int,
    mode: str,
) -> torch.Tensor:
    """Choose complete Main-RoPE pairs, optionally by data-weighted energy."""
    if dim <= 0 or dim > HEAD_DIM or dim % 2:
        raise ValueError(dim)
    pair_count = dim // 2
    quota = _balanced_pair_quota(pair_count)
    decoder = decoder.reshape(HEADS, HEAD_DIM, LATENT_DIM).float()
    # Decoder energy prevents high-Q but absent residual directions from being
    # selected.  Query energy measures actual functional consumption.
    decoder_energy = decoder.square().sum(-1).reshape(HEADS, 64, 2).sum(-1)
    query_energy = decoder_energy.new_zeros(HEADS, 64)
    for case in train_cases:
        q = case["raw"]["current_q_main"].float().reshape(
            case["raw"]["current_q_main"].shape[0], HEADS, -1, 64, 2
        )
        query_energy += q.square().sum(-1).mean((0, 2))
    query_energy /= max(len(train_cases), 1)
    score = decoder_energy * query_energy.clamp_min(1e-12)
    outputs = []
    for head in range(HEADS):
        selected_pairs = []
        for (start, end), count in zip(AXIS_PAIR_RANGES, quota):
            if mode == "balanced":
                local = torch.linspace(start, end - 1, steps=count, device=decoder.device).round().long()
            elif mode == "energy":
                local = score[head, start:end].topk(count, sorted=False).indices + start
                local = local.sort().values
            else:
                raise ValueError(mode)
            selected_pairs.extend(local.tolist())
        channels = [[2 * pair, 2 * pair + 1] for pair in selected_pairs]
        outputs.append(torch.tensor(channels, device=decoder.device).reshape(-1))
    return torch.stack(outputs).long()


@torch.no_grad()
def _svd_native(decoder: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head unconstrained rank-D factorization for Main V."""
    source = decoder.reshape(HEADS, HEAD_DIM, LATENT_DIM).float()
    reads, lifts = [], []
    for head in range(HEADS):
        u, s, vh = torch.linalg.svd(source[head], full_matrices=False)
        reads.append(s[:dim, None] * vh[:dim])
        lifts.append(u[:, :dim].T)
    return torch.stack(reads), torch.stack(lifts)  # [H,D,512], [H,D,128]


def _main_axis_ranks(
    dim: int, requested: tuple[int, int, int] | list[int] | None = None
) -> tuple[int, int, int]:
    """D48 -> 8/20/20, D64 -> 8/28/28, D128 -> 16/56/56."""
    if dim < 8 or dim > HEAD_DIM or dim % 4:
        raise ValueError(dim)
    if requested is None:
        temporal = 16 if dim == HEAD_DIM else 8
        remaining = dim - temporal
        height = remaining // 2
        width = remaining - height
        ranks = (temporal, height, width)
    else:
        ranks = tuple(int(value) for value in requested)
        if len(ranks) != 3 or sum(ranks) != dim:
            raise ValueError(f"axis ranks must contain three values summing to {dim}: {ranks}")
    for rank, (pair_start, pair_end) in zip(ranks, AXIS_PAIR_RANGES):
        if rank <= 0 or rank > 2 * (pair_end - pair_start):
            raise ValueError(f"axis rank {ranks} exceeds native widths")
    return ranks


@torch.no_grad()
def _svd_main_axes(
    decoder: torch.Tensor,
    dim: int,
    axis_ranks: tuple[int, int, int] | list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Factor each Main 3D-RoPE axis without requiring RoPE commutation.

    For token coordinate p, the kernel evaluates q^T R(p) U first and then
    dots that query feature with Bz.  U may therefore mix different native
    frequencies within one axis without ever pretending that low coordinates
    have the original RoPE frequencies.
    """
    source = decoder.reshape(HEADS, HEAD_DIM, LATENT_DIM).float()
    lefts, reads = [], []
    channel_ranges = ((0, 16), (16, 72), (72, 128))
    for (start, end), rank in zip(channel_ranges, _main_axis_ranks(dim, axis_ranks)):
        per_head_left, per_head_read = [], []
        for head in range(HEADS):
            u, s, vh = torch.linalg.svd(source[head, start:end], full_matrices=False)
            per_head_left.append(u[:, :rank])
            per_head_read.append(s[:rank, None] * vh[:rank])
        lefts.append(torch.stack(per_head_left))  # [H,width,r]
        reads.append(torch.stack(per_head_read))  # [H,r,512]
    return lefts, reads


@torch.no_grad()
def _svd_quartet(decoder: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r factorization across 32 PRoPE representation copies.

    Decoder[h] is viewed as [32 copies, 4 representation dims, 512].
    SVD across the copy axis produces D ~= (A x I4) B, which commutes with
    every tiled 4x4 PRoPE transform.
    """
    if dim % 4:
        raise ValueError(dim)
    rank = dim // 4
    source = decoder.reshape(HEADS, 32, 4, LATENT_DIM).float()
    reads, mixes = [], []
    for head in range(HEADS):
        matrix = source[head].reshape(32, 4 * LATENT_DIM)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        reads.append((s[:rank, None] * vh[:rank]).reshape(rank * 4, LATENT_DIM))
        mixes.append(u[:, :rank].T)  # [r,32], low = A @ native.
    return torch.stack(reads), torch.stack(mixes)


def _quartet_reduce(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    native = x.float().reshape(*x.shape[:-1], 32, 4)
    return torch.einsum("hro,bhqoc->bhqrc", mix.float(), native).flatten(-2)


def _quartet_lift(x: torch.Tensor, reduce_mix: torch.Tensor) -> torch.Tensor:
    rank = x.shape[-1] // 4
    low = x.float().reshape(*x.shape[:-1], rank, 4)
    # reduce_mix is [H,r,32], so its transpose is the native lift.
    return torch.einsum("hro,bhqrc->bhqoc", reduce_mix.float(), low).flatten(-2)


class DictionaryResidualAttention(nn.Module):
    """Exact full-anchor Attention plus trainable codec-residual Attention."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        kop: dict,
        vop: dict,
        train_cases: list[dict],
        main_pair_mode: str = "energy",
        main_k_mode: str = "axis_svd",
        main_axis_ranks: tuple[int, int, int] | list[int] | None = None,
        v_anchor_mode: str = "exact",
        k_anchor_mode: str = "exact",
    ):
        super().__init__()
        if key_dim % 4 or value_dim % 4:
            raise ValueError("D must be divisible by four for PRoPE")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.main_k_mode = main_k_mode
        self.main_axis_ranks = _main_axis_ranks(key_dim, main_axis_ranks)
        if v_anchor_mode not in {"exact", "low"}:
            raise ValueError(v_anchor_mode)
        self.v_anchor_mode = v_anchor_mode
        if k_anchor_mode not in {"exact", "low"}:
            raise ValueError(k_anchor_mode)
        self.k_anchor_mode = k_anchor_mode

        if main_k_mode in {"pair_select", "learned_rope"}:
            main_indices = _main_pair_indices(kop["decoder"], train_cases, key_dim, main_pair_mode)
            self.register_buffer("main_indices", main_indices)
            kdecoder = kop["decoder"].reshape(HEADS, HEAD_DIM, LATENT_DIM).float()
            batch = torch.arange(HEADS, device=kdecoder.device)[:, None]
            self.k_main_read = nn.Parameter(kdecoder[batch, main_indices])
            if main_k_mode == "learned_rope":
                identity = torch.eye(HEAD_DIM, device=kdecoder.device)
                reduce = identity[main_indices]
                self.q_main_reduce = nn.Parameter(reduce.clone())
                if self.k_anchor_mode == "low":
                    self.k_main_anchor_reduce = nn.Parameter(reduce.clone())
        elif main_k_mode == "axis_svd":
            lefts, reads = _svd_main_axes(
                kop["decoder"], key_dim, self.main_axis_ranks
            )
            self.k_main_left = nn.ParameterList(nn.Parameter(value) for value in lefts)
            self.k_main_axis_read = nn.ParameterList(nn.Parameter(value) for value in reads)
        else:
            raise ValueError(main_k_mode)
        v_main_read, v_main_lift = _svd_native(vop["decoder"], value_dim)
        k_prope_read, q_prope_mix = _svd_quartet(kop["decoder"], key_dim)
        v_prope_read, v_prope_mix = _svd_quartet(vop["decoder"], value_dim)

        self.v_main_read = nn.Parameter(v_main_read)
        self.v_main_lift = nn.Parameter(v_main_lift)
        self.k_prope_read = nn.Parameter(k_prope_read)
        self.q_prope_mix = nn.Parameter(q_prope_mix)
        self.v_prope_read = nn.Parameter(v_prope_read)
        self.v_prope_mix = nn.Parameter(v_prope_mix)
        if self.v_anchor_mode == "low":
            # Directly project the existing dynamic C64 V centers into the
            # same low-dimensional value space.  This removes the expensive
            # exact 64-bin full128 accumulation from the pairwise kernel while
            # preserving K anchors exactly.  The PRoPE map remains A x I4.
            self.v_main_anchor_reduce = nn.Parameter(v_main_lift.clone())
            self.v_prope_anchor_mix = nn.Parameter(v_prope_mix.clone())
        self.log_residual_scale_main = nn.Parameter(torch.zeros(HEADS))
        self.log_residual_scale_prope = nn.Parameter(torch.zeros(HEADS))

    def _main_axis_logits(self, case: dict, q_full: torch.Tensor) -> torch.Tensor:
        z = case["k_payload"]["dequant"]
        result = q_full.new_zeros(
            q_full.shape[0], HEADS, q_full.shape[2], z.shape[1]
        )
        channel_ranges = ((0, 16), (16, 72), (72, 128))
        for axis, ((start, end), left, read) in enumerate(zip(
            channel_ranges, self.k_main_left, self.k_main_axis_read
        )):
            residual = _decode_low(z, read)
            if self.k_anchor_mode == "low":
                anchor_axis = case["anchor_k_pre"][..., start:end].float()
                anchor_low = torch.einsum(
                    "bhnw,hwr->bhnr", anchor_axis, left.float()
                )
                residual = residual + anchor_low
            unique_cos, unique_sin, inverse = case["main_axis_positions"][axis]
            # [H,width,r] -> each column is a native vector transformed by
            # each unique key coordinate: [H,U,r,width].
            basis = left.float().permute(0, 2, 1)
            positioned = (
                basis[:, None] * unique_cos.float()[None, :, None]
                + _rotate_pairs(basis[:, None]) * unique_sin.float()[None, :, None]
            )
            features = torch.einsum(
                "bhqw,hurw->bhqur", q_full[..., start:end].float(), positioned
            )
            # Coordinate cardinality is small (T<=24,H<=30,W<=53).  A fused
            # kernel performs this gather inside the pairwise tile; the Python
            # gate uses grouped matmuls to avoid a QxNxR materialization.
            axis_logits = result.new_zeros(result.shape)
            for coordinate in range(positioned.shape[1]):
                token_indices = torch.where(inverse == coordinate)[0]
                if token_indices.numel() == 0:
                    continue
                partial = torch.einsum(
                    "bhqr,bhnr->bhqn",
                    features[:, :, :, coordinate],
                    residual.index_select(2, token_indices),
                )
                axis_logits = axis_logits.index_copy(-1, token_indices, partial)
            result = result + axis_logits
        return result

    def _common_logits(
        self,
        q_full: torch.Tensor,
        raw: dict,
        history_logits: torch.Tensor,
        branch: str,
        log_mass: tuple[float, float, float],
    ) -> torch.Tensor:
        current_k = raw[f"current_k_{branch}"].float()
        text = torch.matmul(q_full.float(), raw["text_k"].float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        current = torch.matmul(q_full.float(), current_k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        return torch.cat(
            (text + log_mass[0], history_logits + log_mass[1], current + log_mass[2]), dim=-1
        )

    def main(self, case: dict):
        raw, pos = case["raw"], case["pos"]
        q_full = raw["current_q_main"].float()
        if self.k_anchor_mode == "exact":
            anchor_k = case["anchor_k_main"]
            anchor_logits = torch.matmul(q_full, anchor_k.transpose(-1, -2))
        else:
            anchor_logits = 0.0

        if self.main_k_mode == "pair_select":
            residual_k = _decode_low(case["k_payload"]["dequant"], self.k_main_read)
            if self.k_anchor_mode == "low":
                anchor_low = _gather_last_per_head(
                    case["anchor_k_pre"].float(), self.main_indices
                )
                residual_k = residual_k + anchor_low
            residual_k = _apply_selected_rope(
                residual_k,
                pos["freqs_cos_k"], pos["freqs_sin_k"],
                self.main_indices,
            )
            # q_main is already positioned. Selecting full native pairs is
            # exactly equivalent to selecting before 3D-RoPE.
            q_low = _gather_last_per_head(q_full, self.main_indices)
            residual_logits = torch.matmul(q_low, residual_k.transpose(-1, -2))
            history_length = residual_k.shape[2]
        elif self.main_k_mode == "learned_rope":
            residual_k = _decode_low(case["k_payload"]["dequant"], self.k_main_read)
            if self.k_anchor_mode == "low":
                anchor_low = torch.einsum(
                    "bhno,hdo->bhnd",
                    case["anchor_k_pre"].float(),
                    self.k_main_anchor_reduce.float(),
                )
                residual_k = residual_k + anchor_low
            history_k = _apply_selected_rope(
                residual_k,
                pos["freqs_cos_k"], pos["freqs_sin_k"],
                self.main_indices,
            )
            q_pre_low = torch.einsum(
                "bhqo,hdo->bhqd",
                raw["current_q_pre"].float(),
                self.q_main_reduce.float(),
            )
            q_low = _apply_selected_rope(
                q_pre_low,
                pos["freqs_cos_q"], pos["freqs_sin_q"],
                self.main_indices,
            )
            residual_logits = torch.matmul(q_low, history_k.transpose(-1, -2))
            history_length = history_k.shape[2]
        else:
            residual_logits = self._main_axis_logits(case, q_full)
            history_length = case["k_payload"]["dequant"].shape[1]
        residual_scale = self.log_residual_scale_main.exp()[None, :, None, None]
        history_logits = (anchor_logits + residual_scale * residual_logits) / math.sqrt(HEAD_DIM)
        logits = self._common_logits(q_full, raw, history_logits, "main", case["segment_log_mass"])
        probs = logits.softmax(-1)

        nt = raw["text_k"].shape[2]
        nh = history_length
        text_output = torch.matmul(probs[..., :nt], raw["text_v"].float())
        history_prob = probs[..., nt:nt + nh]
        residual_v = _decode_low(case["v_payload"]["dequant"], self.v_main_read)
        if self.v_anchor_mode == "exact":
            anchor_output = torch.matmul(history_prob, case["anchor_v_main"])
            history_low_values = residual_v
        else:
            anchor_output = 0.0
            anchor_low = torch.einsum(
                "bhno,hdo->bhnd",
                case["anchor_v_pre"].float(),
                self.v_main_anchor_reduce.float(),
            )
            history_low_values = residual_v + anchor_low
        residual_low = torch.matmul(history_prob, history_low_values)
        residual_output = torch.einsum("bhqd,hdo->bhqo", residual_low, self.v_main_lift)
        current_output = torch.matmul(probs[..., nt + nh:], raw["current_v_main"].float())
        return text_output + anchor_output + residual_output + current_output, logits, probs

    def prope(self, case: dict):
        raw, pos = case["raw"], case["pos"]
        q_full = raw["current_q_prope"].float()
        if self.k_anchor_mode == "exact":
            anchor_logits = torch.matmul(q_full, case["anchor_k_prope"].transpose(-1, -2))
        else:
            anchor_logits = 0.0

        q_pre_low = _quartet_reduce(raw["current_q_pre"], self.q_prope_mix)
        apply_q, _, _ = prope_functions(pos, "q", self.key_dim, q_pre_low.device)
        q_low = apply_q(q_pre_low)
        residual_k = _decode_low(case["k_payload"]["dequant"], self.k_prope_read)
        if self.k_anchor_mode == "low":
            residual_k = residual_k + _quartet_reduce(
                case["anchor_k_pre"], self.q_prope_mix
            )
        _, apply_k, _ = prope_functions(pos, "k", self.key_dim, residual_k.device)
        residual_k = apply_k(residual_k)
        residual_logits = torch.matmul(q_low, residual_k.transpose(-1, -2))
        residual_scale = self.log_residual_scale_prope.exp()[None, :, None, None]
        history_logits = (anchor_logits + residual_scale * residual_logits) / math.sqrt(HEAD_DIM)
        logits = self._common_logits(q_full, raw, history_logits, "prope", case["segment_log_mass"])
        probs = logits.softmax(-1)

        nt = raw["text_k"].shape[2]
        nh = residual_k.shape[2]
        native_raw = (
            torch.matmul(probs[..., :nt], raw["text_v"].float())
            + torch.matmul(probs[..., nt + nh:], raw["current_v_prope"].float())
        )
        if self.v_anchor_mode == "exact":
            native_raw = native_raw + torch.matmul(
                probs[..., nt:nt + nh], case["anchor_v_prope"]
            )
        native_output = restore_prope_output(native_raw, pos)

        residual_v = _decode_low(case["v_payload"]["dequant"], self.v_prope_read)
        _, apply_v, _ = prope_functions(pos, "k", self.value_dim, residual_v.device)
        residual_v = apply_v(residual_v)
        if self.v_anchor_mode == "low":
            anchor_low = _quartet_reduce(
                case["anchor_v_pre"], self.v_prope_anchor_mix
            )
            residual_v = residual_v + apply_v(anchor_low)
        residual_low = torch.matmul(probs[..., nt:nt + nh], residual_v)
        _, _, restore_low = prope_functions(pos, "q", self.value_dim, residual_v.device)
        residual_low = restore_low(residual_low)
        residual_full = _quartet_lift(residual_low, self.v_prope_mix)
        residual_full = residual_full.permute(0, 2, 1, 3).reshape(
            residual_full.shape[0], residual_full.shape[2], HEADS * HEAD_DIM
        )
        return native_output + residual_full, logits, probs

    def forward(self, case: dict):
        return self.main(case), self.prope(case)


@torch.no_grad()
def prepare_case(path: Path, layer: int, kop: dict, vop: dict, weights: dict, device: torch.device):
    case = load_formal_case(path, layer, kop, vop, device)
    ktable, kids = _anchor_table(case, kop, "k")
    vtable, vids = _anchor_table(case, vop, "v")
    anchor_k = _assigned(ktable, kids)
    anchor_v = _assigned(vtable, vids)
    case["anchor_k_main"] = (
        anchor_k * case["pos"]["freqs_cos_k"].float()[None, None]
        + _rotate_pairs(anchor_k) * case["pos"]["freqs_sin_k"].float()[None, None]
    ).detach()
    case["anchor_k_pre"] = anchor_k.detach()
    case["anchor_v_main"] = anchor_v.detach()
    case["anchor_v_pre"] = anchor_v.detach()
    _, apply_kv, _ = prope_functions(case["pos"], "k", HEAD_DIM, device)
    case["anchor_k_prope"] = apply_kv(anchor_k).detach()
    case["anchor_v_prope"] = apply_kv(anchor_v).detach()
    case["main_axis_positions"] = []
    for start, end in ((0, 16), (16, 72), (72, 128)):
        cos = case["pos"]["freqs_cos_k"].float()[:, start:end]
        sin = case["pos"]["freqs_sin_k"].float()[:, start:end]
        signature = torch.cat((cos, sin), dim=-1)
        unique, inverse = torch.unique(signature, dim=0, return_inverse=True)
        width = end - start
        case["main_axis_positions"].append(
            (unique[:, :width].detach(), unique[:, width:].detach(), inverse.detach())
        )
    case["teacher_main_selected"] = case["sampled_teacher_main"]
    case["teacher_prope_selected"] = case["sampled_teacher_prope"]
    case["teacher_block_selected"] = replay_block(
        case["raw"]["layer_input"], case["raw"]["vec"],
        case["teacher_main_selected"], case["teacher_prope_selected"], weights,
    ).detach()
    return case


def _centered_history_loss(student: torch.Tensor, teacher: torch.Tensor, nt: int, nh: int):
    student = student[..., nt:nt + nh]
    teacher = teacher[..., nt:nt + nh]
    student = student - student.mean(-1, keepdim=True)
    teacher = teacher - teacher.mean(-1, keepdim=True)
    return normalized_mse(student, teacher)


def _row_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    teacher_prob = teacher.detach().float().softmax(-1)
    return F.kl_div(student.float().log_softmax(-1), teacher_prob, reduction="none").sum(-1).mean()


def training_loss(
    model: DictionaryResidualAttention,
    case: dict,
    weights: dict,
    logit_weight: float,
    kl_weight: float,
    block_weight: float,
):
    main, prope = model(case)
    teacher_main = case["teacher_main_selected"]
    teacher_prope = case["teacher_prope_selected"]
    output = normalized_mse(main[0], teacher_main) + normalized_mse(prope[0], teacher_prope)
    nt = case["raw"]["text_k"].shape[2]
    nh = case["k_payload"]["dequant"].shape[1]
    logits = _centered_history_loss(main[1], case["teacher_main_logits"], nt, nh)
    logits = logits + _centered_history_loss(prope[1], case["teacher_prope_logits"], nt, nh)
    kl = _row_kl(main[1], case["teacher_main_logits"]) + _row_kl(
        prope[1], case["teacher_prope_logits"]
    )
    student_block = replay_block(
        case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0], weights
    )
    block = normalized_mse(student_block, case["teacher_block_selected"])
    total = output + logit_weight * logits + kl_weight * kl + block_weight * block
    return total, {
        "output": output.detach(), "logits": logits.detach(),
        "kl": kl.detach(), "block": block.detach(),
    }


def _topk_recall(student: torch.Tensor, teacher: torch.Tensor, k: int = 32) -> float:
    k = min(k, teacher.shape[-1])
    left = student.topk(k, dim=-1).indices
    right = teacher.topk(k, dim=-1).indices
    return float((left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1).float().mean())


@torch.no_grad()
def evaluate(model: DictionaryResidualAttention, cases: list[dict], weights: dict):
    rows = []
    for case in cases:
        main, prope = model(case)
        block = replay_block(
            case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0], weights
        )
        nt = case["raw"]["text_k"].shape[2]
        nh = case["k_payload"]["dequant"].shape[1]
        teacher_main_hist = case["teacher_main_logits"][..., nt:nt + nh]
        teacher_prope_hist = case["teacher_prope_logits"][..., nt:nt + nh]
        main_hist = main[1][..., nt:nt + nh]
        prope_hist = prope[1][..., nt:nt + nh]
        online_block = case["raw"]["layer_output"].float()
        rows.append({
            "case_id": case["raw"]["case_id"],
            "chunk": int(case["raw"]["ar_chunk_id"]),
            "attention_rel_l2_main": rel_l2(main[0], case["teacher_main_selected"]),
            "attention_rel_l2_prope": rel_l2(prope[0], case["teacher_prope_selected"]),
            "history_centered_logit_rel_l2_main": rel_l2(
                main_hist - main_hist.mean(-1, keepdim=True),
                teacher_main_hist - teacher_main_hist.mean(-1, keepdim=True),
            ),
            "history_centered_logit_rel_l2_prope": rel_l2(
                prope_hist - prope_hist.mean(-1, keepdim=True),
                teacher_prope_hist - teacher_prope_hist.mean(-1, keepdim=True),
            ),
            "history_top32_recall_main": _topk_recall(main_hist, teacher_main_hist),
            "history_top32_recall_prope": _topk_recall(prope_hist, teacher_prope_hist),
            "softmax_kl_main": float(_row_kl(main[1], case["teacher_main_logits"])),
            "softmax_kl_prope": float(_row_kl(prope[1], case["teacher_prope_logits"])),
            "block_rel_l2": rel_l2(block, case["teacher_block_selected"]),
            "sampled_teacher_vs_online_block_rel_l2": rel_l2(
                case["teacher_block_selected"], online_block
            ),
            "student_vs_online_block_rel_l2": rel_l2(block, online_block),
        })
    summary = {}
    for key in rows[0]:
        if key in {"case_id", "chunk"}:
            continue
        values = [row[key] for row in rows]
        summary[key + "_mean"] = sum(values) / len(values)
        summary[key + "_max"] = max(values)
    return rows, summary


def _paths(root: Path, layer: int, prefix: str) -> list[Path]:
    return sorted(root.glob(f"{prefix}_*_layer{layer:02d}_mixed.pt"))


def _serializable_args(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main():
    global LATENT_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument(
        "--heldout-root",
        type=Path,
        help="optional separate root containing test samples for case-holdout training",
    )
    ap.add_argument("--operator", type=Path, required=True)
    ap.add_argument("--world-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--key-dim", type=int, default=48)
    ap.add_argument("--value-dim", type=int, default=48)
    ap.add_argument("--main-pair-mode", choices=("balanced", "energy"), default="energy")
    ap.add_argument(
        "--main-k-mode",
        choices=("axis_svd", "pair_select", "learned_rope"),
        default="axis_svd",
    )
    ap.add_argument(
        "--k-anchor-mode",
        choices=("exact", "low"),
        default="exact",
        help=(
            "exact evaluates the full128 C64 K anchor; low projects each "
            "dynamic C64 K center into the same structured low K space"
        ),
    )
    ap.add_argument(
        "--main-axis-ranks",
        type=int,
        nargs=3,
        metavar=("T", "H", "W"),
        help=(
            "explicit Main axis-SVD ranks; values must sum to --key-dim "
            "(for example D48: 16 16 16)"
        ),
    )
    ap.add_argument(
        "--v-anchor-mode",
        choices=("exact", "low"),
        default="exact",
        help=(
            "exact uses full128 C64 V mass accumulation; low directly projects "
            "dynamic C64 V centers into the trainable low value space"
        ),
    )
    ap.add_argument("--expected-train", type=int, default=36)
    ap.add_argument("--expected-heldout", type=int, default=12)
    ap.add_argument("--heldout-prefix", default="test",
                    help="Filename prefix for held-out development samples in chunk_holdout mode")
    ap.add_argument(
        "--split-mode", choices=("chunk_holdout", "case_holdout"), default="chunk_holdout"
    )
    ap.add_argument("--train-chunks", type=int, nargs="+", default=(2, 4))
    ap.add_argument("--validation-chunks", type=int, nargs="+", default=(6,))
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-validation", type=int, default=0)
    ap.add_argument("--max-heldout", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.05)
    ap.add_argument("--logit-weight", type=float, default=0.20)
    ap.add_argument("--kl-weight", type=float, default=0.05)
    ap.add_argument("--block-weight", type=float, default=0.25)
    ap.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "optional shape-compatible reader checkpoint to fine-tune; the "
            "codec and world-model weights remain frozen"
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--d128-attention-threshold", type=float, default=3e-3)
    ap.add_argument("--d128-block-threshold", type=float, default=1e-3)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)

    all_train_paths = _paths(args.data_root, args.layer, "train")
    heldout_paths = _paths(args.heldout_root or args.data_root, args.layer, args.heldout_prefix)
    if len(all_train_paths) != args.expected_train or len(heldout_paths) != args.expected_heldout:
        raise RuntimeError(
            f"L{args.layer}: expected {args.expected_train}+{args.expected_heldout}, "
            f"got {len(all_train_paths)}+{len(heldout_paths)}"
        )
    if args.split_mode == "chunk_holdout":
        train_paths = [
            path for path in all_train_paths
            if any(f"_chunk{chunk:02d}_" in path.name for chunk in args.train_chunks)
        ]
        validation_paths = [
            path for path in all_train_paths
            if any(f"_chunk{chunk:02d}_" in path.name for chunk in args.validation_chunks)
        ]
        if set(train_paths) & set(validation_paths):
            raise RuntimeError("training and validation chunk sets overlap")
    else:
        train_paths = all_train_paths
        validation_paths = [path for path in heldout_paths if path.name.startswith("test_015_")]
        heldout_paths = [path for path in heldout_paths if path.name.startswith("test_016_")]
    if not train_paths or not validation_paths:
        raise RuntimeError(
            f"empty train/validation split: {len(train_paths)}+{len(validation_paths)}"
        )
    if args.max_train:
        train_paths = train_paths[:args.max_train]
    if args.max_validation:
        validation_paths = validation_paths[:args.max_validation]
    if args.max_heldout:
        heldout_paths = heldout_paths[:args.max_heldout]

    kop = load_operator(args.operator, args.layer, "k", device)
    vop = load_operator(args.operator, args.layer, "v", device)
    LATENT_DIM = int(kop["decoder"].shape[1])
    if int(vop["decoder"].shape[1]) != LATENT_DIM:
        raise RuntimeError("K/V operator ranks differ")
    weights = load_block_weights(str(args.world_model), args.layer, device)
    validation_label = "long-context validation" if args.split_mode == "chunk_holdout" else "case validation"
    print(
        f"L{args.layer} loading {len(train_paths)} train + "
        f"{len(validation_paths)} {validation_label} + {len(heldout_paths)} heldout",
        flush=True,
    )
    train = [prepare_case(path, args.layer, kop, vop, weights, device) for path in train_paths]
    validation = [
        prepare_case(path, args.layer, kop, vop, weights, device) for path in validation_paths
    ]
    heldout = [prepare_case(path, args.layer, kop, vop, weights, device) for path in heldout_paths]
    model = DictionaryResidualAttention(
        args.key_dim,
        args.value_dim,
        kop,
        vop,
        train,
        args.main_pair_mode,
        args.main_k_mode,
        args.main_axis_ranks,
        args.v_anchor_mode,
        args.k_anchor_mode,
    ).to(device)
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        state = initial_checkpoint.get("model", initial_checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"loaded reader initialization: {args.init_checkpoint}", flush=True)
    initial_rows, initial = evaluate(model, heldout, weights)
    validation_initial_rows, validation_initial = evaluate(model, validation, weights)
    print(json.dumps({"initial": initial, "validation_initial": validation_initial}, indent=2), flush=True)
    if args.key_dim == HEAD_DIM and args.value_dim == HEAD_DIM:
        if args.steps != 0:
            raise RuntimeError("D128 is a frozen correctness gate and requires --steps 0")
        attention_max = max(
            initial["attention_rel_l2_main_max"],
            initial["attention_rel_l2_prope_max"],
            validation_initial["attention_rel_l2_main_max"],
            validation_initial["attention_rel_l2_prope_max"],
        )
        block_max = max(
            initial["block_rel_l2_max"], validation_initial["block_rel_l2_max"]
        )
        if attention_max > args.d128_attention_threshold or block_max > args.d128_block_threshold:
            raise RuntimeError(
                f"D128 correctness gate failed: attention_max={attention_max:.6e}, "
                f"block_max={block_max:.6e}"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr * args.lr_min_ratio
    )
    history = []
    validation_history = []
    best_validation = float(validation_initial["block_rel_l2_mean"])
    best_step = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for step in range(args.steps):
        case = train[(step * 17 + step // len(train)) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = training_loss(
            model, case, weights, args.logit_weight, args.kl_weight, args.block_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 50 == 0:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({key: float(value) for key, value in pieces.items()})
            history.append(row)
            print(
                f"L{args.layer} K{args.key_dim}V{args.value_dim} "
                f"{step + 1}/{args.steps} "
                + " ".join(f"{key}={value:.6f}" for key, value in row.items() if key != "step"),
                flush=True,
            )
        if args.eval_every and (step + 1) % args.eval_every == 0:
            _, periodic = evaluate(model, validation, weights)
            periodic_row = {"step": step + 1, **periodic}
            validation_history.append(periodic_row)
            score = float(periodic["block_rel_l2_mean"])
            print(
                f"L{args.layer} validation step={step + 1} "
                f"block={score:.6f} main={periodic['attention_rel_l2_main_mean']:.6f} "
                f"prope={periodic['attention_rel_l2_prope_mean']:.6f}",
                flush=True,
            )
            if score < best_validation:
                best_validation = score
                best_step = step + 1
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

    # Report and deploy the long-context validation winner rather than the
    # last optimizer step; short sampled fits can otherwise silently overfit.
    model.load_state_dict(best_state)
    train_rows, train_final = evaluate(model, train, weights)
    validation_rows, validation_final = evaluate(model, validation, weights)
    final_rows, final = evaluate(model, heldout, weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / (
        f"layer_{args.layer:02d}_k{args.key_dim}_v{args.value_dim}.pt"
    )
    torch.save({
        "model": model.state_dict(),
        "config": _serializable_args(args),
        "main_indices": model.main_indices.cpu() if hasattr(model, "main_indices") else None,
        "main_axis_ranks": list(model.main_axis_ranks) if args.main_k_mode == "axis_svd" else None,
        "history": history,
        "validation_history": validation_history,
        "best_step": best_step,
        "formal_rank": LATENT_DIM,
        "method": (
            "low_c64_kv_anchor_structured_residual_attention"
            if args.k_anchor_mode == "low" and args.v_anchor_mode == "low"
            else "exact_c64_k_anchor_low_v_anchor_residual_attention"
            if args.v_anchor_mode == "low"
            else "low_c64_k_anchor_exact_v_anchor_residual_attention"
            if args.k_anchor_mode == "low"
            else "exact_c64_anchor_residual_lowdim_attention"
        ),
        "numerical_mode": "offline_fp16_semantic_dump_fp32_trainable_reads",
    }, checkpoint)
    result = {
        "config": _serializable_args(args),
        "initial": initial,
        "validation_initial": validation_initial,
        "train_final": train_final,
        "validation_final": validation_final,
        "final": final,
        "initial_rows": initial_rows,
        "validation_initial_rows": validation_initial_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "final_rows": final_rows,
        "history": history,
        "validation_history": validation_history,
        "best_step": best_step,
        "main_indices": model.main_indices.cpu().tolist() if hasattr(model, "main_indices") else None,
        "main_axis_ranks": list(model.main_axis_ranks) if args.main_k_mode == "axis_svd" else None,
        "checkpoint": str(checkpoint),
        "numerical_mode": "offline_fp16_semantic_dump_fp32_trainable_reads",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "initial": initial,
        "train": train_final,
        "validation": validation_final,
        "final": final,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
