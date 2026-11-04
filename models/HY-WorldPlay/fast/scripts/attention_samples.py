"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from position_utils import prope_functions, restore_prope_output, rope_selected
from hyvideo.models.transformers.modules.kv_codecs import _joint_semantic_smooth
from attention_utils import load_operator, normalized_mse, reduced_rope, rel_l2, rope_channel_indices
try:
    from train_codec_native_block_lora_gate import BlockLoRA, centered_logit_loss
except ModuleNotFoundError:
    # The dictionary-reader trainer imports only ``load_case`` from this
    # module.  Keep that maintained path usable after archival cleanup of the
    # rejected block-LoRA experiment; fail explicitly if someone requests it.
    class BlockLoRA:  # pragma: no cover - rejected legacy path
        def __init__(self, *args, **kwargs):
            raise RuntimeError("archived block-LoRA training source is unavailable")

    def centered_logit_loss(student, teacher):
        student = student.float() - student.float().mean(-1, keepdim=True)
        teacher = teacher.float() - teacher.float().mean(-1, keepdim=True)
        return (student - teacher).square().mean() / teacher.square().mean().clamp_min(1e-8)
from block_replay import load_block_weights, replay_block


def _to_device(payload, device):
    for key, value in list(payload.items()):
        if torch.is_tensor(value):
            payload[key] = value.to(device)
        elif isinstance(value, dict):
            payload[key] = _to_device(value, device)
    return payload


def _position_payload(raw):
    return {
        "freqs_cos_q": raw["current_freqs_cos_q"],
        "freqs_sin_q": raw["current_freqs_sin_q"],
        "freqs_cos_k": raw["history_freqs_cos"],
        "freqs_sin_k": raw["history_freqs_sin"],
        "viewmats_q": raw["current_viewmats_q"],
        "Ks_q": raw["current_Ks_q"],
        "viewmats_k": raw["history_viewmats"],
        "Ks_k": raw["history_Ks"],
    }


def _segment_log_mass(raw, key_parts):
    """Correct uniform token subsampling to the online segment population.

    Text is dumped in full.  History/current are uniform deterministic samples,
    so each sampled token represents N/n online tokens in the softmax measure.
    """
    totals = (
        int(key_parts[0].shape[2]),
        math.prod(int(value) for value in raw["history_thw"]),
        math.prod(int(value) for value in raw["current_thw"]),
    )
    return tuple(
        math.log(max(total, 1) / max(int(part.shape[2]), 1))
        for total, part in zip(totals, key_parts)
    )


def _full_attention(q, key_parts, value_parts, log_mass=None):
    if log_mass is None:
        log_mass = (0.0,) * len(key_parts)
    logit_parts = [
        torch.matmul(q.float(), key.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
        for key, bias in zip(key_parts, log_mass)
    ]
    logits = torch.cat(logit_parts, dim=-1)
    values = torch.cat([part.float() for part in value_parts], dim=2)
    probs = logits.softmax(-1)
    return torch.matmul(probs, values), logits


@torch.no_grad()
def encode_formal_payload(pre, layer, op):
    """Return original codec teacher plus pre/post-INT4 payload states."""
    residual, centers, ids = _joint_semantic_smooth(pre, layer)
    b, h, s, d = residual.shape
    rows = residual.permute(0, 2, 1, 3).reshape(-1, h * d).float()
    coeff = F.linear((rows - op["mean"]).to(torch.bfloat16), op["encoder"].to(torch.bfloat16)).float()
    rank = int(op["encoder"].shape[0])
    coeff = coeff.reshape(b, s, rank)
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    q = (coeff / scale).round().clamp(-7, 7)
    dequant = q * scale
    assigned = torch.gather(centers.float(), 1, ids.long()[..., None].expand(-1, -1, h * d))
    decoded = F.linear(
        dequant.reshape(-1, rank).to(torch.bfloat16),
        op["decoder"].to(torch.bfloat16),
        op["mean"].to(torch.bfloat16),
    ).float()
    decoded = decoded + assigned.reshape(-1, h * d)
    recon = decoded.reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()

    # The centre is dynamic per cache.  Project its actual values once into
    # the PCA coordinate system.  Token reads only gather by the existing ID.
    centre_input = centers.float() + op["mean"][None, None, :]
    centre_code = F.linear(
        centre_input.to(torch.bfloat16), op["encoder"].to(torch.bfloat16)
    ).float()
    return recon, {
        "coeff": coeff.detach(),
        "dequant": dequant.detach(),
        "scale": scale.detach(),
        "centre_code": centre_code.detach(),
        "ids": ids.long().detach(),
        "semantic_centers": centers.detach(),
        "semantic_ids": ids.long().detach(),
    }


@torch.no_grad()
def load_dumped_formal_payload(payload, op):
    """Use the exact first-pass formal cache payload captured online."""
    dequant = payload["dequant"].float()
    centers = payload["semantic_centers"].float()
    ids = payload["semantic_ids"].long()
    assigned = torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, centers.shape[-1])
    )
    rank = int(op["encoder"].shape[0])
    decoded = F.linear(
        dequant.reshape(-1, rank).to(torch.bfloat16),
        op["decoder"].to(torch.bfloat16), op["mean"].to(torch.bfloat16),
    ).float()
    batch, sequence = dequant.shape[:2]
    decoded = decoded + assigned.reshape(-1, assigned.shape[-1])
    recon = decoded.reshape(batch, sequence, 16, 128).permute(0, 2, 1, 3).contiguous()
    centre_code = F.linear(
        (centers + op["mean"][None, None]).to(torch.bfloat16),
        op["encoder"].to(torch.bfloat16),
    ).float()
    return recon, {
        # The formal pre-organizer coefficient is not retained separately;
        # dequantized first-pass INT4 is its closest runtime-faithful state.
        "coeff": dequant.detach(),
        "dequant": dequant.detach(),
        "scale": dequant.new_ones((*dequant.shape[:-1], 1)),
        "centre_code": centre_code.detach(),
        "ids": ids.detach(),
        "semantic_centers": centers.detach(),
        "semantic_ids": ids.detach(),
    }


@torch.no_grad()
def encode_dumped_semantic_payload(payload, op):
    """Encode residuals selected after C64 was fitted on full history."""
    residual = payload["residual"].float()
    centers = payload["semantic_centers"].float()
    ids = payload["semantic_ids"].long()
    batch, heads, sequence, dim = residual.shape
    rows = residual.permute(0, 2, 1, 3).reshape(-1, heads * dim)
    rank = int(op["encoder"].shape[0])
    coeff = F.linear(
        (rows - op["mean"]).to(torch.bfloat16), op["encoder"].to(torch.bfloat16)
    ).float().reshape(batch, sequence, rank)
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    dequant = (coeff / scale).round().clamp(-7, 7) * scale
    assigned = torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, heads * dim)
    )
    decoded = F.linear(
        dequant.reshape(-1, rank).to(torch.bfloat16),
        op["decoder"].to(torch.bfloat16), op["mean"].to(torch.bfloat16),
    ).float() + assigned.reshape(-1, heads * dim)
    recon = decoded.reshape(batch, sequence, heads, dim).permute(0, 2, 1, 3).contiguous()
    centre_code = F.linear(
        (centers + op["mean"][None, None]).to(torch.bfloat16),
        op["encoder"].to(torch.bfloat16),
    ).float()
    return recon, {
        "coeff": coeff.detach(), "dequant": dequant.detach(), "scale": scale.detach(),
        "centre_code": centre_code.detach(), "ids": ids.detach(),
        "semantic_centers": centers.detach(), "semantic_ids": ids.detach(),
    }


def load_case(path, layer, kop, vop, device):
    raw = _to_device(torch.load(path, map_location="cpu"), device)
    pos = _position_payload(raw)
    if "history_k_formal_payload" in raw and "history_v_formal_payload" in raw:
        history_k_codec, k_payload = load_dumped_formal_payload(raw["history_k_formal_payload"], kop)
        history_v_codec, v_payload = load_dumped_formal_payload(raw["history_v_formal_payload"], vop)
    elif "history_k_semantic_payload" in raw and "history_v_semantic_payload" in raw:
        history_k_codec, k_payload = encode_dumped_semantic_payload(raw["history_k_semantic_payload"], kop)
        history_v_codec, v_payload = encode_dumped_semantic_payload(raw["history_v_semantic_payload"], vop)
    else:
        history_k_codec, k_payload = encode_formal_payload(raw["history_k_pre"].float(), layer, kop)
        history_v_codec, v_payload = encode_formal_payload(raw["history_v_pre"].float(), layer, vop)

    history_k_main = rope_selected(history_k_codec, pos, "k")
    _, history_apply_kv, _ = prope_functions(pos, "k", 128, device)
    history_k_prope = history_apply_kv(history_k_codec)
    history_v_prope = history_apply_kv(history_v_codec)
    main_keys = [raw["text_k"], history_k_main, raw["current_k_main"]]
    main_values = [raw["text_v"], history_v_codec, raw["current_v_main"]]
    prope_keys = [raw["text_k"], history_k_prope, raw["current_k_prope"]]
    prope_values = [raw["text_v"], history_v_prope, raw["current_v_prope"]]
    log_mass = _segment_log_mass(raw, main_keys)
    sampled_main, teacher_main_logits = _full_attention(
        raw["current_q_main"],
        main_keys, main_values, log_mass,
    )
    sampled_prope_raw, teacher_prope_logits = _full_attention(
        raw["current_q_prope"],
        prope_keys, prope_values, log_mass,
    )
    batch, sequence, hidden = raw["teacher_attention_main"].shape
    teacher_main = raw["teacher_attention_main"].float().reshape(
        batch, sequence, 16, hidden // 16
    ).permute(0, 2, 1, 3).contiguous()
    return {
        "raw": raw,
        "pos": pos,
        "k_payload": k_payload,
        "v_payload": v_payload,
        # These are the actual production outputs computed with every online
        # text/history/current token, not a re-created sampled surrogate.
        "teacher_main": teacher_main.detach(),
        "teacher_prope": raw["teacher_attention_prope"].float().detach(),
        "teacher_block": raw["layer_output"].float().detach(),
        "sampled_teacher_main": sampled_main.detach(),
        "sampled_teacher_prope": restore_prope_output(sampled_prope_raw, pos).detach(),
        "teacher_main_logits": teacher_main_logits.detach(),
        "teacher_prope_logits": teacher_prope_logits.detach(),
        "segment_log_mass": log_mass,
    }


def _group_members(kv_heads):
    if 16 % kv_heads:
        raise ValueError("kv_heads must divide 16")
    width = 16 // kv_heads
    return [list(range(group * width, (group + 1) * width)) for group in range(kv_heads)]


def decoder_group_initialization(decoder, kv_heads, dim):
    """Map PCA coefficients to grouped native-channel coordinates.

    The legacy direct path deliberately keeps a square 512-D organized cache.
    Consequently fewer than 16 KV heads average the native decoder rows of the
    query heads which will later share that KV head.
    """
    channels = rope_channel_indices(dim, decoder.device)
    rows = []
    for members in _group_members(kv_heads):
        for channel in channels.tolist():
            rows.append(torch.stack([
                decoder[head * 128 + channel].float() for head in members
            ]).mean(0))
    weight = torch.stack(rows)
    if weight.shape != (512, 512):
        raise RuntimeError(f"expected square organizer, got {tuple(weight.shape)}")
    return weight


def half_decoder_initialization(decoder, kv_heads, dim):
    """Initialize independent reduced heads directly from the codec decoder.

    Unlike ``decoder_group_initialization``, this matrix is allowed to be
    rectangular.  In the intended H16xD64 setting it maps the *stored* 512-D
    codec latent to 1024 values (16 independent 64-D heads).  No heads are
    averaged or repeated and the 2048-D native KV is never materialized.
    """
    if kv_heads != 16:
        raise ValueError("half_decoder currently requires 16 independent KV heads")
    channels = rope_channel_indices(dim, decoder.device)
    rows = [
        decoder[head * 128 + channel].float()
        for head in range(16)
        for channel in channels.tolist()
    ]
    weight = torch.stack(rows)
    expected = (kv_heads * dim, 512)
    if weight.shape != expected:
        raise RuntimeError(f"expected half decoder {expected}, got {tuple(weight.shape)}")
    return weight


class DirectCodecLatentAttention(nn.Module):
    def __init__(self, k_heads, k_dim, v_heads, v_dim, kop, vop, payload_mode,
                 compress_current=False):
        super().__init__()
        if payload_mode == "half_decoder":
            if k_heads != 16 or v_heads != 16:
                raise ValueError("half_decoder requires 16 independent K/V heads")
        elif k_heads * k_dim != 512 or v_heads * v_dim != 512:
            raise ValueError("legacy K/V organizations must use exactly the 512-D payload")
        self.k_heads, self.k_dim = k_heads, k_dim
        self.v_heads, self.v_dim = v_heads, v_dim
        self.payload_mode = payload_mode
        self.compress_current = bool(compress_current)
        if payload_mode == "half_decoder":
            k_init = half_decoder_initialization(kop["decoder"], k_heads, k_dim)
            v_init = half_decoder_initialization(vop["decoder"], v_heads, v_dim)
            self.k_half_decoder = nn.Parameter(k_init)
            self.v_half_decoder = nn.Parameter(v_init)
            # Aliases keep the common state/read code compact.  These are not
            # cache organizers: their rectangular output is produced on read.
            self.k_organizer = self.k_half_decoder
            self.v_organizer = self.v_half_decoder
        else:
            k_init = decoder_group_initialization(kop["decoder"], k_heads, k_dim)
            v_init = decoder_group_initialization(vop["decoder"], v_heads, v_dim)
        if payload_mode == "frozen":
            self.register_buffer("k_organizer", torch.eye(512, device=k_init.device))
            self.register_buffer("v_organizer", torch.eye(512, device=v_init.device))
        elif payload_mode == "qat_organized":
            self.k_organizer = nn.Parameter(k_init)
            self.v_organizer = nn.Parameter(v_init)
        elif payload_mode == "qat_joint":
            # Start exactly from the validated formal PCA coordinate system.
            # Attention must earn any coordinate change while the learned
            # reverse map continuously preserves the original codec output.
            identity = torch.eye(512, device=k_init.device)
            self.k_organizer = nn.Parameter(identity.clone())
            self.v_organizer = nn.Parameter(identity.clone())
            self.k_reconstructor = nn.Parameter(identity.clone())
            self.v_reconstructor = nn.Parameter(identity.clone())
        elif payload_mode == "qat_orthogonal":
            # Closest orthogonal matrices to the decoder-aware grouped views.
            # They remain invertible and can be folded into encoder/decoder.
            ku, _, kvh = torch.linalg.svd(k_init.float(), full_matrices=False)
            vu, _, vvh = torch.linalg.svd(v_init.float(), full_matrices=False)
            self.k_organizer = nn.Parameter(ku @ kvh)
            self.v_organizer = nn.Parameter(vu @ vvh)
        elif payload_mode != "half_decoder":
            raise ValueError(payload_mode)
        self.register_buffer("k_decoder", kop["decoder"].float())
        self.register_buffer("v_decoder", vop["decoder"].float())

        self.q_weight = nn.Parameter(torch.zeros(16, k_dim, 128))
        if self.compress_current:
            # These post-position maps can be fused into the reduced-attention
            # input stage in production.  They are explicit in this teacher
            # gate so RoPE/PRoPE remains native before the 64-D projection.
            self.current_q_weight = nn.Parameter(torch.zeros(16, k_dim, 128))
            self.current_k_weight = nn.Parameter(torch.zeros(16, k_dim, 128))
            self.current_v_weight = nn.Parameter(torch.zeros(16, v_dim, 128))
        self.o_main = nn.Parameter(torch.zeros(16, 128, v_dim))
        self.o_prope = nn.Parameter(torch.zeros(16, 128, v_dim))
        self.logit_scale_main = nn.Parameter(torch.zeros(16))
        self.logit_scale_prope = nn.Parameter(torch.zeros(16))
        self._initialize_native_channels()

    @torch.no_grad()
    def _initialize_native_channels(self):
        k_channels = rope_channel_indices(self.k_dim, self.q_weight.device)
        v_channels = rope_channel_indices(self.v_dim, self.q_weight.device)
        for head in range(16):
            self.q_weight[head, torch.arange(self.k_dim), k_channels] = 1
            if self.compress_current:
                self.current_q_weight[head, torch.arange(self.k_dim), k_channels] = 1
                self.current_k_weight[head, torch.arange(self.k_dim), k_channels] = 1
                self.current_v_weight[head, torch.arange(self.v_dim), v_channels] = 1
            self.o_main[head, v_channels, torch.arange(self.v_dim)] = 1
            self.o_prope[head, v_channels, torch.arange(self.v_dim)] = 1

    @staticmethod
    def _ste_int4(x):
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
        dequant = (x.float() / scale).round().clamp(-7, 7) * scale
        return x.float() + (dequant - x.float()).detach()

    @torch.no_grad()
    def project_orthogonal(self):
        if self.payload_mode != "qat_orthogonal":
            return
        for weight in (self.k_organizer, self.v_organizer):
            q, r = torch.linalg.qr(weight.float())
            sign = torch.where(torch.diagonal(r) < 0, -torch.ones_like(torch.diagonal(r)), torch.ones_like(torch.diagonal(r)))
            weight.copy_(q * sign[None, :])

    def _state(self, payload, organizer):
        if self.payload_mode == "half_decoder":
            # The formal rank-512 INT4 payload is dequantized once, followed by
            # a rectangular half decode.  There is no 1024-D cache payload and
            # no second quantization.
            residual = F.linear(payload["dequant"].float(), organizer)
            centres = F.linear(payload["centre_code"].float(), organizer)
        elif self.payload_mode == "frozen":
            residual = payload["dequant"].float()
            centres = payload["centre_code"].float()
        else:
            residual = self._ste_int4(F.linear(payload["coeff"].float(), organizer))
            # Project 64 dynamic centres once per cache; no per-token decode.
            centres = F.linear(payload["centre_code"].float(), organizer)
        assigned = torch.gather(
            centres, 1, payload["ids"][..., None].expand(-1, -1, centres.shape[-1])
        )
        return residual + assigned

    def reconstruction_loss(self, case):
        if self.payload_mode != "qat_joint":
            return self.q_weight.new_zeros(())
        total = self.q_weight.new_zeros(())
        for target, organizer, reconstructor, decoder in (
            (case["k_payload"], self.k_organizer, self.k_reconstructor, self.k_decoder),
            (case["v_payload"], self.v_organizer, self.v_reconstructor, self.v_decoder),
        ):
            organized = self._ste_int4(F.linear(target["coeff"].float(), organizer))
            recovered = F.linear(organized, reconstructor)
            candidate = F.linear(recovered.to(torch.bfloat16), decoder.to(torch.bfloat16)).float()
            teacher = F.linear(target["dequant"].to(torch.bfloat16), decoder.to(torch.bfloat16)).float()
            total = total + normalized_mse(candidate, teacher)
        return total

    @staticmethod
    def _to_heads(state, heads, dim):
        return state.reshape(state.shape[0], state.shape[1], heads, dim).permute(0, 2, 1, 3)

    @staticmethod
    def _expand_kv_heads(x, kv_heads):
        return x.repeat_interleave(16 // kv_heads, dim=1)

    def branch(self, case, branch):
        raw, pos = case["raw"], case["pos"]
        q0 = torch.einsum("bhsd,hkd->bhsk", raw["current_q_pre"].float(), self.q_weight)
        k_state = self._state(case["k_payload"], self.k_organizer)
        v_state = self._state(case["v_payload"], self.v_organizer)
        k0 = self._expand_kv_heads(self._to_heads(k_state, self.k_heads, self.k_dim), self.k_heads)
        v0 = self._expand_kv_heads(self._to_heads(v_state, self.v_heads, self.v_dim), self.v_heads)
        if branch == "main":
            q_low = reduced_rope(q0, pos, "q")
            k_low = reduced_rope(k0, pos, "k")
            v_low = v0
            q_full = raw["current_q_main"].float()
            current_k = raw["current_k_main"].float()
            current_v = raw["current_v_main"].float()
            scale = self.logit_scale_main.exp()[None, :, None, None]
            output_weight = self.o_main
            if self.compress_current:
                current_q_low = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_q_main"].float(), self.current_q_weight
                )
                current_k0 = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_k_main"].float(), self.current_k_weight
                )
                current_v_low = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_v_main"].float(), self.current_v_weight
                )
                current_k_low = current_k0
        else:
            apply_q, _, _ = prope_functions(pos, "q", self.k_dim, q0.device)
            _, apply_k, _ = prope_functions(pos, "k", self.k_dim, q0.device)
            _, apply_v, _ = prope_functions(pos, "k", self.v_dim, q0.device)
            q_low, k_low, v_low = apply_q(q0), apply_k(k0), apply_v(v0)
            q_full = raw["current_q_prope"].float()
            current_k = raw["current_k_prope"].float()
            current_v = raw["current_v_prope"].float()
            scale = self.logit_scale_prope.exp()[None, :, None, None]
            output_weight = self.o_prope
            if self.compress_current:
                current_q_low = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_q_prope"].float(), self.current_q_weight
                )
                current_k0 = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_k_prope"].float(), self.current_k_weight
                )
                current_v_low = torch.einsum(
                    "bhsd,hkd->bhsk", raw["current_v_prope"].float(), self.current_v_weight
                )
                current_k_low = current_k0

        text_logits = torch.matmul(q_full, raw["text_k"].float().transpose(-1, -2)) / math.sqrt(128)
        history_logits = torch.matmul(q_low, k_low.transpose(-1, -2)) / math.sqrt(self.k_dim)
        if self.compress_current:
            current_logits = torch.matmul(
                current_q_low, current_k_low.transpose(-1, -2)
            ) / math.sqrt(self.k_dim)
        else:
            current_logits = torch.matmul(q_full, current_k.transpose(-1, -2)) / math.sqrt(128)
        text_mass, history_mass, current_mass = case["segment_log_mass"]
        logits = torch.cat([
            text_logits + text_mass,
            history_logits * scale + history_mass,
            current_logits + current_mass,
        ], dim=-1)
        probs = logits.softmax(-1)
        nt, nh = raw["text_k"].shape[2], k_low.shape[2]
        text_raw = torch.matmul(probs[..., :nt], raw["text_v"].float())
        history_low = torch.matmul(probs[..., nt:nt + nh], v_low)
        if self.compress_current:
            current_low = torch.matmul(probs[..., nt + nh:], current_v_low)
        else:
            current_raw = torch.matmul(probs[..., nt + nh:], current_v)
        if branch == "prope":
            _, _, apply_o = prope_functions(pos, "q", self.v_dim, q0.device)
            history_low = apply_o(history_low)
            history = torch.einsum("bhsd,hod->bhso", history_low, output_weight)
            history = history.permute(0, 2, 1, 3).reshape(history.shape[0], history.shape[2], -1)
            if self.compress_current:
                current_positioned = torch.einsum(
                    "bhsd,hod->bhso", current_low, output_weight
                )
                text_current = restore_prope_output(text_raw + current_positioned, pos)
            else:
                text_current = restore_prope_output(text_raw + current_raw, pos)
            return text_current + history, logits
        history = torch.einsum(
            "bhsd,hod->bhso",
            history_low + current_low if self.compress_current else history_low,
            output_weight,
        )
        return text_raw + history if self.compress_current else text_raw + current_raw + history, logits

    def forward(self, case):
        return self.branch(case, "main"), self.branch(case, "prope")


def attention_loss(model, case, weights, output_weight, logit_weight, functional_weight, reconstruction_weight):
    main, prope = model(case)
    output = normalized_mse(main[0], case["teacher_main"]) + normalized_mse(prope[0], case["teacher_prope"])
    logits = centered_logit_loss(main[1], case["teacher_main_logits"]) + centered_logit_loss(
        prope[1], case["teacher_prope_logits"]
    )
    block = replay_block(case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0], weights)
    functional = normalized_mse(block, case["teacher_block"])
    reconstruction = model.reconstruction_loss(case)
    return (output_weight * output + logit_weight * logits + functional_weight * functional
            + reconstruction_weight * reconstruction)


def healing_loss(model, block, case, output_weight, logit_weight, reconstruction_weight):
    main, prope = model(case)
    candidate = block(case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0])
    functional = normalized_mse(candidate, case["teacher_block"])
    output = normalized_mse(main[0], case["teacher_main"]) + normalized_mse(prope[0], case["teacher_prope"])
    logits = centered_logit_loss(main[1], case["teacher_main_logits"]) + centered_logit_loss(
        prope[1], case["teacher_prope_logits"]
    )
    return functional + output_weight * output + logit_weight * logits + reconstruction_weight * model.reconstruction_loss(case)


@torch.no_grad()
def evaluate_reconstruction(model, cases):
    if model.payload_mode != "qat_joint":
        return None
    rows = []
    for case in cases:
        row = {}
        for name, target, organizer, reconstructor, decoder in (
            ("k", case["k_payload"], model.k_organizer, model.k_reconstructor, model.k_decoder),
            ("v", case["v_payload"], model.v_organizer, model.v_reconstructor, model.v_decoder),
        ):
            organized = model._ste_int4(F.linear(target["coeff"].float(), organizer))
            recovered = F.linear(organized, reconstructor)
            candidate = F.linear(recovered.to(torch.bfloat16), decoder.to(torch.bfloat16)).float()
            teacher = F.linear(target["dequant"].to(torch.bfloat16), decoder.to(torch.bfloat16)).float()
            row[name + "_full_fallback_rel_l2"] = rel_l2(candidate, teacher)
        rows.append(row)
    return {
        key + "_mean": sum(row[key] for row in rows) / len(rows)
        for key in ("k_full_fallback_rel_l2", "v_full_fallback_rel_l2")
    } | {
        key + "_max": max(row[key] for row in rows)
        for key in ("k_full_fallback_rel_l2", "v_full_fallback_rel_l2")
    }


@torch.no_grad()
def evaluate(model, cases, weights, block=None):
    rows = []
    for case in cases:
        main, prope = model(case)
        if block is None:
            candidate = replay_block(case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0], weights)
        else:
            candidate = block(case["raw"]["layer_input"], case["raw"]["vec"], main[0], prope[0])
        rows.append({
            "case_id": case["raw"]["case_id"],
            "attention_rel_l2_main": rel_l2(main[0], case["teacher_main"]),
            "attention_rel_l2_prope": rel_l2(prope[0], case["teacher_prope"]),
            "block_rel_l2": rel_l2(candidate, case["teacher_block"]),
        })
    summary = {}
    for key in ("attention_rel_l2_main", "attention_rel_l2_prope", "block_rel_l2"):
        values = [row[key] for row in rows]
        summary[key + "_mean"] = sum(values) / len(values)
        summary[key + "_max"] = max(values)
    return rows, summary




