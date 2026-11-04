"""Attention-aligned V adaptation conditioned on a frozen K-A1 working point.

This is deliberately separate from :mod:`causal_k_a1`.  K-A1, the V encoder,
the INT4 code coordinates, C64, and the DiT remain frozen; only the per-layer V
decoder is trainable.  The module is inert unless ``MG35_V_A1_TRAIN=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


_MODELS = {}
_LOSSES = []
_METRICS = []


def enabled() -> bool:
    return os.environ.get("MG35_V_A1_TRAIN", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def eval_only() -> bool:
    return os.environ.get("MG35_V_A1_EVAL_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def selected_layers() -> tuple[int, ...]:
    raw = os.environ.get("MG35_V_A1_LAYERS", "")
    layers = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if enabled() and not layers:
        raise RuntimeError("MG35_V_A1_TRAIN requires MG35_V_A1_LAYERS")
    return layers


def selected_chunk(chunk_idx: int) -> bool:
    chunks = {
        int(item.strip())
        for item in os.environ.get("MG35_V_A1_CHUNKS", "1,2,3,4").split(",")
        if item.strip()
    }
    return int(chunk_idx) in chunks


def inner_steps() -> int:
    return int(os.environ.get("MG35_V_A1_INNER_STEPS", "5"))


def teacher_k_mode() -> str:
    """Choose the causal question answered by the V update.

    ``same`` isolates marginal V error under the already deployed K-A1.
    ``clean`` lets V compensate the remaining joint K/V attention error.
    """
    mode = os.environ.get("MG35_V_A1_TEACHER_K_MODE", "clean").strip().lower()
    if mode not in {"same", "clean"}:
        raise RuntimeError(f"unsupported MG35_V_A1_TEACHER_K_MODE={mode!r}")
    return mode


def _output_dir() -> Path:
    return Path(os.environ["MG35_V_A1_OUTPUT_DIR"])


def _initial_dir() -> Path:
    return Path(os.environ["MG35_V_A1_INIT_DIR"])


def _load_v_checkpoint(path: Path, layer: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "mg35_worldplay_v_c64_int4_qat_v1":
        raise RuntimeError(f"bad V initialization/checkpoint: {path}")
    if payload.get("target") != "v_c64" or int(payload.get("layer", -1)) != layer:
        raise RuntimeError(f"V checkpoint provenance mismatch L{layer}: {path}")
    expected_storage = os.environ.get(
        "MG35_KVCODEC_CENTER_STORAGE", "bf16"
    ).strip().lower()
    expected_block = int(os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64"))
    if payload.get("center_storage") != expected_storage or int(
        payload.get("center_block_size", -1)
    ) != expected_block:
        raise RuntimeError(f"V checkpoint center provenance mismatch L{layer}: {path}")
    return payload


def _ensure_model(layer: int, device: torch.device):
    model = _MODELS.get(int(layer))
    if model is not None:
        return model
    output_checkpoint = _output_dir() / f"layer_{layer:02d}.pt"
    initial_checkpoint = _initial_dir() / f"layer_{layer:02d}.pt"
    initial = _load_v_checkpoint(initial_checkpoint, layer)
    saved = (
        _load_v_checkpoint(output_checkpoint, layer)
        if output_checkpoint.is_file()
        else initial
    )
    # The V code coordinates are immutable during this phase.  In particular,
    # resuming an adapted decoder must never silently swap its frozen encoder.
    if not torch.equal(saved["encoder"].float(), initial["encoder"].float()):
        raise RuntimeError(f"V-on-KA1 encoder drift detected L{layer}")
    decoder = torch.nn.Parameter(saved["decoder"].float().to(device))
    optimizer = torch.optim.AdamW(
        [decoder],
        lr=float(os.environ.get("MG35_V_A1_LR", "5e-6")),
        weight_decay=0.0,
    )
    if saved.get("v_a1_optimizer") is not None:
        optimizer.load_state_dict(saved["v_a1_optimizer"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(
                        device=device,
                        dtype=decoder.dtype if key != "step" else value.dtype,
                    )
    model = {
        "layer": int(layer),
        "mean": initial["mean"].float().to(device),
        "encoder": initial["encoder"].float().to(device),
        "decoder": decoder,
        "optimizer": optimizer,
        "step": int(saved.get("v_a1_step", 0)),
        "rank": int(initial["rank"]),
    }
    _MODELS[int(layer)] = model
    return model


def reconstruct(layer: int, clean_v: torch.Tensor, *, track_grad: bool):
    """Apply the deployed C64/INT4 V encoder and the trainable V decoder."""
    model = _ensure_model(layer, clean_v.device)
    from .causal_k_a1 import _semantic_residual

    residual, assigned = _semantic_residual(clean_v.detach(), layer)
    coeff = F.linear(
        residual.reshape(-1, residual.shape[-1]) - model["mean"],
        model["encoder"],
    )
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    ) / 7.0
    stored_scale = scale.to(torch.bfloat16).float()
    exact = (coeff / scale).round().clamp(-7, 7) * stored_scale
    # Decoder-only training does not require STE through the frozen encoder,
    # but retaining this form makes the quantized working point explicit.
    quantized = coeff + (exact - coeff).detach() if track_grad else exact
    decoded = F.linear(quantized, model["decoder"], model["mean"])
    decoded = decoded.reshape_as(residual) + assigned
    return decoded.to(dtype=clean_v.dtype)


def reconstruct_segmented(
    layer: int,
    clean_v: torch.Tensor,
    frames: list[int],
    chunk_ids: list[int],
    *,
    track_grad: bool,
) -> torch.Tensor:
    if not frames or len(frames) != len(chunk_ids):
        raise RuntimeError(
            f"V-on-KA1 invalid cache metadata: frames={len(frames)} "
            f"chunk_ids={len(chunk_ids)}"
        )
    if int(clean_v.shape[1]) % len(frames):
        raise RuntimeError(
            f"V-on-KA1 tokens={clean_v.shape[1]} not divisible by frames={len(frames)}"
        )
    tokens_per_frame = int(clean_v.shape[1]) // len(frames)
    runs = []
    first = 0
    for index in range(1, len(chunk_ids) + 1):
        if index == len(chunk_ids) or chunk_ids[index] != chunk_ids[first]:
            runs.append((first * tokens_per_frame, index * tokens_per_frame))
            first = index
    return torch.cat(
        [
            reconstruct(layer, clean_v[:, start:stop], track_grad=track_grad)
            for start, stop in runs
        ],
        dim=1,
    )


def _copy_cache(cache):
    return {
        "k": cache.get("k"),
        "v": cache.get("v"),
        "positions": list(cache.get("positions", cache.get("frames", []))),
        "frames": list(cache.get("frames", [])),
        "chunk_ids": list(cache.get("chunk_ids", [])),
        "_k_clean_for_a1": cache.get("_k_clean_for_a1"),
        "_v_clean_for_a1": cache.get("_v_clean_for_a1"),
    }


def teacher_cache(caches):
    out = [_copy_cache(cache) for cache in caches]
    for layer in selected_layers():
        clean_v = out[layer].get("_v_clean_for_a1")
        if not torch.is_tensor(clean_v):
            raise RuntimeError(f"V-on-KA1 Teacher missing clean V at layer {layer}")
        out[layer]["v"] = clean_v.detach()
        if teacher_k_mode() == "clean":
            clean_k = out[layer].get("_k_clean_for_a1")
            if not torch.is_tensor(clean_k):
                raise RuntimeError(f"V-on-KA1 Teacher missing clean K at layer {layer}")
            out[layer]["k"] = clean_k.detach()
    return out


def student_cache(caches):
    out = [_copy_cache(cache) for cache in caches]
    for layer in selected_layers():
        clean_v = out[layer].get("_v_clean_for_a1")
        if not torch.is_tensor(clean_v):
            raise RuntimeError(f"V-on-KA1 Student missing clean V at layer {layer}")
        with torch.set_grad_enabled(True):
            out[layer]["v"] = reconstruct_segmented(
                layer,
                clean_v,
                list(out[layer].get("frames", [])),
                list(out[layer].get("chunk_ids", [])),
                track_grad=True,
            )
        if not out[layer]["v"].requires_grad:
            raise RuntimeError(f"V-on-KA1 Student V lost autograd at layer {layer}")
    return out


def reset_atom() -> None:
    _LOSSES.clear()
    _METRICS.clear()


def reset_student_losses() -> None:
    _LOSSES.clear()
    _METRICS.clear()


def record_attention(
    layer: int,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
) -> None:
    if int(layer) not in set(selected_layers()):
        return
    pred = student_output.float()
    ref = teacher_output.detach().to(device=pred.device).float()
    rel = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]),
        ref.reshape(-1, ref.shape[-1]),
        dim=-1,
    ).mean()
    loss = rel + 0.1 * (1.0 - cosine)
    if not loss.requires_grad:
        raise RuntimeError(f"V-on-KA1 attention loss has no autograd graph L{layer}")
    _LOSSES.append(loss)
    _METRICS.append(
        {
            "layer": int(layer),
            "attention_rel_l2": float(rel.detach().cpu()),
            "attention_cosine": float(cosine.detach().cpu()),
        }
    )


def _anchor(caches):
    rels, cosines, losses = [], [], []
    sample_count = 65536
    for layer in selected_layers():
        clean = caches[layer]["_v_clean_for_a1"].detach().float().reshape(-1)
        corrupt = caches[layer]["v"].float().reshape(-1)
        if sample_count < clean.numel():
            stride = max(1, clean.numel() // sample_count)
            ids = torch.arange(0, clean.numel(), stride, device=clean.device)[:sample_count]
            clean = clean.index_select(0, ids)
            corrupt = corrupt.index_select(0, ids)
        rel = (corrupt - clean).norm() / clean.norm().clamp_min(1e-8)
        cosine = F.cosine_similarity(corrupt, clean.to(corrupt.device), dim=0)
        losses.append(0.2 * rel + 0.02 * (1.0 - cosine))
        rels.append(float(rel.detach().cpu()))
        cosines.append(float(cosine.detach().cpu()))
    return torch.stack(losses).mean(), {
        "anchor_rel_l2": sum(rels) / len(rels),
        "anchor_cosine": sum(cosines) / len(cosines),
    }


def zero_grad() -> None:
    for layer in selected_layers():
        _ensure_model(layer, torch.device("cuda"))["optimizer"].zero_grad(
            set_to_none=True
        )


def loss_and_metrics(student_caches):
    if len(_LOSSES) != len(selected_layers()):
        raise RuntimeError(
            f"V-on-KA1 expected {len(selected_layers())} attention losses, "
            f"got {len(_LOSSES)}"
        )
    with torch.enable_grad():
        functional = torch.stack(_LOSSES).mean()
        anchor, anchor_metrics = _anchor(student_caches)
        total = functional + anchor
    if not total.requires_grad or total.grad_fn is None:
        raise RuntimeError("V-on-KA1 combined loss has no autograd graph")
    return total, functional, list(_METRICS), anchor_metrics


def step(loss: torch.Tensor, *, atom: dict) -> dict:
    loss.backward()
    info = {"max_grad_norm": 0.0, "layers": {}}
    for layer in selected_layers():
        model = _MODELS[layer]
        grad = torch.nn.utils.clip_grad_norm_([model["decoder"]], 1.0)
        info["max_grad_norm"] = max(
            info["max_grad_norm"], float(grad.detach().cpu())
        )
        before = model["decoder"].detach().clone()
        model["optimizer"].step()
        model["step"] += 1
        update = float((model["decoder"].detach() - before).float().norm().cpu())
        info["layers"][str(layer)] = {
            "grad_norm": float(grad.detach().cpu()),
            "update_norm": update,
        }
    save(atom=atom)
    return info


def save(*, atom: dict) -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers():
        model = _MODELS[layer]
        payload = {
            "format": "mg35_worldplay_v_c64_int4_qat_v1",
            "target": "v_c64",
            "layer": int(layer),
            "rank": int(model["rank"]),
            "mode": "decoder_only_attention_adaptation",
            "initialization": "formal_v_qat",
            "mean": model["mean"].detach().cpu().float(),
            "encoder": model["encoder"].detach().cpu().float(),
            "decoder": model["decoder"].detach().cpu().float(),
            "v_a1_optimizer": model["optimizer"].state_dict(),
            "v_a1_step": int(model["step"]),
            "center_storage": os.environ.get(
                "MG35_KVCODEC_CENTER_STORAGE", "bf16"
            ).strip().lower(),
            "center_block_size": int(
                os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64")
            ),
            "v_a1_recipe": {
                "condition": "frozen_k_a1",
                "local_target": "attention_output",
                "teacher_k_mode": teacher_k_mode(),
                "inner_steps": inner_steps(),
                "reconstruction_weight": 0.2,
                "cosine_weight": 0.02,
                "decoder_only": True,
                "encoder_frozen": True,
            },
            "last_v_a1_atom": dict(atom),
        }
        tmp = out / f"layer_{layer:02d}.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(out / f"layer_{layer:02d}.pt")


def append_log(record: dict) -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "v_on_ka1_steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
