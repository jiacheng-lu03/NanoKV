"""Strict HY-style K-A1 adaptation for the Matrix-Game causal KV path.

This module intentionally mirrors the frozen HY recipe: a clean-K Teacher and
compressed-K Student replay the same final denoising atom, block hidden losses
are averaged over one layer group, K reconstruction/cosine anchors are added,
and decoder-only optimizers step synchronously.  It is enabled only while
``MG35_K_A1_TRAIN=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


_MODELS = {}
_TARGETS = {}
_LOSSES = []
_METRICS = []
_GATE_ALPHA_CACHE = None


def enabled() -> bool:
    return os.environ.get("MG35_K_A1_TRAIN", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def eval_only() -> bool:
    return os.environ.get("MG35_K_A1_EVAL_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def gate_score_enabled() -> bool:
    """Differentiate frozen A1 decoder displacements through the causal endpoint."""
    return os.environ.get("MG35_K_A1_GATE_SCORE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def terminal_output_enabled() -> bool:
    """Keep the full frozen DiT graph for a terminal/chunk-output objective."""
    return enabled() and loss_mode() in {"terminal", "trajectory", "boundary"}


def selected_layers() -> tuple[int, ...]:
    raw = os.environ.get("MG35_K_A1_LAYERS", "")
    layers = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if enabled() and not layers:
        raise RuntimeError("MG35_K_A1_TRAIN requires MG35_K_A1_LAYERS")
    return layers


def selected_chunk(chunk_idx: int) -> bool:
    chunks = {
        int(item.strip())
        for item in os.environ.get("MG35_K_A1_CHUNKS", "1,2,3,4").split(",")
        if item.strip()
    }
    return int(chunk_idx) in chunks


def inner_steps() -> int:
    return int(os.environ.get("MG35_K_A1_INNER_STEPS", "5"))


def retained_rank(total: int) -> int:
    value = int(os.environ.get("MG35_K_A1_RANK", "0"))
    rank = int(total) if value <= 0 else value
    if rank <= 0 or rank > int(total):
        raise RuntimeError(f"invalid MG35_K_A1_RANK={value} for total={total}")
    return rank


def loss_mode() -> str:
    mode = os.environ.get("MG35_K_A1_LOSS_MODE", "block").strip().lower()
    if mode not in {"block", "attention", "terminal", "trajectory", "boundary"}:
        raise RuntimeError(f"unsupported MG35_K_A1_LOSS_MODE={mode!r}")
    return mode


def _gate_alphas() -> dict[int, float]:
    global _GATE_ALPHA_CACHE
    if _GATE_ALPHA_CACHE is not None:
        return _GATE_ALPHA_CACHE
    path = os.environ.get("MG35_K_A1_GATE_ALPHA_JSON", "").strip()
    raw = {} if not path else json.loads(Path(path).read_text())
    values = {int(key): float(value) for key, value in raw.items()}
    unknown = sorted(set(values) - set(range(30)))
    if unknown:
        raise RuntimeError(f"unknown K-A1 gate layers: {unknown}")
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise RuntimeError("K-A1 gate alpha values must be in [0, 1]")
    _GATE_ALPHA_CACHE = values
    return values


def teacher_v_mode() -> str:
    mode = os.environ.get("MG35_K_A1_TEACHER_V_MODE", "same").strip().lower()
    if mode not in {"same", "clean"}:
        raise RuntimeError(f"unsupported MG35_K_A1_TEACHER_V_MODE={mode!r}")
    return mode


def teacher_q_mode() -> str:
    """Select same-state Q or the clean-cache Teacher's propagated Q state."""
    mode = os.environ.get("MG35_K_A1_TEACHER_Q_MODE", "same").strip().lower()
    if mode not in {"same", "teacher"}:
        raise RuntimeError(f"unsupported MG35_K_A1_TEACHER_Q_MODE={mode!r}")
    return mode


def _output_dir() -> Path:
    return Path(os.environ["MG35_K_A1_OUTPUT_DIR"])


def _basis(layer: int):
    root = Path(os.environ["MG35_KVCODEC_BASIS_DIR"])
    artifact = torch.load(
        root / "k_c64" / f"layer_{layer:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected_storage = os.environ.get("MG35_KVCODEC_CENTER_STORAGE", "bf16").strip().lower()
    expected_block = int(os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64"))
    if artifact.get("center_storage") != expected_storage or int(
        artifact.get("center_block_size", -1)
    ) != expected_block:
        raise RuntimeError(f"K-A1 basis center provenance mismatch L{layer}")
    return artifact


def _ensure_model(layer: int, device: torch.device):
    model = _MODELS.get(int(layer))
    if model is not None:
        return model
    artifact = _basis(layer)
    components = artifact["components"].float()
    rank = retained_rank(int(components.shape[0]))
    encoder = components[-rank:].contiguous().to(device)
    source_root = os.environ.get("MG35_K_A1_SOURCE_DIR", "").strip()
    output_checkpoint = _output_dir() / f"layer_{layer:02d}.pt"
    source_checkpoint = (
        Path(source_root) / f"layer_{layer:02d}.pt" if source_root else None
    )
    # An output checkpoint is a true interrupted-run resume.  A checkpoint in
    # SOURCE_DIR supplies decoder initialization only: its Adam moments and
    # step count belong to a different objective and must not leak into the
    # terminal adaptation schedule.
    if output_checkpoint.is_file():
        checkpoint = output_checkpoint
        resume_checkpoint = True
    else:
        checkpoint = source_checkpoint
        resume_checkpoint = False
    if checkpoint is not None and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_storage = os.environ.get("MG35_KVCODEC_CENTER_STORAGE", "bf16").strip().lower()
        expected_block = int(os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64"))
        if saved.get("center_storage") != expected_storage or int(
            saved.get("center_block_size", -1)
        ) != expected_block:
            raise RuntimeError(f"K-A1 checkpoint center provenance mismatch L{layer}")
        decoder_init = saved["decoder"].float()
    else:
        saved = None
        decoder_init = encoder.t().contiguous().cpu()
    if tuple(decoder_init.shape) != tuple(encoder.t().shape):
        raise RuntimeError(
            f"K-A1 decoder rank mismatch L{layer}: "
            f"decoder={tuple(decoder_init.shape)} encoder={tuple(encoder.shape)}"
        )
    decoder = torch.nn.Parameter(
        decoder_init.to(device), requires_grad=not gate_score_enabled()
    )
    gate = None
    if gate_score_enabled():
        gate = torch.nn.Parameter(
            torch.tensor(
                _gate_alphas().get(int(layer), 0.0),
                device=device,
                dtype=torch.float32,
            )
        )
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            [decoder],
            lr=float(os.environ.get("MG35_K_A1_LR", "5e-6")),
            weight_decay=0.0,
        )
        if resume_checkpoint and saved is not None and saved.get("optimizer") is not None:
            optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(
                            device=device,
                            dtype=decoder.dtype if key != "step" else value.dtype,
                        )
    model = {
        "layer": int(layer),
        "mean": artifact["mean"].float().to(device),
        "encoder": encoder,
        "decoder": decoder,
        "initial_decoder": encoder.t().contiguous(),
        "gate": gate,
        "optimizer": optimizer,
        "step": int(saved.get("step", 0)) if resume_checkpoint else 0,
        "rank": int(rank),
    }
    _MODELS[int(layer)] = model
    return model


def _semantic_residual(value: torch.Tensor, layer: int):
    # Identical C64 construction to the production MG3.5 cache-write codec.
    rows = value.float()
    batch, sequence, dim = rows.shape
    clusters = int(os.environ.get("MG35_KVCODEC_C64_CLUSTERS", "64"))
    iters = int(os.environ.get("MG35_KVCODEC_C64_ITERS", "2"))
    generator = torch.Generator(device="cpu").manual_seed(20260804 + int(layer))
    initial = torch.randperm(sequence, generator=generator)[:clusters].to(rows.device)
    centers = rows.index_select(1, initial).clone()
    row_norm = rows.square().sum(-1, keepdim=True)
    labels = torch.zeros((batch, sequence), dtype=torch.long, device=rows.device)
    for _ in range(iters):
        distances = (
            row_norm + centers.square().sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(rows, centers.transpose(1, 2))
        )
        labels = distances.argmin(-1)
        assignment = F.one_hot(labels, num_classes=clusters).transpose(1, 2).float()
        counts = assignment.sum(-1)
        updated = torch.bmm(assignment, rows) / counts.clamp_min(1.0).unsqueeze(-1)
        centers = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    # Reuse the deployment center round-trip. This is BF16 for strict
    # WorldPlay and block-INT8 only when the explicit MG experiment is enabled.
    from .nanokv_codec import store_centers
    _, centers = store_centers(centers)
    assigned = torch.gather(
        centers, 1, labels[..., None].expand(-1, -1, dim)
    )
    return rows - assigned, assigned


def reconstruct(layer: int, clean_k: torch.Tensor, *, track_grad: bool):
    model = _ensure_model(layer, clean_k.device)
    residual, assigned = _semantic_residual(clean_k.detach(), layer)
    coeff = F.linear(residual.reshape(-1, residual.shape[-1]) - model["mean"], model["encoder"])
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / 7.0
    stored_scale = scale.to(torch.bfloat16).float()
    exact = (coeff / scale).round().clamp(-7, 7) * stored_scale
    quantized = coeff + (exact - coeff).detach() if track_grad else exact
    decoder = model["decoder"]
    if gate_score_enabled():
        initial = model["initial_decoder"]
        decoder = initial + model["gate"] * (decoder - initial)
    decoded = F.linear(quantized, decoder, model["mean"])
    decoded = decoded.reshape_as(residual) + assigned
    return decoded.to(dtype=clean_k.dtype)


def reconstruct_segmented(
    layer: int,
    clean_k: torch.Tensor,
    frames: list[int],
    chunk_ids: list[int],
    *,
    track_grad: bool,
    write_ids: list[int] | None = None,
) -> torch.Tensor:
    """Reproduce the deployed immutable source-chunk payload boundaries.

    Deployment builds C64 centers once per cache write, so Student history must
    be regrouped by write boundaries.  ``write_ids`` carries that explicit
    per-frame write index; ``chunk_ids`` is only a fallback because corrected
    global history labels every generated chunk ``-1`` and would merge writes.
    """
    segment_ids = list(write_ids) if write_ids else list(chunk_ids)
    if not frames or len(frames) != len(segment_ids):
        raise RuntimeError(
            f"Strict K-A1 invalid cache metadata: frames={len(frames)} "
            f"write_ids={len(write_ids) if write_ids else None} "
            f"chunk_ids={len(chunk_ids)}"
        )
    if int(clean_k.shape[1]) % len(frames):
        raise RuntimeError(
            f"Strict K-A1 tokens={clean_k.shape[1]} not divisible by frames={len(frames)}"
        )
    tokens_per_frame = int(clean_k.shape[1]) // len(frames)
    runs = []
    first = 0
    for index in range(1, len(segment_ids) + 1):
        if index == len(segment_ids) or segment_ids[index] != segment_ids[first]:
            runs.append((first * tokens_per_frame, index * tokens_per_frame))
            first = index
    parts = [
        reconstruct(layer, clean_k[:, start:stop], track_grad=track_grad)
        for start, stop in runs
    ]
    return torch.cat(parts, dim=1)


def _copy_cache(cache):
    return {
        "k": cache.get("k"),
        "v": cache.get("v"),
        "positions": list(cache.get("positions", cache.get("frames", []))),
        "frames": list(cache.get("frames", [])),
        "chunk_ids": list(cache.get("chunk_ids", [])),
        "_k_clean_for_a1": cache.get("_k_clean_for_a1"),
        "_v_clean_for_a1": cache.get("_v_clean_for_a1"),
        "_a1_write_ids": list(cache.get("_a1_write_ids", [])),
    }


def teacher_cache(caches):
    layers = set(selected_layers())
    out = [_copy_cache(cache) for cache in caches]
    for layer in layers:
        clean = out[layer].get("_k_clean_for_a1")
        if not torch.is_tensor(clean):
            raise RuntimeError(f"Strict K-A1 Teacher missing clean K at layer {layer}")
        out[layer]["k"] = clean.detach()
        if teacher_v_mode() == "clean":
            clean_v = out[layer].get("_v_clean_for_a1")
            if not torch.is_tensor(clean_v):
                raise RuntimeError(f"Strict K-A1 Teacher missing clean V at layer {layer}")
            out[layer]["v"] = clean_v.detach()
    return out


def student_cache(caches):
    layers = set(selected_layers())
    out = [_copy_cache(cache) for cache in caches]
    for layer in layers:
        clean = out[layer].get("_k_clean_for_a1")
        if not torch.is_tensor(clean):
            raise RuntimeError(f"Strict K-A1 Student missing clean K at layer {layer}")
        # The distilled inference entry is globally no-grad. HY's strict
        # reapply path explicitly overrides that context for Student K too.
        with torch.set_grad_enabled(True):
            out[layer]["k"] = reconstruct_segmented(
                layer,
                clean,
                list(out[layer].get("frames", [])),
                list(out[layer].get("chunk_ids", [])),
                track_grad=True,
                write_ids=list(out[layer].get("_a1_write_ids", [])) or None,
            )
        if not out[layer]["k"].requires_grad:
            raise RuntimeError(f"Strict K-A1 Student K lost autograd at layer {layer}")
    return out


def reset_atom() -> None:
    _TARGETS.clear()
    _LOSSES.clear()
    _METRICS.clear()


def reset_student_losses() -> None:
    """Drain one Student inner-step graph while preserving Teacher targets."""
    _LOSSES.clear()
    _METRICS.clear()


def record_block(pass_name: str, layer: int, hidden: torch.Tensor) -> None:
    if loss_mode() != "block":
        return
    if int(layer) not in set(selected_layers()):
        return
    token_count = min(256, int(hidden.shape[1]))
    channel_count = min(256, int(hidden.shape[2]))
    seed = 2027
    gen_t = torch.Generator(device="cpu").manual_seed(seed + int(layer) * 1009)
    gen_c = torch.Generator(device="cpu").manual_seed(seed + int(layer) * 2003 + 17)
    token_idx = torch.randperm(int(hidden.shape[1]), generator=gen_t)[:token_count].sort().values.to(hidden.device)
    channel_idx = torch.randperm(int(hidden.shape[2]), generator=gen_c)[:channel_count].sort().values.to(hidden.device)
    sketch = hidden[:, token_idx][:, :, channel_idx]
    if pass_name == "teacher":
        _TARGETS[int(layer)] = sketch.detach()
        return
    if pass_name != "student":
        return
    target = _TARGETS.get(int(layer))
    if target is None:
        raise RuntimeError(f"Strict K-A1 missing Teacher block target L{layer}")
    pred = sketch.float()
    ref = target.to(device=pred.device).float()
    rel = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]), ref.reshape(-1, ref.shape[-1]), dim=-1
    ).mean()
    loss = rel + 0.1 * (1.0 - cosine)
    if not loss.requires_grad:
        raise RuntimeError(f"Strict K-A1 block loss has no autograd graph L{layer}")
    _LOSSES.append(loss)
    _METRICS.append({
        "layer": int(layer),
        "block_rel_l2": float(rel.detach().cpu()),
        "block_cosine": float(cosine.detach().cpu()),
    })


def record_attention(
    layer: int,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
) -> None:
    """Match historical-K readout with identical current Q/V and hidden state.

    Append-only models write the generated state back into later cache entries.
    A full block-hidden relative loss is heavily diluted by the residual stream,
    cross-attention, and MLP.  This counterfactual instead isolates the causal
    operation that the K decoder controls: routing clean V with compressed K.
    """
    if loss_mode() != "attention" or int(layer) not in set(selected_layers()):
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
        raise RuntimeError(f"Strict K-A1 attention loss has no autograd graph L{layer}")
    _LOSSES.append(loss)
    _METRICS.append({
        "layer": int(layer),
        "attention_rel_l2": float(rel.detach().cpu()),
        "attention_cosine": float(cosine.detach().cpu()),
    })


def record_teacher_attention(layer: int, teacher_output: torch.Tensor) -> None:
    """Store the direct clean-cache Teacher readout for a propagated-Q target."""
    if (
        loss_mode() != "attention"
        or teacher_q_mode() != "teacher"
        or int(layer) not in set(selected_layers())
    ):
        return
    _TARGETS[int(layer)] = teacher_output.detach()


def teacher_attention_target(layer: int) -> torch.Tensor:
    target = _TARGETS.get(int(layer))
    if target is None:
        raise RuntimeError(f"Strict K-A1 missing Teacher attention target L{layer}")
    return target


def _anchor(caches):
    rels, cosines, losses = [], [], []
    sample_count = 65536
    for layer in selected_layers():
        clean = caches[layer]["_k_clean_for_a1"].detach().float().reshape(-1)
        corrupt = caches[layer]["k"].float().reshape(-1)
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
        model = _ensure_model(layer, torch.device("cuda"))
        if gate_score_enabled():
            model["gate"].grad = None
        else:
            model["optimizer"].zero_grad(set_to_none=True)


def paired_prediction_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Return the functional clean-cache alignment loss for one denoise step."""
    with torch.enable_grad():
        pred = student_prediction.float()
        ref = teacher_prediction.detach().to(device=pred.device).float()
        rel = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
        cosine = F.cosine_similarity(
            pred.flatten(1), ref.flatten(1), dim=-1
        ).mean()
        functional = rel + 0.1 * (1.0 - cosine)
    if not functional.requires_grad:
        raise RuntimeError("K-A1 paired prediction loss has no autograd graph")
    return functional, {
        "prediction_rel_l2": float(rel.detach().cpu()),
        "prediction_cosine": float(cosine.detach().cpu()),
    }


def trajectory_loss_and_metrics(
    step_losses: list[torch.Tensor],
    step_metrics: list[dict],
    student_caches,
    *,
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
):
    """Compose one full-denoising chunk loss and one reconstruction guard.

    The Student graph spans every causal-DMD denoise transition in the current
    chunk.  Teacher states are detached.  All selected K decoders therefore
    receive one synchronous update from the coupled chunk endpoint instead of
    one update per denoise atom.
    """
    if loss_mode() != "trajectory":
        raise RuntimeError("trajectory loss requested outside trajectory mode")
    if not step_losses or len(step_losses) != len(step_metrics):
        raise RuntimeError(
            "K-A1 trajectory requires matched non-empty step losses/metrics"
        )
    raw_weights = os.environ.get(
        "MG35_K_A1_TRAJECTORY_STEP_WEIGHTS", "0.25,0.5,1.0"
    )
    weights = [float(item.strip()) for item in raw_weights.split(",") if item.strip()]
    if len(weights) != len(step_losses) or any(value < 0.0 for value in weights):
        raise RuntimeError(
            "MG35_K_A1_TRAJECTORY_STEP_WEIGHTS must provide one non-negative "
            f"weight per denoise step; got {weights} for {len(step_losses)} steps"
        )
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise RuntimeError("K-A1 trajectory step weights must have positive sum")

    with torch.enable_grad():
        step_functional = sum(
            weight * loss for weight, loss in zip(weights, step_losses)
        ) / weight_sum
        pred = student_latent.float()
        ref = teacher_latent.detach().to(device=pred.device).float()
        chunk_rel = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
        chunk_cosine = F.cosine_similarity(
            pred.flatten(1), ref.flatten(1), dim=-1
        ).mean()
        chunk_functional = chunk_rel + 0.1 * (1.0 - chunk_cosine)
        chunk_weight = float(
            os.environ.get("MG35_K_A1_TRAJECTORY_CHUNK_WEIGHT", "1.0")
        )
        if chunk_weight < 0.0:
            raise RuntimeError("MG35_K_A1_TRAJECTORY_CHUNK_WEIGHT must be non-negative")
        functional = (
            step_functional + chunk_weight * chunk_functional
        ) / (1.0 + chunk_weight)
        anchor, anchor_metrics = _anchor(student_caches)
        total = functional + anchor
    if not total.requires_grad or total.grad_fn is None:
        raise RuntimeError("K-A1 trajectory combined loss has no autograd graph")
    metrics = {
        "steps": [dict(row) for row in step_metrics],
        "step_weights": weights,
        "chunk_weight": chunk_weight,
        "chunk_rel_l2": float(chunk_rel.detach().cpu()),
        "chunk_cosine": float(chunk_cosine.detach().cpu()),
    }
    return total, functional, metrics, anchor_metrics


def loss_and_metrics(
    student_caches,
    *,
    student_prediction: torch.Tensor | None = None,
    teacher_prediction: torch.Tensor | None = None,
):
    if not isinstance(student_caches, list) or any(
        not isinstance(cache, dict) for cache in student_caches
    ):
        raise TypeError("Strict K-A1 loss expects the per-layer cache list")
    if loss_mode() in {"terminal", "boundary"}:
        if student_prediction is None or teacher_prediction is None:
            raise RuntimeError(
                "K-A1 terminal/boundary loss requires paired predictions"
            )
        # The public distilled runner is globally no-grad.  The Student
        # prediction already carries the full cache-gate Jacobian, but the
        # terminal objective itself must also be composed inside a grad island
        # or PyTorch returns a detached scalar.
        functional, pair_metrics = paired_prediction_loss(
            student_prediction, teacher_prediction
        )
        metrics = [{
            "terminal_rel_l2": pair_metrics["prediction_rel_l2"],
            "terminal_cosine": pair_metrics["prediction_cosine"],
        }]
        if gate_score_enabled():
            return functional, functional, metrics, {
                "anchor_rel_l2": 0.0,
                "anchor_cosine": 1.0,
            }
        # Decoder training retains the same reconstruction guard as block-A1.
        # The terminal target is allowed to coordinate all selected layers,
        # while this anchor prevents a short calibration trajectory from
        # learning an arbitrary endpoint-specific inverse.
        with torch.enable_grad():
            anchor, anchor_metrics = _anchor(student_caches)
            total = functional + anchor
        if not total.requires_grad or total.grad_fn is None:
            raise RuntimeError("K-A1 terminal combined loss has no autograd graph")
        return total, functional, metrics, anchor_metrics
    if len(_LOSSES) != len(selected_layers()):
        raise RuntimeError(
            f"Strict K-A1 expected {len(selected_layers())} block losses, got {len(_LOSSES)}"
        )
    # The public distilled runner is no-grad; HY explicitly re-enables grad for
    # both its reconstruction anchor and the final objective composition.
    with torch.enable_grad():
        functional = torch.stack(_LOSSES).mean()
        anchor, anchor_metrics = _anchor(student_caches)
        total = functional + anchor
    if not total.requires_grad or total.grad_fn is None:
        raise RuntimeError("Strict K-A1 combined loss has no autograd graph")
    return total, functional, list(_METRICS), anchor_metrics


def step(loss: torch.Tensor, *, atom: dict) -> dict:
    loss.backward()
    if gate_score_enabled():
        info = {"gate_score": True, "layers": {}}
        for layer in selected_layers():
            model = _MODELS[layer]
            grad = model["gate"].grad
            if grad is None or not torch.isfinite(grad):
                raise RuntimeError(f"invalid K-A1 gate gradient L{layer}")
            value = float(grad.detach().cpu())
            info["layers"][str(layer)] = {
                "alpha": float(model["gate"].detach().cpu()),
                "gradient": value,
                "directional_utility": -value,
            }
        return info
    info = {"max_grad_norm": 0.0, "layers": {}}
    for layer in selected_layers():
        model = _MODELS[layer]
        grad = torch.nn.utils.clip_grad_norm_([model["decoder"]], 1.0)
        info["max_grad_norm"] = max(info["max_grad_norm"], float(grad.detach().cpu()))
        before = model["decoder"].detach().clone()
        model["optimizer"].step()
        model["step"] += 1
        update = float((model["decoder"].detach() - before).float().norm().cpu())
        info["layers"][str(layer)] = {"grad_norm": float(grad.detach().cpu()), "update_norm": update}
    save(atom=atom)
    return info


def save(*, atom: dict) -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers():
        model = _MODELS[layer]
        payload = {
            "format": "mg35_strict_hy_k_a1_v1",
            "target": "k_c64",
            "layer": int(layer),
            "decoder": model["decoder"].detach().cpu().float(),
            "optimizer": model["optimizer"].state_dict(),
            "step": int(model["step"]),
            "rank": int(model["rank"]),
            "center_storage": os.environ.get(
                "MG35_KVCODEC_CENTER_STORAGE", "bf16"
            ).strip().lower(),
            "center_block_size": int(
                os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64")
            ),
            "recipe": {
                "teacher_cache_mode": "same_state_group",
                "loss_mode": loss_mode(),
                "local_target": (
                    "attention_output"
                    if loss_mode() == "attention"
                    else "next_chunk_first_prediction_after_cache_fill"
                    if loss_mode() == "boundary"
                    else "terminal_chunk_prediction"
                    if loss_mode() == "terminal"
                    else "full_denoising_chunk_trajectory"
                    if loss_mode() == "trajectory"
                    else "hidden"
                ),
                "teacher_v_mode": teacher_v_mode(),
                "token_samples": 256,
                "channel_samples": 256,
                "inner_steps": inner_steps(),
                "reconstruction_weight": 0.2,
                "cosine_weight": 0.02,
                "decoder_only": True,
            },
            "last_atom": dict(atom),
        }
        tmp = out / f"layer_{layer:02d}.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(out / f"layer_{layer:02d}.pt")


def append_log(record: dict) -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "strict_a1_steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
