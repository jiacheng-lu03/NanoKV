"""Opt-in K-routing distillation for codec-native old-history attention.

The formal Teacher and low-D Student must be two consumers of one physical
rank-512 INT4+C64 payload.  This module never encodes or decodes a cache.  It
only records small routing targets from already-positioned Teacher K and
compares them with the already-positioned D32/D64 Student K.

For each sampled current query, the objective separates the routing problem
into the two factors which completely determine old-history probability when
the native text/recent/current branch is fixed:

* centered logits / conditional KL inside old history; and
* old-history log-sum-exp relative to the native branch log-sum-exp.

Everything is dormant unless ``HY_CODEC_NATIVE_ROUTING_DISTILL_ENABLE=1`` and
the functional trainer is in its explicit ``teacher`` or ``student`` pass.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F


_TARGETS: dict[tuple[int, int, int], dict[str, Any]] = {}
_LOSSES: list[torch.Tensor] = []
_METRICS: list[dict[str, Any]] = []
_CONTRACT_CACHE_KEY: Optional[tuple[Any, ...]] = None
_CONTRACT_CACHE_VALUE: Optional[dict[str, Any]] = None


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enabled() -> bool:
    return _flag("HY_CODEC_NATIVE_ROUTING_DISTILL_ENABLE")


def _parse_layers(name: str, fallback: str = "") -> tuple[int, ...]:
    value = os.environ.get(name, fallback).strip()
    try:
        layers = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not layers:
        raise ValueError(f"{name} must not be empty for routing distillation")
    if layers[0] < 0 or layers[-1] > 53:
        raise ValueError(f"{name} must contain only WorldPlay layers L0-53")
    return layers


def selected_layers() -> tuple[int, ...]:
    fallback = os.environ.get(
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS",
        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", ""),
    )
    return _parse_layers("HY_CODEC_NATIVE_ROUTING_DISTILL_LAYERS", fallback)


def _normalized_teacher_cache_mode() -> str:
    value = os.environ.get(
        "HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE", "clean_all"
    ).strip().lower()
    aliases = {
        "same_state": "same_state_group",
        "local": "same_state_group",
        "local_group": "same_state_group",
        "selected_clean": "same_state_group",
    }
    return aliases.get(value, value)


def validate_runtime_contract() -> dict[str, Any]:
    """Hard-gate the only supported, query-only causal training topology."""

    global _CONTRACT_CACHE_KEY, _CONTRACT_CACHE_VALUE
    names = (
        "HY_CODEC_NATIVE_ROUTING_DISTILL_ENABLE",
        "HY_FUNCTIONAL_BLOCK_FT_LOSS_MODE",
        "HY_FUNCTIONAL_BLOCK_FT_TARGET",
        "HY_FUNCTIONAL_BLOCK_FT_LAYERS",
        "HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE",
        "HY_DIRTY_HISTORY_RECOVERY_TEACHER_MODE",
        "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER",
        "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_ENABLE",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_TARGETS",
        "HY_CODEC_NATIVE_ROUTING_DISTILL_LAYERS",
        "HY_DIRECT_CODEC_LATENT_ENABLE",
        "HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES",
        "HY_DIRECT_CODEC_RECENT_NATIVE_LAYERS",
        "HY_DIRECT_CODEC_FORMAL_ANCHOR_STRIDE",
        "HY_DIRECT_CODEC_FORMAL_ANCHOR_IDENTITY_SINK_FRAMES",
        "HY_CODEC_NATIVE_ROUTING_QUERY_SAMPLES",
        "HY_CODEC_NATIVE_ROUTING_CANDIDATES",
        "HY_CODEC_NATIVE_ROUTING_TOPK",
        "HY_CODEC_NATIVE_ROUTING_TEMPERATURE",
        "HY_DIRTY_HISTORY_OUTPUT_ADAPTER_CLEAN_WEIGHT",
    )
    cache_key = tuple(os.environ.get(name) for name in names)
    if cache_key == _CONTRACT_CACHE_KEY and _CONTRACT_CACHE_VALUE is not None:
        return dict(_CONTRACT_CACHE_VALUE)

    if not enabled():
        raise RuntimeError("routing contract requested while distillation is disabled")
    exact_values = {
        "HY_FUNCTIONAL_BLOCK_FT_LOSS_MODE": "routing",
        "HY_FUNCTIONAL_BLOCK_FT_TARGET": "kv",
        "HY_DIRTY_HISTORY_RECOVERY_TEACHER_MODE": "local_exact",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_TARGETS": "query",
    }
    mismatches = {
        name: (os.environ.get(name, "").strip().lower(), expected)
        for name, expected in exact_values.items()
        if os.environ.get(name, "").strip().lower() != expected
    }
    required_flags = (
        "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_ENABLE",
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN",
        "HY_DIRECT_CODEC_LATENT_ENABLE",
    )
    disabled_flags = [name for name in required_flags if not _flag(name)]
    if mismatches or disabled_flags:
        raise RuntimeError(
            "routing distillation requires query-only local-exact training: "
            f"mismatches={mismatches} disabled_flags={disabled_flags}"
        )
    if _normalized_teacher_cache_mode() != "same_state_group":
        raise RuntimeError(
            "routing distillation requires same_state_group formal Teacher cache"
        )

    routing_layers = selected_layers()
    layer_sets = {
        "functional": _parse_layers("HY_FUNCTIONAL_BLOCK_FT_LAYERS"),
        "adapter": _parse_layers("HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS"),
        "formal_teacher": _parse_layers(
            "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS"
        ),
    }
    differing = {
        name: value for name, value in layer_sets.items() if value != routing_layers
    }
    if differing:
        raise RuntimeError(
            "routing/functional/adapter/formal-Teacher layer sets must be exact: "
            f"routing={routing_layers} differing={differing}"
        )

    recent_frames = int(
        os.environ.get("HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES", "0") or 0
    )
    if recent_frames <= 0:
        raise RuntimeError("routing distillation requires a positive native recent tail")
    recent_layers = os.environ.get(
        "HY_DIRECT_CODEC_RECENT_NATIVE_LAYERS", ""
    ).strip()
    if recent_layers and _parse_layers(
        "HY_DIRECT_CODEC_RECENT_NATIVE_LAYERS"
    ) != routing_layers:
        raise RuntimeError(
            "a recent-native layer filter must exactly cover routing layers"
        )
    anchor_stride = int(
        os.environ.get("HY_DIRECT_CODEC_FORMAL_ANCHOR_STRIDE", "0") or 0
    )
    anchor_sink = int(
        os.environ.get(
            "HY_DIRECT_CODEC_FORMAL_ANCHOR_IDENTITY_SINK_FRAMES", "0"
        )
        or 0
    )
    if anchor_stride or anchor_sink:
        raise RuntimeError(
            "minimal routing distillation forbids formal anchors; Teacher/Student "
            "old-history rows must have one identical dense ordering"
        )

    query_samples = int(
        os.environ.get("HY_CODEC_NATIVE_ROUTING_QUERY_SAMPLES", "8")
    )
    candidates = int(
        os.environ.get("HY_CODEC_NATIVE_ROUTING_CANDIDATES", "256")
    )
    topk = int(os.environ.get("HY_CODEC_NATIVE_ROUTING_TOPK", "16"))
    temperature = float(
        os.environ.get("HY_CODEC_NATIVE_ROUTING_TEMPERATURE", "1.0")
    )
    if query_samples <= 0 or candidates <= 1 or not 0 < topk <= candidates:
        raise RuntimeError(
            "routing samples require query_samples>0 and 0<topk<=candidates"
        )
    if temperature <= 0.0:
        raise RuntimeError("routing KL temperature must be positive")
    clean_weight = float(
        os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_CLEAN_WEIGHT", "0")
    )
    if clean_weight != 0.0:
        raise RuntimeError(
            "routing-only training forbids the extra clean-adapter prediction pass"
        )

    report = {
        "layers": routing_layers,
        "query_samples": query_samples,
        "candidates": candidates,
        "topk": topk,
        "temperature": temperature,
        "recent_frames": recent_frames,
        "teacher_cache_mode": "same_state_group",
        "teacher_payload": "single_rank512_int4_c64_alias",
        "trainable_consumer": "codec_native_query_only",
        "formal_anchors": 0,
    }
    _CONTRACT_CACHE_KEY = cache_key
    _CONTRACT_CACHE_VALUE = dict(report)
    return report


def active_for_layer(layer_idx: int, pass_name: Optional[str] = None) -> bool:
    if not enabled():
        return False
    if pass_name is None:
        pass_name = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
    if pass_name.strip().lower() not in {"teacher", "student"}:
        return False
    contract = validate_runtime_contract()
    return int(layer_idx) in set(contract["layers"])


def _step_key(layer_idx: int) -> tuple[int, int, int]:
    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    if chunk < 0 or step < 0:
        raise RuntimeError(
            "routing distillation requires non-negative rollout chunk and step"
        )
    return chunk, step, int(layer_idx)


def _tensor_identity(tensor: Any, name: str) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"single-payload routing descriptor lacks tensor {name}")
    return {
        "data_ptr": int(tensor.data_ptr()),
        "shape": tuple(int(value) for value in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _payload_identity(layer_cache: Mapping[str, Any], *, formal: bool) -> dict[str, Any]:
    if not layer_cache.get("_codec_native_causal_payload_verified") or not layer_cache.get(
        "_codec_native_causal_single_encode_verified"
    ):
        raise RuntimeError("routing distillation requires a verified single-encode cache")
    if bool(layer_cache.get("_codec_native_force_formal_teacher")) != bool(formal):
        raise RuntimeError("formal/student routing cache role is inconsistent")
    if bool(layer_cache.get("direct_codec_latent")) == bool(formal):
        raise RuntimeError("formal Teacher and direct Student cache roles are inverted")
    payload = layer_cache.get("k_vision")
    if not isinstance(payload, Mapping):
        raise RuntimeError("routing distillation requires a dictionary K payload")
    code_name = "latent_int8" if formal else "latent_int4"
    identity = {
        "codes": _tensor_identity(payload.get(code_name), code_name),
        "scales": _tensor_identity(payload.get("latent_scale"), "latent_scale"),
        "ids": _tensor_identity(payload.get("semantic_ids"), "semantic_ids"),
        "latent_shape": tuple(int(value) for value in payload.get("latent_shape", ())),
        "latent_numel": int(payload.get("latent_numel", -1)),
        "shape": tuple(int(value) for value in payload.get("shape", ())),
        "layer_idx": int(payload.get("layer_idx", -1)),
        "tensor_name": payload.get("tensor_name"),
    }
    if formal:
        identity["operator_path"] = payload.get("single_encode_operator_path")
        identity["operator_sha256"] = payload.get("single_encode_operator_sha256")
        centers = payload.get("semantic_centers")
        if not isinstance(centers, torch.Tensor):
            raise RuntimeError("formal routing Teacher is missing its sole C64 table")
        identity["centers"] = _tensor_identity(centers, "semantic_centers")
        if (
            not isinstance(identity["operator_path"], str)
            or not identity["operator_path"]
            or not isinstance(identity["operator_sha256"], str)
            or len(identity["operator_sha256"]) != 64
            or centers.ndim != 3
            or tuple(centers.shape[-2:]) != (64, 2048)
        ):
            raise RuntimeError(
                "formal routing Teacher lacks a bound rank512 operator/C64 descriptor"
            )
    elif payload.get("payload_mode") != "dictionary_lowkv":
        raise RuntimeError("routing Student must consume dictionary_lowkv payload")
    if identity["tensor_name"] != "k" or identity["layer_idx"] < 0:
        raise RuntimeError(f"malformed routing K payload identity: {identity}")
    return identity


def _same_student_payload(teacher: Mapping[str, Any], student: Mapping[str, Any]) -> bool:
    for name in ("codes", "scales", "ids"):
        if teacher[name] != student[name]:
            return False
    for name in ("latent_shape", "latent_numel", "shape", "layer_idx", "tensor_name"):
        if teacher[name] != student[name]:
            return False
    return True


def _old_history_tokens(
    layer_cache: Mapping[str, Any], layer_idx: int, history_length: int
) -> int:
    thw = layer_cache.get("vision_thw")
    if not isinstance(thw, (tuple, list)) or len(thw) != 3:
        raise RuntimeError("routing history requires vision_thw=(T,H,W)")
    temporal, height, width = (int(value) for value in thw)
    if min(temporal, height, width) <= 0 or temporal * height * width != int(
        history_length
    ):
        raise RuntimeError(
            f"routing history is not dense time-major THW: thw={thw} "
            f"length={history_length}"
        )
    recent = validate_runtime_contract()["recent_frames"]
    recent_tokens = min(int(recent), temporal) * height * width
    old_tokens = int(history_length) - recent_tokens
    if old_tokens <= 0:
        raise RuntimeError(
            f"routing L{int(layer_idx)} has no old history after recent={recent}"
        )
    return old_tokens


def _query_indices(length: int, layer_idx: int, device: torch.device) -> torch.Tensor:
    contract = validate_runtime_contract()
    count = min(int(contract["query_samples"]), int(length))
    chunk, step, _ = _step_key(layer_idx)
    seed = int(os.environ.get("HY_CODEC_NATIVE_ROUTING_SEED", "20260831"))
    generator = torch.Generator(device="cpu").manual_seed(
        seed + int(layer_idx) * 1009 + chunk * 97 + step * 17
    )
    return (
        torch.randperm(int(length), generator=generator)[:count]
        .sort()
        .values.to(device=device)
    )


def _gather_candidate_keys(
    key: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # key [B,H,K,D], indices [B,H,Q,C] -> [B,H,Q,C,D]
    if key.ndim != 4 or indices.ndim != 4 or key.shape[:2] != indices.shape[:2]:
        raise ValueError(
            f"candidate gather shape mismatch key={tuple(key.shape)} "
            f"indices={tuple(indices.shape)}"
        )
    expanded = key[:, :, None].expand(
        -1, -1, indices.shape[2], -1, -1
    )
    gather_indices = indices[..., None].expand(-1, -1, -1, -1, key.shape[-1])
    return torch.gather(expanded, 3, gather_indices)


@torch.no_grad()
def _teacher_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact full-history LSE and top-k with bounded transient logits."""

    scale = 1.0 / math.sqrt(128.0)
    running_lse = torch.full(
        query.shape[:-1], -torch.inf, device=query.device, dtype=torch.float32
    )
    top_values = torch.empty(
        (*query.shape[:-1], 0), device=query.device, dtype=torch.float32
    )
    top_indices = torch.empty(
        (*query.shape[:-1], 0), device=query.device, dtype=torch.long
    )
    for start in range(0, int(key.shape[2]), int(chunk_size)):
        end = min(start + int(chunk_size), int(key.shape[2]))
        logits = torch.matmul(
            query.float(), key[:, :, start:end].float().transpose(-1, -2)
        ) * scale
        running_lse = torch.logaddexp(running_lse, torch.logsumexp(logits, -1))
        local_k = min(int(topk), end - start)
        values, indices = logits.topk(local_k, dim=-1)
        indices = indices + start
        if top_values.shape[-1]:
            values = torch.cat((top_values, values), -1)
            indices = torch.cat((top_indices, indices), -1)
        keep = min(int(topk), values.shape[-1])
        top_values, positions = values.topk(keep, dim=-1)
        top_indices = torch.gather(indices, -1, positions)
    return running_lse, top_values, top_indices


def _dense_lse(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2))
    return torch.logsumexp(logits / math.sqrt(128.0), -1)


def _branch_metric(value: torch.Tensor, start: int, end: int) -> float:
    return float(value[start:end].detach().float().mean().cpu())


def record_teacher(
    layer_idx: int,
    query_bhqd: torch.Tensor,
    history_key_bhkd: torch.Tensor,
    layer_cache: Mapping[str, Any],
) -> bool:
    """Record a small D128 formal routing target; never retains full K/Q."""

    if not active_for_layer(layer_idx, "teacher"):
        return False
    if os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "").strip().lower() != "teacher":
        return False
    if (
        query_bhqd.ndim != 4
        or history_key_bhkd.ndim != 4
        or query_bhqd.shape[:2] != history_key_bhkd.shape[:2]
        or query_bhqd.shape[-1] != 128
        or history_key_bhkd.shape[-1] != 128
    ):
        raise RuntimeError(
            "formal routing Teacher expects matched [2B,H,Q/K,128] tensors: "
            f"q={tuple(query_bhqd.shape)} k={tuple(history_key_bhkd.shape)}"
        )
    if query_bhqd.shape[0] % 2:
        raise RuntimeError("formal routing Teacher requires stacked Main/PRoPE batch")
    payload_identity = _payload_identity(layer_cache, formal=True)
    if payload_identity["layer_idx"] != int(layer_idx):
        raise RuntimeError("formal routing Teacher layer/payload index mismatch")
    old_tokens = _old_history_tokens(
        layer_cache, int(layer_idx), int(history_key_bhkd.shape[2])
    )
    history_key = history_key_bhkd[:, :, :old_tokens]
    query_indices = _query_indices(
        int(query_bhqd.shape[2]), int(layer_idx), query_bhqd.device
    )
    query = query_bhqd.index_select(2, query_indices)
    contract = validate_runtime_contract()
    topk = min(int(contract["topk"]), old_tokens)
    candidate_count = min(int(contract["candidates"]), old_tokens)
    topk = min(topk, candidate_count)
    chunk_size = max(
        1, int(os.environ.get("HY_CODEC_NATIVE_ROUTING_LOGIT_CHUNK", "8192"))
    )
    history_lse, _top_values, top_indices = _teacher_summary(
        query, history_key, topk=topk, chunk_size=chunk_size
    )

    negative_count = candidate_count - topk
    if negative_count:
        chunk, step, _ = _step_key(layer_idx)
        seed = int(os.environ.get("HY_CODEC_NATIVE_ROUTING_SEED", "20260831"))
        generator = torch.Generator(device="cpu").manual_seed(
            seed + int(layer_idx) * 2029 + chunk * 193 + step * 31 + 7
        )
        negative = torch.randperm(old_tokens, generator=generator)[:negative_count]
        negative = negative.to(device=query.device, dtype=torch.long)
        negative = negative.view(1, 1, 1, -1).expand(
            query.shape[0], query.shape[1], query.shape[2], -1
        )
        candidate_indices = torch.cat((top_indices, negative), -1)
    else:
        candidate_indices = top_indices
    candidate_key = _gather_candidate_keys(history_key, candidate_indices)
    candidate_logits = torch.einsum(
        "bhqd,bhqcd->bhqc", query.float(), candidate_key.float()
    ) / math.sqrt(128.0)

    key = _step_key(layer_idx)
    if key in _TARGETS:
        raise RuntimeError(f"duplicate formal routing Teacher target for {key}")
    _TARGETS[key] = {
        "payload_identity": payload_identity,
        "query_indices": query_indices.detach().cpu(),
        "candidate_indices": candidate_indices.detach().to("cpu", torch.int32),
        "candidate_logits": candidate_logits.detach().to("cpu", torch.float32),
        "history_lse": history_lse.detach().to("cpu", torch.float32),
        "teacher_topk": top_indices.detach().to("cpu", torch.int32),
        "old_tokens": int(old_tokens),
        "query_tokens": int(query_bhqd.shape[2]),
        "branch_batch": int(query_bhqd.shape[0]),
        "heads": int(query_bhqd.shape[1]),
        "teacher_dim": 128,
    }
    return True


def record_student(
    layer_idx: int,
    adapted_query_bhqd: torch.Tensor,
    old_key_bhkd: torch.Tensor,
    old_value_bhkd: torch.Tensor,
    native_lse_bhq: torch.Tensor,
    layer_cache: Mapping[str, Any],
    *,
    flash_lse_fn: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor]]] = None,
) -> bool:
    """Append one differentiable query-only routing loss for a low-D layer."""

    if not active_for_layer(layer_idx, "student"):
        return False
    if os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "").strip().lower() != "student":
        return False
    key = _step_key(layer_idx)
    target = _TARGETS.get(key)
    if target is None:
        raise RuntimeError(f"missing formal routing Teacher target for {key}")
    student_identity = _payload_identity(layer_cache, formal=False)
    if not _same_student_payload(target["payload_identity"], student_identity):
        raise RuntimeError(
            f"routing L{int(layer_idx)} Teacher/Student do not alias one K payload"
        )
    if student_identity["layer_idx"] != int(layer_idx):
        raise RuntimeError("routing Student layer/payload index mismatch")
    if (
        adapted_query_bhqd.ndim != 4
        or old_key_bhkd.ndim != 4
        or old_value_bhkd.shape != old_key_bhkd.shape
        or native_lse_bhq.shape != adapted_query_bhqd.shape[:-1]
        or adapted_query_bhqd.shape[:2] != old_key_bhkd.shape[:2]
        or adapted_query_bhqd.shape[-1] != old_key_bhkd.shape[-1]
    ):
        raise RuntimeError(
            "low-D routing Student shape mismatch: "
            f"q={tuple(adapted_query_bhqd.shape)} k={tuple(old_key_bhkd.shape)} "
            f"v={tuple(old_value_bhkd.shape)} native_lse={tuple(native_lse_bhq.shape)}"
        )
    if int(old_key_bhkd.shape[2]) != int(target["old_tokens"]):
        raise RuntimeError(
            f"routing old-history row mismatch Student={old_key_bhkd.shape[2]} "
            f"Teacher={target['old_tokens']}"
        )
    if (
        int(adapted_query_bhqd.shape[2]) != int(target["query_tokens"])
        or int(adapted_query_bhqd.shape[0]) != int(target["branch_batch"])
        or int(adapted_query_bhqd.shape[1]) != int(target["heads"])
    ):
        raise RuntimeError("routing Teacher/Student query layout changed")

    device = adapted_query_bhqd.device
    query_indices = target["query_indices"].to(device=device, dtype=torch.long)
    candidate_indices = target["candidate_indices"].to(
        device=device, dtype=torch.long
    )
    query = adapted_query_bhqd.index_select(2, query_indices)
    candidate_key = _gather_candidate_keys(old_key_bhkd, candidate_indices)
    student_logits = torch.einsum(
        "bhqd,bhqcd->bhqc", query.float(), candidate_key.float()
    ) / math.sqrt(128.0)
    teacher_logits = target["candidate_logits"].to(
        device=device, dtype=torch.float32
    )

    teacher_centered = teacher_logits - teacher_logits.mean(-1, keepdim=True)
    student_centered = student_logits - student_logits.mean(-1, keepdim=True)
    teacher_variance = (
        teacher_centered.square().mean(-1, keepdim=True).detach().clamp_min(1e-4)
    )
    centered_per_row = (
        (student_centered - teacher_centered).square().mean(-1)
        / teacher_variance.squeeze(-1)
    )
    centered_loss = centered_per_row.mean()

    temperature = float(validate_runtime_contract()["temperature"])
    teacher_probability = torch.softmax(teacher_logits / temperature, -1)
    teacher_log_probability = torch.log_softmax(teacher_logits / temperature, -1)
    student_log_probability = torch.log_softmax(student_logits / temperature, -1)
    kl_per_row = (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(-1) * (temperature * temperature)
    kl_loss = kl_per_row.mean()

    if flash_lse_fn is None:
        student_history_lse = _dense_lse(query, old_key_bhkd)
    else:
        _unused_output, student_history_lse = flash_lse_fn(
            query,
            old_key_bhkd,
            old_value_bhkd,
            scale=1.0 / math.sqrt(128.0),
        )
    teacher_history_lse = target["history_lse"].to(
        device=device, dtype=torch.float32
    )
    native_lse = native_lse_bhq.index_select(2, query_indices).detach().float()
    teacher_log_odds = teacher_history_lse - native_lse
    student_log_odds = student_history_lse.float() - native_lse
    log_odds_per_row = F.smooth_l1_loss(
        student_log_odds,
        teacher_log_odds,
        reduction="none",
        beta=float(os.environ.get("HY_CODEC_NATIVE_ROUTING_LOG_ODDS_BETA", "0.5")),
    )
    log_odds_loss = log_odds_per_row.mean()

    centered_weight = float(
        os.environ.get("HY_CODEC_NATIVE_ROUTING_CENTERED_WEIGHT", "1.0")
    )
    kl_weight = float(os.environ.get("HY_CODEC_NATIVE_ROUTING_KL_WEIGHT", "0.5"))
    mass_weight = float(
        os.environ.get("HY_CODEC_NATIVE_ROUTING_MASS_WEIGHT", "0.25")
    )
    total = (
        centered_weight * centered_loss
        + kl_weight * kl_loss
        + mass_weight * log_odds_loss
    )
    if not adapted_query_bhqd.requires_grad or not total.requires_grad:
        raise RuntimeError(
            "routing Student query graph is detached before loss collection: "
            f"layer={int(layer_idx)} pass="
            f"{os.environ.get('HY_FUNCTIONAL_BLOCK_FT_PASS', '')!r} "
            f"grad_enabled={torch.is_grad_enabled()} "
            f"query_requires_grad={adapted_query_bhqd.requires_grad} "
            f"loss_requires_grad={total.requires_grad}"
        )
    _LOSSES.append(total)

    topk = min(
        int(validate_runtime_contract()["topk"]), student_logits.shape[-1]
    )
    student_positions = student_logits.detach().topk(topk, dim=-1).indices
    student_global = torch.gather(candidate_indices, -1, student_positions)
    teacher_topk = target["teacher_topk"].to(device=device, dtype=torch.long)
    topk_recall = (
        (teacher_topk.unsqueeze(-1) == student_global.unsqueeze(-2))
        .any(-1)
        .float()
        .mean(-1)
    )
    history_mass_mae = (
        torch.sigmoid(student_log_odds.detach())
        - torch.sigmoid(teacher_log_odds.detach())
    ).abs()
    log_odds_abs = (student_log_odds.detach() - teacher_log_odds).abs()
    branch_batch = int(adapted_query_bhqd.shape[0])
    split = branch_batch // 2
    metrics: dict[str, Any] = {
        "layer": int(layer_idx),
        "low_dim": int(adapted_query_bhqd.shape[-1]),
        "query_samples": int(query.shape[2]),
        "candidate_count": int(candidate_indices.shape[-1]),
        "centered_nmse": float(centered_loss.detach().cpu()),
        "conditional_kl": float(kl_loss.detach().cpu()),
        "history_log_odds_smooth_l1": float(log_odds_loss.detach().cpu()),
        "history_log_odds_mae": float(log_odds_abs.float().mean().cpu()),
        "history_mass_mae": float(history_mass_mae.float().mean().cpu()),
        "candidate_topk_recall": float(topk_recall.float().mean().cpu()),
    }
    if branch_batch % 2 == 0:
        for name, value in (
            ("centered_nmse", centered_per_row),
            ("conditional_kl", kl_per_row),
            ("history_log_odds", log_odds_abs),
            ("history_mass_mae", history_mass_mae),
            ("candidate_topk_recall", topk_recall),
        ):
            metrics[f"main_{name}"] = _branch_metric(value, 0, split)
            metrics[f"prope_{name}"] = _branch_metric(value, split, branch_batch)
    _METRICS.append(metrics)
    return True


def pop_loss(*, preserve_targets: bool = False):
    if not _LOSSES:
        if not preserve_targets:
            _TARGETS.clear()
        return None, []
    # The online pipeline is globally ``@torch.no_grad`` and opens autograd
    # only around the Student transformer call.  The per-layer losses retain
    # those graphs after that call returns, but aggregating them here under the
    # restored outer no-grad mode would create a detached ``stack(...).mean()``
    # result.  Own this tiny grad island at the collection boundary so the
    # query-only adapter graph survives exactly until the functional update.
    with torch.enable_grad():
        loss = torch.stack(_LOSSES).mean()
    if not loss.requires_grad:
        raise RuntimeError(
            "routing loss aggregation detached every Student query graph"
        )
    metrics = list(_METRICS)
    _LOSSES.clear()
    _METRICS.clear()
    if not preserve_targets:
        _TARGETS.clear()
    return loss, metrics


def reset() -> None:
    _TARGETS.clear()
    _LOSSES.clear()
    _METRICS.clear()


def reset_for_tests() -> None:
    global _CONTRACT_CACHE_KEY, _CONTRACT_CACHE_VALUE
    reset()
    _CONTRACT_CACHE_KEY = None
    _CONTRACT_CACHE_VALUE = None


def target_count() -> int:
    return len(_TARGETS)
