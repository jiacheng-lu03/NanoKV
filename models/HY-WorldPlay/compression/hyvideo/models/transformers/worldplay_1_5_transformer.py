# Modified for NanoKV: KV-cache codec integration or evaluation configuration.
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, Tuple, Optional, Union, Dict
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from einops import rearrange, repeat
from loguru import logger

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .modules.activation_layers import get_activation_layer
from .modules.norm_layers import get_norm_layer
from .modules.embed_layers import (
    TimestepEmbedder,
    PatchEmbed,
    TextProjection,
    VisionProjection,
)
from .modules.attention import (
    parallel_attention,
    pop_k_attention_audit_metrics,
    sequence_parallel_attention_txt,
    sequence_parallel_attention_vision,
)
from .modules import position_profile
from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .modules.mlp_layers import MLP, MLPEmbedder, FinalLayer, LinearWarpforSingle
from .modules.modulate_layers import ModulateDiT, modulate, apply_gate
from .modules.token_refiner import SingleTokenRefiner

from hyvideo.utils.communications import all_gather
from hyvideo.utils.infer_utils import torch_compile_wrapper
from hyvideo.models.text_encoders.byT5 import ByT5Mapper
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.prope.camera_rope import prope_qkv
from hyvideo.models.transformers.modules.kv_codec_training import (
    compress_layer_cache,
    dump_attention_sample,
    dump_qk_subset_pre_position,
)
from hyvideo.models.transformers.modules.kv_codecs import apply_codec_to_layer_cache
from hyvideo.models.transformers.modules.qvg_adapter import (
    compress_layer_cache as compress_qvg_layer_cache,
)
from hyvideo.models.transformers.modules.jensen_kv_adapter import (
    compress_layer_cache as compress_jensen_layer_cache,
)
from hyvideo.models.transformers.modules.turboquant_adapter import (
    compress_layer_cache as compress_turboquant_layer_cache,
)
from hyvideo.models.transformers.modules.kvtc_adapter import (
    compress_ready_group as compress_kvtc_ready_group,
    dump_calibration_sample as dump_kvtc_calibration_sample,
    enabled as kvtc_enabled,
)


def _kv_resident_numel_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel(), obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        if obj.get("codec") == "kvtc_layer_v1":
            shared = obj.get("shared", {})
            if int(obj.get("layer_idx", -1)) != int(shared.get("group_start", -2)):
                return 0, 0
        total_numel = 0
        total_bytes = 0
        for value in obj.values():
            sub_numel, sub_bytes = _kv_resident_numel_bytes(value)
            total_numel += sub_numel
            total_bytes += sub_bytes
        return total_numel, total_bytes
    if isinstance(obj, (list, tuple)):
        total_numel = 0
        total_bytes = 0
        for value in obj:
            sub_numel, sub_bytes = _kv_resident_numel_bytes(value)
            total_numel += sub_numel
            total_bytes += sub_bytes
        return total_numel, total_bytes
    return 0, 0


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _parse_int_set(value: str):
    value = value.strip().lower()
    if value == "all":
        return "all"
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _layer_selected(layer_idx: int, layers) -> bool:
    return layers == "all" or int(layer_idx) in layers


def _p1_causal_audit_enabled() -> bool:
    return _env_enabled("HY_K_P1_CAUSAL_AUDIT")


def _audit_tensor_fingerprint(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if not torch.is_tensor(tensor):
        return None
    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _audit_layer_cache_fingerprint(kv_cache, layer_idx: int) -> Dict[str, Any]:
    if kv_cache is None or int(layer_idx) >= len(kv_cache):
        return {"missing": True}
    layer_cache = kv_cache[int(layer_idx)]
    return {
        name: _audit_tensor_fingerprint(layer_cache.get(name))
        for name in ("k_vision", "v_vision")
    }


def _append_p1_causal_audit(record: Dict[str, Any]) -> None:
    path_raw = os.environ.get("HY_K_P1_CAUSAL_AUDIT_JSONL", "").strip()
    if not path_raw:
        return
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _fixed_indices(length: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    count = min(int(count), int(length))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(int(length), generator=gen)[:count].sort().values
    return idx.to(device=device)


_FUNCTIONAL_BLOCK_TARGETS: Dict[Tuple[int, int, int], torch.Tensor] = {}
_FUNCTIONAL_WINDOW_ENTRIES: Dict[Tuple[int, int], torch.Tensor] = {}
_FUNCTIONAL_BLOCK_LOCAL_LOSSES = []
_FUNCTIONAL_BLOCK_PROP1_LOSSES = []
_FUNCTIONAL_BLOCK_PROP1_LAYERS = []
_FUNCTIONAL_BLOCK_ENDPOINT_LOSSES = []
_FUNCTIONAL_BLOCK_METRICS = []


def reset_functional_block_losses() -> None:
    _FUNCTIONAL_BLOCK_TARGETS.clear()
    _FUNCTIONAL_WINDOW_ENTRIES.clear()
    _FUNCTIONAL_BLOCK_LOCAL_LOSSES.clear()
    _FUNCTIONAL_BLOCK_PROP1_LOSSES.clear()
    _FUNCTIONAL_BLOCK_PROP1_LAYERS.clear()
    _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES.clear()
    _FUNCTIONAL_BLOCK_METRICS.clear()


def pop_functional_block_loss(preserve_targets: bool = False):
    if (
        not _FUNCTIONAL_BLOCK_LOCAL_LOSSES
        and not _FUNCTIONAL_BLOCK_PROP1_LOSSES
        and not _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES
    ):
        return None, []
    all_losses = (
        _FUNCTIONAL_BLOCK_LOCAL_LOSSES
        + _FUNCTIONAL_BLOCK_PROP1_LOSSES
        + _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES
    )
    needs_grad = any(loss.requires_grad for loss in all_losses)
    with torch.set_grad_enabled(needs_grad):
        local_weight = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LOCAL_WEIGHT", "1.0"))
        prop1_weight = float(
            os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PROP1_WEIGHT", "0.0")
        )
        endpoint_beta = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ENDPOINT_WEIGHT", "0.0"))
        numerator = None
        denominator = 0.0
        if local_weight > 0.0 and _FUNCTIONAL_BLOCK_LOCAL_LOSSES:
            local_loss = torch.stack(_FUNCTIONAL_BLOCK_LOCAL_LOSSES).mean()
            numerator = local_weight * local_loss
            denominator += local_weight
        if prop1_weight > 0.0 and _FUNCTIONAL_BLOCK_PROP1_LOSSES:
            prop1_loss = torch.stack(_FUNCTIONAL_BLOCK_PROP1_LOSSES).mean()
            prop1_term = prop1_weight * prop1_loss
            numerator = prop1_term if numerator is None else numerator + prop1_term
            denominator += prop1_weight
        if endpoint_beta > 0.0 and _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES:
            endpoint_loss = torch.stack(_FUNCTIONAL_BLOCK_ENDPOINT_LOSSES).mean()
            endpoint_term = endpoint_beta * endpoint_loss
            numerator = endpoint_term if numerator is None else numerator + endpoint_term
            denominator += endpoint_beta
        if numerator is None:
            loss = None
        elif _env_enabled("HY_FUNCTIONAL_BLOCK_FT_NO_WEIGHT_NORMALIZE"):
            loss = numerator
        else:
            loss = numerator / max(denominator, 1e-12)
    metrics = list(_FUNCTIONAL_BLOCK_METRICS)
    _FUNCTIONAL_BLOCK_LOCAL_LOSSES.clear()
    _FUNCTIONAL_BLOCK_PROP1_LOSSES.clear()
    _FUNCTIONAL_BLOCK_PROP1_LAYERS.clear()
    _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES.clear()
    _FUNCTIONAL_BLOCK_METRICS.clear()
    if not preserve_targets:
        _FUNCTIONAL_BLOCK_TARGETS.clear()
        _FUNCTIONAL_WINDOW_ENTRIES.clear()
    return loss, metrics


def get_functional_block_loss_components() -> Dict[str, Any]:
    """Return live functional loss tensors without clearing their graphs."""
    with torch.enable_grad():
        local_by_layer: Dict[int, list] = {}
        for loss, metric in zip(
            _FUNCTIONAL_BLOCK_LOCAL_LOSSES, _FUNCTIONAL_BLOCK_METRICS
        ):
            local_by_layer.setdefault(int(metric["layer"]), []).append(loss)
        local_components = {
            layer: torch.stack(losses).mean()
            for layer, losses in local_by_layer.items()
        }
        endpoint = (
            torch.stack(_FUNCTIONAL_BLOCK_ENDPOINT_LOSSES).mean()
            if _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES
            else None
        )
        local = (
            torch.stack(_FUNCTIONAL_BLOCK_LOCAL_LOSSES).mean()
            if _FUNCTIONAL_BLOCK_LOCAL_LOSSES
            else None
        )
        prop1 = (
            torch.stack(_FUNCTIONAL_BLOCK_PROP1_LOSSES).mean()
            if _FUNCTIONAL_BLOCK_PROP1_LOSSES
            else None
        )
        prop1_by_layer: Dict[int, list] = {}
        for layer, loss in zip(
            _FUNCTIONAL_BLOCK_PROP1_LAYERS, _FUNCTIONAL_BLOCK_PROP1_LOSSES
        ):
            prop1_by_layer.setdefault(int(layer), []).append(loss)
        prop1_components = {
            layer: torch.stack(losses).mean()
            for layer, losses in prop1_by_layer.items()
        }
    return {
        "endpoint": endpoint,
        "local": local,
        "local_by_layer": local_components,
        "prop1": prop1,
        "prop1_by_layer": prop1_components,
    }


def _functional_prop1_enabled() -> bool:
    return _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PROP1_ENABLE")


def _functional_prop_horizon() -> int:
    horizon = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PROP_HORIZON", "1"))
    if horizon < 1:
        raise ValueError(f"Propagation horizon must be positive, got {horizon}")
    return horizon


def _record_functional_prop1(
    source_layer: int,
    source_entry: torch.Tensor,
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    *,
    observation_input: Optional[torch.Tensor] = None,
    observation_trainable_parameter_count: int = -1,
    cache_audit: Optional[Dict[str, Any]] = None,
) -> None:
    """Record source-only error after a frozen downstream block horizon.

    Both branches use the exact same compressed historical K downstream.
    The denominator is the sampled Teacher update from the common
    source entry, matching A1's residual-update normalization without letting
    the large residual-stream baseline hide the propagated error.
    """
    horizon = _functional_prop_horizon()
    token_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TOKEN_SAMPLES", "256"))
    channel_count = int(
        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_CHANNEL_SAMPLES", "256")
    )
    seed = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_SEED", "2027"))
    token_idx = _fixed_indices(
        student_out.shape[1],
        token_count,
        seed + int(source_layer) * 1009,
        student_out.device,
    )
    channel_idx = _fixed_indices(
        student_out.shape[2],
        channel_count,
        seed + int(source_layer) * 2003 + 17,
        student_out.device,
    )
    student = student_out[:, token_idx][:, :, channel_idx].float()
    teacher = teacher_out[:, token_idx][:, :, channel_idx].to(student).float()
    entry = source_entry[:, token_idx][:, :, channel_idx].to(student).float()
    diff_sq = (student - teacher).pow(2).sum()
    teacher_update_sq = (teacher - entry).pow(2).sum().clamp_min(1e-8)
    loss = diff_sq / teacher_update_sq
    if (
        _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT")
        and not _env_enabled("HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY")
    ):
        if not student_out.requires_grad or student_out.grad_fn is None:
            raise RuntimeError(
                "Propagation Student output lost graph after "
                f"L{int(source_layer) + horizon}"
            )
        if not loss.requires_grad or loss.grad_fn is None:
            raise RuntimeError(f"P1 loss has no graph for source L{source_layer}")
    _FUNCTIONAL_BLOCK_PROP1_LOSSES.append(loss)
    _FUNCTIONAL_BLOCK_PROP1_LAYERS.append(int(source_layer))
    rel_l2 = float(torch.sqrt(loss.clamp_min(0.0)).detach().cpu())
    if _p1_causal_audit_enabled():
        observation_input_grad_norm = 0.0
        observation_input_grad_nonzero = False
        if torch.is_tensor(observation_input) and observation_input.requires_grad:
            observation_input_grad = torch.autograd.grad(
                loss,
                observation_input,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if observation_input_grad is not None:
                observation_input_grad_norm = float(
                    observation_input_grad.detach().float().norm().cpu()
                )
                observation_input_grad_nonzero = observation_input_grad_norm > 0.0
        audit_record = {
            "chunk": int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1")),
            "step": int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1")),
            "source_layer": int(source_layer),
            "observation_layer": int(source_layer) + horizon,
            "propagation_horizon": horizon,
            "trainable_codec_layer": int(source_layer),
            "branch_order": os.environ.get(
                "HY_K_P1_BRANCH_ORDER", "teacher_first"
            ).strip().lower(),
            "source_student_clean": _env_enabled(
                "HY_K_P1_SOURCE_STUDENT_CLEAN"
            ),
            "observation_teacher_clean": _env_enabled(
                "HY_K_P1_OBSERVATION_TEACHER_CLEAN"
            ),
            "share_current_qkv": _env_enabled(
                "HY_K_P1_SHARE_CURRENT_QKV"
            ),
            "prop1_loss": float(loss.detach().cpu()),
            "prop1_rel_l2": rel_l2,
            "observation_input_requires_grad": bool(
                torch.is_tensor(observation_input)
                and observation_input.requires_grad
            ),
            "observation_input_grad_norm": observation_input_grad_norm,
            "observation_input_grad_nonzero": observation_input_grad_nonzero,
            "observation_trainable_parameter_count": int(
                observation_trainable_parameter_count
            ),
            "student_output": _audit_tensor_fingerprint(student_out),
            "teacher_output": _audit_tensor_fingerprint(teacher_out),
            "cache": cache_audit or {},
        }
        _append_p1_causal_audit(audit_record)
        print("[KP1CausalAudit] " + json.dumps(audit_record, sort_keys=True), flush=True)
    for metric in reversed(_FUNCTIONAL_BLOCK_METRICS):
        if int(metric.get("layer", -1)) == int(source_layer):
            metric["prop1_rel_l2"] = rel_l2
            metric["prop1_loss"] = float(loss.detach().cpu())
            metric["prop1_target_layer"] = int(source_layer) + horizon
            break


def _functional_block_ft_enabled() -> bool:
    return _env_enabled("HY_FUNCTIONAL_BLOCK_FT_ENABLE")


def _functional_window_key() -> Tuple[int, int]:
    return (
        int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1")),
        int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1")),
    )


def _functional_local_teacher_cache(kv_cache, layer_idx: int):
    """Replace only the current layer's historical K or V with its clean tensor."""
    if kv_cache is None or int(layer_idx) >= len(kv_cache):
        return None
    target = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TARGET", "k").strip().lower()
    copied_layer = dict(kv_cache[int(layer_idx)])
    pairs = (
        (("k_vision", "_k_vision_clean_for_ft"), ("v_vision", "_v_vision_clean_for_ft"))
        if target == "kv"
        else (("v_vision", "_v_vision_clean_for_ft"),)
        if target == "v"
        else (("k_vision", "_k_vision_clean_for_ft"),)
    )
    for cache_name, clean_name in pairs:
        clean = copied_layer.get(clean_name)
        if not torch.is_tensor(clean):
            if _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT"):
                raise RuntimeError(
                    f"Functional local teacher missing {clean_name} for layer {int(layer_idx)}"
                )
            return None
        copied_layer[cache_name] = clean
    copied_cache = list(kv_cache)
    copied_cache[int(layer_idx)] = copied_layer
    return copied_cache


def _functional_source_clean_counterfactual_cache(kv_cache, layer_idx: int):
    """Use clean historical K numerically while retaining a zero-valued codec graph.

    The causal clean counterfactual should make the Student and Teacher source
    inputs identical.  A direct clean-tensor replacement would bypass the
    decoder entirely and trip the strict graph assertion, so attach
    ``corrupt - corrupt.detach()``: it is exactly zero in the forward pass but
    preserves the diagnostic autograd path.
    """
    if kv_cache is None or int(layer_idx) >= len(kv_cache):
        return None
    copied_layer = dict(kv_cache[int(layer_idx)])
    clean = copied_layer.get("_k_vision_clean_for_ft")
    corrupt = copied_layer.get("k_vision")
    if not (torch.is_tensor(clean) and torch.is_tensor(corrupt)):
        raise RuntimeError(
            f"P1 source-clean counterfactual missing K tensors at L{layer_idx}"
        )
    copied_layer["k_vision"] = clean.to(corrupt) + (
        corrupt - corrupt.detach()
    )
    copied_cache = list(kv_cache)
    copied_cache[int(layer_idx)] = copied_layer
    return copied_cache


def _k_placement_oracle_active(layer_idx: int) -> bool:
    mode = os.environ.get("HY_K_PLACEMENT_ORACLE_MODE", "").strip().lower()
    if mode not in {"base", "attention", "block"}:
        return False
    if os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "denoise") != "denoise":
        return False
    layers = _parse_int_set(
        os.environ.get("HY_K_PLACEMENT_ORACLE_LAYERS", "0,1,2,3")
    )
    chunks = _parse_int_set(os.environ.get("HY_K_PLACEMENT_ORACLE_CHUNKS", ""))
    steps = _parse_int_set(
        os.environ.get("HY_K_PLACEMENT_ORACLE_STEPS", "0,1,2,3")
    )
    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    return (
        _layer_selected(layer_idx, layers)
        and (chunks == "all" or chunk in chunks)
        and (steps == "all" or step in steps)
    )


def _k_placement_clean_cache(kv_cache, layer_idx: int):
    """Return a shallow same-state cache with only this layer's K made clean."""
    if kv_cache is None or int(layer_idx) >= len(kv_cache):
        return None
    copied_layer = dict(kv_cache[int(layer_idx)])
    clean = copied_layer.get("_k_vision_clean_for_ft")
    if not torch.is_tensor(clean):
        raise RuntimeError(
            f"K placement oracle missing clean pre-position K at layer {layer_idx}"
        )
    copied_layer["k_vision"] = clean
    copied_cache = list(kv_cache)
    copied_cache[int(layer_idx)] = copied_layer
    return copied_cache


def _k_placement_rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred_f = pred.detach().float()
    ref_f = ref.detach().float()
    return float(((pred_f - ref_f).norm() / ref_f.norm().clamp_min(1e-8)).cpu())


def _record_k_placement_oracle(
    layer_idx: int,
    mode: str,
    clean_attention_update: torch.Tensor,
    student_attention_update: torch.Tensor,
    clean_block_output: torch.Tensor,
    student_block_output: torch.Tensor,
    injected_block_output: torch.Tensor,
    attention_audit: Optional[Dict[str, Any]] = None,
    clean_readout_main: Optional[torch.Tensor] = None,
    student_readout_main: Optional[torch.Tensor] = None,
    clean_readout_prope: Optional[torch.Tensor] = None,
    student_readout_prope: Optional[torch.Tensor] = None,
) -> None:
    path_raw = os.environ.get("HY_K_PLACEMENT_ORACLE_JSONL", "").strip()
    if not path_raw:
        return
    clean_attn_f = clean_attention_update.detach().float()
    student_attn_f = student_attention_update.detach().float()
    attn_delta = clean_attn_f - student_attn_f
    clean_block_f = clean_block_output.detach().float()
    student_block_f = student_block_output.detach().float()
    injected_block_f = injected_block_output.detach().float()
    record = {
        "mode": mode,
        "chunk": int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1")),
        "step": int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1")),
        "layer": int(layer_idx),
        "attention_update_rel_l2": _k_placement_rel_l2(
            student_attn_f, clean_attn_f
        ),
        "attention_delta_rms": float(attn_delta.pow(2).mean().sqrt().cpu()),
        "student_block_rel_l2": _k_placement_rel_l2(
            student_block_f, clean_block_f
        ),
        "injected_block_rel_l2": _k_placement_rel_l2(
            injected_block_f, clean_block_f
        ),
        "clean_block_rms": float(clean_block_f.pow(2).mean().sqrt().cpu()),
    }
    if attention_audit is not None:
        record["runtime_attention"] = attention_audit
    if torch.is_tensor(clean_readout_main) and torch.is_tensor(student_readout_main):
        record["readout_main_rel_l2"] = _k_placement_rel_l2(
            student_readout_main, clean_readout_main
        )
    if torch.is_tensor(clean_readout_prope) and torch.is_tensor(student_readout_prope):
        record["readout_prope_rel_l2"] = _k_placement_rel_l2(
            student_readout_prope, clean_readout_prope
        )
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _record_functional_block(
    layer_idx: int,
    img_in: torch.Tensor,
    img_out: torch.Tensor,
    local_teacher_out: Optional[torch.Tensor] = None,
) -> None:
    if not _functional_block_ft_enabled():
        return
    if os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "denoise") != "denoise":
        return
    layers = _parse_int_set(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "13,40,53"))
    if not _layer_selected(layer_idx, layers):
        return
    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    if chunk < 0 or step < 0:
        return
    mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
    key = (chunk, step, int(layer_idx))
    token_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TOKEN_SAMPLES", "256"))
    channel_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_CHANNEL_SAMPLES", "256"))
    seed = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_SEED", "2027"))
    token_idx = _fixed_indices(
        img_out.shape[1], token_count, seed + layer_idx * 1009, img_out.device
    )
    channel_idx = _fixed_indices(
        img_out.shape[2], channel_count, seed + layer_idx * 2003 + 17, img_out.device
    )
    sketch = img_out[:, token_idx][:, :, channel_idx]
    if mode == "teacher":
        _FUNCTIONAL_BLOCK_TARGETS[key] = sketch.detach()
        return
    if mode != "student":
        return
    target = _FUNCTIONAL_BLOCK_TARGETS.get(key)
    if target is None:
        return
    target_mode = os.environ.get(
        "HY_FUNCTIONAL_BLOCK_FT_LOCAL_TARGET", "hidden"
    ).strip().lower()
    if target_mode in {"residual", "residual_update", "block_update", "delta"}:
        if local_teacher_out is None:
            raise RuntimeError(
                f"Residual-update functional target missing local teacher output at layer {layer_idx}"
            )
        sampled_in = img_in[:, token_idx][:, :, channel_idx]
        sampled_teacher = local_teacher_out[:, token_idx][:, :, channel_idx]
        pred = (sketch - sampled_in).float()
        ref = (sampled_teacher - sampled_in.detach()).float()
        diff_sq = (pred - ref).pow(2).sum()
        ref_sq = ref.pow(2).sum().clamp_min(1e-8)
        local_loss = diff_sq / ref_sq
        rel_l2 = torch.sqrt(local_loss.clamp_min(0.0))
    else:
        pred = sketch.float()
        ref = target.to(device=sketch.device, dtype=sketch.dtype).float()
        rel_l2 = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
        local_loss = rel_l2 + 0.1 * (
            1.0
            - torch.nn.functional.cosine_similarity(
                pred.reshape(-1, pred.shape[-1]),
                ref.reshape(-1, ref.shape[-1]),
                dim=-1,
            ).mean()
        )
    cosine = torch.nn.functional.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]),
        ref.reshape(-1, ref.shape[-1]),
        dim=-1,
    ).mean()
    _FUNCTIONAL_BLOCK_LOCAL_LOSSES.append(local_loss)
    endpoint_layer = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ENDPOINT_LAYER", "-1"))
    is_endpoint = int(layer_idx) == endpoint_layer
    endpoint_rel_l2 = 0.0
    endpoint_loss_value = 0.0
    if is_endpoint:
        window_entry = _FUNCTIONAL_WINDOW_ENTRIES.get(_functional_window_key())
        if window_entry is None:
            if _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT"):
                raise RuntimeError("Missing shared compressed-prefix window entry")
        else:
            sampled_entry = window_entry[:, token_idx][:, :, channel_idx]
            endpoint_ref = target.to(device=sketch.device, dtype=sketch.dtype)
            endpoint_diff_sq = (sketch.float() - endpoint_ref.float()).pow(2).sum()
            endpoint_ref_sq = (
                endpoint_ref.float() - sampled_entry.float()
            ).pow(2).sum().clamp_min(1e-8)
            endpoint_loss = endpoint_diff_sq / endpoint_ref_sq
            if _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT"):
                if not sketch.requires_grad or sketch.grad_fn is None:
                    raise RuntimeError("Student endpoint hidden has no autograd graph")
                if not endpoint_loss.requires_grad or endpoint_loss.grad_fn is None:
                    raise RuntimeError("Functional endpoint loss has no autograd graph")
            _FUNCTIONAL_BLOCK_ENDPOINT_LOSSES.append(endpoint_loss)
            endpoint_rel_l2 = float(torch.sqrt(endpoint_loss.clamp_min(0.0)).detach().cpu())
            endpoint_loss_value = float(endpoint_loss.detach().cpu())
            student_hidden_rms = float(
                sketch.detach().float().pow(2).mean().sqrt().cpu()
            )
            teacher_hidden_rms = float(
                endpoint_ref.detach().float().pow(2).mean().sqrt().cpu()
            )
            entry_hidden_rms = float(
                sampled_entry.detach().float().pow(2).mean().sqrt().cpu()
            )
    if not is_endpoint or window_entry is None:
        student_hidden_rms = 0.0
        teacher_hidden_rms = 0.0
        entry_hidden_rms = 0.0
    _FUNCTIONAL_BLOCK_METRICS.append(
        {
            "chunk": chunk,
            "step": step,
            "layer": int(layer_idx),
            "block_rel_l2": float(rel_l2.detach().cpu()),
            "block_cosine": float(cosine.detach().cpu()),
            "is_endpoint": bool(is_endpoint),
            "endpoint_rel_l2": endpoint_rel_l2,
            "endpoint_loss": endpoint_loss_value,
            "endpoint_student_hidden_rms": student_hidden_rms,
            "endpoint_teacher_hidden_rms": teacher_hidden_rms,
            "endpoint_entry_hidden_rms": entry_hidden_rms,
        }
    )


@torch.no_grad()
def _dump_rollout_block_sketch(layer_idx: int, img: torch.Tensor) -> None:
    if not (
        _env_enabled("HY_ROLLOUT_DRIFT_DUMP_ENABLE")
        or _env_enabled("HY_ROLLOUT_BLOCK_SKETCH_DUMP_ENABLE")
    ):
        return
    layers = _parse_int_set(os.environ.get("HY_ROLLOUT_DRIFT_LAYERS", "13,40,53"))
    if not _layer_selected(layer_idx, layers):
        return
    if os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "denoise") != "denoise":
        return

    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    if chunk < 0 or step < 0:
        return
    max_chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_MAX_CHUNK", "-1"))
    if max_chunk >= 0 and chunk > max_chunk:
        return

    token_count = int(os.environ.get("HY_ROLLOUT_DRIFT_TOKEN_SAMPLES", "256"))
    channel_count = int(os.environ.get("HY_ROLLOUT_DRIFT_CHANNEL_SAMPLES", "256"))
    seed = int(os.environ.get("HY_ROLLOUT_DRIFT_SEED", "1234"))
    token_idx = _fixed_indices(img.shape[1], token_count, seed + layer_idx * 1009, img.device)
    channel_idx = _fixed_indices(img.shape[2], channel_count, seed + layer_idx * 2003 + 17, img.device)
    sketch = img.detach()[:, token_idx][:, :, channel_idx].to("cpu", dtype=torch.float16).contiguous()

    out_dir = Path(os.environ.get("HY_ROLLOUT_DRIFT_DUMP_DIR", "kv_ae_outputs/ae16x_v2/rollout_drift"))
    run_name = os.environ.get("HY_ROLLOUT_DRIFT_RUN_NAME", "run")
    path = out_dir / run_name / f"chunk{chunk:02d}_step{step:02d}_layer{int(layer_idx):02d}_block.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "block_sketch",
            "chunk": chunk,
            "step": step,
            "layer": int(layer_idx),
            "shape": tuple(int(x) for x in img.shape),
            "token_idx": token_idx.to("cpu"),
            "channel_idx": channel_idx.to("cpu"),
            "tensor": sketch,
        },
        path,
    )


@torch.no_grad()
def _dump_54layer_audit(layer_idx: int, img: torch.Tensor, cache_vision: bool) -> None:
    """Record per-layer block output for the 54-layer online sketch audit.

    Controlled by HY_54L_AUDIT_DIR. Records during the first generation
    forward (cache_vision=True context recompute or cache_vision=False
    denoise step) into <dir>/layer{idx:02d}.pt. Each run writes into its own
    directory (set per group: clean / base / sketch) so the per-layer error
    is computed offline against the clean run.
    """
    out_dir = os.environ.get("HY_54L_AUDIT_DIR", "").strip()
    if not out_dir:
        return
    pass_name = os.environ.get("HY_54L_AUDIT_PASS", "gen").strip()
    stage = "write" if cache_vision else "read"
    expected_stage = os.environ.get("HY_54L_AUDIT_STAGE", "read").strip()
    if stage != expected_stage:
        return
    path = Path(out_dir) / f"{pass_name}_layer{int(layer_idx):02d}_block.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "54l_audit",
            "pass": pass_name,
            "stage": stage,
            "layer": int(layer_idx),
            "shape": tuple(int(x) for x in img.shape),
            "tensor": img.detach().to("cpu", dtype=torch.float16).contiguous(),
        },
        path,
    )


def _replay_dump_tensor(value: Any):
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu")
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.bfloat16)
        return tensor.contiguous()
    if isinstance(value, tuple):
        return tuple(_replay_dump_tensor(item) for item in value)
    if isinstance(value, list):
        return [_replay_dump_tensor(item) for item in value]
    if isinstance(value, dict):
        return {key: _replay_dump_tensor(item) for key, item in value.items()}
    return value


def _replay_dump_selected_cache(layer_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(layer_cache, dict):
        return {}
    keys = (
        "k_vision",
        "v_vision",
        "_k_vision_clean_for_ft",
        "_k_vision_base_for_ft",
        "k_txt",
        "v_txt",
        "vision_thw",
        "vision_kv_position_mode",
        "vision_freqs_cis",
        "vision_viewmats",
        "vision_Ks",
        "vision_block_idx",
    )
    return {key: _replay_dump_tensor(layer_cache[key]) for key in keys if key in layer_cache}


@torch.no_grad()
def _dump_k_replay_state(
    layer_idx: int,
    img_in: torch.Tensor,
    img_out: torch.Tensor,
    vec: torch.Tensor,
    freqs_cis,
    viewmats: Optional[torch.Tensor],
    Ks: Optional[torch.Tensor],
    layer_cache: Optional[Dict[str, Any]],
    vision_thw,
) -> None:
    if not _env_enabled("HY_K_REPLAY_DUMP_ENABLE"):
        return
    if os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "denoise") != "denoise":
        return
    layers = _parse_int_set(os.environ.get("HY_K_REPLAY_DUMP_LAYERS", "13,40,53"))
    if not _layer_selected(layer_idx, layers):
        return
    chunks = _parse_int_set(os.environ.get("HY_K_REPLAY_DUMP_CHUNKS", ""))
    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    target_step = int(os.environ.get("HY_K_REPLAY_DUMP_STEP", "3"))
    if chunk < 0 or step != target_step:
        return
    if chunks and not _layer_selected(chunk, chunks):
        return

    out_dir = Path(os.environ.get("HY_K_REPLAY_DUMP_DIR", "kv_ae_outputs/ae16x_v2/k_replay_dumps"))
    run_name = os.environ.get("HY_K_REPLAY_DUMP_RUN_NAME", "run")
    path = out_dir / run_name / f"chunk{chunk:02d}_step{step:02d}_layer{int(layer_idx):02d}.pt"
    if path.exists() and not _env_enabled("HY_K_REPLAY_DUMP_FORCE"):
        return

    cache_dump = _replay_dump_selected_cache(layer_cache)
    if "_k_vision_clean_for_ft" not in cache_dump:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "k_replay_state_v1",
            "chunk": chunk,
            "step": step,
            "layer": int(layer_idx),
            "img_in": _replay_dump_tensor(img_in),
            "img_out_compressed": _replay_dump_tensor(img_out),
            "vec": _replay_dump_tensor(vec),
            "freqs_cis": _replay_dump_tensor(freqs_cis),
            "viewmats": _replay_dump_tensor(viewmats),
            "Ks": _replay_dump_tensor(Ks),
            "vision_thw": vision_thw,
            "layer_cache": cache_dump,
        },
        path,
    )


def _profile_cuda_call(bucket: str, block_idx: Optional[int], fn, *args, **kwargs):
    return position_profile.cuda_call(bucket, block_idx, fn, *args, **kwargs)


def is_blocks(n: str, m) -> bool:
    print("is_blocks", n, flush=True)
    is_valid = ("double_blocks" in n and str.isdigit(n.split(".")[-1])) or (
        "single_blocks" in n and str.isdigit(n.split(".")[-1])
    )
    return is_valid


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.factory_kwargs = factory_kwargs

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_attn_q = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.img_attn_k = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.img_attn_v = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_q = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_k = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_v = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    # modulating module for txt embedding
    def modulate_txt(self, vec_txt, txt):
        # Generate modulation parameters (shift, scale, gate) for text embeddings using adaptive norm
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        return (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        )

    # modulating module for img embedding
    def modulate_img(self, vec, img):
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        return (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        )

    @torch_compile_wrapper()
    def forward_txt(
        self,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        kv_cache: Optional[dict] = None,
        cache_txt: bool = False,
    ) -> Tuple[torch.Tensor]:
        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        attn_mode = "torch_causal"  # for ar model, the default mode is torch_causal
        txt_attn, t_kv = sequence_parallel_attention_txt(
            (txt_q),
            (txt_k),
            (txt_v),
            img_q_len=txt_q.shape[1],
            img_kv_len=txt_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_txt=cache_txt,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )
        return txt, t_kv

    @torch_compile_wrapper()
    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        attn_param=None,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_vision: bool = False,
        vision_thw=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_block_input = img
        (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.modulate_img(vec, img)
        if _p1_causal_audit_enabled() and _env_enabled(
            "HY_K_P1_SHARE_CURRENT_QKV"
        ):
            functional_pass = os.environ.get(
                "HY_FUNCTIONAL_BLOCK_FT_PASS", ""
            ).strip().lower()
            if functional_pass == "p1_teacher":
                self._hy_p1_teacher_current_qkv = (
                    img_q.detach(),
                    img_k.detach(),
                    img_v.detach(),
                )
            elif functional_pass == "student" and hasattr(
                self, "_hy_p1_teacher_current_qkv"
            ):
                teacher_q, teacher_k, teacher_v = self._hy_p1_teacher_current_qkv
                img_q = teacher_q.to(img_q)
                img_k = teacher_k.to(img_k)
                img_v = teacher_v.to(img_v)
                delattr(self, "_hy_p1_teacher_current_qkv")
        img_k_pre_position = img_k
        img_v_pre_position = img_v
        if cache_vision and _env_enabled("HY_KV_QK_SUBSET_DUMP_ENABLE"):
            dump_qk_subset_pre_position(
                block_idx,
                img_q,
                img_k_pre_position,
                vision_thw,
            )

        # Add camera pose conditioning through ProPE (Projective Positional Encoding)
        # delete rope components (original attn included)
        # Apply ProPE transformation to Q, K, V using camera view matrices and intrinsics
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = _profile_cuda_call(
            "forward_prope_qkv",
            block_idx,
            prope_qkv,
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]

        if freqs_cis is not None:
            img_qq, img_kk = _profile_cuda_call(
                "forward_rope",
                block_idx,
                apply_rotary_emb,
                img_q,
                img_k,
                freqs_cis,
                head_first=False,
            )
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        img_attn, img_attn_prope, vision_kv = sequence_parallel_attention_vision(
            (img_q, img_q_prope),
            (img_k, img_k_prope),
            (img_v, img_v_prope),
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_vision=cache_vision,
            attn_param=attn_param,
            vision_thw=vision_thw,
        )
        img_attn_prope_raw = img_attn_prope
        if cache_vision and _env_enabled("HY_PRE_POSITION_VISION_KV_CACHE"):
            from hyvideo.models.transformers.modules.ar_sparse_operation import (
                SparseBlockLayout,
                masked_block_mean,
                tile_visual_tokens,
            )

            vision_kv["k_vision_pre_position"] = (
                img_k_pre_position.transpose(1, 2).contiguous()
            )
            vision_kv["v_vision_pre_position"] = (
                img_v_pre_position.transpose(1, 2).contiguous()
            )
            if (
                vision_thw is not None
                and (
                    bool((attn_param or {}).get("sparse_decode_enabled", False))
                    or _env_enabled("HY_SPARSE_DECODE_ENABLE")
                )
                and (
                    not os.environ.get("HY_SPARSE_DECODE_LAYERS", "").strip()
                    or int(block_idx)
                    in {
                        int(item)
                        for item in os.environ["HY_SPARSE_DECODE_LAYERS"].split(",")
                        if item.strip()
                    }
                )
            ):
                tile_size = tuple(
                    int(value)
                    for value in (attn_param or {}).get(
                        "sparse_tile_size", (4, 8, 4)
                    )
                )
                proxy_layout = SparseBlockLayout.build(
                    tuple(int(value) for value in vision_thw),
                    tile_size,
                    device=img_k.device,
                )
                k_rope = img_k.transpose(1, 2).contiguous()
                k_prope = img_k_prope.transpose(1, 2).contiguous()
                vision_kv["k_proxy_rope"] = masked_block_mean(
                    tile_visual_tokens(k_rope, proxy_layout), proxy_layout
                )
                vision_kv["k_proxy_prope"] = masked_block_mean(
                    tile_visual_tokens(k_prope, proxy_layout), proxy_layout
                )
                vision_kv["sparse_proxy_tile_size"] = tile_size

        img_attn_prope = rearrange(
            img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num
        )
        img_attn_prope = apply_fn_o(
            img_attn_prope
        )  # [batch, num_heads, seqlen, head_dim]
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        img_attn_prope_restored = img_attn_prope

        if _env_enabled("HY_K_PLACEMENT_ORACLE_CAPTURE"):
            self._hy_k_placement_readout_main = img_attn.detach()
            self._hy_k_placement_readout_prope = img_attn_prope.detach()

        attention_update = apply_gate(
            self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope),
            gate=img_mod1_gate,
        )
        if _env_enabled("HY_K_PLACEMENT_ORACLE_CAPTURE"):
            self._hy_k_placement_attention_update = attention_update.detach()
        img = img + attention_update
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )
        if _env_enabled("HY_K_PLACEMENT_ORACLE_CAPTURE"):
            self._hy_k_placement_block_output = img.detach()

        v_refit_dump = _env_enabled("HY_KV_AE3D_ATTN_DUMP_V_REFIT_ONLY")
        if (cache_vision or v_refit_dump) and vision_thw is not None:
            layer_cache = (
                kv_cache[block_idx]
                if kv_cache is not None and block_idx < len(kv_cache)
                else {}
            )
            dump_attention_sample(
                block_idx,
                img_q,
                img_k,
                img_v,
                img_q_prope,
                img_k_prope,
                img_v_prope,
                img_attn,
                img_attn_prope_raw,
                vision_thw,
                freqs_cis=freqs_cis,
                viewmats=viewmats,
                Ks=Ks,
                layer_input=img_block_input,
                layer_output=img,
                vec=vec,
                text_k=layer_cache.get("k_txt"),
                text_v=layer_cache.get("v_txt"),
                attn_prope_restored=img_attn_prope_restored,
            )

        return img, vision_kv

    @torch_compile_wrapper()
    def forward_bi(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.modulate_img(vec, img)
        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        # add camera pose through prope
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn, txt_attn = (
            attn[:, : img_q.shape[1]].contiguous(),
            attn[:, img_q.shape[1] :].contiguous(),
        )

        attn_prope = parallel_attention(
            (img_q_prope, txt_q),
            (img_k_prope, txt_k),
            (img_v_prope, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn_prope = attn_prope[:, : img_q_prope.shape[1]].contiguous()
        img_attn_prope = rearrange(
            img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num
        )
        img_attn_prope = apply_fn_o(
            img_attn_prope
        )  # [batch, num_heads, seqlen, head_dim]
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")

        img = img + apply_gate(
            self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope),
            gate=img_mod1_gate,
        )
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )
        return img, txt

    @torch_compile_wrapper()
    def forward_sr(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.modulate_img(vec, img)
        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )

        img_attn, txt_attn = (
            attn[:, : img_q.shape[1]].contiguous(),
            attn[:, img_q.shape[1] :].contiguous(),
        )

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt

    @torch_compile_wrapper()
    def forward(
        self,
        bi_inference=True,
        ar_txt_inference=False,
        ar_vision_inference=False,
        **kwargs,
    ):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            return self.forward_sr(**kwargs)


class MMSingleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(
            hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs
        )
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        """Forward pass for the single stream block."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        txt_mod_shift, txt_mod_scale, txt_mod_gate = self.modulation(vec_txt).chunk(
            3, dim=-1
        )

        x_pre_norm = self.pre_norm(x)
        img_token, txt_token = x_pre_norm[:, :-txt_len, :], x_pre_norm[:, -txt_len:, :]

        img_mod = modulate(img_token, shift=mod_shift, scale=mod_scale)
        txt_mod = modulate(txt_token, shift=txt_mod_shift, scale=txt_mod_scale)
        x_mod = torch.cat([img_mod, txt_mod], dim=1)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = "flash"
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))
        img_output, txt_output = output[:, :-txt_len, :], output[:, -txt_len:, :]

        img_output = apply_gate(img_output, gate=mod_gate)
        txt_output = apply_gate(txt_output, gate=txt_mod_gate)
        gate_output = torch.cat([img_output, txt_output], dim=1)

        return x + gate_output


class HunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
    ):
        super().__init__()
        factory_kwargs = {}

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False,
            )

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )
        self.vector_in = (
            MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            )
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if use_meanflow
            else None
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # STA
        if attn_param is None:
            self.attn_param = {
                # STA
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                # SSTA
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                # flex-block-attn:
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param

        sparse_defaults = {
            "sparse_decode_enabled": False,
            "sparse_tile_size": (4, 8, 4),
            "sparse_topk_ratio": 1.0,
            "sparse_recent_protect": 0,
            "sparse_sim_metric": "cosine",
            "sparse_dual_branch_index": True,
            "sparse_compact_selected_kv": True,
            "sparse_debug_compare_dense": False,
            "sparse_oracle_indices": False,
        }
        for key, value in sparse_defaults.items():
            self.attn_param.setdefault(key, value)

        if attn_mode == "flex-block-attn":
            self.register_to_config(attn_param=self.attn_param)

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert (
                self.glyph_byT5_v2
            ), "text type embedding is only used when glyph_byT5_v2 is True"
            assert (
                vision_projection is not None
            ), "text type embedding is only used when vision_projection is not None"
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None

        self.gradient_checkpointing = False

    def load_hunyuan_state_dict(self, model_path):
        load_key = "module"
        bare_model = "unknown"

        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )

        result = self.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            logger.info("[load.py] Missing keys when loading state_dict:")
            for key in result.missing_keys:
                logger.info(f"[load.py] Missing key: {key}")
        if result.unexpected_keys:
            logger.info("[load.py] Unexpected keys when loading state_dict:")
            for key in result.unexpected_keys:
                logger.info(f"[load.py] Unexpected key: {key}")
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}"
            )

        return result

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(
        self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True
    ):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            pad_byt5,
                            pad_text,
                        ],
                        dim=0,
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(
                dtype=torch.int64
            )

        return reorder_txt, reorder_mask

    def add_action_parameters(self):
        # Initialize the embedding module for action conditioning (using Silu activation).
        # This method adds action control capabilities to the pre-trained transformer
        self.action_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu")
        )

        # Zero-initialize the weights of the final layer (mlp[2]) of the embedder.
        # This ensures the initial output is zero, preventing the action conditioning
        # from affecting the model's behavior at the start of training.
        nn.init.zeros_(self.action_in.mlp[2].weight)

        # Zero-initialize the bias term if it exists.
        if self.action_in.mlp[2].bias is not None:
            nn.init.zeros_(self.action_in.mlp[2].bias)

        # Iterate through each transformer double-block to inject action-specific projections.
        for block in self.double_blocks:
            # Create a new linear projection layer for image attention properties.
            block.img_attn_prope_proj = nn.Linear(
                block.hidden_size,
                block.hidden_size,
                bias=block.qkv_bias,
                **block.factory_kwargs,
            )

            # Zero-initialize the weights of this new projection layer.
            # Again, this ensures the new layer acts as an identity function initially.
            nn.init.zeros_(block.img_attn_prope_proj.weight)

            # Zero-initialize the bias term for the projection layer.
            if block.img_attn_prope_proj.bias is not None:
                nn.init.zeros_(block.img_attn_prope_proj.bias)

    def get_text_and_mask(
        self,
        encoder_attention_mask,
        text_states,
        timestep_txt,
        extra_kwargs,
        vision_states,
        mask_type,
    ):
        text_mask = encoder_attention_mask
        txt = text_states
        bs = txt.shape[0]

        # Prepare modulation vectors
        vec_txt = self.time_in(timestep_txt)

        # Embed text tokens
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(
                txt, timestep_txt, text_mask if self.use_attention_mask else None
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(
                    txt[:, :, 0], device=text_mask.device, dtype=torch.long
                )
            )
            txt = txt + cond_emb

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(
                        byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long
                    )
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            # If t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                # Set vision tokens to zero to mitigate potential block mask error in SSTA
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            # Ensure valid tokens precede padding tokens
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2
                    * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

            txt, text_mask = self.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )
        return txt, text_mask, vec_txt

    # txt embedding kv cache
    def forward_txt(
        self,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor = None,
        mask_type="t2v",
        extra_kwargs=None,
        kv_cache: Optional[dict] = None,
        cache_txt: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Forward pass for text embeddings only, optionally caching K/V for autoregressive generation
        if cache_txt:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for _ in range(transformer_num_layers):
                _kv_cache_new.append(
                    {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
                )

        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type,
        )

        # mask the txt tokens in advance for efficiency
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            txt, t_kv = block(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                txt=txt,
                vec_txt=vec_txt,
                text_mask=None,  # we have masked txt tokens already, set None here
                attn_param=None,
                is_flash=False,
                block_idx=index,
                kv_cache=kv_cache,
                cache_txt=cache_txt,
            )

            if cache_txt:
                _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
                _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]

        if cache_txt:
            return _kv_cache_new

    # using kv cache to calculate vision embedding
    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_r=None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_vision: bool = False,
        rope_temporal_size=4,
        start_rope_start_idx=0,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache_vision:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for i in range(transformer_num_layers):
                _kv_cache_new.append(
                    {
                        "k_vision": None,
                        "v_vision": None,
                        "k_txt": kv_cache[i]["k_txt"],
                        "v_txt": kv_cache[i]["v_txt"],
                    }
                )

        img = x = hidden_states
        t = timestep
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param["thw"] = [tt, th, tw]
        rope_temporal_size = rope_temporal_size // self.patch_size[0]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                (rope_temporal_size, th, tw)
            )
            per_latent_size = th * tw
            start_index = start_rope_start_idx * per_latent_size
            end_index = (start_rope_start_idx + tt) * per_latent_size
            freqs_cos = freqs_cos[start_index:end_index, ...]
            freqs_sin = freqs_sin[start_index:end_index, ...]

        img = self.img_in(img)

        action = action.reshape(-1)
        t = t.reshape(-1)

        # Prepare modulation vectors
        vec = self.time_in(t)
        vec = vec + self.action_in(action)  # inject discrete action

        # boardcast to match the sequence length of video latents
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)
        viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
        Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)

        # Sequence parallel: split spatial tokens across multiple GPUs for memory efficiency
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            # Verify sequence length is sufficient for parallelization
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (
                    sp_size - 1
                ), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")
            viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
            Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
        else:
            vec = rearrange(vec, "B S C->(B S) C")

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Propagation carries the clean source output through a frozen suffix.
        # The Student source output remains in ``img`` with its autograd graph.
        p1_pending = None

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            ft_enabled = _functional_block_ft_enabled()
            ft_pass = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
            ft_layers = _parse_int_set(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "13,40,53"))
            ft_selected = ft_enabled and _layer_selected(index, ft_layers)
            ft_prop_horizon = _functional_prop_horizon()
            ft_prop1_source = (
                ft_enabled
                and ft_pass == "student"
                and _functional_prop1_enabled()
                and ft_selected
                and index + ft_prop_horizon < len(self.double_blocks)
            )
            ft_prop1_target = (
                ft_enabled
                and ft_pass == "student"
                and p1_pending is not None
                and int(p1_pending["source_layer"]) < index
                and index
                <= int(p1_pending["source_layer"])
                + int(p1_pending["horizon"])
            )
            ft_detach_mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_DETACH_MODE", "per_layer").strip().lower()
            ft_window_mode = ft_enabled and ft_detach_mode in {"window", "physical_window"}
            ft_window_start = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_WINDOW_START", "-1"))
            ft_window_end = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_WINDOW_END", "-1"))
            ft_window_checkpoint = _env_enabled("HY_FUNCTIONAL_BLOCK_FT_WINDOW_CHECKPOINT")
            ft_final_pred_suffix_grad = (
                ft_enabled
                and ft_pass == "student"
                and _env_enabled("HY_FUNCTIONAL_BLOCK_FT_FINAL_PRED_SUFFIX_GRAD")
                and index > ft_window_end
            )
            ft_suffix_checkpoint = _env_enabled(
                "HY_FUNCTIONAL_BLOCK_FT_SUFFIX_CHECKPOINT"
            )
            ft_reuse_prefix = ft_window_mode and _env_enabled(
                "HY_FUNCTIONAL_BLOCK_FT_REUSE_PREFIX_TENSOR"
            )
            ft_window_range = (
                ft_window_mode
                and ft_window_start >= 0
                and ft_window_end >= ft_window_start
                and ft_window_start <= index <= ft_window_end
            )
            # In strict causal-window mode the Teacher computes the compressed
            # prefix once. Student inner passes skip those blocks and consume the
            # exact same detached entry tensor at the first window layer.
            if (
                ft_reuse_prefix
                and ft_pass == "student"
                and index < ft_window_start
            ):
                continue
            if ft_reuse_prefix and index == ft_window_start:
                entry_key = _functional_window_key()
                if ft_pass == "teacher":
                    _FUNCTIONAL_WINDOW_ENTRIES[entry_key] = img.detach()
                elif ft_pass == "student":
                    shared_entry = _FUNCTIONAL_WINDOW_ENTRIES.get(entry_key)
                    if shared_entry is None:
                        if _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT"):
                            raise RuntimeError(
                                f"Missing Teacher compressed-prefix tensor for key={entry_key}"
                            )
                    else:
                        img = shared_entry
                        if (
                            int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_INNER_INDEX", "0")) == 0
                            and _env_enabled("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_DEBUG")
                        ):
                            print(
                                "[FunctionalWindowPrefix] "
                                f"key={entry_key} layer={index} "
                                f"data_ptr={img.data_ptr()} shape={tuple(img.shape)}",
                                flush=True,
                            )
            ft_in_window = (
                ft_window_range
                and ft_pass == "student"
            )
            if ft_enabled:
                if ft_in_window:
                    if (
                        index == ft_window_start
                        or _env_enabled("HY_FUNCTIONAL_BLOCK_FT_WINDOW_INNER_DETACH")
                    ) and not _env_enabled("HY_K_T0_PRESERVE_DENOISE_GRAPH"):
                        img = img.detach()
                elif ft_final_pred_suffix_grad or ft_prop1_target:
                    # Frozen clean suffix: preserve only the hidden-state VJP back
                    # to the trainable early codecs. Suffix parameters are frozen
                    # by the pipeline before this forward.
                    pass
                else:
                    img = img.detach()
            grad_ctx = torch.enable_grad() if (
                ft_pass == "student"
                and (
                    ft_selected
                    or ft_in_window
                    or ft_final_pred_suffix_grad
                    or ft_prop1_target
                )
            ) else torch.no_grad()
            if not ft_enabled:
                grad_ctx = torch.enable_grad() if torch.is_grad_enabled() else torch.no_grad()
            replay_dump_img_in = img
            local_teacher_img = None
            placement_clean_img = None
            placement_clean_attention = None
            placement_clean_readout_main = None
            placement_clean_readout_prope = None
            placement_active = (
                not cache_vision and _k_placement_oracle_active(index)
            )
            if placement_active:
                placement_clean_cache = _k_placement_clean_cache(kv_cache, index)
                previous_audit_pass = os.environ.get("HY_K_ATTENTION_AUDIT_PASS")
                os.environ["HY_K_ATTENTION_AUDIT_PASS"] = "clean"
                try:
                    with torch.no_grad():
                        placement_clean_img = block(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            img=img.detach(),
                            vec=vec,
                            freqs_cis=freqs_cis,
                            attn_param=self.attn_param,
                            block_idx=index,
                            viewmats=viewmats,
                            Ks=Ks,
                            kv_cache=placement_clean_cache,
                            cache_vision=False,
                            vision_thw=(tt, th, tw),
                        )[0].detach()
                finally:
                    if previous_audit_pass is None:
                        os.environ.pop("HY_K_ATTENTION_AUDIT_PASS", None)
                    else:
                        os.environ["HY_K_ATTENTION_AUDIT_PASS"] = previous_audit_pass
                placement_clean_attention = getattr(
                    block, "_hy_k_placement_attention_update", None
                )
                if not torch.is_tensor(placement_clean_attention):
                    raise RuntimeError(
                        f"K placement oracle did not capture clean attention at L{index}"
                    )
                placement_clean_attention = placement_clean_attention.detach()
                placement_clean_readout_main = getattr(
                    block, "_hy_k_placement_readout_main", None
                )
                placement_clean_readout_prope = getattr(
                    block, "_hy_k_placement_readout_prope", None
                )
            local_target_mode = os.environ.get(
                "HY_FUNCTIONAL_BLOCK_FT_LOCAL_TARGET", "hidden"
            ).strip().lower()
            if (
                ft_pass == "student"
                and ft_selected
                and (
                    local_target_mode
                    in {"residual", "residual_update", "block_update", "delta"}
                    or ft_prop1_source
                )
            ):
                local_teacher_cache = _functional_local_teacher_cache(kv_cache, index)
                if local_teacher_cache is not None:
                    previous_pass = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
                    os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "local_teacher"
                    try:
                        with torch.no_grad():
                            local_teacher_img = block(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                img=img.detach(),
                                vec=vec,
                                freqs_cis=freqs_cis,
                                attn_param=self.attn_param,
                                block_idx=index,
                                viewmats=viewmats,
                                Ks=Ks,
                                kv_cache=local_teacher_cache,
                                cache_vision=False,
                                vision_thw=(tt, th, tw),
                            )[0].detach()
                    finally:
                        os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = previous_pass
            p1_teacher_out = None
            p1_cache_audit = None
            p1_branch_order = os.environ.get(
                "HY_K_P1_BRANCH_ORDER", "teacher_first"
            ).strip().lower()
            if p1_branch_order not in {"teacher_first", "student_first"}:
                raise RuntimeError(
                    f"Unknown HY_K_P1_BRANCH_ORDER={p1_branch_order}"
                )
            if (
                ft_prop1_target
                and p1_branch_order == "student_first"
                and _env_enabled("HY_K_P1_SHARE_CURRENT_QKV")
            ):
                raise RuntimeError(
                    "Current-QKV sharing audit requires teacher_first order"
                )
            if ft_prop1_target and _p1_causal_audit_enabled():
                p1_cache_audit = {
                    "before_first_branch": _audit_layer_cache_fingerprint(
                        kv_cache, index
                    )
                }
            if ft_prop1_target and p1_branch_order == "teacher_first":
                previous_pass = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
                os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "p1_teacher"
                teacher_cache = kv_cache
                if _env_enabled("HY_K_P1_OBSERVATION_TEACHER_CLEAN"):
                    teacher_cache = _functional_local_teacher_cache(kv_cache, index)
                try:
                    # Frozen target branch: same compressed historical K as the
                    # Student, but current Q/K/V arise naturally from its own
                    # propagated hidden state.
                    with torch.no_grad():
                        p1_teacher_out = block(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            img=p1_pending["teacher_hidden"],
                            vec=vec,
                            freqs_cis=freqs_cis,
                            attn_param=self.attn_param,
                            block_idx=index,
                            viewmats=viewmats,
                            Ks=Ks,
                            kv_cache=teacher_cache,
                            cache_vision=False,
                            vision_thw=(tt, th, tw),
                        )[0].detach()
                finally:
                    os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = previous_pass
                if p1_cache_audit is not None:
                    p1_cache_audit["after_teacher"] = (
                        _audit_layer_cache_fingerprint(kv_cache, index)
                    )
            main_block_cache = kv_cache
            if (
                ft_prop1_source
                and _env_enabled("HY_K_P1_SOURCE_STUDENT_CLEAN")
            ):
                main_block_cache = _functional_source_clean_counterfactual_cache(
                    kv_cache, index
                )
            with grad_ctx:
                previous_audit_pass = os.environ.get("HY_K_ATTENTION_AUDIT_PASS")
                if placement_active:
                    os.environ["HY_K_ATTENTION_AUDIT_PASS"] = "student"
                if (
                    (
                        (ft_in_window and ft_window_checkpoint)
                        or (ft_final_pred_suffix_grad and ft_suffix_checkpoint)
                    )
                    and not cache_vision
                ):
                    # The main Transformer is frozen during functional FT. Recompute its
                    # window activations in backward so endpoint losses can reach early
                    # codecs without retaining every intermediate block activation.
                    def _checkpointed_block(
                        img_in,
                        _block=block,
                        _index=index,
                        _attn_param=dict(self.attn_param),
                    ):
                        return _block(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            img=img_in,
                            vec=vec,
                            freqs_cis=freqs_cis,
                            attn_param=_attn_param,
                            block_idx=_index,
                            viewmats=viewmats,
                            Ks=Ks,
                            kv_cache=kv_cache,
                            cache_vision=False,
                            vision_thw=(tt, th, tw),
                        )[0]

                    img = activation_checkpoint(
                        _checkpointed_block,
                        img,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    vision_kv = None
                else:
                    img, vision_kv = block(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        img=img,
                        vec=vec,
                        freqs_cis=freqs_cis,
                        attn_param=self.attn_param,
                        block_idx=index,
                        viewmats=viewmats,
                        Ks=Ks,
                        kv_cache=main_block_cache,
                        cache_vision=cache_vision,
                        vision_thw=(tt, th, tw),
                    )
                if placement_active:
                    if previous_audit_pass is None:
                        os.environ.pop("HY_K_ATTENTION_AUDIT_PASS", None)
                    else:
                        os.environ["HY_K_ATTENTION_AUDIT_PASS"] = previous_audit_pass
                _dump_54layer_audit(index, img, cache_vision)
            if ft_prop1_target and p1_cache_audit is not None:
                p1_cache_audit["after_student"] = _audit_layer_cache_fingerprint(
                    kv_cache, index
                )
            if ft_prop1_target and p1_branch_order == "student_first":
                previous_pass = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
                os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "p1_teacher"
                teacher_cache = kv_cache
                if _env_enabled("HY_K_P1_OBSERVATION_TEACHER_CLEAN"):
                    teacher_cache = _functional_local_teacher_cache(kv_cache, index)
                try:
                    with torch.no_grad():
                        p1_teacher_out = block(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            img=p1_pending["teacher_hidden"],
                            vec=vec,
                            freqs_cis=freqs_cis,
                            attn_param=self.attn_param,
                            block_idx=index,
                            viewmats=viewmats,
                            Ks=Ks,
                            kv_cache=teacher_cache,
                            cache_vision=False,
                            vision_thw=(tt, th, tw),
                        )[0].detach()
                finally:
                    os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = previous_pass
                if p1_cache_audit is not None:
                    p1_cache_audit["after_teacher"] = (
                        _audit_layer_cache_fingerprint(kv_cache, index)
                    )
            if placement_active:
                student_block_img = img
                student_attention = getattr(
                    block, "_hy_k_placement_attention_update", None
                )
                student_readout_main = getattr(
                    block, "_hy_k_placement_readout_main", None
                )
                student_readout_prope = getattr(
                    block, "_hy_k_placement_readout_prope", None
                )
                if not torch.is_tensor(student_attention):
                    raise RuntimeError(
                        f"K placement oracle did not capture Student attention at L{index}"
                    )
                mode = os.environ.get(
                    "HY_K_PLACEMENT_ORACLE_MODE", "base"
                ).strip().lower()
                if mode == "attention":
                    # Deployment-oriented post-block attention residual: correct
                    # only the additive attention contribution and deliberately
                    # do not replay the MLP on the corrected intermediate state.
                    img = student_block_img + (
                        placement_clean_attention.to(student_block_img)
                        - student_attention.to(student_block_img)
                    )
                elif mode == "block":
                    img = placement_clean_img.to(student_block_img)
                else:
                    img = student_block_img
                _record_k_placement_oracle(
                    index,
                    mode,
                    placement_clean_attention,
                    student_attention,
                    placement_clean_img,
                    student_block_img,
                    img,
                    attention_audit=pop_k_attention_audit_metrics(index),
                    clean_readout_main=placement_clean_readout_main,
                    student_readout_main=student_readout_main,
                    clean_readout_prope=placement_clean_readout_prope,
                    student_readout_prope=student_readout_prope,
                )
            if (
                ft_enabled
                and ft_pass == "student"
                and ft_selected
                and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_DEBUG_GRAD", "0").lower() in {"1", "true", "yes", "on"}
            ):
                cache_target = None
                cache_base = None
                debug_target = os.environ.get(
                    "HY_FUNCTIONAL_BLOCK_FT_TARGET", "k"
                ).strip().lower()
                target_name = "v_vision" if debug_target == "v" else "k_vision"
                base_name = (
                    "_v_vision_base_for_ft"
                    if target_name == "v_vision"
                    else "_k_vision_base_for_ft"
                )
                try:
                    cache_target = (
                        kv_cache[index].get(target_name) if kv_cache is not None else None
                    )
                    cache_base = (
                        kv_cache[index].get(base_name) if kv_cache is not None else None
                    )
                except Exception:
                    cache_target = None
                    cache_base = None
                print(
                    "[FunctionalBlockFTDebug] "
                    f"layer={index} input_req={replay_dump_img_in.requires_grad} "
                    f"img_req={img.requires_grad} "
                    f"img_grad_fn={type(img.grad_fn).__name__ if img.grad_fn is not None else None} "
                    f"target={target_name} target_req={getattr(cache_target, 'requires_grad', None)} "
                    f"target_grad_fn={type(cache_target.grad_fn).__name__ if torch.is_tensor(cache_target) and cache_target.grad_fn is not None else None} "
                    f"base_req={getattr(cache_base, 'requires_grad', None)}",
                    flush=True,
                )
            record_ctx = torch.enable_grad() if (ft_pass == "student" and ft_selected) else torch.no_grad()
            with record_ctx:
                _record_functional_block(index, replay_dump_img_in, img, local_teacher_img)
            if ft_prop1_target:
                p1_pending["teacher_hidden"] = p1_teacher_out
                observation_distance = index - int(p1_pending["source_layer"])
                if observation_distance == int(p1_pending["horizon"]):
                    with torch.enable_grad():
                        _record_functional_prop1(
                            int(p1_pending["source_layer"]),
                            p1_pending["source_entry"],
                            img,
                            p1_teacher_out,
                            observation_input=replay_dump_img_in,
                            observation_trainable_parameter_count=sum(
                                int(parameter.requires_grad)
                                for parameter in block.parameters()
                            ),
                            cache_audit=p1_cache_audit,
                        )
                    p1_pending = None
            if ft_prop1_source:
                if local_teacher_img is None:
                    raise RuntimeError(
                        f"P1 source L{index} did not produce a clean local branch"
                    )
                p1_pending = {
                    "source_layer": int(index),
                    "source_entry": replay_dump_img_in.detach(),
                    "teacher_hidden": local_teacher_img.detach(),
                    "horizon": int(ft_prop_horizon),
                }
            if not cache_vision:
                _dump_rollout_block_sketch(index, img)
                replay_layer_cache = None
                if kv_cache is not None:
                    try:
                        replay_layer_cache = kv_cache[index]
                    except Exception:
                        replay_layer_cache = None
                _dump_k_replay_state(
                    index,
                    replay_dump_img_in,
                    img,
                    vec,
                    freqs_cis,
                    viewmats,
                    Ks,
                    replay_layer_cache,
                    (tt, th, tw),
                )
            if cache_vision:
                if _env_enabled("HY_PRE_POSITION_VISION_KV_CACHE"):
                    _kv_cache_new[index]["k_vision"] = vision_kv[
                        "k_vision_pre_position"
                    ]
                    _kv_cache_new[index]["v_vision"] = vision_kv[
                        "v_vision_pre_position"
                    ]
                    _kv_cache_new[index]["vision_kv_position_mode"] = "pre"
                    _kv_cache_new[index]["vision_freqs_cis"] = freqs_cis
                    _kv_cache_new[index]["vision_viewmats"] = viewmats
                    _kv_cache_new[index]["vision_Ks"] = Ks
                    _kv_cache_new[index]["vision_block_idx"] = index
                    _kv_cache_new[index]["vision_thw"] = (tt, th, tw)
                    dump_kvtc_calibration_sample(index, "k", _kv_cache_new[index]["k_vision"])
                    dump_kvtc_calibration_sample(index, "v", _kv_cache_new[index]["v_vision"])
                    for sparse_key in (
                        "k_proxy_rope",
                        "k_proxy_prope",
                        "sparse_proxy_tile_size",
                    ):
                        if sparse_key in vision_kv:
                            _kv_cache_new[index][sparse_key] = vision_kv[sparse_key]
                    if kvtc_enabled():
                        # KVTC jointly encodes a consecutive layer group.  Keep
                        # layers raw until the group is complete, then replace
                        # every member with a shared transform-coded payload.
                        compress_kvtc_ready_group(_kv_cache_new, index)
                    else:
                        _kv_cache_new[index] = compress_layer_cache(
                            _kv_cache_new[index], index
                        )
                        _kv_cache_new[index] = apply_codec_to_layer_cache(
                            _kv_cache_new[index], index
                        )
                        _kv_cache_new[index] = compress_qvg_layer_cache(
                            _kv_cache_new[index], index
                        )
                        _kv_cache_new[index] = compress_jensen_layer_cache(
                            _kv_cache_new[index], index
                        )
                        _kv_cache_new[index] = compress_turboquant_layer_cache(
                            _kv_cache_new[index], index
                        )
                    empty_interval = int(
                        os.environ.get(
                            "HY_KV_EFF_OPT_EMPTY_CACHE_INTERVAL_LAYERS", "0"
                        )
                        or 0
                    )
                    if (
                        empty_interval > 0
                        and (index + 1) % empty_interval == 0
                        and (index + 1) < len(self.double_blocks)
                    ):
                        torch.cuda.empty_cache()
                else:
                    _kv_cache_new[index]["k_vision"] = vision_kv["k_vision"]
                    _kv_cache_new[index]["v_vision"] = vision_kv["v_vision"]
                    _kv_cache_new[index]["vision_kv_position_mode"] = "post"
                    _kv_cache_new[index]["vision_block_idx"] = index
                    _kv_cache_new[index] = compress_jensen_layer_cache(
                        _kv_cache_new[index], index
                    )

        if cache_vision:
            if _env_enabled("HY_PRE_POSITION_VISION_KV_PROFILE"):
                total_numel = 0
                total_bytes = 0
                for layer_cache in _kv_cache_new:
                    for name in ("k_vision", "v_vision"):
                        sub_numel, sub_bytes = _kv_resident_numel_bytes(
                            layer_cache.get(name)
                        )
                        total_numel += sub_numel
                        total_bytes += sub_bytes
                mode = _kv_cache_new[0].get("vision_kv_position_mode", "post")
                print(
                    "[Vision KV Cache] "
                    f"mode={mode} resident={total_numel} numel "
                    f"{total_bytes / (1024 ** 2):.2f} MB"
                )
            return _kv_cache_new

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def forward_bi(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = x = hidden_states
        t = timestep
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)
        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        if action is not None:
            vec = vec + self.action_in(action)  # 添加离散的action

        # boardcast to match the sequence length of video latents
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)
        if viewmats is not None:
            viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
            Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)

        # sequence parallel
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (
                    sp_size - 1
                ), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")
            if viewmats is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
        else:
            vec = rearrange(vec, "B S C->(B S) C")

        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type,
        )

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            if viewmats is not None:
                img, txt = block(
                    bi_inference=True,
                    ar_txt_inference=False,
                    ar_vision_inference=False,
                    img=img,
                    txt=txt,
                    vec_txt=vec_txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,  # we have masked txt tokens already, set None here
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                    viewmats=viewmats,
                    Ks=Ks,
                )
            else:
                img, txt = block(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=False,
                    img=img,
                    txt=txt,
                    vec_txt=vec_txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,  # we have masked txt tokens already, set None here
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                )

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def forward(
        self,
        bi_inference=True,
        ar_txt_inference=False,
        ar_vision_inference=False,
        **kwargs,
    ):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise NotImplementedError

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        # Set the attention mode for the current object
        self.attn_mode = attn_mode

        # Iterate through each block in the double_blocks list
        for block in self.double_blocks:
            # Set the attention mode for each block in double_blocks
            block.attn_mode = attn_mode

        # Iterate through each block in the single_blocks list
        for block in self.single_blocks:
            # Set the attention mode for each block in single_blocks
            block.attn_mode = attn_mode
