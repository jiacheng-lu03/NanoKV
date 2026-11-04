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

import os
import math
import einops
import torch
from typing import Optional
from loguru import logger
import numpy as np
import torch.nn.functional as F
from einops import repeat

from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.communications import (
    all_gather,
    all_to_all_4D,
)
from hyvideo.utils.flash_attn_no_pad import (
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
)
from hyvideo.commons import maybe_fallback_attn_mode

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")

from hyvideo.models.transformers.modules.ssta_attention import ssta_3d_attention
from hyvideo.commons.infer_state import get_infer_state
from hyvideo.models.transformers.modules import position_profile

_FREQS_CIS_CUDA_CACHE = {}
_FUNCTIONAL_READOUT_TARGETS = {}
_FUNCTIONAL_READOUT_LOSSES = []
_FUNCTIONAL_READOUT_METRICS = []
_K_ATTENTION_AUDIT_TARGETS = {}
_K_ATTENTION_AUDIT_METRICS = {}


def _k_attention_audit_key(block_idx: int):
    return (
        int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1")),
        int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1")),
        int(block_idx),
    )


def pop_k_attention_audit_metrics(block_idx: int):
    """Return and remove the exact-runtime clean/student attention comparison."""
    return _K_ATTENTION_AUDIT_METRICS.pop(_k_attention_audit_key(block_idx), None)


def _audit_rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((pred - ref).norm() / ref.norm().clamp_min(1e-12)).cpu())


@torch.no_grad()
def _record_k_attention_audit(
    block_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Audit tensors at the exact point consumed by SDPA (after position/cache cat).

    We sample query positions but retain every key position, so probabilities and
    readouts use the real normalization domain.  Clean references are discarded as
    soon as the immediately following student pass has been compared.
    """
    if os.environ.get("HY_K_ATTENTION_AUDIT_ENABLE", "0").lower() not in {
        "1", "true", "yes", "on"
    }:
        return
    pass_name = os.environ.get("HY_K_ATTENTION_AUDIT_PASS", "").strip().lower()
    if pass_name not in {"clean", "student"}:
        return
    selected = {
        int(item.strip())
        for item in os.environ.get("HY_K_PLACEMENT_ORACLE_LAYERS", "0,1,2,3").split(",")
        if item.strip()
    }
    if int(block_idx) not in selected:
        return
    audit_key = _k_attention_audit_key(block_idx)
    if audit_key[0] < 0 or audit_key[1] < 0:
        return

    sample_count = min(
        int(os.environ.get("HY_K_ATTENTION_AUDIT_QUERY_SAMPLES", "8")),
        int(query.shape[2]),
    )
    seed = int(os.environ.get("HY_K_ATTENTION_AUDIT_SEED", "20260722"))
    generator = torch.Generator(device="cpu").manual_seed(
        seed + int(block_idx) * 1009 + audit_key[0] * 97 + audit_key[1] * 17
    )
    query_idx = torch.randperm(int(query.shape[2]), generator=generator)[:sample_count]
    query_idx = query_idx.sort().values.to(query.device)
    q_sample = query[:, :, query_idx, :].float()
    logits = torch.matmul(q_sample, key.float().transpose(-2, -1)) / math.sqrt(query.shape[-1])
    probs = torch.softmax(logits, dim=-1)
    readout = torch.matmul(probs, value.float())

    if pass_name == "clean":
        # Kept only until the adjacent student block call.
        _K_ATTENTION_AUDIT_TARGETS[audit_key] = {
            "query": q_sample.detach(),
            "logits": logits.detach(),
            "probs": probs.detach(),
            "readout": readout.detach(),
            "query_indices": query_idx.detach().cpu().tolist(),
            "query_length": int(query.shape[2]),
            "key_length": int(key.shape[2]),
            "batch_branches": int(query.shape[0]),
        }
        return

    target = _K_ATTENTION_AUDIT_TARGETS.pop(audit_key, None)
    if target is None:
        raise RuntimeError(f"Missing clean attention audit target for {audit_key}")
    ref_q = target["query"].to(q_sample.device)
    ref_logits = target["logits"].to(logits.device)
    ref_probs = target["probs"].to(probs.device)
    ref_readout = target["readout"].to(readout.device)
    eps = 1e-12

    def branch_metrics(start: int, end: int):
        lp, lr = logits[start:end], ref_logits[start:end]
        pp, pr = probs[start:end], ref_probs[start:end]
        rp, rr = readout[start:end], ref_readout[start:end]
        qp, qr = q_sample[start:end], ref_q[start:end]
        topk = min(int(os.environ.get("HY_K_ATTENTION_AUDIT_TOPK", "16")), lp.shape[-1])
        pred_top = lp.topk(topk, dim=-1).indices
        ref_top = lr.topk(topk, dim=-1).indices
        overlap = (pred_top.unsqueeze(-1) == ref_top.unsqueeze(-2)).any(dim=-1).float().mean()
        kl = (pr * ((pr + eps).log() - (pp + eps).log())).sum(dim=-1).mean()
        return {
            "query_rel_l2": _audit_rel_l2(qp, qr),
            "query_max_abs": float((qp - qr).abs().max().cpu()),
            "logits_rel_l2": _audit_rel_l2(lp, lr),
            "logits_cosine": float(F.cosine_similarity(lp.flatten(), lr.flatten(), dim=0).cpu()),
            "prob_rel_l2": _audit_rel_l2(pp, pr),
            "clean_to_student_kl": float(kl.cpu()),
            "topk_overlap": float(overlap.cpu()),
            "readout_rel_l2": _audit_rel_l2(rp, rr),
            "readout_cosine": float(F.cosine_similarity(rp.flatten(), rr.flatten(), dim=0).cpu()),
        }

    branch_count = int(query.shape[0])
    split = branch_count // 2
    metrics = {
        "query_samples": int(sample_count),
        "query_indices": target["query_indices"],
        "query_length": target["query_length"],
        "key_length": target["key_length"],
        "all": branch_metrics(0, branch_count),
    }
    if branch_count % 2 == 0:
        metrics["rope"] = branch_metrics(0, split)
        metrics["prope"] = branch_metrics(split, branch_count)
    _K_ATTENTION_AUDIT_METRICS[audit_key] = metrics


def reset_functional_readout_losses() -> None:
    _FUNCTIONAL_READOUT_TARGETS.clear()
    _FUNCTIONAL_READOUT_LOSSES.clear()
    _FUNCTIONAL_READOUT_METRICS.clear()


def pop_functional_readout_loss(preserve_targets: bool = False):
    if not _FUNCTIONAL_READOUT_LOSSES:
        return None, []
    loss = torch.stack(_FUNCTIONAL_READOUT_LOSSES).mean()
    metrics = list(_FUNCTIONAL_READOUT_METRICS)
    _FUNCTIONAL_READOUT_LOSSES.clear()
    _FUNCTIONAL_READOUT_METRICS.clear()
    if not preserve_targets:
        _FUNCTIONAL_READOUT_TARGETS.clear()
    return loss, metrics


def _readout_fixed_indices(length: int, count: int, seed: int, device) -> torch.Tensor:
    count = min(int(count), int(length))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(length), generator=generator)[:count].sort().values.to(device)


def _record_functional_readout(layer_idx: int, output: torch.Tensor) -> None:
    if not _env_enabled("HY_FUNCTIONAL_BLOCK_FT_READOUT_ENABLE", False):
        return
    if os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "denoise") != "denoise":
        return
    layers = {
        int(item.strip())
        for item in os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "13,40,53").split(",")
        if item.strip()
    }
    if int(layer_idx) not in layers:
        return
    mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "")
    if mode not in {"teacher", "student"}:
        return
    chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
    step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
    if chunk < 0 or step < 0:
        return
    token_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_READOUT_TOKEN_SAMPLES", "256"))
    channel_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_READOUT_CHANNEL_SAMPLES", "256"))
    seed = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_SEED", "2027"))
    token_idx = _readout_fixed_indices(
        output.shape[1], token_count, seed + int(layer_idx) * 3001, output.device
    )
    channel_idx = _readout_fixed_indices(
        output.shape[2], channel_count, seed + int(layer_idx) * 4001 + 29, output.device
    )
    sketch = output[:, token_idx][:, :, channel_idx]
    key = (chunk, step, int(layer_idx))
    if mode == "teacher":
        _FUNCTIONAL_READOUT_TARGETS[key] = sketch.detach()
        return
    target = _FUNCTIONAL_READOUT_TARGETS.get(key)
    if target is None:
        return
    pred = sketch.float()
    ref = target.to(device=sketch.device, dtype=sketch.dtype).float()
    rel_l2 = (pred - ref).norm() / ref.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]),
        ref.reshape(-1, ref.shape[-1]),
        dim=-1,
    ).mean()
    loss = rel_l2 + 0.1 * (1.0 - cosine)
    _FUNCTIONAL_READOUT_LOSSES.append(loss)
    _FUNCTIONAL_READOUT_METRICS.append(
        {
            "chunk": chunk,
            "step": step,
            "layer": int(layer_idx),
            "readout_rel_l2": float(rel_l2.detach().cpu()),
            "readout_cosine": float(cosine.detach().cpu()),
        }
    )


def _position_profile_enabled():
    import os

    return os.environ.get("HY_POSITION_ENCODING_PROFILE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _record_profile(bucket, block_idx, elapsed_ms):
    try:
        infer_state = get_infer_state()
        if not hasattr(infer_state, "position_encoding_profile"):
            infer_state.position_encoding_profile = {}
        stats = infer_state.position_encoding_profile.setdefault(
            f"{bucket}:layer_{block_idx}", {"count": 0, "total_ms": 0.0}
        )
        stats["count"] += 1
        stats["total_ms"] += float(elapsed_ms)
    except Exception:
        pass


def _profile_cuda_call(bucket, block_idx, fn, *args, **kwargs):
    return position_profile.cuda_call(bucket, block_idx, fn, *args, **kwargs)


def _profile_cuda_region(bucket, block_idx):
    class _Region:
        def __enter__(self_inner):
            if not position_profile.enabled():
                self_inner.start = None
                return self_inner
            torch.cuda.synchronize()
            self_inner.start = torch.cuda.Event(enable_timing=True)
            self_inner.end = torch.cuda.Event(enable_timing=True)
            self_inner.start.record()
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            if self_inner.start is None:
                return False
            self_inner.end.record()
            torch.cuda.synchronize()
            position_profile.record(
                bucket, block_idx, self_inner.start.elapsed_time(self_inner.end)
            )
            return False

    return _Region()


def _env_enabled(name, default=False):
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _predecode_next_enabled():
    return _env_enabled("HY_LAYERWISE_PREDECODE_NEXT", False)


def _maybe_take_predecoded_kv(layer_cache, dtype, device):
    pending = layer_cache.pop("_layerwise_predecoded_kv", None)
    if not pending:
        return None
    if pending.get("dtype") != dtype or pending.get("device") != device:
        return None
    event = pending.get("event")
    if event is not None and device.type == "cuda":
        torch.cuda.current_stream(device).wait_event(event)
    return pending["k"], pending["v"]


def _schedule_predecode_for_layer(layer_cache, dtype, device):
    if not _predecode_next_enabled() or device.type != "cuda":
        return
    if layer_cache.get("_layerwise_predecoded_kv") is not None:
        return
    k_obj = layer_cache.get("k_vision")
    v_obj = layer_cache.get("v_vision")
    if not (
        isinstance(k_obj, dict)
        and k_obj.get("layerwise_ae")
        and isinstance(v_obj, dict)
        and v_obj.get("layerwise_ae")
    ):
        return
    from hyvideo.models.transformers.modules.kv_codec_training import maybe_decompress

    stream = layer_cache.get("_layerwise_predecode_stream")
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        layer_cache["_layerwise_predecode_stream"] = stream
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        k_pre = maybe_decompress(k_obj, dtype, device)
        v_pre = maybe_decompress(v_obj, dtype, device)
        event.record(stream)
    layer_cache["_layerwise_predecoded_kv"] = {
        "k": k_pre,
        "v": v_pre,
        "event": event,
        "dtype": dtype,
        "device": device,
    }


def _schedule_next_layer_predecode(kv_cache, block_idx, dtype, device):
    if not _predecode_next_enabled() or kv_cache is None:
        return
    next_idx = int(block_idx) + 1 if block_idx is not None else None
    if next_idx is None or next_idx >= len(kv_cache):
        return
    next_cache = kv_cache[next_idx]
    if next_cache.get("vision_kv_position_mode") != "pre":
        return
    _schedule_predecode_for_layer(next_cache, dtype, device)


def _get_cuda_freqs_cis(freqs_cis, device):
    if not isinstance(freqs_cis, tuple):
        return freqs_cis
    key = (
        id(freqs_cis[0]),
        id(freqs_cis[1]),
        device,
        tuple(freqs_cis[0].shape),
        tuple(freqs_cis[1].shape),
        freqs_cis[0].dtype,
        freqs_cis[1].dtype,
    )
    cached = _FREQS_CIS_CUDA_CACHE.get(key)
    if cached is None:
        cached = (freqs_cis[0].to(device), freqs_cis[1].to(device))
        _FREQS_CIS_CUDA_CACHE[key] = cached
    return cached


def _restore_pre_position_vision_kv(layer_cache, return_branches=False):
    from hyvideo.models.transformers.modules.posemb_layers import (
        apply_rotary_emb,
        apply_rotary_emb_one,
    )
    from hyvideo.models.transformers.modules.kv_codec_training import maybe_decompress
    from hyvideo.prope.camera_rope import (
        _prepare_apply_fns_all_dim,
        prope_qkv,
    )

    restore_dtype = layer_cache["v_txt"].dtype
    restore_device = layer_cache["v_txt"].device
    from hyvideo.models.transformers.modules.jensen_kv_adapter import (
        decompress_pair as decompress_jensen_pair,
    )
    from hyvideo.models.transformers.modules.kvtc_adapter import decompress_pair as decompress_kvtc_pair
    from hyvideo.models.transformers.modules.qvg_adapter import (
        maybe_decompress as maybe_decompress_qvg,
    )
    from hyvideo.models.transformers.modules.turboquant_adapter import (
        decompress_pair as decompress_turboquant_pair,
    )

    turboquant_pair = decompress_turboquant_pair(
        layer_cache.get("k_vision"), layer_cache.get("v_vision"),
        restore_dtype, restore_device,
    )
    jensen_pair = None if turboquant_pair is not None else decompress_jensen_pair(
        layer_cache.get("k_vision"), layer_cache.get("v_vision"),
        restore_dtype, restore_device,
    )
    kvtc_pair = None if turboquant_pair is not None or jensen_pair is not None else decompress_kvtc_pair(
        layer_cache.get("k_vision"), layer_cache.get("v_vision"), restore_dtype, restore_device
    )
    qvg_key, qvg_value = maybe_decompress_qvg(
        layer_cache.get("k_vision"), layer_cache.get("v_vision")
    ) if turboquant_pair is None and jensen_pair is None and kvtc_pair is None else (None, None)
    predecoded = (
        _maybe_take_predecoded_kv(layer_cache, restore_dtype, restore_device)
        if turboquant_pair is None and jensen_pair is None and kvtc_pair is None
        else None
    )
    if turboquant_pair is not None:
        key_pre, value_pre = turboquant_pair
    elif jensen_pair is not None:
        key_pre, value_pre = jensen_pair
    elif kvtc_pair is not None:
        key_pre, value_pre = kvtc_pair
    elif predecoded is not None:
        key_pre, value_pre = predecoded
    elif qvg_key is not layer_cache.get("k_vision"):
        key_pre = qvg_key.to(device=restore_device, dtype=restore_dtype)
        value_pre = qvg_value.to(device=restore_device, dtype=restore_dtype)
    else:
        key_pre = maybe_decompress(layer_cache["k_vision"], restore_dtype, restore_device)
        value_pre = maybe_decompress(layer_cache["v_vision"], restore_dtype, restore_device)
    freqs_cis = layer_cache.get("vision_freqs_cis")
    viewmats = layer_cache.get("vision_viewmats")
    Ks = layer_cache.get("vision_Ks")
    if freqs_cis is None or viewmats is None:
        raise RuntimeError(
            "pre-position vision KV cache requires vision_freqs_cis and vision_viewmats"
        )

    block_idx = layer_cache.get("vision_block_idx")
    if (
        _env_enabled("HY_PRE_POSITION_ROPE_FREQ_CACHE", True)
        and
        isinstance(freqs_cis, tuple)
        and (not freqs_cis[0].is_cuda or freqs_cis[0].device != key_pre.device)
    ):
        freqs_cis = _get_cuda_freqs_cis(freqs_cis, key_pre.device)
    with _profile_cuda_region("restore_transpose_pre", block_idx):
        key_pre_bshd = key_pre.transpose(1, 2)
    if _env_enabled("HY_PRE_POSITION_RESTORE_SINGLE_ROPE", True):
        key_rope_bshd = _profile_cuda_call(
            "restore_rope",
            block_idx,
            apply_rotary_emb_one,
            key_pre_bshd,
            freqs_cis,
            head_first=False,
        )
    else:
        _, key_rope_bshd = _profile_cuda_call(
            "restore_rope",
            block_idx,
            apply_rotary_emb,
            key_pre_bshd,
            key_pre_bshd,
            freqs_cis,
            head_first=False,
        )
    with _profile_cuda_region("restore_rope_transpose", block_idx):
        key_rope = key_rope_bshd.transpose(1, 2).contiguous()
        value_rope = value_pre

    if _env_enabled("HY_PRE_POSITION_RESTORE_PROPE_KV_ONLY", True):
        use_prepare_cache = _env_enabled("HY_PRE_POSITION_PROPE_PREPARE_CACHE", True)
        apply_fn_kv = layer_cache.get("_vision_prope_apply_fn_kv")
        cache_key = layer_cache.get("_vision_prope_apply_fn_kv_key")
        current_key = (
            key_pre.shape[-1],
            id(viewmats),
            tuple(viewmats.shape),
            viewmats.data_ptr(),
            id(Ks) if Ks is not None else None,
            tuple(Ks.shape) if Ks is not None else None,
            Ks.data_ptr() if Ks is not None else None,
        )
        if not use_prepare_cache or apply_fn_kv is None or cache_key != current_key:
            _, apply_fn_kv, _ = _profile_cuda_call(
                "restore_prope_prepare",
                block_idx,
                _prepare_apply_fns_all_dim,
                head_dim=key_pre.shape[-1],
                viewmats=viewmats,
                Ks=Ks,
                patches_x=None,
                patches_y=None,
                image_width=None,
                image_height=None,
            )
            if use_prepare_cache:
                layer_cache["_vision_prope_apply_fn_kv"] = apply_fn_kv
                layer_cache["_vision_prope_apply_fn_kv_key"] = current_key
        if _env_enabled("HY_PRE_POSITION_PROPE_JOINT_KV", False):
            with _profile_cuda_region("restore_prope_apply_joint_kv", block_idx):
                prope_kv = apply_fn_kv(torch.cat([key_pre, value_pre], dim=1))
                key_prope, value_prope = torch.split(
                    prope_kv, [key_pre.shape[1], value_pre.shape[1]], dim=1
                )
        else:
            key_prope = _profile_cuda_call(
                "restore_prope_apply_k",
                block_idx,
                apply_fn_kv,
                key_pre,
            )
            value_prope = _profile_cuda_call(
                "restore_prope_apply_v",
                block_idx,
                apply_fn_kv,
                value_pre,
            )
    else:
        _, key_prope, value_prope, _ = _profile_cuda_call(
            "restore_prope_qkv",
            block_idx,
            prope_qkv,
            key_pre,
            key_pre,
            value_pre,
            viewmats=viewmats,
            Ks=Ks,
        )

    if return_branches:
        return (key_rope, key_prope), (value_rope, value_prope)

    with _profile_cuda_region("restore_cat", block_idx):
        restored_key = torch.cat([key_rope, key_prope], dim=0)
        restored_value = torch.cat([value_rope, value_prope], dim=0)
    if (
        _env_enabled("HY_FUNCTIONAL_JOINT_PATH_AUDIT_INTERMEDIATES")
        and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "") == "student"
    ):
        layer_cache["_joint_audit_key_pre"] = key_pre
        layer_cache["_joint_audit_key_rope"] = key_rope
        layer_cache["_joint_audit_key_prope"] = key_prope
        layer_cache["_joint_audit_restored_key"] = restored_key
        layer_cache["_joint_audit_value_pre"] = value_pre
        layer_cache["_joint_audit_restored_value"] = restored_value
    return restored_key, restored_value


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)  # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=drop_rate,
            is_causal=causal,
        )

        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(
            qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None
        )
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def sequence_parallel_attention_txt(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    kv_cache=None,
    cache_txt=False,
):
    encoder_query = q
    encoder_key = k
    encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_group = parallel_dims.sp_group
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

    if enable_sp:

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    encoder_query = encoder_query.transpose(1, 2)
    encoder_key = encoder_key.transpose(1, 2)
    encoder_value = encoder_value.transpose(1, 2)
    t_kv = {}
    if cache_txt:
        t_kv["k_txt"] = encoder_key
        t_kv["v_txt"] = encoder_value

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state.enable_sageattn and block_idx in infer_state.sage_blocks_range
    )
    if enable_sageattn:
        from sageattention import sageattn

        encoder_hidden_states = sageattn(
            encoder_query,
            encoder_key,
            encoder_value,
            tensor_layout="HND",
            is_causal=False,
        )
    else:
        encoder_hidden_states = F.scaled_dot_product_attention(
            encoder_query, encoder_key, encoder_value, dropout_p=0.0, is_causal=False
        )

    # transpose back
    encoder_hidden_states = encoder_hidden_states.transpose(1, 2)  # [B, S, H, D]

    if enable_sp:
        encoder_hidden_states = all_gather(
            encoder_hidden_states, dim=2, group=sp_group
        ).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(q.dtype)

    b, s, a, d = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(b, s, -1)

    return encoder_hidden_states, t_kv


@torch.compiler.disable
def sequence_parallel_attention_vision(
    q,
    k,
    v,
    block_idx=None,
    kv_cache=None,
    cache_vision=False,
    attn_param=None,
    vision_thw=None,
):
    assert kv_cache is not None
    query, query_prope = q
    key, key_prope = k
    value, value_prope = v

    with _profile_cuda_region("vision_branch_cat", block_idx):
        query = torch.cat([query, query_prope], dim=0)
        key = torch.cat([key, key_prope], dim=0)
        value = torch.cat([value, value_prope], dim=0)

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_group = parallel_dims.sp_group

    if enable_sp:
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, sp_group, scatter_dim=2, gather_dim=1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    layer_cache = kv_cache[block_idx]
    cache_vision_key = layer_cache["k_vision"]  # previous key
    cache_vision_value = layer_cache["v_vision"]  # previous value
    jensen_active = bool(layer_cache.get("jensen_quantized", False))
    jensen_history_tokens = 0
    jensen_correction = jensen_active and _env_enabled(
        "HY_JENSEN_TAYLOR_CORRECTION", True
    )
    if jensen_active:
        shape = cache_vision_key.get("shape") if isinstance(cache_vision_key, dict) else None
        if shape is None:
            raise RuntimeError("Jensen cache is missing its packed K shape")
        jensen_history_tokens = int(shape[2])
        if layer_cache.get("vision_kv_position_mode") == "post":
            from hyvideo.models.transformers.modules.jensen_kv_adapter import (
                decompress_pair as decompress_jensen_pair,
            )

            restored_pair = decompress_jensen_pair(
                cache_vision_key,
                cache_vision_value,
                query.dtype,
                query.device,
            )
            if restored_pair is None:
                raise RuntimeError("failed to restore Jensen post-position cache")
            cache_vision_key, cache_vision_value = restored_pair

    vision_kv = {}
    if cache_vision:
        vision_kv["k_vision"] = key
        vision_kv["v_vision"] = value

    sparse_params = dict(attn_param or {})
    sparse_enabled = bool(sparse_params.get("sparse_decode_enabled", False))
    sparse_enabled = sparse_enabled or _env_enabled("HY_SPARSE_DECODE_ENABLE")
    sparse_layers = {
        int(item)
        for item in os.environ.get("HY_SPARSE_DECODE_LAYERS", "").split(",")
        if item.strip()
    }
    if sparse_layers and int(block_idx) not in sparse_layers:
        sparse_enabled = False
    if os.environ.get("HY_SPARSE_TOPK_RATIO"):
        sparse_params["sparse_topk_ratio"] = float(
            os.environ["HY_SPARSE_TOPK_RATIO"]
        )
    if os.environ.get("HY_SPARSE_RECENT_PROTECT"):
        sparse_params["sparse_recent_protect"] = int(
            os.environ["HY_SPARSE_RECENT_PROTECT"]
        )
    if os.environ.get("HY_SPARSE_SIM_METRIC"):
        sparse_params["sparse_sim_metric"] = os.environ["HY_SPARSE_SIM_METRIC"]
    if (
        sparse_enabled
        and not cache_vision
        and cache_vision_key is not None
        and layer_cache.get("vision_kv_position_mode") == "pre"
    ):
        if not bool(sparse_params.get("sparse_compact_selected_kv", True)):
            raise NotImplementedError(
                "selective pre-KV decode requires compact selected K/V"
            )
        required = (
            "k_proxy_rope",
            "k_proxy_prope",
            "sparse_proxy_tile_size",
            "vision_thw",
            "vision_freqs_cis",
            "vision_viewmats",
        )
        missing = [name for name in required if name not in layer_cache]
        if missing:
            raise RuntimeError(f"sparse pre-KV cache is missing {missing}")
        if enable_sp:
            raise NotImplementedError(
                "selective pre-KV decode does not yet support sequence parallelism"
            )
        if vision_thw is None:
            raise RuntimeError("sparse vision attention requires current vision_thw")
        from hyvideo.models.transformers.modules.ar_sparse_operation import (
            SparseBlockLayout,
            _write_sparse_stats,
            generate_oracle_visual_indices,
            sparse_pre_position_attention,
            tile_visual_tokens,
        )

        encoder_key = repeat(
            layer_cache["k_txt"], "B H S D->(B R) H S D", R=2
        )
        encoder_value = repeat(
            layer_cache["v_txt"], "B H S D->(B R) H S D", R=2
        )
        dense_history_k = None
        dense_history_v = None
        if bool(sparse_params.get("sparse_oracle_indices", False)):
            dense_history_k, dense_history_v = _restore_pre_position_vision_kv(
                layer_cache
            )
            tile_size = tuple(
                int(value)
                for value in sparse_params.get(
                    "sparse_tile_size", (4, 8, 4)
                )
            )
            history_layout = SparseBlockLayout.build(
                tuple(int(value) for value in layer_cache["vision_thw"]),
                tile_size,
                device=query.device,
            )
            current_layout = SparseBlockLayout.build(
                tuple(int(value) for value in vision_thw),
                tile_size,
                device=query.device,
            )
            q_tiled = tile_visual_tokens(query, current_layout)
            history_tiled = tile_visual_tokens(
                dense_history_k, history_layout
            )
            current_tiled = tile_visual_tokens(key, current_layout)
            visual_tiled = torch.cat(
                [history_tiled, current_tiled], dim=2
            )
            virtual_batch = query.shape[0]
            visual_valid = torch.cat(
                [
                    history_layout.block_valid_mask[None].expand(
                        virtual_batch, -1, -1
                    ),
                    current_layout.block_valid_mask[None].expand(
                        virtual_batch, -1, -1
                    ),
                ],
                dim=1,
            )
            current_ids = torch.arange(
                history_layout.num_blocks,
                history_layout.num_blocks + current_layout.num_blocks,
                device=query.device,
            )
            recent = min(
                int(sparse_params.get("sparse_recent_protect", 0)),
                history_layout.num_blocks,
            )
            if recent:
                recent_ids = torch.arange(
                    history_layout.num_blocks - recent,
                    history_layout.num_blocks,
                    device=query.device,
                )
                mandatory_ids = torch.cat([recent_ids, current_ids])
            else:
                mandatory_ids = current_ids
            oracle_indices, oracle_counts, _ = generate_oracle_visual_indices(
                q_tiled,
                encoder_key,
                visual_tiled,
                visual_valid,
                float(sparse_params.get("sparse_topk_ratio", 1.0)),
                mandatory_ids,
            )
            sparse_params["_indices_override"] = oracle_indices
            sparse_params["_counts_override"] = oracle_counts
        hidden_states = sparse_pre_position_attention(
            query,
            key,
            value,
            encoder_key,
            encoder_value,
            layer_cache,
            tuple(int(value) for value in vision_thw),
            sparse_params,
            int(block_idx),
        )
        if bool(sparse_params.get("sparse_debug_compare_dense", False)) or _env_enabled(
            "HY_SPARSE_DECODE_DEBUG"
        ):
            if dense_history_k is None:
                dense_history_k, dense_history_v = _restore_pre_position_vision_kv(
                    layer_cache
                )
            dense_k = torch.cat(
                [encoder_key, dense_history_k, key], dim=2
            )
            dense_v = torch.cat(
                [encoder_value, dense_history_v, value], dim=2
            )
            dense_output = F.scaled_dot_product_attention(
                query,
                dense_k,
                dense_v,
                dropout_p=0.0,
                is_causal=False,
            )
            sparse_dense_rel_l2 = float(
                (hidden_states.float() - dense_output.float()).norm()
                / dense_output.float().norm().clamp_min(1e-12)
            )
            _write_sparse_stats(
                {
                    "layer": int(block_idx),
                    "debug_sparse_vs_dense_rel_l2": sparse_dense_rel_l2,
                    "debug_topk_ratio": float(
                        sparse_params.get("sparse_topk_ratio", 1.0)
                    ),
                }
            )
            print(
                f"[SparseDecodeDebug] layer={block_idx} "
                f"sparse_vs_dense_rel_l2={sparse_dense_rel_l2:.6e}",
                flush=True,
            )
        hidden_states = hidden_states.transpose(1, 2)
        b, s, a, d = hidden_states.shape
        hidden_states = hidden_states.reshape(b, s, -1)
        hidden_states, hidden_states_prope = torch.chunk(
            hidden_states, chunks=2, dim=0
        )
        return hidden_states, hidden_states_prope, vision_kv

    restored_cache_key = None
    restored_cache_value = None
    restored_cache_branches = None
    if not cache_vision and cache_vision_key is not None:
        if layer_cache.get("vision_kv_position_mode") == "pre":
            direct_final_cat = _env_enabled(
                "HY_PRE_POSITION_DIRECT_FINAL_CAT", False
            )
            if direct_final_cat:
                restored_cache_branches = _restore_pre_position_vision_kv(
                    layer_cache, return_branches=True
                )
                cache_vision_key = restored_cache_branches[0][0]
                cache_vision_value = restored_cache_branches[1][0]
            else:
                cache_vision_key, cache_vision_value = _restore_pre_position_vision_kv(
                    layer_cache
                )
            if (
                os.environ.get("HY_FUNCTIONAL_BLOCK_FT_DEBUG_GRAD", "0").lower()
                in {"1", "true", "yes", "on"}
                and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "") == "student"
            ):
                selected = {
                    int(item.strip())
                    for item in os.environ.get(
                        "HY_FUNCTIONAL_BLOCK_FT_LAYERS", ""
                    ).split(",")
                    if item.strip()
                }
                if int(block_idx) in selected:
                    raw_v = layer_cache.get("v_vision")
                    print(
                        "[FunctionalAttentionGradDebug] "
                        f"layer={block_idx} raw_v_req={getattr(raw_v, 'requires_grad', None)} "
                        f"restored_v_req={cache_vision_value.requires_grad} "
                        f"restored_v_grad_fn={type(cache_vision_value.grad_fn).__name__ if cache_vision_value.grad_fn is not None else None}",
                        flush=True,
                    )
            _schedule_next_layer_predecode(
                kv_cache, block_idx, cache_vision_key.dtype, cache_vision_key.device
            )
        if restored_cache_branches is not None:
            pass
        elif _env_enabled("HY_KV_EFF_OPT_FUSED_CACHE_CAT", False):
            restored_cache_key = cache_vision_key
            restored_cache_value = cache_vision_value
        else:
            with _profile_cuda_region("decode_cache_current_cat", block_idx):
                key = torch.cat([cache_vision_key, key], dim=2)
                value = torch.cat([cache_vision_value, value], dim=2)

    with _profile_cuda_region("text_kv_repeat", block_idx):
        encoder_key = kv_cache[block_idx]["k_txt"]
        encoder_value = kv_cache[block_idx]["v_txt"]
        encoder_key = repeat(encoder_key, "B H S D->(B R) H S D", R=2)
        encoder_value = repeat(encoder_value, "B H S D->(B R) H S D", R=2)

    if restored_cache_branches is not None:
        (history_key_main, history_key_prope), (
            history_value_main,
            history_value_prope,
        ) = restored_cache_branches
        batch = history_key_main.shape[0]
        total_sequence = (
            encoder_key.shape[2] + history_key_main.shape[2] + key.shape[2]
        )
        final_shape = (
            batch * 2,
            key.shape[1],
            total_sequence,
            key.shape[3],
        )
        with _profile_cuda_region("direct_final_cat", block_idx):
            if _env_enabled("HY_PRE_POSITION_TRITON_FINAL_CAT", False):
                from .kv_position_triton import concat_final_kv
                final_key, final_value = concat_final_kv(
                    encoder_key, encoder_value,
                    history_key_main, history_value_main,
                    history_key_prope, history_value_prope,
                    key, value,
                )
            else:
                final_key = torch.empty(
                    final_shape, device=key.device, dtype=key.dtype
                )
                final_value = torch.empty(
                    final_shape, device=value.device, dtype=value.dtype
                )
                torch.cat(
                    [encoder_key[:batch], history_key_main, key[:batch]],
                    dim=2,
                    out=final_key[:batch],
                )
                torch.cat(
                    [encoder_key[batch:], history_key_prope, key[batch:]],
                    dim=2,
                    out=final_key[batch:],
                )
                torch.cat(
                    [encoder_value[:batch], history_value_main, value[:batch]],
                    dim=2,
                    out=final_value[:batch],
                )
                torch.cat(
                    [encoder_value[batch:], history_value_prope, value[batch:]],
                    dim=2,
                    out=final_value[batch:],
                )
            key, value = final_key, final_value
    elif restored_cache_key is not None:
        with _profile_cuda_region("fused_text_cache_current_cat", block_idx):
            key = torch.cat([encoder_key, restored_cache_key, key], dim=2)
            value = torch.cat([encoder_value, restored_cache_value, value], dim=2)
    else:
        with _profile_cuda_region("text_vision_cat", block_idx):
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

    if (
        _env_enabled("HY_FUNCTIONAL_JOINT_PATH_AUDIT_INTERMEDIATES")
        and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "") == "student"
    ):
        layer_cache["_joint_audit_sdpa_key"] = key
        layer_cache["_joint_audit_sdpa_value"] = value

    _record_k_attention_audit(block_idx, query, key, value)

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state.enable_sageattn and block_idx in infer_state.sage_blocks_range
    )
    if jensen_correction:
        if enable_sageattn:
            raise RuntimeError("Jensen Taylor baseline requires FlexAttention, not SageAttention")
        from hyvideo.models.transformers.modules.jensen_attention import (
            corrected_flex_attention,
        )

        hidden_states = _profile_cuda_call(
            "jensen_flex_vision",
            block_idx,
            corrected_flex_attention,
            query,
            key,
            value,
            layer_cache=layer_cache,
            text_tokens=int(encoder_key.shape[2]),
            history_tokens=jensen_history_tokens,
        )
    elif enable_sageattn:
        from sageattention import sageattn

        hidden_states = _profile_cuda_call(
            "sdpa_vision",
            block_idx,
            sageattn,
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=False,
        )
    else:
        hidden_states = _profile_cuda_call(
            "sdpa_vision",
            block_idx,
            F.scaled_dot_product_attention,
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

    # transpose back
    hidden_states = hidden_states.transpose(1, 2)  # [B, S, H, D]

    if enable_sp:
        hidden_states = all_to_all_4D(
            hidden_states, sp_group, scatter_dim=1, gather_dim=2
        )
        hidden_states = hidden_states.to(query.dtype)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)
    if (
        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_DEBUG_GRAD", "0").lower()
        in {"1", "true", "yes", "on"}
        and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "") == "student"
    ):
        selected = {
            int(item.strip())
            for item in os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "").split(",")
            if item.strip()
        }
        if int(block_idx) in selected:
            print(
                "[FunctionalAttentionGradDebug] "
                f"layer={block_idx} attn_out_req={hidden_states.requires_grad} "
                f"attn_out_grad_fn={type(hidden_states.grad_fn).__name__ if hidden_states.grad_fn is not None else None}",
                flush=True,
            )
    _record_functional_readout(block_idx, hidden_states)
    hidden_states, hidden_states_prope = torch.chunk(hidden_states, chunks=2, dim=0)

    return hidden_states, hidden_states_prope, vision_kv


@torch.compiler.disable
def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
):
    return sequence_parallel_attention(
        q,
        k,
        v,
        img_q_len,
        img_kv_len,
        attn_mode,
        text_mask,
        attn_param=attn_param,
        block_idx=block_idx,
    )


def sequence_parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_group = parallel_dims.sp_group
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

    if enable_sp:
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, sp_group, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = maybe_fallback_attn_mode(attn_mode, get_infer_state(), block_idx)

    if attn_mode == "sageattn":
        from sageattention import sageattn

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(
            query, key, value, tensor_layout="NHD", is_causal=False
        )
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )

        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)  # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim
        if attn_mask is not None:
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    # add new attn for chunk-wise attn
    elif attn_mode == "torch_causal":  # now: we set text_mask = None
        # attention: here we concat the encoder text sequence first, then apply causal attention
        vision_seq_length = query.shape[1]
        text_seq_length = encoder_query.shape[1]
        total_seq_length = vision_seq_length + text_seq_length

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        # prepare causal mask for chunk-wise attention
        latent_seq_length = attn_param["thw"][-1] * attn_param["thw"][-2]
        chunk_seq_length = latent_seq_length * 4
        chunk_num = (vision_seq_length) // chunk_seq_length
        causal_mask = torch.zeros(
            (total_seq_length, total_seq_length), device=query.device
        )
        causal_mask[:, :text_seq_length] = 1  # no attention for the rest
        for i in range(chunk_num):
            start_i = text_seq_length + i * chunk_seq_length
            end_i = min(start_i + chunk_seq_length, total_seq_length)
            for j in range(i + 1):
                start_j = text_seq_length + j * chunk_seq_length
                end_j = min(start_j + chunk_seq_length, total_seq_length)
                # full attention within chunk i for j == i, causal for j < i
                causal_mask[start_i:end_i, start_j:end_j] = 1

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # 1, 1, S, S
        causal_mask = causal_mask.expand(query.shape[0], 1, -1, -1)
        causal_mask = causal_mask.to(torch.bool)  # Force bool dtype

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask, dropout_p=0.0, is_causal=False
        )

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)  # [B, S, H, D]

        # return back to the original order: [query, encoder_query]
        hidden_states, encoder_hidden_states = (
            hidden_states[:, text_seq_length:],
            hidden_states[:, :text_seq_length],
        )
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(
            qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
        )

    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(
            qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
        )

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(
                    f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]"
                )
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31:  # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(f"Unsupported attention mode: {attn_mode}.")

    if enable_sp:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
        hidden_states = all_to_all_4D(
            hidden_states, sp_group, scatter_dim=1, gather_dim=2
        )
        encoder_hidden_states = all_gather(
            encoder_hidden_states, dim=2, group=sp_group
        ).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
