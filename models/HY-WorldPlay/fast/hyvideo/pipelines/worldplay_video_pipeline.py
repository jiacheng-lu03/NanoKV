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


import copy
import hashlib
import inspect
import gc
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import loguru

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from torch import distributed as dist
from einops import repeat

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging
from diffusers.utils.torch_utils import randn_tensor

from hyvideo.commons import (
    PIPELINE_CONFIGS,
    SR_PIPELINE_CONFIGS,
    TRANSFORMER_VERSION_TO_SR_VERSION,
    auto_offload_model,
    get_gpu_memory,
    get_rank,
    is_flash3_available,
    is_sparse_attn_supported,
)
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.commons.infer_state import get_infer_state

from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat

from hyvideo.models.transformers.worldplay_1_5_transformer import (
    _FUNCTIONAL_BLOCK_TARGETS,
    HunyuanVideo_1_5_DiffusionTransformer,
    get_functional_block_loss_components,
    pop_functional_block_loss,
    reset_functional_block_losses,
)
from hyvideo.models.transformers.modules.upsample import (
    SRTo720pUpsampler,
    SRTo1080pUpsampler,
)
from hyvideo.models.transformers.modules.attention import (
    pop_functional_readout_loss,
    reset_functional_readout_losses,
)
from hyvideo.models.transformers.modules.codec_native_routing_distill import (
    pop_loss as pop_codec_native_routing_loss,
    reset as reset_codec_native_routing_losses,
)
from hyvideo.models.transformers.modules.kv_ae3d import write_summary as write_kv_ae3d_summary
from hyvideo.models.transformers.modules.qvg_adapter import write_summary as write_qvg_summary
from hyvideo.models.transformers.modules.jensen_kv_adapter import (
    write_summary as write_jensen_summary,
)
from hyvideo.models.transformers.modules.turboquant_adapter import (
    write_summary as write_turboquant_summary,
)
from hyvideo.models.transformers.modules.kv_ae3d import (
    backward_k_functional_audit,
    detach_k_functional_cache_for_rollout,
    detach_v_functional_cache_for_rollout,
    k_functional_anchor_loss_for_cache,
    joint_functional_trust_anchor_loss_for_cache,
    k_functional_reconstruction_losses_for_cache,
    reapply_k_corruption_for_ft,
    reapply_v_corruption_for_ft,
    reset_v_final_prediction_tensor_audit,
    run_k_pca_tail_score_audit,
    run_k_p1_gradient_audit,
    run_joint_functional_path_audit,
    run_v_final_prediction_gradient_audit,
    run_v_functional_gradient_audit,
    save_k_functional_gradient_snapshot,
    save_k_functional_train_checkpoints,
    save_v_functional_train_checkpoints,
    step_k_functional_train_optimizers,
    v_functional_anchor_loss_for_cache,
    v_functional_loss_grad_norm,
    v_functional_reconstruction_losses_for_cache,
    zero_k_functional_train_grads,
)
from hyvideo.models.transformers.modules.kv_codecs import write_summary as write_kv_codec_summary
from hyvideo.models.vision_encoder import VisionEncoder

from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from hyvideo.utils.data_utils import (
    generate_crop_size_list,
    get_closest_ratio,
    resize_and_center_crop,
)
from hyvideo.utils.multitask_utils import (
    merge_tensor_by_mask,
)
from hyvideo.utils.retrieval_context import (
    generate_points_in_sphere,
    select_aligned_memory_frames,
)


from .pipeline_utils import retrieve_timesteps, rescale_noise_cfg


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _k_t0_tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable content hash for immutable T0 atom/Teacher audit tensors."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

_ROLLOUT_CONTEXT_CORRECTOR_CACHE = {}
_ROLLOUT_RESIDUAL_BASIS_CACHE = {}


class _DirtyHistoryOutputAdapterLinear(nn.Module):
    """Frozen DiT output head plus a zero-init residual correction branch."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        adapter_type = os.environ.get(
            "HY_DIRTY_HISTORY_OUTPUT_ADAPTER_TYPE", "linear"
        ).strip().lower()
        if adapter_type == "linear":
            self.delta = nn.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            nn.init.zeros_(self.delta.weight)
            if self.delta.bias is not None:
                nn.init.zeros_(self.delta.bias)
        elif adapter_type == "mlp":
            bottleneck = int(
                os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_BOTTLENECK", "64")
            )
            if bottleneck <= 0:
                raise ValueError("Dirty-history output-adapter bottleneck must be positive")
            down = nn.Linear(
                base.in_features,
                bottleneck,
                bias=True,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            up = nn.Linear(
                bottleneck,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            nn.init.zeros_(up.weight)
            if up.bias is not None:
                nn.init.zeros_(up.bias)
            self.delta = nn.Sequential(down, nn.SiLU(), up)
        else:
            raise ValueError(
                "Unsupported HY_DIRTY_HISTORY_OUTPUT_ADAPTER_TYPE="
                f"{adapter_type!r}; expected linear or mlp"
            )
        self.adapter_type = adapter_type
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + self.delta(x) if self.enabled else out


class _DirtyHistoryLoRALinear(nn.Module):
    """Frozen linear layer with a zero-init, mergeable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("Dirty-history LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha) / float(rank)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + self.lora_b(self.lora_a(x)) * self.scale


class _DirtyHistoryDynamicsAdapter:
    """Owns LoRA updates spread across the DiT visual dynamics path."""

    def __init__(self, modules: Dict[str, _DirtyHistoryLoRALinear], optimizer):
        self.modules = modules
        self.optimizer = optimizer

    @property
    def enabled(self) -> bool:
        return all(module.enabled for module in self.modules.values())

    @enabled.setter
    def enabled(self, value: bool) -> None:
        for module in self.modules.values():
            module.enabled = bool(value)

    def parameters(self):
        for module in self.modules.values():
            yield from module.lora_a.parameters()
            yield from module.lora_b.parameters()


def _dirty_history_dynamics_adapter_enabled() -> bool:
    return _drift_env_enabled("HY_DIRTY_HISTORY_DYNAMICS_LORA_ENABLE")


def _codec_native_history_adapter_enabled() -> bool:
    from hyvideo.models.transformers.modules.codec_native_history_adapter import (
        enabled,
    )

    return enabled()


def _codec_native_history_adapter_training_enabled() -> bool:
    from hyvideo.models.transformers.modules.codec_native_history_adapter import (
        training_enabled,
    )

    return training_enabled()


def _dirty_history_any_adapter_enabled() -> bool:
    return (
        _dirty_history_output_adapter_enabled()
        or _dirty_history_dynamics_adapter_enabled()
        or _codec_native_history_adapter_enabled()
    )


def _dirty_history_trainable_adapter_enabled() -> bool:
    return (
        _dirty_history_output_adapter_enabled()
        or _dirty_history_dynamics_adapter_enabled()
        or _codec_native_history_adapter_training_enabled()
    )


def _ensure_codec_native_history_adapter(transformer):
    if not _codec_native_history_adapter_enabled():
        return None
    if _dirty_history_output_adapter_enabled() or _dirty_history_dynamics_adapter_enabled():
        raise RuntimeError(
            "codec-native history adapter cannot be combined with the old "
            "full-block/output adapters in one gate"
        )
    from hyvideo.models.transformers.modules.direct_codec_latent_attention import (
        ensure_history_adapter_runtime,
    )
    from hyvideo.models.transformers.modules.codec_native_history_adapter import (
        optimizer,
    )

    reference = next(transformer.parameters())
    adapter = ensure_history_adapter_runtime(reference.device, reference.dtype)
    if _codec_native_history_adapter_training_enabled():
        optimizer()
    if get_rank() == 0 and not getattr(
        transformer, "_hy_codec_native_history_adapter_announced", False
    ):
        print(
            "[CodecNativeHistoryAdapter] installed "
            f"layers={len(adapter.layers)} "
            f"params={sum(parameter.numel() for parameter in adapter.parameters())} "
            f"training={int(_codec_native_history_adapter_training_enabled())}",
            flush=True,
        )
        transformer._hy_codec_native_history_adapter_announced = True
    return adapter


def _ensure_dirty_history_dynamics_adapter(transformer):
    if not _dirty_history_dynamics_adapter_enabled():
        return None
    adapter = getattr(transformer, "_hy_dirty_history_dynamics_adapter", None)
    if adapter is None:
        rank = int(os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_RANK", "8"))
        alpha = float(os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_ALPHA", str(rank)))
        layers_value = os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_LAYERS", "")
        layer_ids = (
            {int(item.strip()) for item in layers_value.split(",") if item.strip()}
            if layers_value.strip()
            else set(range(len(transformer.double_blocks)))
        )
        target_names = {
            item.strip()
            for item in os.environ.get(
                "HY_DIRTY_HISTORY_DYNAMICS_LORA_TARGETS",
                "img_attn_q,img_attn_proj,img_mlp_fc2",
            ).split(",")
            if item.strip()
        }
        modules: Dict[str, _DirtyHistoryLoRALinear] = {}
        for layer_idx, block in enumerate(transformer.double_blocks):
            if layer_idx not in layer_ids:
                continue
            candidates = {
                "img_attn_q": (block, "img_attn_q"),
                "img_attn_proj": (block, "img_attn_proj"),
                "img_attn_prope_proj": (block, "img_attn_prope_proj"),
                "img_mlp_fc2": (block.img_mlp, "fc2"),
            }
            for target_name in sorted(target_names):
                if target_name not in candidates:
                    raise ValueError(f"Unsupported dynamics LoRA target: {target_name}")
                owner, attr = candidates[target_name]
                base = getattr(owner, attr)
                if not isinstance(base, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at L{layer_idx}.{target_name}, got {type(base)}"
                    )
                wrapped = _DirtyHistoryLoRALinear(base, rank=rank, alpha=alpha)
                setattr(owner, attr, wrapped)
                modules[f"layer_{layer_idx:02d}.{target_name}"] = wrapped
        params = [param for module in modules.values() for param in (
            module.lora_a.weight, module.lora_b.weight
        )]
        lr = float(os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_LR", "1e-5"))
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        adapter = _DirtyHistoryDynamicsAdapter(modules, optimizer)
        transformer._hy_dirty_history_dynamics_adapter = adapter
        if get_rank() == 0:
            print(
                "[DirtyHistoryDynamicsLoRA] installed "
                f"layers={len(layer_ids)} modules={len(modules)} rank={rank} "
                f"params={sum(param.numel() for param in params)}",
                flush=True,
            )

    ckpt_path = os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_CKPT", "").strip()
    loaded_path = getattr(transformer, "_hy_dirty_history_dynamics_adapter_loaded", "")
    if ckpt_path and ckpt_path != loaded_path:
        payload = torch.load(ckpt_path, map_location="cpu")
        checkpoint_modules = payload["modules"]
        if set(checkpoint_modules) != set(adapter.modules):
            raise RuntimeError(
                "Dynamics LoRA module set mismatch: "
                f"checkpoint={sorted(checkpoint_modules)} runtime={sorted(adapter.modules)}"
            )
        for name, module in adapter.modules.items():
            module.lora_a.load_state_dict(checkpoint_modules[name]["lora_a"])
            module.lora_b.load_state_dict(checkpoint_modules[name]["lora_b"])
        transformer._hy_dirty_history_dynamics_adapter_loaded = ckpt_path
        if get_rank() == 0:
            print(f"[DirtyHistoryDynamicsLoRA] loaded={ckpt_path}", flush=True)
    adapter.enabled = True
    return adapter


def _dirty_history_output_adapter_enabled() -> bool:
    return _drift_env_enabled("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_ENABLE")


def _dirty_history_output_adapter_chunk_enabled(chunk: int) -> bool:
    """Optionally restrict a checkpointed adapter to its trained rollout states."""
    value = os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_ACTIVE_CHUNKS", "").strip()
    if not value:
        return True
    return int(chunk) in {int(item.strip()) for item in value.split(",") if item.strip()}


def _set_dirty_history_adapters_chunk_enabled(transformer, chunk: int) -> bool:
    """Apply the active-chunk policy to every adapter ownership model.

    The codec-native history bank deliberately is not registered on the frozen
    Transformer: registration would make ``transformer.requires_grad_(False)``
    freeze it and would pollute the base model state dict.  It must therefore
    be toggled explicitly alongside the two Transformer-owned legacy adapters.
    """

    active = _dirty_history_output_adapter_chunk_enabled(chunk)
    adapter = getattr(transformer, "_hy_dirty_history_output_adapter", None)
    if adapter is not None:
        adapter.enabled = active
    dynamics_adapter = getattr(
        transformer, "_hy_dirty_history_dynamics_adapter", None
    )
    if dynamics_adapter is not None:
        dynamics_adapter.enabled = active
    if _codec_native_history_adapter_enabled():
        from hyvideo.models.transformers.modules.codec_native_history_adapter import (
            bank as history_adapter_bank,
        )

        history_adapter_bank().enabled = active
    return active


def _ensure_dirty_history_output_adapter(transformer):
    """Install/load the small final-head residual adapter once per transformer."""
    if not _dirty_history_output_adapter_enabled():
        return None
    adapter = getattr(transformer, "_hy_dirty_history_output_adapter", None)
    if adapter is None:
        linear = transformer.final_layer.linear
        if isinstance(linear, _DirtyHistoryOutputAdapterLinear):
            adapter = linear
        else:
            adapter = _DirtyHistoryOutputAdapterLinear(linear)
            transformer.final_layer.linear = adapter
        transformer._hy_dirty_history_output_adapter = adapter
        if get_rank() == 0:
            print(
                "[DirtyHistoryOutputAdapter] installed "
                f"type={adapter.adapter_type} "
                f"params={sum(param.numel() for param in adapter.delta.parameters())}",
                flush=True,
            )

    ckpt_path = os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_CKPT", "").strip()
    loaded_path = getattr(transformer, "_hy_dirty_history_output_adapter_loaded", "")
    if ckpt_path and ckpt_path != loaded_path:
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dirty-history output adapter checkpoint missing: {path}")
        payload = torch.load(path, map_location="cpu")
        checkpoint_type = payload.get("adapter_type")
        if checkpoint_type is not None and checkpoint_type != adapter.adapter_type:
            raise RuntimeError(
                "Dirty-history output adapter type mismatch: "
                f"checkpoint={checkpoint_type!r}, requested={adapter.adapter_type!r}"
            )
        adapter.delta.load_state_dict(payload["delta"])
        transformer._hy_dirty_history_output_adapter_loaded = ckpt_path
        if get_rank() == 0:
            print(f"[DirtyHistoryOutputAdapter] loaded={path}", flush=True)

    if not hasattr(transformer, "_hy_dirty_history_output_adapter_optim"):
        lr = float(os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_LR", "1e-4"))
        transformer._hy_dirty_history_output_adapter_optim = torch.optim.AdamW(
            adapter.delta.parameters(), lr=lr, weight_decay=0.0
        )
    adapter.enabled = True
    return adapter


@contextmanager
def _dirty_history_output_adapter_disabled(transformer):
    adapters = [
        getattr(transformer, "_hy_dirty_history_output_adapter", None),
        getattr(transformer, "_hy_dirty_history_dynamics_adapter", None),
    ]
    if _codec_native_history_adapter_enabled():
        from hyvideo.models.transformers.modules.codec_native_history_adapter import (
            bank as history_adapter_bank,
        )

        adapters.append(history_adapter_bank())
    adapters = [adapter for adapter in adapters if adapter is not None]
    if not adapters:
        yield
        return
    previous = [adapter.enabled for adapter in adapters]
    for adapter in adapters:
        adapter.enabled = False
    try:
        yield
    finally:
        for adapter, enabled in zip(adapters, previous):
            adapter.enabled = enabled


def _zero_dirty_history_output_adapter_grads(transformer) -> None:
    if _codec_native_history_adapter_training_enabled():
        from hyvideo.models.transformers.modules.codec_native_history_adapter import (
            zero_grad as zero_history_adapter_grad,
        )

        zero_history_adapter_grad()
    optimizer = getattr(transformer, "_hy_dirty_history_output_adapter_optim", None)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    dynamics = getattr(transformer, "_hy_dirty_history_dynamics_adapter", None)
    if dynamics is not None:
        dynamics.optimizer.zero_grad(set_to_none=True)


def _step_dirty_history_output_adapter(transformer, loss: torch.Tensor) -> Dict[str, float]:
    if _codec_native_history_adapter_training_enabled():
        from hyvideo.models.transformers.modules.codec_native_history_adapter import (
            step as step_history_adapter,
        )

        return step_history_adapter(loss)
    dynamics = getattr(transformer, "_hy_dirty_history_dynamics_adapter", None)
    optimizer = (
        dynamics.optimizer
        if dynamics is not None
        else getattr(transformer, "_hy_dirty_history_output_adapter_optim", None)
    )
    if optimizer is None:
        raise RuntimeError("Dirty-history output adapter optimizer was not initialized")
    loss.backward()
    params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
    grad_sq = sum(float(param.grad.detach().float().pow(2).sum().cpu()) for param in params)
    grad_norm = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
    clip = float(os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_GRAD_CLIP", "1.0"))
    if params and clip > 0.0:
        torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    adapter = dynamics or transformer._hy_dirty_history_output_adapter
    weight_norm = math.sqrt(
        sum(
            float(param.detach().float().pow(2).sum().cpu())
            for param in (
                adapter.parameters() if dynamics is not None else adapter.delta.parameters()
            )
        )
    )
    return {
        "num_optimizers": 1.0,
        "max_grad_norm": grad_norm,
        "adapter_grad_norm": grad_norm,
        "adapter_weight_norm": weight_norm,
    }


def _save_dirty_history_output_adapter(transformer, output_root: str) -> Optional[str]:
    if _codec_native_history_adapter_enabled():
        from hyvideo.models.transformers.modules.codec_native_history_adapter import (
            save as save_history_adapter,
        )

        return save_history_adapter(output_root)
    dynamics = getattr(transformer, "_hy_dirty_history_dynamics_adapter", None)
    if dynamics is not None:
        path = Path(output_root) / "checkpoint_dynamics_lora.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "modules": {
                    name: {
                        "lora_a": {
                            key: value.detach().cpu()
                            for key, value in module.lora_a.state_dict().items()
                        },
                        "lora_b": {
                            key: value.detach().cpu()
                            for key, value in module.lora_b.state_dict().items()
                        },
                    }
                    for name, module in dynamics.modules.items()
                },
                "rank": int(os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_RANK", "8")),
                "alpha": float(os.environ.get("HY_DIRTY_HISTORY_DYNAMICS_LORA_ALPHA", "8")),
                "targets": os.environ.get(
                    "HY_DIRTY_HISTORY_DYNAMICS_LORA_TARGETS",
                    "img_attn_q,img_attn_proj,img_mlp_fc2",
                ),
            },
            path,
        )
        return str(path)
    adapter = getattr(transformer, "_hy_dirty_history_output_adapter", None)
    if adapter is None:
        return None
    path = Path(output_root) / "checkpoint_output_adapter.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta": {name: value.detach().cpu() for name, value in adapter.delta.state_dict().items()},
            "in_features": adapter.base.in_features,
            "out_features": adapter.base.out_features,
            "adapter_type": adapter.adapter_type,
            "train_args": {
                "lr": float(os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_LR", "1e-4")),
                "clean_weight": float(os.environ.get("HY_DIRTY_HISTORY_OUTPUT_ADAPTER_CLEAN_WEIGHT", "0.2")),
            },
        },
        path,
    )
    return str(path)


def _sync_cuda(device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _elapsed_since(start, device=None):
    _sync_cuda(device)
    return time.perf_counter() - start


def _drift_env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _rollout_boundary_chunks(name: str) -> set[int]:
    value = os.environ.get(name, "").strip()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _prediction_oracle_boundary_enabled(boundary_chunk: int) -> bool:
    state_dir = os.environ.get("HY_DIRTY_HISTORY_PRED_ORACLE_STATE_DIR", "").strip()
    recovery_root = os.environ.get("HY_DIRTY_HISTORY_RECOVERY_CONTEXT_ROOT", "").strip()
    if state_dir:
        boundaries = _rollout_boundary_chunks(
            "HY_DIRTY_HISTORY_PRED_ORACLE_BOUNDARIES"
        )
    elif recovery_root:
        boundaries = _rollout_boundary_chunks(
            "HY_DIRTY_HISTORY_RECOVERY_BOUNDARIES"
        )
    else:
        return False
    return int(boundary_chunk) in boundaries


def _prediction_oracle_step_enabled(step: int) -> bool:
    value = os.environ.get("HY_DIRTY_HISTORY_PRED_ORACLE_STEPS", "3").strip().lower()
    if value in {"all", "*"}:
        return True
    return int(step) in {int(item.strip()) for item in value.split(",") if item.strip()}


def _dirty_history_recovery_teacher_mode() -> str:
    mode = os.environ.get(
        "HY_DIRTY_HISTORY_RECOVERY_TEACHER_MODE", "exact_clean"
    ).strip().lower()
    aliases = {
        "exact": "exact_clean",
        "full": "exact_clean",
        "compressed": "compressed_clean",
        "codec": "compressed_clean",
        "local": "local_exact",
        "same_state": "local_exact",
        "hybrid": "hybrid_exact",
        "ar_hybrid": "hybrid_exact",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"exact_clean", "compressed_clean", "local_exact", "hybrid_exact"}:
        raise ValueError(
            "HY_DIRTY_HISTORY_RECOVERY_TEACHER_MODE must be one of "
            f"exact_clean, compressed_clean, local_exact, hybrid_exact; got {mode!r}"
        )
    return mode


def _hybrid_teacher_context(
    student_context: torch.Tensor,
    clean_context: torch.Tensor,
    selected_frame_indices,
    current_frame_idx: int,
    chunk_latent_frames: int,
) -> torch.Tensor:
    """Clean older history while retaining the latest Student-generated chunks."""
    recent_chunks = max(
        1, int(os.environ.get("HY_DIRTY_HISTORY_RECOVERY_HYBRID_RECENT_CHUNKS", "1"))
    )
    beta = float(os.environ.get("HY_DIRTY_HISTORY_RECOVERY_HYBRID_BETA", "1.0"))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("HY_DIRTY_HISTORY_RECOVERY_HYBRID_BETA must be in [0, 1]")
    recent_start = int(current_frame_idx) - recent_chunks * int(chunk_latent_frames)
    replace_mask = torch.tensor(
        [int(frame_idx) < recent_start for frame_idx in selected_frame_indices],
        device=student_context.device,
        dtype=student_context.dtype,
    ).view(1, 1, -1, 1, 1)
    clean_delta = clean_context - student_context
    return student_context + replace_mask * beta * clean_delta


@torch.no_grad()
def _load_prediction_oracle_clean_context(
    selected_frame_indices,
    boundary_chunk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    state_dir = os.environ.get("HY_DIRTY_HISTORY_PRED_ORACLE_STATE_DIR", "").strip()
    if state_dir:
        path = Path(state_dir) / f"chunk{int(boundary_chunk):02d}_latent_prefix.pt"
    else:
        root = Path(os.environ["HY_DIRTY_HISTORY_RECOVERY_CONTEXT_ROOT"])
        split = os.environ.get("HY_DIRTY_HISTORY_RECOVERY_SPLIT", "train").strip()
        case_id = os.environ.get("HY_DIRTY_HISTORY_RECOVERY_CASE_ID", "").strip()
        if not case_id:
            raise RuntimeError(
                "HY_DIRTY_HISTORY_RECOVERY_CASE_ID is required for paired context training"
            )
        path = (
            root
            / split
            / case_id
            / "clean"
            / f"boundary{int(boundary_chunk):02d}_context.pt"
        )
    payload = torch.load(path, map_location="cpu")
    selected_indices = [int(index) for index in selected_frame_indices]
    if "context_latents" in payload:
        reference = payload["context_latents"]
        reference_indices = [int(index) for index in payload["selected_frame_indices"]]
        if reference_indices != selected_indices:
            raise RuntimeError(
                f"Prediction-recovery retrieval mismatch at boundary {boundary_chunk}: "
                f"runtime={selected_indices} paired={reference_indices} path={path}"
            )
        return reference.to(device=device, dtype=dtype)
    reference = payload["tensor"]
    if selected_indices and max(selected_indices) >= reference.shape[2]:
        raise RuntimeError(
            f"Prediction-oracle state does not cover selected frame "
            f"{max(selected_indices)} at boundary {boundary_chunk}: "
            f"state={tuple(reference.shape)} path={path}"
        )
    return reference[:, :, selected_indices].to(device=device, dtype=dtype)


@torch.no_grad()
def _load_clean_trajectory_tensor(
    kind: str,
    chunk: int,
    step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load a tensor from the matched pure-clean rollout trajectory.

    Unlike the clean-history Teacher path, this target was produced from a
    clean current state *and* clean history.  It therefore provides a
    self-consistent recovery target for an on-policy codec-native Student.
    """
    root_value = os.environ.get("HY_CLEAN_TRAJECTORY_ROOT", "").strip()
    if not root_value:
        raise RuntimeError(
            "HY_CLEAN_TRAJECTORY_ROOT is required for trajectory-recovery FT"
        )
    split = os.environ.get("HY_DIRTY_HISTORY_RECOVERY_SPLIT", "train").strip()
    case_id = os.environ.get("HY_DIRTY_HISTORY_RECOVERY_CASE_ID", "").strip()
    if not case_id:
        raise RuntimeError(
            "HY_DIRTY_HISTORY_RECOVERY_CASE_ID is required for trajectory-recovery FT"
        )
    kind = str(kind).strip().lower()
    if kind not in {"x_in", "noise_pred", "x_next"}:
        raise ValueError(f"Unsupported clean trajectory tensor kind={kind!r}")
    path = (
        Path(root_value)
        / split
        / case_id
        / "clean"
        / "trajectory"
        / "clean"
        / f"chunk{int(chunk):02d}_step{int(step):02d}_{kind}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing matched clean trajectory tensor: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("tensor")):
        raise RuntimeError(f"Malformed clean trajectory payload: {path}")
    if str(payload.get("kind", "")).strip().lower() != kind:
        raise RuntimeError(
            f"Clean trajectory kind mismatch: expected={kind} "
            f"actual={payload.get('kind')!r} path={path}"
        )
    if int(payload.get("chunk", -1)) != int(chunk) or int(payload.get("step", -1)) != int(step):
        raise RuntimeError(
            f"Clean trajectory coordinate mismatch: expected=({chunk},{step}) "
            f"actual=({payload.get('chunk')},{payload.get('step')}) path={path}"
        )
    tensor = payload["tensor"]
    recorded_shape = tuple(int(value) for value in payload.get("shape", tensor.shape))
    if recorded_shape != tuple(tensor.shape):
        raise RuntimeError(
            f"Clean trajectory shape metadata mismatch: metadata={recorded_shape} "
            f"tensor={tuple(tensor.shape)} path={path}"
        )
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def _maybe_apply_rollout_context_corrector(
    context_latents: torch.Tensor,
    selected_frame_indices,
    boundary_chunk: int,
    current_frame_idx: int,
    chunk_latent_frames: int,
    context_action: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
) -> torch.Tensor:
    checkpoint_path = os.environ.get(
        "HY_ROLLOUT_CONTEXT_CORRECTOR_CKPT", ""
    ).strip()
    if not checkpoint_path or int(boundary_chunk) not in _rollout_boundary_chunks(
        "HY_ROLLOUT_CONTEXT_CORRECTOR_BOUNDARIES"
    ):
        return context_latents

    from hyvideo.models.rollout_context_corrector import RolloutContextCorrector

    cache_key = (checkpoint_path, str(context_latents.device))
    model = _ROLLOUT_CONTEXT_CORRECTOR_CACHE.get(cache_key)
    if model is None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config", {})
        model = RolloutContextCorrector(
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 3)),
            num_heads=int(config.get("num_heads", 4)),
            ffn_dim=int(config.get("ffn_dim", 256)),
            use_gate=bool(config.get("use_gate", True)),
            use_pose=bool(config.get("use_pose", False)),
        )
        model.load_state_dict(payload["model"])
        model.to(device=context_latents.device, dtype=torch.float32).eval()
        _ROLLOUT_CONTEXT_CORRECTOR_CACHE[cache_key] = model
        if get_rank() == 0:
            print(
                f"[RolloutContextCorrector] loaded checkpoint={checkpoint_path} "
                f"params={model.num_parameters}",
                flush=True,
            )

    selected_indices = [int(index) for index in selected_frame_indices]
    frame_indices = torch.tensor(
        selected_indices, device=context_latents.device, dtype=torch.long
    ).unsqueeze(0).expand(context_latents.shape[0], -1)
    distance = torch.tensor(
        [int(current_frame_idx) - index for index in selected_indices],
        device=context_latents.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(context_latents.shape[0], -1)
    anchor = torch.tensor(
        [index < int(chunk_latent_frames) for index in selected_indices],
        device=context_latents.device,
        dtype=torch.bool,
    ).unsqueeze(0).expand(context_latents.shape[0], -1)
    boundary = torch.full(
        (context_latents.shape[0],),
        int(boundary_chunk),
        device=context_latents.device,
        dtype=torch.long,
    )
    corrected, correction, gate = model(
        context_latents.float(),
        frame_indices,
        distance,
        anchor,
        context_action.long(),
        boundary,
        context_viewmats=viewmats[:, selected_indices].to(context_latents.device),
        context_Ks=Ks[:, selected_indices].to(context_latents.device),
        query_viewmats=viewmats[
            :, current_frame_idx : current_frame_idx + chunk_latent_frames
        ].to(context_latents.device),
        query_Ks=Ks[
            :, current_frame_idx : current_frame_idx + chunk_latent_frames
        ].to(context_latents.device),
    )
    scale = float(os.environ.get("HY_ROLLOUT_CONTEXT_CORRECTOR_SCALE", "1.0"))
    output = context_latents.float() + scale * correction.float()
    if get_rank() == 0:
        print(
            f"[RolloutContextCorrector] boundary={boundary_chunk} "
            f"scale={scale:g} gate={gate.float().mean().item():.4f} "
            f"correction_rms={correction.float().square().mean().sqrt().item():.6f}",
            flush=True,
        )
    return output.to(dtype=context_latents.dtype)


def _text_only_kv_cache(kv_cache):
    if kv_cache is None:
        return None
    return [
        {
            "k_vision": None,
            "v_vision": None,
            "k_txt": layer_cache.get("k_txt"),
            "v_txt": layer_cache.get("v_txt"),
        }
        for layer_cache in kv_cache
    ]


def _prepare_vision_cache_rebuild(kv_cache, *, label: str):
    """Keep text KV while releasing the obsolete vision-cache generation.

    A vision-cache refill only consumes the cached text entries.  Passing the
    complete previous cache keeps old and new vision payloads alive together
    until the refill returns, creating a transient peak without changing the
    result.  This mirrors the lifetime fix in the frozen codec repository.
    """
    if not _drift_env_enabled("HY_KV_DROP_OLD_BEFORE_REBUILD"):
        return kv_cache
    text_only = _text_only_kv_cache(kv_cache)
    if get_rank() == 0:
        print(
            f"[KVCacheLifecycle] released old vision KV before {label} rebuild",
            flush=True,
        )
    return text_only


@torch.no_grad()
def _maybe_dump_rollout_boundary_state(
    latents: torch.Tensor, end_idx: int, chunk: int
) -> None:
    if int(chunk) not in _rollout_boundary_chunks("HY_ROLLOUT_STATE_DUMP_CHUNKS"):
        return
    out_dir = Path(os.environ["HY_ROLLOUT_STATE_DUMP_DIR"])
    path = out_dir / f"chunk{int(chunk):02d}_latent_prefix.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = latents[:, :, : int(end_idx)].detach().to("cpu", dtype=torch.float16).contiguous()
    torch.save({"chunk": int(chunk), "end_idx": int(end_idx), "tensor": prefix}, path)
    if get_rank() == 0:
        print(f"[RolloutStateOracle] dumped chunk={chunk} prefix={tuple(prefix.shape)} path={path}", flush=True)


@torch.no_grad()
def _maybe_reset_rollout_boundary_state(
    latents: torch.Tensor, start_idx: int, end_idx: int, chunk: int
) -> None:
    if int(chunk) not in _rollout_boundary_chunks("HY_ROLLOUT_STATE_RESET_CHUNKS"):
        return
    scope = os.environ.get("HY_ROLLOUT_STATE_RESET_SCOPE", "prefix").strip().lower()
    if scope in {"selected_context", "shadow_context"}:
        return
    ref_dir = Path(os.environ["HY_ROLLOUT_STATE_RESET_DIR"])
    path = ref_dir / f"chunk{int(chunk):02d}_latent_prefix.pt"
    payload = torch.load(path, map_location="cpu")
    reference = payload["tensor"].to(device=latents.device, dtype=latents.dtype)
    full_expected = latents[:, :, : int(end_idx)]
    if tuple(reference.shape) != tuple(full_expected.shape):
        raise RuntimeError(
            f"Boundary-state shape mismatch for chunk {chunk}: "
            f"reference={tuple(reference.shape)} expected={tuple(full_expected.shape)}"
        )
    chunk_width = int(end_idx) - int(start_idx)
    if scope == "prefix":
        slice_start = 0
    elif scope == "latest":
        slice_start = int(start_idx)
    elif scope == "recent2":
        slice_start = max(0, int(start_idx) - chunk_width)
    else:
        raise ValueError(f"Unsupported HY_ROLLOUT_STATE_RESET_SCOPE={scope!r}")
    expected = latents[:, :, slice_start : int(end_idx)]
    reference = reference[:, :, slice_start : int(end_idx)]
    alpha = float(os.environ.get("HY_ROLLOUT_STATE_RESET_ALPHA", "1.0"))
    expected.copy_(expected + alpha * (reference - expected))
    if get_rank() == 0:
        print(
            f"[RolloutStateOracle] reset chunk={chunk} scope={scope} alpha={alpha:g} "
            f"range={slice_start}:{end_idx} state={tuple(reference.shape)} path={path}",
            flush=True,
        )


@torch.no_grad()
def _maybe_reset_rollout_selected_context(
    latents: torch.Tensor,
    selected_frame_indices,
    boundary_chunk: int,
    shadow_memory: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    selected_indices = [int(idx) for idx in selected_frame_indices]
    current = latents[:, :, selected_indices]
    scope = os.environ.get("HY_ROLLOUT_STATE_RESET_SCOPE", "prefix").strip().lower()
    if scope not in {"selected_context", "shadow_context"}:
        return current

    if scope == "shadow_context" and shadow_memory:
        current = current.clone()
        for position, frame_idx in enumerate(selected_indices):
            corrected = shadow_memory.get(frame_idx)
            if corrected is not None:
                current[:, :, position].copy_(
                    corrected.to(device=current.device, dtype=current.dtype)
                )

    if int(boundary_chunk) not in _rollout_boundary_chunks(
        "HY_ROLLOUT_STATE_RESET_CHUNKS"
    ):
        return current
    ref_dir = Path(os.environ["HY_ROLLOUT_STATE_RESET_DIR"])
    path = ref_dir / f"chunk{int(boundary_chunk):02d}_latent_prefix.pt"
    payload = torch.load(path, map_location="cpu")
    reference = payload["tensor"].to(device=latents.device, dtype=latents.dtype)
    valid_positions = [
        (position, frame_idx)
        for position, frame_idx in enumerate(selected_indices)
        if frame_idx < reference.shape[2]
    ]
    valid_indices = [frame_idx for _, frame_idx in valid_positions]
    if not valid_indices:
        raise RuntimeError(f"No valid selected context indices for boundary chunk {boundary_chunk}")
    alpha = float(os.environ.get("HY_ROLLOUT_STATE_RESET_ALPHA", "1.0"))
    current = current.clone()
    current_valid = torch.stack(
        [current[:, :, position] for position, _ in valid_positions], dim=2
    )
    residual = reference[:, :, valid_indices] - current_valid
    projection = os.environ.get(
        "HY_ROLLOUT_STATE_RESET_PROJECTION", "full"
    ).strip().lower()
    def spatial_lowpass(value: torch.Tensor, factor: int) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        value_2d = value.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        low_height = math.ceil(height / factor)
        low_width = math.ceil(width / factor)
        low_resolution = F.adaptive_avg_pool2d(
            value_2d.float(), (low_height, low_width)
        )
        projected_2d = F.interpolate(
            low_resolution,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=value.dtype)
        return projected_2d.reshape(
            batch, frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)

    if projection in {"full", "none"}:
        projected_residual = residual
    elif projection in {"spatial2", "spatial4"}:
        factor = int(projection[-1])
        projected_residual = spatial_lowpass(residual, factor)
    elif projection in {"highpass2", "highpass4"}:
        factor = int(projection[-1])
        projected_residual = residual - spatial_lowpass(residual, factor)
    elif projection == "frame_mean":
        projected_residual = residual.mean(dim=(-2, -1), keepdim=True)
    elif projection in {"global_shift1", "global_shift2"}:
        # Oracle: allow one integer spatial translation per retrieved frame. This
        # distinguishes globally coherent drift from the local high-frequency
        # structure preserved by the full residual.
        max_shift = int(projection.removeprefix("global_shift"))
        best_shifted = current_valid.clone()
        best_error = torch.full(
            (current_valid.shape[0], current_valid.shape[2]),
            float("inf"),
            device=current_valid.device,
            dtype=torch.float32,
        )
        for shift_y in range(-max_shift, max_shift + 1):
            for shift_x in range(-max_shift, max_shift + 1):
                shifted = torch.roll(
                    current_valid, shifts=(shift_y, shift_x), dims=(-2, -1)
                )
                error = (shifted.float() - reference[:, :, valid_indices].float()).square().mean(
                    dim=(1, 3, 4)
                )
                select = error < best_error
                best_error = torch.where(select, error, best_error)
                best_shifted = torch.where(
                    select[:, None, :, None, None], shifted, best_shifted
                )
        projected_residual = best_shifted - current_valid
    elif projection in {"local_shift1", "local_shift2"}:
        # Oracle for local correspondence: each spatial location may select a
        # nearby source location from the student context. Unlike global_shift,
        # this preserves the full latent grid while testing whether local motion
        # alignment explains the high-frequency correction signal.
        max_shift = int(projection.removeprefix("local_shift"))
        target = reference[:, :, valid_indices]
        best_shifted = current_valid.clone()
        best_error = (current_valid.float() - target.float()).square().mean(dim=1)
        for shift_y in range(-max_shift, max_shift + 1):
            for shift_x in range(-max_shift, max_shift + 1):
                if shift_x == 0 and shift_y == 0:
                    continue
                shifted = torch.roll(
                    current_valid, shifts=(shift_y, shift_x), dims=(-2, -1)
                )
                error = (shifted.float() - target.float()).square().mean(dim=1)
                select = error < best_error
                best_error = torch.where(select, error, best_error)
                best_shifted = torch.where(
                    select[:, None], shifted, best_shifted
                )
        projected_residual = best_shifted - current_valid
    elif projection in {"spatial_top10", "spatial_top25", "spatial_top50"}:
        # Retain the exact residual only at its largest spatial locations. This
        # measures whether a sparse confidence mask can carry the useful part of
        # the correction while preserving the full spatial grid where it matters.
        keep_percent = int(projection.removeprefix("spatial_top"))
        spatial_energy = residual.float().square().mean(dim=1, keepdim=True)
        flat_energy = spatial_energy.flatten(start_dim=3)
        keep_count = max(1, math.ceil(flat_energy.shape[-1] * keep_percent / 100.0))
        threshold = flat_energy.topk(keep_count, dim=-1).values[..., -1:]
        mask = (flat_energy >= threshold).reshape_as(spatial_energy)
        projected_residual = residual * mask.to(dtype=residual.dtype)
    elif projection in {"channel_pca4", "channel_pca8", "channel_pca16"}:
        rank = int(projection.removeprefix("channel_pca"))
        basis_path = os.environ.get(
            "HY_ROLLOUT_STATE_RESET_CHANNEL_BASIS", ""
        ).strip()
        if not basis_path:
            raise RuntimeError(
                "HY_ROLLOUT_STATE_RESET_CHANNEL_BASIS is required for channel PCA"
            )
        cache_key = (basis_path, str(residual.device), str(residual.dtype))
        basis = _ROLLOUT_RESIDUAL_BASIS_CACHE.get(cache_key)
        if basis is None:
            payload = torch.load(basis_path, map_location="cpu")
            if bool(payload.get("centered", True)):
                raise RuntimeError("Channel basis must be fitted without centering")
            basis = payload["basis"].to(
                device=residual.device, dtype=residual.dtype
            )
            _ROLLOUT_RESIDUAL_BASIS_CACHE[cache_key] = basis
        basis = basis[:, :rank]
        coefficients = torch.einsum("cr,bcthw->brthw", basis, residual)
        projected_residual = torch.einsum(
            "cr,brthw->bcthw", basis, coefficients
        )
    else:
        raise ValueError(
            f"Unsupported HY_ROLLOUT_STATE_RESET_PROJECTION={projection!r}"
        )
    corrected = current_valid + alpha * projected_residual
    writeback = _drift_env_enabled("HY_ROLLOUT_STATE_RESET_WRITEBACK", "1")
    if writeback:
        latents[:, :, valid_indices] = corrected
    elif scope == "shadow_context":
        if shadow_memory is None:
            raise RuntimeError("shadow_context requires a persistent shadow_memory")
        for corrected_position, (current_position, frame_idx) in enumerate(valid_positions):
            corrected_frame = corrected[:, :, corrected_position]
            current[:, :, current_position].copy_(corrected_frame)
            shadow_memory[frame_idx] = corrected_frame.detach().to("cpu").clone()
    else:
        for corrected_position, (current_position, _) in enumerate(valid_positions):
            current[:, :, current_position].copy_(corrected[:, :, corrected_position])
    if get_rank() == 0:
        residual_rms = residual.float().square().mean().sqrt()
        projected_rms = projected_residual.float().square().mean().sqrt()
        print(
            f"[RolloutStateOracle] reset chunk={boundary_chunk} scope={scope} "
            f"alpha={alpha:g} writeback={int(writeback)} frames={valid_indices} "
            f"projection={projection} residual_rms={residual_rms.item():.6f} "
            f"projected_rms={projected_rms.item():.6f} "
            f"rms_ratio={(projected_rms / residual_rms.clamp_min(1e-12)).item():.6f} "
            f"shadow_size={len(shadow_memory or {})} path={path}",
            flush=True,
        )
    return current


@torch.no_grad()
def _maybe_dump_rollout_retrieval_context(
    latents: torch.Tensor,
    selected_frame_indices,
    boundary_chunk: int,
    current_frame_idx: int,
    chunk_latent_frames: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    action: torch.Tensor,
) -> None:
    if int(boundary_chunk) not in _rollout_boundary_chunks(
        "HY_ROLLOUT_CONTEXT_DUMP_BOUNDARIES"
    ):
        return
    out_dir_value = os.environ.get("HY_ROLLOUT_CONTEXT_DUMP_DIR", "").strip()
    if not out_dir_value:
        raise RuntimeError(
            "HY_ROLLOUT_CONTEXT_DUMP_DIR is required when context dumping is enabled"
        )
    selected_indices = [int(idx) for idx in selected_frame_indices]
    query_end = int(current_frame_idx) + int(chunk_latent_frames)

    def cpu_tensor(value):
        if value is None:
            return None
        return value.detach().to("cpu").contiguous()

    payload = {
        "format_version": 1,
        "case_id": os.environ.get("HY_ROLLOUT_CONTEXT_DUMP_CASE", ""),
        "role": os.environ.get("HY_ROLLOUT_CONTEXT_DUMP_ROLE", ""),
        "boundary_chunk": int(boundary_chunk),
        "next_chunk": int(boundary_chunk) + 1,
        "selected_frame_indices": torch.tensor(selected_indices, dtype=torch.int64),
        "anchor_mask": torch.tensor(
            [frame_idx < int(chunk_latent_frames) for frame_idx in selected_indices],
            dtype=torch.bool,
        ),
        "distance_to_query": torch.tensor(
            [int(current_frame_idx) - frame_idx for frame_idx in selected_indices],
            dtype=torch.int64,
        ),
        "context_latents": cpu_tensor(
            latents[:, :, selected_indices].to(dtype=torch.float16)
        ),
        "context_viewmats": cpu_tensor(viewmats[:, selected_indices]),
        "context_Ks": cpu_tensor(Ks[:, selected_indices]),
        "context_action": cpu_tensor(action[:, selected_indices]),
        "query_viewmats": cpu_tensor(viewmats[:, current_frame_idx:query_end]),
        "query_Ks": cpu_tensor(Ks[:, current_frame_idx:query_end]),
        "query_action": cpu_tensor(action[:, current_frame_idx:query_end]),
    }
    out_dir = Path(out_dir_value)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"boundary{int(boundary_chunk):02d}_context.pt"
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    if get_rank() == 0:
        print(
            f"[RolloutContextDump] case={payload['case_id']} role={payload['role']} "
            f"boundary={boundary_chunk} frames={selected_indices} "
            f"latent={tuple(payload['context_latents'].shape)} path={path}",
            flush=True,
        )


@torch.no_grad()
def _dump_rollout_latent_tensor(kind: str, tensor: torch.Tensor, chunk: int, step: int) -> None:
    if not _drift_env_enabled("HY_ROLLOUT_DRIFT_DUMP_ENABLE"):
        return
    chunk_filter = os.environ.get("HY_ROLLOUT_DRIFT_DUMP_CHUNKS", "").strip()
    if chunk_filter and int(chunk) not in {
        int(item.strip()) for item in chunk_filter.split(",") if item.strip()
    }:
        return
    step_filter = os.environ.get("HY_ROLLOUT_DRIFT_DUMP_STEPS", "").strip()
    if step_filter and int(step) not in {
        int(item.strip()) for item in step_filter.split(",") if item.strip()
    }:
        return
    max_chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_MAX_CHUNK", "-1"))
    if max_chunk >= 0 and int(chunk) > max_chunk:
        return
    out_dir = Path(os.environ.get("HY_ROLLOUT_DRIFT_DUMP_DIR", "kv_ae_outputs/ae16x_v2/rollout_drift"))
    run_name = os.environ.get("HY_ROLLOUT_DRIFT_RUN_NAME", "run")
    path = out_dir / run_name / f"chunk{int(chunk):02d}_step{int(step):02d}_{kind}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": kind,
            "chunk": int(chunk),
            "step": int(step),
            "shape": tuple(int(x) for x in tensor.shape),
            "tensor": tensor.detach().to("cpu", dtype=torch.float16).contiguous(),
        },
        path,
    )


def _functional_ft_clean_kv_cache(kv_cache, clean_layers=None):
    if kv_cache is None:
        return None
    clean_layer_set = None
    if clean_layers is not None:
        clean_layer_set = {int(layer) for layer in clean_layers}
    out = []
    for layer_idx, layer_cache in enumerate(kv_cache):
        copied = dict(layer_cache)
        clean_k = copied.get("_k_vision_clean_for_ft")
        if torch.is_tensor(clean_k) and (clean_layer_set is None or int(layer_idx) in clean_layer_set):
            copied["k_vision"] = clean_k
        out.append(copied)
    return out


def _functional_ft_teacher_kv_cache(kv_cache, target_layers):
    mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE", "clean_all").strip().lower()
    if mode in {"same_state", "same_state_group", "local", "local_group", "selected_clean"}:
        return _functional_ft_clean_kv_cache(kv_cache, clean_layers=target_layers)
    return _functional_ft_clean_kv_cache(kv_cache)


def _functional_ft_target() -> str:
    target = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TARGET", "k").strip().lower()
    if target not in {"k", "v", "kv"}:
        raise ValueError(f"Unknown HY_FUNCTIONAL_BLOCK_FT_TARGET={target!r}")
    return target


def _functional_ft_loss_mode() -> str:
    mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LOSS_MODE", "block").strip().lower()
    if mode in {
        "trajectory_recovery",
        "clean_trajectory",
        "clean_x_next",
        "trajectory_x_next",
    }:
        return "trajectory_recovery"
    if mode in {"routing", "route", "routing_distill", "k_routing"}:
        return "routing"
    if mode in {"prediction", "pred", "noise_pred", "recovery"}:
        return "prediction"
    if mode in {"final_latent", "latent", "x_next"}:
        return "final_latent"
    if mode in {"readout", "attention", "attention_output", "pv"}:
        return "readout"
    return "block"


def _scheduler_step_no_advance(scheduler, model_output, timestep, sample):
    """Evaluate one scheduler step on an independent full-state copy.

    Functional Teacher and Student calls must not share or partially restore a
    mutable scheduler. A deep copy preserves every scheduler field, including
    any state added beyond ``_step_index``, and makes call order irrelevant.
    """
    scheduler_copy = copy.deepcopy(scheduler)
    return scheduler_copy.step(model_output, timestep, sample, return_dict=False)[0]


def _relative_spatiotemporal_gradient_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Relative first-difference loss for latent time and spatial detail.

    The rollout latent is ``[B, C, T, H, W]``.  Normalizing each axis by the
    corresponding clean-gradient energy keeps the optional loss independent
    of latent scale and prevents the large low-frequency latent values from
    hiding texture or temporal-transition errors.
    """
    if student.shape != teacher.shape or student.ndim != 5:
        raise ValueError(
            "trajectory gradient loss expects equal [B,C,T,H,W] tensors; "
            f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    losses = []
    for axis in (2, 3, 4):
        if student.shape[axis] <= 1:
            continue
        current = [slice(None)] * student.ndim
        previous = [slice(None)] * student.ndim
        current[axis] = slice(1, None)
        previous[axis] = slice(None, -1)
        student_delta = student[tuple(current)] - student[tuple(previous)]
        teacher_delta = teacher[tuple(current)] - teacher[tuple(previous)]
        delta_error = student_delta - teacher_delta
        losses.append(
            delta_error.pow(2).sum()
            / teacher_delta.pow(2).sum().clamp_min(1e-8)
        )
    if not losses:
        return student.new_zeros(())
    return torch.stack(losses).mean()


def _functional_ft_clean_target_cache(kv_cache, clean_layers=None):
    target = _functional_ft_target()
    if target == "k":
        return _functional_ft_clean_kv_cache(kv_cache, clean_layers=clean_layers)
    if kv_cache is None:
        return None
    clean_layer_set = None
    if clean_layers is not None:
        clean_layer_set = {int(layer) for layer in clean_layers}
    out = []
    for layer_idx, layer_cache in enumerate(kv_cache):
        copied = dict(layer_cache)
        selected = clean_layer_set is None or int(layer_idx) in clean_layer_set
        clean_v = copied.get("_v_vision_clean_for_ft")
        if torch.is_tensor(clean_v) and selected:
            copied["v_vision"] = clean_v
        if target == "kv":
            clean_k = copied.get("_k_vision_clean_for_ft")
            if torch.is_tensor(clean_k) and selected:
                copied["k_vision"] = clean_k
        out.append(copied)
    return out


def _functional_ft_teacher_target_cache(kv_cache, target_layers):
    mode = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE", "clean_all").strip().lower()
    if _drift_env_enabled("HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER"):
        if mode not in {
            "same_state",
            "same_state_group",
            "local",
            "local_group",
            "selected_clean",
        }:
            raise RuntimeError(
                "codec-native causal formal Teacher requires "
                "HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE=same_state_group"
            )
        if _functional_ft_target() != "kv":
            raise RuntimeError(
                "codec-native causal formal Teacher must replace both K and V; "
                "set HY_FUNCTIONAL_BLOCK_FT_TARGET=kv"
            )
        from hyvideo.models.transformers.modules.codec_native_causal_window import (
            build_formal_teacher_cache,
        )

        # Ordinarily the layers supervised by the functional loss and the
        # layers decoded formally in the Teacher are identical.  A focused
        # residual adapter, however, needs a *full-context* Teacher while only
        # owning a small number of trainable/loss layers.  Keep the old
        # behavior unless the training-only context override is explicit.
        context_text = os.environ.get(
            "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_CONTEXT_LAYERS", ""
        ).strip()
        if context_text:
            try:
                teacher_context_layers = {
                    int(item.strip())
                    for item in context_text.split(",")
                    if item.strip()
                }
            except ValueError as error:
                raise ValueError(
                    "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_CONTEXT_LAYERS "
                    "must be a comma-separated integer list"
                ) from error
            if not teacher_context_layers:
                raise ValueError(
                    "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_CONTEXT_LAYERS "
                    "must not be empty when set"
                )
        else:
            teacher_context_layers = target_layers
        return build_formal_teacher_cache(kv_cache, teacher_context_layers)
    if mode in {"same_state", "same_state_group", "local", "local_group", "selected_clean"}:
        return _functional_ft_clean_target_cache(kv_cache, clean_layers=target_layers)
    return _functional_ft_clean_target_cache(kv_cache)


def _functional_ft_student_target_cache(kv_cache, target_layers):
    """Build the K/V clean-suffix Student cache without forcing current tokens.

    Only historical cached target tensors are replaced. Current-token hidden and
    Q/K/V continue to be computed by the Student blocks themselves.
    """
    if kv_cache is None or not _drift_env_enabled(
        "HY_FUNCTIONAL_BLOCK_FT_CLEAN_SUFFIX"
    ):
        return kv_cache
    target = _functional_ft_target()
    pairs = (
        (("k_vision", "_k_vision_clean_for_ft"), ("v_vision", "_v_vision_clean_for_ft"))
        if target == "kv"
        else (("v_vision", "_v_vision_clean_for_ft"),)
        if target == "v"
        else (("k_vision", "_k_vision_clean_for_ft"),)
    )
    suffix_start = int(
        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_CLEAN_SUFFIX_START", "4")
    )
    target_layer_set = {int(layer) for layer in target_layers}
    out = []
    for layer_idx, layer_cache in enumerate(kv_cache):
        copied = dict(layer_cache)
        if layer_idx >= suffix_start and layer_idx not in target_layer_set:
            for cache_name, clean_name in pairs:
                clean_value = copied.get(clean_name)
                if not torch.is_tensor(clean_value):
                    raise RuntimeError(
                        f"Clean-suffix Student missing {clean_name} at layer {layer_idx}"
                    )
                copied[cache_name] = clean_value.detach()
        out.append(copied)
    return out


def _functional_ft_cache_debug(kv_cache, chunk: int, step: int) -> None:
    if not _functional_ft_enabled() or get_rank() != 0:
        return
    if int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_CACHE_DEBUG", "1")) <= 0:
        return
    layers = {
        int(item.strip())
        for item in os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "13,40,53").split(",")
        if item.strip()
    }
    rows = []
    for layer_idx, layer_cache in enumerate(kv_cache or []):
        if layer_idx not in layers:
            continue
        names = ("k", "v") if _functional_ft_target() == "kv" else (_functional_ft_target(),)
        for name in names:
            clean = layer_cache.get(f"_{name}_vision_clean_for_ft")
            corrupt = layer_cache.get(f"{name}_vision")
            if torch.is_tensor(clean) and torch.is_tensor(corrupt):
                rel = (corrupt.float() - clean.float()).norm() / clean.float().norm().clamp_min(1e-8)
                rows.append(f"L{layer_idx}{name.upper()}={float(rel.detach().cpu()):.6e}")
    if rows:
        print(f"[FunctionalBlockFT] cache_delta chunk={chunk} step={step} " + " ".join(rows), flush=True)


def _functional_ft_enabled() -> bool:
    return _drift_env_enabled("HY_FUNCTIONAL_BLOCK_FT_ENABLE")


def _functional_ft_step_enabled(step: int) -> bool:
    if not _functional_ft_enabled():
        return False
    # T0 owns the whole four-step graph and performs a single endpoint
    # backward.  Never also enter the legacy per-denoise-step trainer.
    if _drift_env_enabled("HY_K_T0_ENABLE"):
        return False
    chunks = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_CHUNKS", "").strip()
    if chunks:
        chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
        allowed = {
            int(item.strip())
            for item in chunks.split(",")
            if item.strip()
        }
        if chunk not in allowed:
            return False
    only_step = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ONLY_STEP", "").strip()
    if only_step:
        return int(step) == int(only_step)
    step_set = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_STEP_SET", "").strip()
    if step_set:
        allowed_steps = {
            int(item.strip())
            for item in step_set.split(",")
            if item.strip()
        }
        return int(step) in allowed_steps
    max_steps = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_STEPS_PER_CHUNK", "1"))
    return int(step) < max_steps


def _k_t0_chunk_enabled(chunk: int) -> bool:
    if not _drift_env_enabled("HY_K_T0_ENABLE"):
        return False
    chunks = os.environ.get("HY_K_T0_CHUNKS", "").strip()
    if not chunks:
        return True
    return int(chunk) in {
        int(item.strip()) for item in chunks.split(",") if item.strip()
    }


def _functional_ft_inner_steps_for_chunk(chunk: int) -> int:
    mapping = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_INNER_STEPS_BY_CHUNK", "").strip()
    if mapping:
        for item in mapping.split(","):
            if not item.strip() or ":" not in item:
                continue
            key, value = item.split(":", 1)
            if int(key.strip()) == int(chunk):
                return int(value.strip())
    return int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_INNER_STEPS", "1"))


def _functional_ft_teacher_force_all_steps() -> bool:
    return _functional_ft_enabled() and _drift_env_enabled(
        "HY_FUNCTIONAL_BLOCK_FT_TEACHER_FORCE_ALL_STEPS"
    )


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    sr_videos: Optional[Union[torch.Tensor, np.ndarray]] = None


class HunyuanVideo_1_5_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->text_encoder_2->byt5_model->transformer->vae"
    _optional_components = ["text_encoder_2"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HunyuanVideo_1_5_DiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        flow_shift: float = 7.0,
        guidance_scale: float = 6.0,
        embedded_guidance_scale: Optional[float] = None,
        progress_bar_config: Dict[str, Any] = None,
        vision_num_semantic_tokens=729,
        vision_states_dim=1152,
        glyph_byT5_v2=True,
        byt5_model=None,
        byt5_tokenizer=None,
        byt5_max_length=256,
        prompt_format=None,
        execution_device=None,
        vision_encoder=None,
        enable_offloading=False,
    ):
        super().__init__()

        self.register_to_config(
            glyph_byT5_v2=glyph_byT5_v2,
            byt5_max_length=byt5_max_length,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=embedded_guidance_scale,
        )

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        if glyph_byT5_v2:
            self.byt5_max_length = byt5_max_length
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
                byt5_model=byt5_model,
                byt5_tokenizer=byt5_tokenizer,
            )
            self.prompt_format = prompt_format
        else:
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
            )
            self.byt5_model = None
            self.byt5_tokenizer = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_len = text_encoder.max_length
        self.target_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.autocast_enabled = True
        self.vae_autocast_enabled = True
        self.enable_offloading = enable_offloading
        self.execution_device = torch.device(execution_device)

        if vision_encoder:
            self.register_modules(vision_encoder=vision_encoder)
        else:
            self.vision_encoder = None

        # Default i2v target size configurations
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }

    @classmethod
    def _create_scheduler(cls, flow_shift):
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        return scheduler

    @classmethod
    def _load_byt5(cls, cached_folder, glyph_byT5_v2, byt5_max_length, device):
        if not glyph_byT5_v2:
            byt5_kwargs = None
            prompt_format = None
            return byt5_kwargs, prompt_format
        try:
            load_from = os.path.join(cached_folder, "text_encoder")
            glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
            if not os.path.exists(glyph_root):
                raise RuntimeError(
                    f"Glyph checkpoint not found from '{glyph_root}'. \n"
                    "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                    "- Required files:\n"
                    "    Glyph-SDXL-v2\n"
                    "    ├── assets\n"
                    "    │   ├── color_idx.json\n"
                    "    │   └── multilingual_10-lang_idx.json\n"
                    "    └── checkpoints\n"
                    "        └── byt5_model.pt\n"
                )

            byT5_google_path = os.path.join(load_from, "byt5-small")
            if not os.path.exists(byT5_google_path):
                loguru.logger.warning(
                    f"ByT5 google path not found from: {byT5_google_path}. "
                    f"Try downloading from https://huggingface.co/google/byt5-small."
                )
                byT5_google_path = "google/byt5-small"

            multilingual_prompt_format_color_path = os.path.join(
                glyph_root, "assets/color_idx.json"
            )
            multilingual_prompt_format_font_path = os.path.join(
                glyph_root, "assets/multilingual_10-lang_idx.json"
            )

            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=byt5_max_length,
            )

            byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
            prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path,
                color_path=multilingual_prompt_format_color_path,
            )
            return byt5_kwargs, prompt_format
        except Exception as e:
            raise RuntimeError("Error loading byT5 glyph processor") from e

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
                Text encoder to use. If None, uses the pipeline's default text encoder.
            data_type (`str`, *optional*):
                Type of data being encoded. Defaults to "image".
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(
                prompt, data_type=data_type, max_length=self.text_len
            )
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(
                uncond_tokens, data_type=data_type, max_length=self.text_len
            )

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, is_uncond=True
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def prepare_extra_func_kwargs(self, func, kwargs):
        """
        Prepare extra keyword arguments for scheduler functions.

        Filters kwargs to only include parameters that the function accepts.
        This is useful since not all schedulers have the same signature.
        """
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare latents for video generation.

        Args:
            batch_size: Batch size for generation.
            num_channels_latents: Number of channels in latent space.
            latent_height: Height of latent tensors.
            latent_width: Width of latent tensors.
            video_length: Number of frames in the video.
            dtype: Data type for latents.
            device: Target device for latents.
            generator: Random number generator.
            latents: Pre-computed latents. If None, random latents are generated.

        Returns:
            torch.Tensor: Prepared latents tensor.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.
    # pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        pattern = r"\"(.*?)\"|“(.*?)”"
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros(
            (1, self.byt5_max_length), device=device, dtype=torch.int64
        )

        glyph_texts = self._extract_glyph_texts(prompt_text)

        if len(glyph_texts) > 0:
            text_styles = [
                {"color": None, "font-family": None} for _ in range(len(glyph_texts))
            ]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)

            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)

            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.

        Args:
            prompts: List of prompt strings or single prompt string.
            device: Target device for tensors.

        Returns:
            dict: Dictionary containing:
                - "byt5_text_states": Combined embeddings tensor.
                - "byt5_text_mask": Combined attention mask tensor.
                Returns empty dict if glyph_byT5_v2 is disabled.
        """
        if not self.config.glyph_byT5_v2:
            return {}

        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)

        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)

            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {"byt5_text_states": byt5_embeddings, "byt5_text_mask": byt5_masks}

    def extract_image_features(self, reference_image):
        """
        Extract features from a reference image using VisionEncoder.

        Args:
            reference_image: numpy array of shape (H, W, 3) with dtype uint8.

        Returns:
            VisionEncoderModelOutput: Encoded image features.
        """
        assert isinstance(reference_image, np.ndarray)
        assert reference_image.ndim == 3 and reference_image.shape[2] == 3
        assert reference_image.dtype == np.uint8

        image_encoder_output = self.vision_encoder.encode_images(reference_image)

        return image_encoder_output

    def _prepare_vision_states(
        self, reference_image, target_resolution, latents, device
    ):
        """
        Prepare vision states for multitask training.

        Args:
            reference_image: Reference image for i2v tasks (None for t2v tasks).
            target_resolution: Target size for i2v tasks.
            latents: Latent tensors.
            device: Target device.

        Returns:
            torch.Tensor or None: Vision states tensor or None if vision encoder is unavailable.
        """
        if reference_image is None:
            vision_states = torch.zeros(
                latents.shape[0],
                self.config.vision_num_semantic_tokens,
                self.config.vision_states_dim,
            ).to(latents.device)
        else:
            reference_image = (
                np.array(reference_image)
                if isinstance(reference_image, Image.Image)
                else reference_image
            )
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]

            height, width = self.get_closest_resolution_given_reference_image(
                reference_image, target_resolution
            )

            # Encode reference image to vision states
            if self.vision_encoder is not None:
                input_image_np = resize_and_center_crop(
                    reference_image, target_width=width, target_height=height
                )
                vision_states = self.vision_encoder.encode_images(input_image_np)
                vision_states = vision_states.last_hidden_state.to(
                    device=device, dtype=self.target_dtype
                )
            else:
                vision_states = None

        # Repeat image features for batch size if needed (for classifier-free guidance)
        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)

        return vision_states

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == "i2v":
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
            ).to(latents.device)

        mask_zeros = torch.zeros(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_ones = torch.ones(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_concat = merge_tensor_by_mask(
            mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2
        ).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def get_closest_resolution_given_reference_image(
        self, reference_image, target_resolution
    ):
        """
        Get closest supported resolution for a reference image.

        Args:
            reference_image: PIL Image or numpy array.
            target_resolution: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        assert reference_image is not None

        if isinstance(reference_image, Image.Image):
            origin_size = reference_image.size
        elif isinstance(reference_image, np.ndarray):
            H, W, C = reference_image.shape
            origin_size = (W, H)
        else:
            raise ValueError(
                f"Unsupported reference_image type: {type(reference_image)}. "
                f"Must be PIL Image or numpy array"
            )

        return self.get_closest_resolution_given_original_size(
            origin_size, target_resolution
        )

    def get_closest_resolution_given_original_size(self, origin_size, target_size):
        """
        Get closest supported resolution for given original size and target resolution.

        Args:
            origin_size: Tuple of (width, height) of original image.
            target_size: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        bucket_hw_base_size = self.target_size_config[target_size][
            "bucket_hw_base_size"
        ]
        bucket_hw_bucket_stride = self.target_size_config[target_size][
            "bucket_hw_bucket_stride"
        ]

        assert bucket_hw_base_size in [
            128,
            256,
            480,
            512,
            640,
            720,
            960,
            1440,
        ], f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960, 1440], but got {bucket_hw_base_size}"

        crop_size_list = generate_crop_size_list(
            bucket_hw_base_size, bucket_hw_bucket_stride
        )
        aspect_ratios = np.array(
            [round(float(h) / float(w), 5) for h, w in crop_size_list]
        )
        closest_size, closest_ratio = get_closest_ratio(
            origin_size[1], origin_size[0], aspect_ratios, crop_size_list
        )

        height = closest_size[0]
        width = closest_size[1]

        return height, width

    def get_image_condition_latents(self, task_type, reference_image, height, width):

        if task_type == "t2v":
            cond_latents = None

        elif task_type == "i2v":
            origin_size = reference_image.size

            target_height, target_width = height, width
            original_width, original_height = origin_size

            scale_factor = max(
                target_width / original_width, target_height / original_height
            )
            resize_width = int(round(original_width * scale_factor))
            resize_height = int(round(original_height * scale_factor))

            ref_image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (resize_height, resize_width),
                        interpolation=transforms.InterpolationMode.LANCZOS,
                    ),
                    transforms.CenterCrop((target_height, target_width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            ref_images_pixel_values = ref_image_transform(reference_image)
            ref_images_pixel_values = (
                ref_images_pixel_values.unsqueeze(0)
                .unsqueeze(2)
                .to(self.execution_device)
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                cond_latents = self.vae.encode(
                    ref_images_pixel_values
                ).latent_dist.mode()
                cond_latents.mul_(self.vae.config.scaling_factor)

        else:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Must be 't2v' or 'i2v'"
            )

        return cond_latents

    @property
    def vae_spatial_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_spatial"):
            return self.vae.config.ffactor_spatial
        else:
            return 16

    @property
    def vae_temporal_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_temporal"):
            return self.vae.config.ffactor_temporal
        else:
            return 4

    def get_latent_size(self, video_length, height, width):
        spatial_compression_ratio = self.vae_spatial_compression_ratio
        temporal_compression_ratio = self.vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        height, width = (
            height // spatial_compression_ratio,
            width // spatial_compression_ratio,
        )

        assert height > 0 and width > 0 and video_length > 0, (
            f"Invalid dimensions - height: {height}, width: {width}, "
            f"video_length: {video_length}"
        )

        return video_length, height, width

    def init_kv_cache(self):
        self._kv_cache = []
        self._kv_cache_neg = []
        transformer_num_layers = len(self.transformer.double_blocks)
        for i in range(transformer_num_layers):
            self._kv_cache.append(
                {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            )
            self._kv_cache_neg.append(
                {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            )

    def ar_rollout(
        self,
        latents,
        timesteps,
        prompt_embeds,
        prompt_mask,
        vision_states,
        cond_latents,
        task_type,
        extra_kwargs,
        viewmats,
        Ks,
        action,
        device,
        transformer_resident=False,
    ):
        # When transformer_resident is True, keep the transformer on GPU for the
        # entire AR rollout instead of loading/offloading it per chunk.  This does
        # not increase peak VRAM (other large modules are already offloaded before
        # ar_rollout is called) but removes per-chunk transfer overhead.
        if transformer_resident and self.enable_offloading:
            with auto_offload_model(
                self.transformer, self.execution_device, enabled=True
            ):
                orig = self.enable_offloading
                self.enable_offloading = False
                try:
                    return self._ar_rollout_inner(
                        latents, timesteps, prompt_embeds, prompt_mask,
                        vision_states, cond_latents, task_type, extra_kwargs,
                        viewmats, Ks, action, device,
                    )
                finally:
                    self.enable_offloading = orig
        else:
            return self._ar_rollout_inner(
                latents, timesteps, prompt_embeds, prompt_mask,
                vision_states, cond_latents, task_type, extra_kwargs,
                viewmats, Ks, action, device,
            )

    def _ar_rollout_inner(
        self,
        latents,
        timesteps,
        prompt_embeds,
        prompt_mask,
        vision_states,
        cond_latents,
        task_type,
        extra_kwargs,
        viewmats,
        Ks,
        action,
        device,
    ):
        rollout_start = time.perf_counter()
        perf_stats = {
            "chunks": [],
            "total_history_selection_time_s": 0.0,
            "total_recompute_time_s": 0.0,
            "total_denoise_time_s": 0.0,
            "total_dit_core_time_s": 0.0,
            "total_ar_rollout_inner_time_s": 0.0,
        }
        self.init_kv_cache()
        # Evaluation-only adapter runs need the checkpoint installed before the
        # first denoise forward. Adapter-only training installs it below, after
        # freezing the base DiT, so resumed weights participate causally in the
        # complete rollout without opening any pre-target autograd graph.
        if _dirty_history_any_adapter_enabled() and not _functional_ft_enabled():
            if _codec_native_history_adapter_enabled():
                _ensure_codec_native_history_adapter(self.transformer)
            elif _dirty_history_dynamics_adapter_enabled():
                _ensure_dirty_history_dynamics_adapter(self.transformer)
            else:
                _ensure_dirty_history_output_adapter(self.transformer)
        positive_idx = 1 if self.do_classifier_free_guidance else 0
        stabilization_level = 15
        # text, siglip, byt5 embedding cache
        with (
            torch.autocast(
                device_type="cuda",
                dtype=self.target_dtype,
                enabled=self.autocast_enabled,
            ),
            auto_offload_model(
                self.transformer, self.execution_device, enabled=self.enable_offloading
            ),
        ):
            extra_kwargs_pos = {
                "byt5_text_states": extra_kwargs["byt5_text_states"][
                    positive_idx, None, ...
                ],
                "byt5_text_mask": extra_kwargs["byt5_text_mask"][
                    positive_idx, None, ...
                ],
            }
            t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
            self._kv_cache = self.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds[positive_idx, None, ...],
                encoder_attention_mask=prompt_mask[positive_idx, None, ...],
                vision_states=vision_states[positive_idx, None, ...],
                mask_type=task_type,
                extra_kwargs=extra_kwargs_pos,
                kv_cache=self._kv_cache,
                cache_txt=True,
            )
            if self.do_classifier_free_guidance:
                extra_kwargs_neg = {
                    "byt5_text_states": extra_kwargs["byt5_text_states"][0, None, ...],
                    "byt5_text_mask": extra_kwargs["byt5_text_mask"][0, None, ...],
                }
                t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
                self._kv_cache_neg = self.transformer(
                    bi_inference=False,
                    ar_txt_inference=True,
                    ar_vision_inference=False,
                    timestep_txt=t_expand_txt,
                    text_states=prompt_embeds[0, None, ...],
                    encoder_attention_mask=prompt_mask[0, None, ...],
                    vision_states=vision_states[0, None, ...],
                    mask_type=task_type,
                    extra_kwargs=extra_kwargs_neg,
                    kv_cache=self._kv_cache_neg,
                    cache_txt=True,
                )

        retrieval_temporal_context_size = int(
            os.environ.get("HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE", "12")
        )
        retrieval_memory_frames = int(
            os.environ.get("HY_WORLDPLAY_MEMORY_FRAMES", "20")
        )
        if retrieval_temporal_context_size < 0:
            raise ValueError("HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE must be non-negative")
        if retrieval_memory_frames < retrieval_temporal_context_size + 4:
            raise ValueError(
                "HY_WORLDPLAY_MEMORY_FRAMES must reserve the first 4 latent frames "
                "in addition to HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE; got "
                f"memory={retrieval_memory_frames}, "
                f"temporal={retrieval_temporal_context_size}"
            )
        if get_rank() == 0:
            print(
                "[WorldPlayRetrieval] "
                f"memory_frames={retrieval_memory_frames} "
                f"temporal_context_size={retrieval_temporal_context_size}",
                flush=True,
            )

        selected_frame_indices = []
        rollout_shadow_memory: dict[int, torch.Tensor] = {}

        # Adapter-only functional training never updates the base DiT. Freeze it
        # before any history recompute so pre-target chunks do not build useless
        # parameter autograd graphs. The adapter is installed later at the first
        # selected atom and retains its own trainable parameters.
        if (
            _functional_ft_enabled()
            and _dirty_history_trainable_adapter_enabled()
            and not getattr(self.transformer, "_hy_functional_ft_frozen", False)
        ):
            self.transformer.requires_grad_(False)
            self.transformer._hy_functional_ft_frozen = True
            if get_rank() == 0:
                print(
                    "[FunctionalBlockFT] froze main Transformer before AR rollout",
                    flush=True,
                )
        if _functional_ft_enabled() and _codec_native_history_adapter_enabled():
            _ensure_codec_native_history_adapter(self.transformer)

        for chunk_i in range(self.chunk_num):
            transient_cache_restore = None
            transient_cache_restore_neg = None
            prediction_oracle_cache = None
            prediction_oracle_cache_neg = None
            _set_dirty_history_adapters_chunk_enabled(self.transformer, chunk_i)
            chunk_stats = {
                "chunk": chunk_i,
                "history_selection_time_s": 0.0,
                "recompute_time_s": 0.0,
                "denoise_time_s": 0.0,
            }
            if chunk_i > 0:
                history_start = time.perf_counter()
                current_frame_idx = (
                    chunk_i * self.chunk_latent_frames
                )  # the current frame index to generate

                selected_frame_indices = []
                for chunk_start_idx in range(
                    current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4
                ):
                    selected_history_frame_id = select_aligned_memory_frames(
                        viewmats[0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=retrieval_memory_frames,
                        temporal_context_size=retrieval_temporal_context_size,
                        pred_latent_size=4,
                        points_local=self.points_local,
                        device=device,
                    )
                    selected_frame_indices += selected_history_frame_id
                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                to_remove = list(
                    range(
                        current_frame_idx, current_frame_idx + self.chunk_latent_frames
                    )
                )
                selected_frame_indices = [
                    x for x in selected_frame_indices if x not in to_remove
                ]
                if _drift_env_enabled("HY_ROLLOUT_LOG_RETRIEVAL_INDICES") and get_rank() == 0:
                    print(
                        f"[RolloutRetrieval] chunk={chunk_i} boundary={chunk_i - 1} "
                        f"frames={selected_frame_indices}",
                        flush=True,
                    )
                _maybe_dump_rollout_retrieval_context(
                    latents=latents,
                    selected_frame_indices=selected_frame_indices,
                    boundary_chunk=chunk_i - 1,
                    current_frame_idx=current_frame_idx,
                    chunk_latent_frames=self.chunk_latent_frames,
                    viewmats=viewmats,
                    Ks=Ks,
                    action=action,
                )
                reset_scope = os.environ.get(
                    "HY_ROLLOUT_STATE_RESET_SCOPE", "prefix"
                ).strip().lower()
                reset_writeback = _drift_env_enabled(
                    "HY_ROLLOUT_STATE_RESET_WRITEBACK", "1"
                )
                transient_selected = (
                    reset_scope == "selected_context"
                    and not reset_writeback
                    and int(chunk_i - 1)
                    in _rollout_boundary_chunks("HY_ROLLOUT_STATE_RESET_CHUNKS")
                )
                if transient_selected:
                    transient_cache_restore = _text_only_kv_cache(self._kv_cache)
                    transient_cache_restore_neg = _text_only_kv_cache(self._kv_cache_neg)
                chunk_stats["history_selection_time_s"] = _elapsed_since(
                    history_start, device
                )

                context_latents = _maybe_reset_rollout_selected_context(
                    latents,
                    selected_frame_indices,
                    chunk_i - 1,
                    rollout_shadow_memory,
                )
                context_cond_latents_input = cond_latents[:, :, selected_frame_indices]
                context_action = action[:, selected_frame_indices].to(device)
                context_latents = _maybe_apply_rollout_context_corrector(
                    context_latents,
                    selected_frame_indices,
                    chunk_i - 1,
                    current_frame_idx,
                    self.chunk_latent_frames,
                    context_action,
                    viewmats,
                    Ks,
                )
                context_latents_input = torch.concat(
                    [context_latents, context_cond_latents_input], dim=1
                )

                context_viewmats = viewmats[:, selected_frame_indices].to(device)
                context_Ks = Ks[:, selected_frame_indices].to(device)
                context_timestep = torch.full(
                    (len(selected_frame_indices),),
                    stabilization_level - 1,
                    device=device,
                    dtype=timesteps.dtype,
                )
                # compute kv cache
                # Mark recomputation unconditionally so the low-dimensional
                # Attention runtime can keep numerically sensitive cache
                # writes on the strict backend.
                os.environ["HY_ROLLOUT_DRIFT_STAGE"] = "recompute"
                os.environ["HY_ROLLOUT_DRIFT_CHUNK"] = str(chunk_i)
                os.environ["HY_ROLLOUT_DRIFT_STEP"] = "-1"
                recompute_start = time.perf_counter()
                os.environ["HY_KV_MULTIPOS_AR_CHUNK_ID"] = str(chunk_i)
                if (
                    os.environ.get("HY_KV_MULTIPOS_DUMP_ENABLE", "0").lower()
                    in {"1", "true", "yes", "on"}
                    or os.environ.get("HY_KV_QK_SUBSET_DUMP_ENABLE", "0").lower()
                    in {"1", "true", "yes", "on"}
                ):
                    actual_context_len = int(
                        math.ceil(
                            len(selected_frame_indices)
                            / max(1, int(self.chunk_latent_frames))
                        )
                    )
                    os.environ["HY_KV_MULTIPOS_AR_CHUNK_ID"] = str(chunk_i)
                    os.environ["HY_KV_MULTIPOS_SELECTED_FRAME_INDICES"] = json.dumps(
                        selected_frame_indices
                    )
                    os.environ["HY_KV_MULTIPOS_CONTEXT_LEN"] = str(
                        actual_context_len
                    )
                    os.environ["HY_KV_MULTIPOS_CONTEXT_START"] = str(
                        min(selected_frame_indices) if selected_frame_indices else -1
                    )
                    os.environ["HY_KV_MULTIPOS_CONTEXT_END"] = str(
                        max(selected_frame_indices) if selected_frame_indices else -1
                    )
                    os.environ["HY_KV_MULTIPOS_EXPECTED_CONTEXT_LEN"] = (
                        "5" if chunk_i >= 4 else str(chunk_i + 1)
                    )
                if (
                    os.environ.get("HY_KV_AE3D_ATTN_DUMP_ENABLE", "0").lower()
                    in {"1", "true", "yes", "on"}
                    or os.environ.get(
                        "HY_CODEC_NATIVE_MIXED_DUMP_ENABLE", "0"
                    ).lower() in {"1", "true", "yes", "on"}
                ):
                    os.environ["HY_KV_AE3D_AR_CHUNK_ID"] = str(chunk_i)
                    os.environ["HY_KV_AE3D_SELECTED_FRAME_INDICES"] = json.dumps(
                        selected_frame_indices
                    )
                with (
                    torch.enable_grad() if _functional_ft_enabled() else nullcontext(),
                    torch.autocast(
                        device_type="cuda",
                        dtype=self.target_dtype,
                        enabled=self.autocast_enabled,
                    ),
                    auto_offload_model(
                        self.transformer,
                        self.execution_device,
                        enabled=self.enable_offloading,
                    ),
                ):
                    os.environ["HY_KV_MULTIPOS_DUMP_PASS"] = "pos"
                    rebuild_cache = _prepare_vision_cache_rebuild(
                        self._kv_cache, label="positive"
                    )
                    if rebuild_cache is not self._kv_cache:
                        self._kv_cache = None
                    self._kv_cache = self.transformer(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=context_latents_input,
                        timestep=context_timestep,
                        timestep_r=None,
                        mask_type=task_type,
                        return_dict=False,
                        viewmats=context_viewmats.to(self.target_dtype),
                        Ks=context_Ks.to(self.target_dtype),
                        action=context_action.to(self.target_dtype),
                        kv_cache=rebuild_cache,
                        cache_vision=True,
                        rope_temporal_size=context_latents_input.shape[2],
                        start_rope_start_idx=0,
                    )
                    del rebuild_cache
                    if self.do_classifier_free_guidance:
                        os.environ["HY_KV_MULTIPOS_DUMP_PASS"] = "neg"
                        rebuild_cache_neg = _prepare_vision_cache_rebuild(
                            self._kv_cache_neg, label="negative"
                        )
                        if rebuild_cache_neg is not self._kv_cache_neg:
                            self._kv_cache_neg = None
                        self._kv_cache_neg = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=context_latents_input,
                            timestep=context_timestep,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=context_viewmats.to(self.target_dtype),
                            Ks=context_Ks.to(self.target_dtype),
                            action=context_action.to(self.target_dtype),
                            kv_cache=rebuild_cache_neg,
                            cache_vision=True,
                            rope_temporal_size=context_latents_input.shape[2],
                            start_rope_start_idx=0,
                        )
                        del rebuild_cache_neg
                    os.environ["HY_KV_MULTIPOS_DUMP_PASS"] = "pos"

                    if (
                        _prediction_oracle_boundary_enabled(chunk_i - 1)
                        and _dirty_history_recovery_teacher_mode() != "local_exact"
                    ):
                        clean_context_latents = _load_prediction_oracle_clean_context(
                            selected_frame_indices,
                            chunk_i - 1,
                            context_latents.device,
                            context_latents.dtype,
                        )
                        teacher_mode = _dirty_history_recovery_teacher_mode()
                        if teacher_mode == "hybrid_exact":
                            clean_context_latents = _hybrid_teacher_context(
                                context_latents,
                                clean_context_latents,
                                selected_frame_indices,
                                current_frame_idx,
                                self.chunk_latent_frames,
                            )
                        clean_context_input = torch.concat(
                            [clean_context_latents, context_cond_latents_input], dim=1
                        )
                        compression_off = {
                            "HY_LAYERWISE_RAW_V_AE": "0",
                            "HY_KV_CORRUPTION_ENABLE": "0",
                            "HY_KV_CORRUPTION_K_ENABLE": "0",
                            "HY_KV_AE3D_ENABLE": "0",
                            "HY_KV_CODEC_ENABLE": "0",
                            "HY_PREKV_AE_ENABLE": "0",
                        }
                        compression_context = (
                            _temporary_env(compression_off)
                            if teacher_mode in {"exact_clean", "hybrid_exact"}
                            else nullcontext()
                        )
                        with compression_context, torch.no_grad():
                            prediction_oracle_cache = self.transformer(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                hidden_states=clean_context_input,
                                timestep=context_timestep,
                                timestep_r=None,
                                mask_type=task_type,
                                return_dict=False,
                                viewmats=context_viewmats.to(self.target_dtype),
                                Ks=context_Ks.to(self.target_dtype),
                                action=context_action.to(self.target_dtype),
                                kv_cache=_text_only_kv_cache(self._kv_cache),
                                cache_vision=True,
                                rope_temporal_size=clean_context_input.shape[2],
                                start_rope_start_idx=0,
                            )
                            if self.do_classifier_free_guidance:
                                prediction_oracle_cache_neg = self.transformer(
                                    bi_inference=False,
                                    ar_txt_inference=False,
                                    ar_vision_inference=True,
                                    hidden_states=clean_context_input,
                                    timestep=context_timestep,
                                    timestep_r=None,
                                    mask_type=task_type,
                                    return_dict=False,
                                    viewmats=context_viewmats.to(self.target_dtype),
                                    Ks=context_Ks.to(self.target_dtype),
                                    action=context_action.to(self.target_dtype),
                                    kv_cache=_text_only_kv_cache(self._kv_cache_neg),
                                    cache_vision=True,
                                    rope_temporal_size=clean_context_input.shape[2],
                                    start_rope_start_idx=0,
                                )
                        if get_rank() == 0:
                            print(
                                f"[DirtyHistoryPredOracle] prepared boundary={chunk_i - 1} "
                                f"frames={selected_frame_indices} clean_kv=1",
                                f"teacher_mode={teacher_mode} "
                                f"hybrid_beta={os.environ.get('HY_DIRTY_HISTORY_RECOVERY_HYBRID_BETA', '1.0')}",
                                flush=True,
                            )

                chunk_stats["recompute_time_s"] = _elapsed_since(
                    recompute_start, device
                )
                self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            start_idx = chunk_i * self.chunk_latent_frames
            end_idx = chunk_i * self.chunk_latent_frames + self.chunk_latent_frames

            denoise_start = time.perf_counter()
            if _k_t0_chunk_enabled(chunk_i):
                t0_eval_only = _drift_env_enabled("HY_K_T0_EVAL_ONLY")
                if self.num_inference_steps != 4 or len(timesteps) != 4:
                    raise RuntimeError(
                        "K T0 Gate-1 requires exactly four denoise steps, got "
                        f"num_inference_steps={self.num_inference_steps} "
                        f"timesteps={len(timesteps)}"
                    )
                if self.do_classifier_free_guidance:
                    raise RuntimeError(
                        "K T0 Gate-1 currently requires guidance_scale=1/no CFG; "
                        "CFG support must preserve both positive and negative K54 graphs"
                    )
                if not getattr(self.transformer, "_hy_functional_ft_frozen", False):
                    self.transformer.requires_grad_(False)
                    self.transformer._hy_functional_ft_frozen = True
                target_layers = set(range(len(self.transformer.double_blocks)))
                if len(target_layers) != 54:
                    raise RuntimeError(
                        f"K T0 expected 54 Transformer layers, got {len(target_layers)}"
                    )
                clean_cache = _functional_ft_teacher_target_cache(
                    self._kv_cache, target_layers
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                t0_forward_start = time.perf_counter()

                # Teacher is fully evaluated before any differentiable K54
                # decoder graph exists.  Targets are moved to CPU immediately,
                # so clean Teacher and Student activations never overlap.
                cond_chunk = cond_latents[:, :, start_idx:end_idx]
                view_chunk = viewmats[:, start_idx:end_idx].to(device)
                intrinsics_chunk = Ks[:, start_idx:end_idx].to(device)
                action_chunk = action[:, start_idx:end_idx].to(device)
                teacher_scheduler = copy.deepcopy(self.scheduler)
                teacher_x = latents[:, :, start_idx:end_idx].detach().clone()
                atom_initial_sha256 = _k_t0_tensor_sha256(teacher_x)
                teacher_states_cpu = []
                teacher_predictions_cpu = []
                reset_functional_block_losses()
                for t0_step, t0_timestep in enumerate(timesteps):
                    os.environ["HY_ROLLOUT_DRIFT_STEP"] = str(t0_step)
                    t0_timestep_input = torch.full(
                        (self.chunk_latent_frames,),
                        t0_timestep,
                        device=device,
                        dtype=timesteps.dtype,
                    )
                    teacher_input = torch.concat([teacher_x, cond_chunk], dim=1)
                    teacher_input = teacher_scheduler.scale_model_input(
                        teacher_input, t0_timestep
                    )
                    os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "teacher"
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type="cuda",
                            dtype=self.target_dtype,
                            enabled=self.autocast_enabled,
                        ),
                    ):
                        teacher_prediction = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=teacher_input,
                            timestep=t0_timestep_input,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=view_chunk.to(self.target_dtype),
                            Ks=intrinsics_chunk.to(self.target_dtype),
                            action=action_chunk.to(self.target_dtype),
                            kv_cache=clean_cache,
                            cache_vision=False,
                            rope_temporal_size=teacher_input.shape[2]
                            + len(selected_frame_indices),
                            start_rope_start_idx=len(selected_frame_indices),
                        )[0]
                        teacher_x = teacher_scheduler.step(
                            teacher_prediction,
                            t0_timestep,
                            teacher_x,
                            return_dict=False,
                        )[0].detach()
                    teacher_predictions_cpu.append(
                        teacher_prediction.to(device="cpu", dtype=torch.float16)
                    )
                    teacher_states_cpu.append(
                        teacher_x.to(device="cpu", dtype=torch.float16)
                    )
                    del teacher_input, teacher_prediction
                teacher_endpoint_cpu = teacher_states_cpu[-1]
                del teacher_scheduler, teacher_x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                student_cache = [dict(layer_cache) for layer_cache in self._kv_cache]
                zero_k_functional_train_grads()
                reset_v_final_prediction_tensor_audit()
                os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "student"
                offload_saved_tensors = _drift_env_enabled(
                    "HY_K_T0_SAVE_AUTOGRAD_ON_CPU"
                )
                for layer_idx in sorted(target_layers):
                    saved_tensor_context = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if offload_saved_tensors
                        else nullcontext()
                    )
                    with saved_tensor_context:
                        student_cache[layer_idx] = reapply_k_corruption_for_ft(
                            student_cache[layer_idx], layer_idx
                        )
                sampled_layers = (0, 13, 27, 40, 53)
                for layer_idx in sampled_layers:
                    restored_k = student_cache[layer_idx].get("k_vision")
                    if (
                        not t0_eval_only
                        and (
                        not torch.is_tensor(restored_k)
                        or not restored_k.requires_grad
                        or restored_k.grad_fn is None
                        )
                    ):
                        raise RuntimeError(
                            f"K T0 restored K lost autograd graph at L{layer_idx}"
                        )

                k_reconstruction = []
                with torch.no_grad():
                    for layer_idx in sorted(target_layers):
                        restored_k = student_cache[layer_idx].get("k_vision")
                        clean_k = student_cache[layer_idx].get(
                            "_k_vision_clean_for_ft"
                        )
                        if torch.is_tensor(restored_k) and torch.is_tensor(clean_k):
                            restored_float = restored_k.detach().float()
                            clean_float = clean_k.detach().float()
                            k_reconstruction.append(
                                {
                                    "layer": int(layer_idx),
                                    "rel_l2": float(
                                        (
                                            (restored_float - clean_float).norm()
                                            / clean_float.norm().clamp_min(1e-8)
                                        ).cpu()
                                    ),
                                    "mse": float(
                                        (restored_float - clean_float)
                                        .pow(2).mean().cpu()
                                    ),
                                }
                            )

                student_scheduler = copy.deepcopy(self.scheduler)
                student_x = latents[:, :, start_idx:end_idx].detach().clone()
                student_states = []
                student_predictions = []
                block_metrics_all = []
                for t0_step, t0_timestep in enumerate(timesteps):
                    os.environ["HY_ROLLOUT_DRIFT_STEP"] = str(t0_step)
                    t0_timestep_input = torch.full(
                        (self.chunk_latent_frames,),
                        t0_timestep,
                        device=device,
                        dtype=timesteps.dtype,
                    )
                    student_input = torch.concat([student_x, cond_chunk], dim=1)
                    student_input = student_scheduler.scale_model_input(
                        student_input, t0_timestep
                    )
                    os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "student"
                    saved_tensor_context = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if offload_saved_tensors
                        else nullcontext()
                    )
                    with (
                        saved_tensor_context,
                        torch.enable_grad(),
                        torch.autocast(
                            device_type="cuda",
                            dtype=self.target_dtype,
                            enabled=self.autocast_enabled,
                        ),
                    ):
                        student_prediction = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=student_input,
                            timestep=t0_timestep_input,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=view_chunk.to(self.target_dtype),
                            Ks=intrinsics_chunk.to(self.target_dtype),
                            action=action_chunk.to(self.target_dtype),
                            kv_cache=student_cache,
                            cache_vision=False,
                            rope_temporal_size=student_input.shape[2]
                            + len(selected_frame_indices),
                            start_rope_start_idx=len(selected_frame_indices),
                        )[0]
                        student_x = student_scheduler.step(
                            student_prediction,
                            t0_timestep,
                            student_x,
                            return_dict=False,
                        )[0]
                    unused_block_loss, block_metrics = pop_functional_block_loss(
                        preserve_targets=True
                    )
                    block_metrics_all.extend(block_metrics)
                    del unused_block_loss
                    if (
                        not t0_eval_only
                        and (not student_x.requires_grad or student_x.grad_fn is None)
                    ):
                        raise RuntimeError(
                            f"K T0 Student x{t0_step + 1} lost autograd graph"
                        )
                    if student_x.requires_grad:
                        student_x.retain_grad()
                    if student_prediction.requires_grad:
                        student_prediction.retain_grad()
                    student_states.append(student_x)
                    student_predictions.append(student_prediction)

                # ``__call__`` is inference-decorated, so even though student_x
                # carries the four-step graph, endpoint arithmetic would be
                # silently detached unless it explicitly overrides no_grad.
                with torch.enable_grad():
                    teacher_endpoint = teacher_endpoint_cpu.to(
                        device=student_x.device, dtype=student_x.dtype
                    )
                    endpoint_diff = student_x.float() - teacher_endpoint.float()
                    endpoint_loss = endpoint_diff.pow(2).mean()
                if (
                    not t0_eval_only
                    and (not endpoint_loss.requires_grad or endpoint_loss.grad_fn is None)
                ):
                    raise RuntimeError("K T0 endpoint loss has no autograd graph")
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t0_forward_s = time.perf_counter() - t0_forward_start
                if t0_eval_only:
                    audit = {
                        "loss": float(endpoint_loss.detach().float().cpu()),
                        "num_layers": 0,
                        "rows": [],
                    }
                    t0_backward_s = 0.0
                else:
                    t0_backward_start = time.perf_counter()
                    audit = backward_k_functional_audit(endpoint_loss)
                    gradient_snapshot_path = os.environ.get(
                        "HY_K_T0_GRADIENT_SNAPSHOT", ""
                    ).strip()
                    if gradient_snapshot_path:
                        audit["gradient_snapshot"] = save_k_functional_gradient_snapshot(
                            gradient_snapshot_path,
                            atom_id=os.environ.get(
                                "HY_K_T0_ATOM_ID", f"chunk{int(chunk_i):02d}"
                            ),
                            loss=float(endpoint_loss.detach().float().cpu()),
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    t0_backward_s = time.perf_counter() - t0_backward_start
                audit["chunk"] = int(chunk_i)
                audit["mode"] = "K54_decoder_only_four_step_endpoint_gate1"
                audit["forward_s"] = float(t0_forward_s)
                audit["backward_s"] = float(t0_backward_s)
                audit["peak_allocated_gb"] = (
                    float(torch.cuda.max_memory_allocated(device) / (1024**3))
                    if torch.cuda.is_available()
                    else 0.0
                )
                audit["peak_reserved_gb"] = (
                    float(torch.cuda.max_memory_reserved(device) / (1024**3))
                    if torch.cuda.is_available()
                    else 0.0
                )
                audit["states"] = [
                    {
                        "step": int(step_idx),
                        "student_requires_grad": bool(state.requires_grad),
                        "student_grad_norm": 0.0
                        if state.grad is None
                        else float(state.grad.detach().float().norm().cpu()),
                        "prediction_grad_norm": 0.0
                        if student_predictions[step_idx].grad is None
                        else float(
                            student_predictions[step_idx]
                            .grad.detach().float().norm().cpu()
                        ),
                        "mse_to_teacher": float(
                            (
                                state.detach().float()
                                - teacher_states_cpu[step_idx]
                                .to(device=state.device, dtype=state.dtype)
                                .float()
                            ).pow(2).mean().cpu()
                        ),
                    }
                    for step_idx, state in enumerate(student_states)
                ]
                audit["prediction_mse"] = [
                    float(
                        (
                            prediction.detach().float()
                            - teacher_predictions_cpu[step_idx]
                            .to(device=prediction.device, dtype=prediction.dtype)
                            .float()
                        ).pow(2).mean().cpu()
                    )
                    for step_idx, prediction in enumerate(student_predictions)
                ]
                audit["atom_initial_sha256"] = atom_initial_sha256
                audit["teacher_state_sha256"] = [
                    _k_t0_tensor_sha256(value) for value in teacher_states_cpu
                ]
                audit["teacher_prediction_sha256"] = [
                    _k_t0_tensor_sha256(value) for value in teacher_predictions_cpu
                ]
                audit["k_reconstruction"] = k_reconstruction
                audit["block_metrics"] = block_metrics_all
                audit["sampled_restored_k"] = {
                    str(layer_idx): {
                        "requires_grad": bool(
                            student_cache[layer_idx]["k_vision"].requires_grad
                        ),
                        "has_grad_fn": (
                            student_cache[layer_idx]["k_vision"].grad_fn is not None
                        ),
                    }
                    for layer_idx in sampled_layers
                }
                if not t0_eval_only and audit["num_layers"] != 54:
                    raise RuntimeError(
                        f"K T0 audit found {audit['num_layers']} trainable layers, expected 54"
                    )
                bad_rows = [
                    row
                    for row in audit["rows"]
                    if (
                        not row["finite"]
                        or row["decoder_grad_norm"] <= 0.0
                        or row["encoder_numel"] != 0
                    )
                ]
                if not t0_eval_only and bad_rows:
                    raise RuntimeError(
                        "K T0 decoder-only gradient gate failed: "
                        + json.dumps(bad_rows[:8], sort_keys=True)
                    )
                out_root = Path(
                    os.environ.get(
                        "HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT",
                        "kv_ae_outputs/ae16x_v2/k_t0_gate1",
                    )
                )
                out_root.mkdir(parents=True, exist_ok=True)
                audit_name = "eval" if t0_eval_only else "gate1"
                audit_path = out_root / f"{audit_name}_chunk{int(chunk_i):02d}.json"
                audit_path.write_text(
                    json.dumps(audit, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if get_rank() == 0:
                    print(
                        "[K-T0-Gate1] "
                        f"chunk={chunk_i} endpoint={audit['loss']:.8e} "
                        f"layers={audit['num_layers']} "
                        f"forward={t0_forward_s:.2f}s backward={t0_backward_s:.2f}s "
                        f"peak={audit['peak_allocated_gb']:.2f}GiB "
                        f"path={audit_path}",
                        flush=True,
                    )
                os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = ""
                zero_k_functional_train_grads()
                del student_cache, clean_cache, student_states, teacher_states_cpu
                del student_predictions, endpoint_loss, endpoint_diff
                del teacher_endpoint, teacher_endpoint_cpu
                del teacher_predictions_cpu, k_reconstruction, block_metrics_all
                reset_functional_block_losses()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            with (
                self.progress_bar(total=self.num_inference_steps) as progress_bar,
                auto_offload_model(
                    self.transformer,
                    self.execution_device,
                    enabled=self.enable_offloading,
                ),
            ):
                stop_after_functional_update = False
                for i, t in enumerate(timesteps):
                    # Backend quality guards need the location even when no
                    # diagnostic dump or fine-tuning feature is enabled.
                    os.environ["HY_ROLLOUT_DRIFT_STAGE"] = "denoise"
                    os.environ["HY_ROLLOUT_DRIFT_CHUNK"] = str(chunk_i)
                    os.environ["HY_ROLLOUT_DRIFT_STEP"] = str(i)
                    if (
                        _drift_env_enabled("HY_ROLLOUT_DRIFT_DUMP_ENABLE")
                        or _drift_env_enabled("HY_K_REPLAY_DUMP_ENABLE")
                        or _functional_ft_enabled()
                        or bool(
                            os.environ.get(
                                "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_STEPS", ""
                            ).strip()
                        )
                        or bool(
                            os.environ.get(
                                "HY_DIRECT_CODEC_PERIODIC_REFRESH_STEPS", ""
                            ).strip()
                        )
                    ):
                        os.environ["HY_ROLLOUT_DRIFT_STAGE"] = "denoise"
                        os.environ["HY_ROLLOUT_DRIFT_CHUNK"] = str(chunk_i)
                        os.environ["HY_ROLLOUT_DRIFT_STEP"] = str(i)
                    timestep_input = torch.full(
                        (self.chunk_latent_frames,),
                        t,
                        device=device,
                        dtype=timesteps.dtype,
                    )
                    latent_model_input = latents[:, :, start_idx:end_idx]
                    cond_latents_input = cond_latents[:, :, start_idx:end_idx]
                    _dump_rollout_latent_tensor("x_in", latent_model_input, chunk_i, i)

                    viewmats_input = viewmats[:, start_idx:end_idx].to(device)
                    Ks_input = Ks[:, start_idx:end_idx].to(device)
                    action_input = action[:, start_idx:end_idx].to(device)

                    latents_concat = torch.concat(
                        [latent_model_input, cond_latents_input], dim=1
                    )
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    teacher_force_all = _functional_ft_teacher_force_all_steps()
                    if _functional_ft_step_enabled(i):
                        if not getattr(self.transformer, "_hy_functional_ft_frozen", False):
                            self.transformer.requires_grad_(False)
                            self.transformer._hy_functional_ft_frozen = True
                            if get_rank() == 0:
                                print(
                                    "[FunctionalBlockFT] froze main Transformer parameters",
                                    flush=True,
                                )
                        adapter_training = _dirty_history_trainable_adapter_enabled()
                        if adapter_training:
                            adapter = (
                                _ensure_codec_native_history_adapter(self.transformer)
                                if _codec_native_history_adapter_training_enabled()
                                else (
                                    _ensure_dirty_history_dynamics_adapter(self.transformer)
                                    if _dirty_history_dynamics_adapter_enabled()
                                    else _ensure_dirty_history_output_adapter(self.transformer)
                                )
                            )
                            adapter.enabled = _dirty_history_output_adapter_chunk_enabled(chunk_i)
                        reset_functional_block_losses()
                        reset_functional_readout_losses()
                        reset_codec_native_routing_losses()
                        _functional_ft_cache_debug(self._kv_cache, chunk_i, i)
                        functional_loss_mode = _functional_ft_loss_mode()
                        trajectory_input_rel_l2 = 0.0
                        target_layers = {
                            int(item.strip())
                            for item in os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "13,40,53").split(",")
                            if item.strip()
                        }
                        if functional_loss_mode == "trajectory_recovery":
                            # The target comes from a separately collected, fully clean
                            # rollout.  Running a local Teacher here would recreate the
                            # inconsistent dirty-current + clean-history target that this
                            # mode is specifically designed to avoid.
                            clean_cache = None
                        elif functional_loss_mode == "prediction":
                            recovery_teacher_mode = _dirty_history_recovery_teacher_mode()
                            if recovery_teacher_mode == "local_exact":
                                clean_cache = _functional_ft_teacher_target_cache(
                                    self._kv_cache, target_layers
                                )
                            else:
                                if prediction_oracle_cache is None:
                                    raise RuntimeError(
                                        "Dirty-history prediction FT requires a matched clean-history "
                                        f"cache for chunk={chunk_i} boundary={chunk_i - 1} "
                                        f"teacher_mode={recovery_teacher_mode}"
                                    )
                                clean_cache = prediction_oracle_cache
                        else:
                            clean_cache = _functional_ft_teacher_target_cache(
                                self._kv_cache, target_layers
                            )
                        if functional_loss_mode == "trajectory_recovery":
                            teacher_noise_pred = None
                            clean_trajectory_x_in = _load_clean_trajectory_tensor(
                                "x_in",
                                chunk_i,
                                i,
                                device=latent_model_input.device,
                                dtype=latent_model_input.dtype,
                            )
                            if tuple(clean_trajectory_x_in.shape) != tuple(latent_model_input.shape):
                                raise RuntimeError(
                                    "Clean trajectory x_in shape does not match Student current "
                                    f"latent: clean={tuple(clean_trajectory_x_in.shape)} "
                                    f"student={tuple(latent_model_input.shape)} chunk={chunk_i} step={i}"
                                )
                            trajectory_input_rel_l2 = float(
                                (
                                    (latent_model_input.float() - clean_trajectory_x_in.float()).norm()
                                    / clean_trajectory_x_in.float().norm().clamp_min(1e-8)
                                ).cpu()
                            )
                            del clean_trajectory_x_in
                            teacher_latent_target = _load_clean_trajectory_tensor(
                                "x_next",
                                chunk_i,
                                i,
                                device=latent_model_input.device,
                                dtype=latent_model_input.dtype,
                            )
                            if tuple(teacher_latent_target.shape) != tuple(latent_model_input.shape):
                                raise RuntimeError(
                                    "Clean trajectory x_next shape does not match Student current "
                                    f"latent: clean={tuple(teacher_latent_target.shape)} "
                                    f"student={tuple(latent_model_input.shape)} chunk={chunk_i} step={i}"
                                )
                        else:
                            os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "teacher"
                            with (
                                _dirty_history_output_adapter_disabled(self.transformer),
                                torch.no_grad(),
                                torch.autocast(
                                    device_type="cuda",
                                    dtype=self.target_dtype,
                                    enabled=self.autocast_enabled,
                                ),
                            ):
                                teacher_noise_pred = self.transformer(
                                    bi_inference=False,
                                    ar_txt_inference=False,
                                    ar_vision_inference=True,
                                    hidden_states=latents_concat.detach(),
                                    timestep=timestep_input,
                                    timestep_r=None,
                                    mask_type=task_type,
                                    return_dict=False,
                                    viewmats=viewmats_input.to(self.target_dtype),
                                    Ks=Ks_input.to(self.target_dtype),
                                    action=action_input.to(self.target_dtype),
                                    kv_cache=clean_cache,
                                    cache_vision=False,
                                    rope_temporal_size=latents_concat.shape[2]
                                    + len(selected_frame_indices),
                                    start_rope_start_idx=len(selected_frame_indices),
                                )[0]
                                if functional_loss_mode == "final_latent":
                                    teacher_latent_target = _scheduler_step_no_advance(
                                        self.scheduler,
                                        teacher_noise_pred,
                                        t,
                                        latent_model_input.detach(),
                                    ).detach()
                                else:
                                    teacher_latent_target = None
                        os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = ""
                        zero_k_functional_train_grads()
                        _zero_dirty_history_output_adapter_grads(self.transformer)

                        # Inner loop: multiple student forward+backward steps on same input
                        inner_steps = _functional_ft_inner_steps_for_chunk(chunk_i)
                        log_every = max(1, int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LOG_EVERY", "1")))
                        student_rollout_noise_pred = None

                        # Drain stale losses (should be none)
                        pop_functional_block_loss(preserve_targets=True)
                        pop_functional_readout_loss(preserve_targets=True)
                        _functional_ft_total_step = 0
                        functional_guard_baselines = {"local": {}, "reconstruction": {}}
                        functional_guard_totals = {
                            "local_triggers": 0,
                            "local_checks": 0,
                            "recon_triggers": 0,
                            "recon_checks": 0,
                            "active_steps": 0,
                            "guard_ratio_sum": 0.0,
                            "guard_ratio_max": 0.0,
                        }

                        for ft_inner in range(inner_steps):
                            os.environ["HY_FUNCTIONAL_BLOCK_FT_INNER_INDEX"] = str(ft_inner)
                            zero_k_functional_train_grads()
                            _zero_dirty_history_output_adapter_grads(self.transformer)
                            reset_v_final_prediction_tensor_audit()
                            # Re-run K codec to refresh corrupted K in cache with updated weights
                            if not adapter_training:
                                for layer_idx in target_layers:
                                    if layer_idx < len(self._kv_cache):
                                        if _functional_ft_target() in {"k", "kv"}:
                                            self._kv_cache[layer_idx] = reapply_k_corruption_for_ft(
                                                self._kv_cache[layer_idx], layer_idx
                                            )
                                        if _functional_ft_target() in {"v", "kv"}:
                                            self._kv_cache[layer_idx] = reapply_v_corruption_for_ft(
                                                self._kv_cache[layer_idx], layer_idx
                                            )

                            student_cache = _functional_ft_student_target_cache(
                                self._kv_cache, target_layers
                            )
                            os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = "student"
                            student_grad_enabled = not _drift_env_enabled(
                                "HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY"
                            )
                            saved_tensor_context = (
                                torch.autograd.graph.save_on_cpu(pin_memory=True)
                                if student_grad_enabled
                                and _drift_env_enabled(
                                    "HY_FUNCTIONAL_BLOCK_FT_SAVE_AUTOGRAD_ON_CPU"
                                )
                                else nullcontext()
                            )
                            with (
                                torch.set_grad_enabled(student_grad_enabled),
                                saved_tensor_context,
                                torch.autocast(
                                    device_type="cuda",
                                    dtype=self.target_dtype,
                                    enabled=self.autocast_enabled,
                                ),
                            ):
                                noise_pred = self.transformer(
                                    bi_inference=False,
                                    ar_txt_inference=False,
                                    ar_vision_inference=True,
                                    hidden_states=latents_concat.detach(),
                                    timestep=timestep_input,
                                    timestep_r=None,
                                    mask_type=task_type,
                                    return_dict=False,
                                    viewmats=viewmats_input.to(self.target_dtype),
                                    Ks=Ks_input.to(self.target_dtype),
                                    action=action_input.to(self.target_dtype),
                                    kv_cache=student_cache,
                                    cache_vision=False,
                                    rope_temporal_size=latents_concat.shape[2]
                                    + len(selected_frame_indices),
                                    start_rope_start_idx=len(selected_frame_indices),
                                )[0]
                            # Keep only the final Student prediction value for continued rollout.
                            # Its training graph is released below after the functional update.
                            student_rollout_noise_pred = noise_pred.detach()
                            clean_adapter_noise_pred = None
                            clean_adapter_weight = float(
                                os.environ.get(
                                    "HY_DIRTY_HISTORY_OUTPUT_ADAPTER_CLEAN_WEIGHT", "0.2"
                                )
                            )
                            if (
                                adapter_training
                                and clean_adapter_weight > 0.0
                                and clean_cache is not None
                            ):
                                with (
                                    torch.enable_grad(),
                                    torch.autocast(
                                        device_type="cuda",
                                        dtype=self.target_dtype,
                                        enabled=self.autocast_enabled,
                                    ),
                                ):
                                    clean_adapter_noise_pred = self.transformer(
                                        bi_inference=False,
                                        ar_txt_inference=False,
                                        ar_vision_inference=True,
                                        hidden_states=latents_concat.detach(),
                                        timestep=timestep_input,
                                        timestep_r=None,
                                        mask_type=task_type,
                                        return_dict=False,
                                        viewmats=viewmats_input.to(self.target_dtype),
                                        Ks=Ks_input.to(self.target_dtype),
                                        action=action_input.to(self.target_dtype),
                                        kv_cache=clean_cache,
                                        cache_vision=False,
                                        rope_temporal_size=latents_concat.shape[2]
                                        + len(selected_frame_indices),
                                        start_rope_start_idx=len(selected_frame_indices),
                                    )[0]
                            block_loss = None
                            block_metrics = []
                            readout_loss = None
                            readout_metrics = []
                            latent_loss = None
                            latent_mse = 0.0
                            latent_rel_l2 = 0.0
                            trajectory_grad_rel_mse = 0.0
                            prediction_loss = None
                            raw_prediction_mse = 0.0
                            prediction_rel_l2 = 0.0
                            prediction_cosine = 0.0
                            prediction_solver_mse = 0.0
                            prediction_solver_rel_l2 = 0.0
                            clean_preserve_loss = None
                            clean_preserve_rel_l2 = 0.0
                            routing_loss = None
                            routing_metrics = []
                            block_components = {}
                            if functional_loss_mode == "block":
                                block_components = get_functional_block_loss_components()
                                block_loss, block_metrics = pop_functional_block_loss(preserve_targets=True)
                                readout_loss, readout_metrics = pop_functional_readout_loss(preserve_targets=True)
                            elif functional_loss_mode == "readout":
                                pop_functional_block_loss(preserve_targets=True)
                                readout_loss, readout_metrics = pop_functional_readout_loss(preserve_targets=True)
                            elif functional_loss_mode == "prediction":
                                pop_functional_block_loss(preserve_targets=True)
                                pop_functional_readout_loss(preserve_targets=True)
                                with torch.enable_grad():
                                    student_pred_float = noise_pred.float()
                                    teacher_pred_float = teacher_noise_pred.float()
                                    prediction_diff = student_pred_float - teacher_pred_float
                                    raw_prediction_mse = float(
                                        prediction_diff.detach().pow(2).mean().cpu()
                                    )
                                    prediction_rel = prediction_diff.norm() / teacher_pred_float.norm().clamp_min(1e-8)
                                    prediction_cos = F.cosine_similarity(
                                        student_pred_float.reshape(1, -1),
                                        teacher_pred_float.reshape(1, -1),
                                        dim=1,
                                    ).mean()
                                    prediction_cos_weight = float(
                                        os.environ.get(
                                            "HY_FUNCTIONAL_BLOCK_FT_PRED_COS_WEIGHT", "0.1"
                                        )
                                    )
                                    prediction_objective = os.environ.get(
                                        "HY_FUNCTIONAL_BLOCK_FT_PRED_OBJECTIVE", "relative_cosine"
                                    ).strip().lower()
                                    if prediction_objective in {"mse", "mean_squared_error"}:
                                        prediction_loss = prediction_diff.pow(2).mean()
                                    else:
                                        prediction_loss = prediction_rel + prediction_cos_weight * (
                                            1.0 - prediction_cos
                                        )
                                    if clean_adapter_noise_pred is not None:
                                        clean_pred_float = clean_adapter_noise_pred.float()
                                        clean_diff = clean_pred_float - teacher_pred_float
                                        clean_preserve_rel = clean_diff.norm() / teacher_pred_float.norm().clamp_min(1e-8)
                                        clean_preserve_loss = (
                                            clean_adapter_weight * clean_preserve_rel
                                        )
                                        prediction_loss = prediction_loss + clean_preserve_loss
                                        clean_preserve_rel_l2 = float(clean_preserve_rel.detach().cpu())
                                    prediction_rel_l2 = float(prediction_rel.detach().cpu())
                                    prediction_cosine = float(prediction_cos.detach().cpu())
                            elif functional_loss_mode == "routing":
                                pop_functional_block_loss(preserve_targets=True)
                                pop_functional_readout_loss(preserve_targets=True)
                                routing_loss, routing_metrics = (
                                    pop_codec_native_routing_loss(
                                        preserve_targets=True
                                    )
                                )
                                if routing_loss is None:
                                    raise RuntimeError(
                                        "routing loss mode produced no selected-layer "
                                        "Teacher/Student routing pairs"
                                    )
                            elif teacher_latent_target is not None:
                                # Drain block sketches created by PASS=student so their graphs do not
                                # accumulate across inner steps. The final-latent mode does not train
                                # on this loss.
                                pop_functional_block_loss(preserve_targets=True)
                                pop_functional_readout_loss(preserve_targets=True)
                                with torch.enable_grad():
                                    student_latent_target = _scheduler_step_no_advance(
                                        self.scheduler,
                                        noise_pred,
                                        t,
                                        latent_model_input.detach(),
                                    )
                                    teacher_latent_dev = teacher_latent_target.to(
                                        device=student_latent_target.device,
                                        dtype=student_latent_target.dtype,
                                    )
                                    latent_diff = student_latent_target.float() - teacher_latent_dev.float()
                                    if teacher_noise_pred is not None:
                                        raw_prediction_mse = float(
                                            (
                                                noise_pred.detach().float()
                                                - teacher_noise_pred.detach().float()
                                            ).pow(2).mean().cpu()
                                        )
                                    latent_energy = (
                                        teacher_latent_dev.float().pow(2).sum().clamp_min(1e-8)
                                    )
                                    latent_rel_mse = latent_diff.pow(2).sum() / latent_energy
                                    latent_rel = latent_rel_mse.sqrt()
                                    latent_mse_tensor = latent_diff.pow(2).mean()
                                    latent_weight = float(
                                        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LATENT_WEIGHT", "1.0")
                                    )
                                    latent_objective = os.environ.get(
                                        "HY_FUNCTIONAL_BLOCK_FT_LATENT_OBJECTIVE", "relative_mse"
                                    ).strip().lower()
                                    if latent_objective in {"mse", "mean_squared_error"}:
                                        latent_loss = latent_mse_tensor * latent_weight
                                    elif latent_objective in {
                                        "relative_mse",
                                        "normalized_mse",
                                        "relative_squared_l2",
                                    }:
                                        latent_loss = latent_rel_mse * latent_weight
                                    else:
                                        latent_loss = latent_rel * latent_weight
                                    if functional_loss_mode == "trajectory_recovery":
                                        trajectory_grad_weight = float(
                                            os.environ.get(
                                                "HY_FUNCTIONAL_BLOCK_FT_TRAJECTORY_GRAD_WEIGHT",
                                                "0.0",
                                            )
                                        )
                                        if trajectory_grad_weight < 0.0:
                                            raise ValueError(
                                                "HY_FUNCTIONAL_BLOCK_FT_TRAJECTORY_GRAD_WEIGHT "
                                                "must be non-negative"
                                            )
                                        if trajectory_grad_weight > 0.0:
                                            trajectory_grad_loss = (
                                                _relative_spatiotemporal_gradient_mse(
                                                    student_latent_target.float(),
                                                    teacher_latent_dev.float(),
                                                )
                                            )
                                            latent_loss = latent_loss + (
                                                trajectory_grad_weight
                                                * trajectory_grad_loss
                                            )
                                            trajectory_grad_rel_mse = float(
                                                trajectory_grad_loss.detach().cpu()
                                            )
                                    latent_mse = float(latent_mse_tensor.detach().cpu())
                                    latent_rel_l2 = float(latent_rel.detach().cpu())

                            if functional_loss_mode == "readout":
                                functional_loss = readout_loss
                            elif functional_loss_mode == "block":
                                functional_loss = block_loss
                                if readout_loss is not None:
                                    readout_weight = float(
                                        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_READOUT_WEIGHT", "0.0")
                                    )
                                    functional_loss = functional_loss + readout_weight * readout_loss
                            elif functional_loss_mode == "prediction":
                                functional_loss = prediction_loss
                            elif functional_loss_mode == "routing":
                                functional_loss = routing_loss
                            else:
                                functional_loss = latent_loss
                            if functional_loss is not None:
                                eval_only = _drift_env_enabled(
                                    "HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY"
                                )
                                if eval_only:
                                    if functional_loss_mode == "prediction":
                                        with torch.no_grad():
                                            student_solver_target = _scheduler_step_no_advance(
                                                self.scheduler,
                                                noise_pred.detach(),
                                                t,
                                                latent_model_input.detach(),
                                            )
                                            teacher_solver_target = _scheduler_step_no_advance(
                                                self.scheduler,
                                                teacher_noise_pred.detach(),
                                                t,
                                                latent_model_input.detach(),
                                            )
                                            solver_diff = (
                                                student_solver_target.float()
                                                - teacher_solver_target.float()
                                            )
                                            prediction_solver_mse = float(
                                                solver_diff.pow(2).mean().cpu()
                                            )
                                            prediction_solver_rel_l2 = float(
                                                (
                                                    solver_diff.norm()
                                                    / teacher_solver_target.float()
                                                    .norm()
                                                    .clamp_min(1e-8)
                                                ).cpu()
                                            )
                                    eval_path_raw = os.environ.get(
                                        "HY_FUNCTIONAL_BLOCK_FT_EVAL_JSONL", ""
                                    ).strip()
                                    if eval_path_raw and get_rank() == 0:
                                        eval_path = Path(eval_path_raw)
                                        eval_path.parent.mkdir(parents=True, exist_ok=True)
                                        eval_record = {
                                            "atom_prefix": os.environ.get(
                                                "HY_FUNCTIONAL_ACCUM_ATOM_PREFIX",
                                                "rollout",
                                            ),
                                            "chunk": int(chunk_i),
                                            "step": int(i),
                                            "inner": int(ft_inner),
                                            "loss": float(
                                                functional_loss.detach().float().cpu()
                                            ),
                                            "prediction_rel_l2": prediction_rel_l2,
                                            "prediction_cosine": prediction_cosine,
                                            "raw_prediction_mse": raw_prediction_mse,
                                            "solver_mse": prediction_solver_mse,
                                            "solver_rel_l2": prediction_solver_rel_l2,
                                            "latent_mse": latent_mse,
                                            "latent_rel_l2": latent_rel_l2,
                                        }
                                        if functional_loss_mode == "routing":
                                            eval_record["routing_metrics"] = routing_metrics
                                        if functional_loss_mode == "block":
                                            eval_record["block_metrics"] = block_metrics
                                            eval_record["local_loss"] = (
                                                None
                                                if block_components.get("local") is None
                                                else float(
                                                    block_components["local"]
                                                    .detach()
                                                    .float()
                                                    .cpu()
                                                )
                                            )
                                            eval_record["prop1_loss"] = (
                                                None
                                                if block_components.get("prop1") is None
                                                else float(
                                                    block_components["prop1"]
                                                    .detach()
                                                    .float()
                                                    .cpu()
                                                )
                                            )
                                            if _functional_ft_target() == "k":
                                                with torch.no_grad():
                                                    k_reconstruction = (
                                                        k_functional_reconstruction_losses_for_cache(
                                                            self._kv_cache,
                                                            target_layers,
                                                        )
                                                    )
                                                eval_record["k_reconstruction_by_layer"] = {
                                                    str(int(layer)): {
                                                        "rel": float(
                                                            values["rel"]
                                                            .detach()
                                                            .float()
                                                            .cpu()
                                                        ),
                                                        "cosine": float(
                                                            values["cosine"]
                                                            .detach()
                                                            .float()
                                                            .cpu()
                                                        ),
                                                    }
                                                    for layer, values in k_reconstruction.items()
                                                }
                                        with eval_path.open("a", encoding="utf-8") as handle:
                                            handle.write(json.dumps(eval_record) + "\n")
                                    if get_rank() == 0:
                                        print(
                                            "[FunctionalBlockFTEval] "
                                            f"chunk={chunk_i} step={i} "
                                            f"loss={float(functional_loss.detach().cpu()):.8e} "
                                            f"pred_rel_l2={prediction_rel_l2:.6f} "
                                            f"pred_cosine={prediction_cosine:.6f} "
                                            f"solver_mse={prediction_solver_mse:.8e} "
                                            f"solver_rel_l2={prediction_solver_rel_l2:.6f}",
                                            flush=True,
                                        )
                                    continue
                                local_guard_enabled = (
                                    os.environ.get("HY_FUNCTIONAL_LOCAL_GUARD_ENABLE", "0")
                                    .strip()
                                    .lower()
                                    in {"1", "true", "yes", "on"}
                                )
                                reconstruction_guard_enabled = (
                                    os.environ.get("HY_FUNCTIONAL_RECON_GUARD_ENABLE", "0")
                                    .strip()
                                    .lower()
                                    in {"1", "true", "yes", "on"}
                                )
                                guard_loss = None
                                guard_metrics = {
                                    "local_trigger_rate": 0.0,
                                    "recon_trigger_rate": 0.0,
                                    "guard_grad_ratio": 0.0,
                                }
                                reconstruction_components = {}
                                if local_guard_enabled or reconstruction_guard_enabled:
                                    if functional_loss_mode != "block" or _functional_ft_target() != "v":
                                        raise RuntimeError(
                                            "Functional guards require V block-loss training"
                                        )
                                    with torch.enable_grad():
                                        reconstruction_components = (
                                            v_functional_reconstruction_losses_for_cache(
                                                self._kv_cache, target_layers
                                            )
                                        )
                                        guard_groups = []
                                        local_triggered = 0
                                        recon_triggered = 0
                                        if local_guard_enabled:
                                            local_guard_terms = []
                                            local_weight = float(
                                                os.environ.get(
                                                    "HY_FUNCTIONAL_LOCAL_GUARD_WEIGHT", "1.0"
                                                )
                                            )
                                            for layer, local_sq in block_components.get(
                                                "local_by_layer", {}
                                            ).items():
                                                local_rel = torch.sqrt(local_sq.clamp_min(0.0))
                                                baseline = functional_guard_baselines["local"].setdefault(
                                                    int(layer), local_rel.detach()
                                                )
                                                tau = float(
                                                    os.environ.get(
                                                        f"HY_FUNCTIONAL_LOCAL_GUARD_TAU_L{int(layer)}",
                                                        os.environ.get(
                                                            "HY_FUNCTIONAL_LOCAL_GUARD_TAU", "0.01"
                                                        ),
                                                    )
                                                )
                                                violation = torch.relu(
                                                    local_rel / baseline.clamp_min(1e-8) - 1.0 - tau
                                                )
                                                local_triggered += int(
                                                    float(violation.detach().cpu()) > 0.0
                                                )
                                                guard_metrics[
                                                    f"local_ratio_L{int(layer)}"
                                                ] = float(
                                                    (local_rel / baseline.clamp_min(1e-8))
                                                    .detach()
                                                    .cpu()
                                                )
                                                local_guard_terms.append(violation.square())
                                            guard_metrics["local_trigger_rate"] = (
                                                float(local_triggered)
                                                / max(1, len(block_components.get("local_by_layer", {})))
                                            )
                                            functional_guard_totals[
                                                "local_triggers"
                                            ] += local_triggered
                                            functional_guard_totals["local_checks"] += len(
                                                block_components.get("local_by_layer", {})
                                            )
                                            if local_guard_terms:
                                                guard_groups.append(
                                                    local_weight
                                                    * torch.stack(local_guard_terms).mean()
                                                )
                                        if reconstruction_guard_enabled:
                                            recon_guard_terms = []
                                            recon_weight = float(
                                                os.environ.get(
                                                    "HY_FUNCTIONAL_RECON_GUARD_WEIGHT", "1.0"
                                                )
                                            )
                                            recon_tau = float(
                                                os.environ.get(
                                                    "HY_FUNCTIONAL_RECON_GUARD_TAU", "0.005"
                                                )
                                            )
                                            for layer, components in reconstruction_components.items():
                                                recon_rel = components["rel"]
                                                baseline = functional_guard_baselines[
                                                    "reconstruction"
                                                ].setdefault(int(layer), recon_rel.detach())
                                                violation = torch.relu(
                                                    recon_rel / baseline.clamp_min(1e-8)
                                                    - 1.0
                                                    - recon_tau
                                                )
                                                recon_triggered += int(
                                                    float(violation.detach().cpu()) > 0.0
                                                )
                                                guard_metrics[
                                                    f"recon_ratio_L{int(layer)}"
                                                ] = float(
                                                    (recon_rel / baseline.clamp_min(1e-8))
                                                    .detach()
                                                    .cpu()
                                                )
                                                recon_guard_terms.append(violation.square())
                                            guard_metrics["recon_trigger_rate"] = (
                                                float(recon_triggered)
                                                / max(1, len(reconstruction_components))
                                            )
                                            functional_guard_totals[
                                                "recon_triggers"
                                            ] += recon_triggered
                                            functional_guard_totals["recon_checks"] += len(
                                                reconstruction_components
                                            )
                                            if recon_guard_terms:
                                                guard_groups.append(
                                                    recon_weight
                                                    * torch.stack(recon_guard_terms).mean()
                                                )
                                        if guard_groups:
                                            guard_loss = torch.stack(guard_groups).sum()
                                gradient_audit = (
                                    os.environ.get("HY_FUNCTIONAL_GRAD_AUDIT", "0")
                                    .strip()
                                    .lower()
                                    in {"1", "true", "yes", "on"}
                                )
                                if gradient_audit:
                                    if adapter_training or _functional_ft_target() != "v":
                                        raise RuntimeError(
                                            "HY_FUNCTIONAL_GRAD_AUDIT requires V codec training"
                                        )
                                    endpoint_component = block_components.get("endpoint")
                                    local_components = block_components.get(
                                        "local_by_layer", {}
                                    )
                                    if endpoint_component is None:
                                        raise RuntimeError(
                                            "Gradient audit requires block endpoint loss"
                                        )
                                    with torch.enable_grad():
                                        if not reconstruction_components:
                                            reconstruction_components = (
                                                v_functional_reconstruction_losses_for_cache(
                                                    self._kv_cache, target_layers
                                                )
                                            )
                                        audit_payload = run_v_functional_gradient_audit(
                                            endpoint_component,
                                            local_components,
                                            reconstruction_components,
                                            chunk=chunk_i,
                                            step=i,
                                        )
                                    if get_rank() == 0:
                                        print(
                                            "[FunctionalGradientAudit] complete "
                                            f"chunk={chunk_i} step={i} "
                                            f"rows={len(audit_payload.get('rows', []))} "
                                            f"path={audit_payload.get('path', '')}",
                                            flush=True,
                                        )
                                    continue
                                final_prediction_gradient_audit = (
                                    os.environ.get(
                                        "HY_FUNCTIONAL_FINAL_PRED_GRAD_AUDIT", "0"
                                    )
                                    .strip()
                                    .lower()
                                    in {"1", "true", "yes", "on"}
                                )
                                if final_prediction_gradient_audit:
                                    if functional_loss_mode != "prediction":
                                        raise RuntimeError(
                                            "Final-prediction gradient audit requires prediction loss"
                                        )
                                    audit_payload = run_v_final_prediction_gradient_audit(
                                        prediction_loss,
                                        target_layers,
                                        chunk=chunk_i,
                                        step=i,
                                    )
                                    if any(param.requires_grad for param in self.transformer.parameters()):
                                        raise RuntimeError(
                                            "Main Transformer parameters are trainable during final-prediction audit"
                                        )
                                    if get_rank() == 0:
                                        print(
                                            "[FinalPredictionGradientAudit] complete "
                                            f"chunk={chunk_i} step={i} "
                                            f"rows={len(audit_payload.get('rows', []))} "
                                            f"path={audit_payload.get('path', '')}",
                                            flush=True,
                                        )
                                    continue
                                k_pca_tail_score = (
                                    os.environ.get(
                                        "HY_K_PCA_TAIL_SCORE_ENABLE", "0"
                                    )
                                    .strip()
                                    .lower()
                                    in {"1", "true", "yes", "on"}
                                )
                                if k_pca_tail_score:
                                    if (
                                        functional_loss_mode != "block"
                                        or _functional_ft_target() != "k"
                                        or len(target_layers) != 1
                                    ):
                                        raise RuntimeError(
                                            "K PCA tail score requires one-layer K block loss"
                                        )
                                    audit_payload = run_k_pca_tail_score_audit(
                                        functional_loss,
                                        layer=int(next(iter(target_layers))),
                                        chunk=chunk_i,
                                        step=i,
                                    )
                                    if get_rank() == 0:
                                        print(
                                            "[KPCATailScore] complete "
                                            f"path={audit_payload.get('path', '')}",
                                            flush=True,
                                        )
                                    continue
                                with torch.enable_grad():
                                    if adapter_training:
                                        anchor_loss, anchor_metrics = None, {}
                                    elif _functional_ft_target() == "kv":
                                        anchor_loss, anchor_metrics = joint_functional_trust_anchor_loss_for_cache(
                                            self._kv_cache, target_layers
                                        )
                                    elif _functional_ft_target() == "v":
                                        anchor_loss, anchor_metrics = v_functional_anchor_loss_for_cache(
                                            self._kv_cache, target_layers
                                        )
                                    else:
                                        anchor_loss, anchor_metrics = k_functional_anchor_loss_for_cache(
                                            self._kv_cache, target_layers
                                        )
                                    if (
                                        _drift_env_enabled(
                                            "HY_K_P1_COMPONENT_GRAD_AUDIT"
                                        )
                                        and _functional_ft_target() == "k"
                                    ):
                                        prop1_components = block_components.get(
                                            "prop1_by_layer", {}
                                        )
                                        local_components = block_components.get(
                                            "local_by_layer", {}
                                        )
                                        reconstruction_components = (
                                            k_functional_reconstruction_losses_for_cache(
                                                self._kv_cache, target_layers
                                            )
                                        )
                                        audit_payload = run_k_p1_gradient_audit(
                                            prop1_components,
                                            local_components,
                                            reconstruction_components,
                                            chunk=chunk_i,
                                            step=i,
                                        )
                                        if not audit_payload.get("rows"):
                                            raise RuntimeError(
                                                "K P1 component audit produced no layer rows"
                                            )
                                        if get_rank() == 0:
                                            print(
                                                "[KP1GradientAudit] complete "
                                                f"rows={len(audit_payload['rows'])} "
                                                f"path={audit_payload.get('path', '')}",
                                                flush=True,
                                            )
                                        continue
                                    train_loss = functional_loss
                                    if anchor_loss is not None:
                                        train_loss = train_loss + anchor_loss
                                    if guard_loss is not None:
                                        if float(guard_loss.detach().cpu()) > 0.0:
                                            endpoint_component = block_components.get("endpoint")
                                            endpoint_grad_norm = v_functional_loss_grad_norm(
                                                endpoint_component
                                            )
                                            raw_guard_grad_norm = v_functional_loss_grad_norm(
                                                guard_loss
                                            )
                                            raw_guard_ratio = raw_guard_grad_norm / max(
                                                endpoint_grad_norm, 1e-12
                                            )
                                            max_guard_ratio = float(
                                                os.environ.get(
                                                    "HY_FUNCTIONAL_GUARD_GRAD_MAX_RATIO", "0.4"
                                                )
                                            )
                                            guard_scale = min(
                                                1.0,
                                                max_guard_ratio / max(raw_guard_ratio, 1e-12),
                                            )
                                            guard_loss = guard_loss * guard_scale
                                            guard_metrics["raw_guard_grad_ratio"] = raw_guard_ratio
                                            guard_metrics["guard_scale"] = guard_scale
                                            guard_metrics["guard_grad_ratio"] = (
                                                raw_guard_ratio * guard_scale
                                            )
                                            functional_guard_totals["active_steps"] += 1
                                            functional_guard_totals["guard_ratio_sum"] += (
                                                guard_metrics["guard_grad_ratio"]
                                            )
                                            functional_guard_totals["guard_ratio_max"] = max(
                                                functional_guard_totals["guard_ratio_max"],
                                                guard_metrics["guard_grad_ratio"],
                                            )
                                        train_loss = train_loss + guard_loss
                                if not train_loss.requires_grad:
                                    if get_rank() == 0:
                                        print(
                                            "[FunctionalBlockFT] skip_no_grad "
                                            f"chunk={chunk_i} step={i} inner={ft_inner} "
                                            f"mode={functional_loss_mode} "
                                            f"func_req={getattr(functional_loss, 'requires_grad', None)} "
                                            f"anchor_req={getattr(anchor_loss, 'requires_grad', None)} "
                                            f"metrics={block_metrics}",
                                            flush=True,
                                        )
                                    continue
                                if _drift_env_enabled("HY_FUNCTIONAL_JOINT_PATH_AUDIT"):
                                    if _functional_ft_target() != "kv":
                                        raise RuntimeError(
                                            "Joint path audit requires HY_FUNCTIONAL_BLOCK_FT_TARGET=kv"
                                        )
                                    run_joint_functional_path_audit(
                                        functional_loss,
                                        self._kv_cache,
                                        target_layers,
                                        chunk=chunk_i,
                                        step=i,
                                    )
                                    continue
                                atom_prefix = os.environ.get(
                                    "HY_FUNCTIONAL_ACCUM_ATOM_PREFIX", "rollout"
                                ).strip()
                                atom_id = (
                                    atom_prefix
                                    if _drift_env_enabled(
                                        "HY_FUNCTIONAL_ACCUM_ATOM_ID_EXACT"
                                    )
                                    else (
                                        f"{atom_prefix}:chunk{int(chunk_i):02d}:"
                                        f"step{int(i):02d}:inner{int(ft_inner):02d}"
                                    )
                                )
                                step_info = (
                                    _step_dirty_history_output_adapter(self.transformer, train_loss)
                                    if adapter_training
                                    else step_k_functional_train_optimizers(
                                        train_loss, atom_id=atom_id
                                    )
                                )
                                if (
                                    int(step_info.get("optimizer_step", 0)) > 0
                                    and _drift_env_enabled(
                                        "HY_FUNCTIONAL_BLOCK_FT_STOP_AFTER_UPDATE"
                                    )
                                ):
                                    stop_after_functional_update = True
                                step_jsonl = os.environ.get(
                                    "HY_FUNCTIONAL_BLOCK_FT_STEP_JSONL", ""
                                ).strip()
                                if step_jsonl and get_rank() == 0:
                                    step_path = Path(step_jsonl)
                                    step_path.parent.mkdir(parents=True, exist_ok=True)
                                    step_record = {
                                        "atom": atom_id,
                                        "chunk": int(chunk_i),
                                        "step": int(i),
                                        "inner": int(ft_inner),
                                        "target": _functional_ft_target(),
                                        "loss_mode": functional_loss_mode,
                                        "functional_loss": float(
                                            functional_loss.detach().float().cpu()
                                        ),
                                        "train_loss": float(
                                            train_loss.detach().float().cpu()
                                        ),
                                        "latent_rel_l2": float(latent_rel_l2),
                                        "trajectory_input_rel_l2": float(
                                            trajectory_input_rel_l2
                                        ),
                                        "trajectory_grad_rel_mse": float(
                                            trajectory_grad_rel_mse
                                        ),
                                        "anchor": anchor_metrics,
                                        "optimizer": step_info,
                                    }
                                    if functional_loss_mode == "routing":
                                        step_record["routing_metrics"] = routing_metrics
                                    with step_path.open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(step_record) + "\n")
                                _functional_ft_total_step += 1
                                if (
                                    int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_VERBOSE", "1")) > 0
                                    and get_rank() == 0
                                    and (_functional_ft_total_step % log_every == 0 or ft_inner == inner_steps - 1)
                                ):
                                    rels = {m["layer"]: m["block_rel_l2"] for m in block_metrics}
                                    coss = {m["layer"]: m["block_cosine"] for m in block_metrics}
                                    l13_r = rels.get(13, 0.0)
                                    l40_r = rels.get(40, 0.0)
                                    l53_r = rels.get(53, 0.0)
                                    c13 = coss.get(13, 0.0)
                                    c40 = coss.get(40, 0.0)
                                    c53 = coss.get(53, 0.0)
                                    mean_r = sum(rels.values()) / max(1, len(rels))
                                    mean_c = sum(coss.values()) / max(1, len(coss))
                                    endpoint_rels = [
                                        float(m.get("endpoint_rel_l2", 0.0))
                                        for m in block_metrics
                                        if m.get("is_endpoint", False)
                                    ]
                                    mean_endpoint_rel = sum(endpoint_rels) / max(
                                        1, len(endpoint_rels)
                                    )
                                    endpoint_metric = next(
                                        (
                                            m
                                            for m in block_metrics
                                            if m.get("is_endpoint", False)
                                        ),
                                        {},
                                    )
                                    local_by_layer = ",".join(
                                        f"L{layer}={rels[layer]:.6f}"
                                        for layer in sorted(rels)
                                    )
                                    recon_by_layer = ",".join(
                                        f"L{layer}={anchor_metrics.get(f'anchor_rel_L{layer}', 0.0):.6f}"
                                        for layer in sorted(target_layers)
                                    )
                                    readout_rels = {
                                        m["layer"]: m["readout_rel_l2"] for m in readout_metrics
                                    }
                                    readout_coss = {
                                        m["layer"]: m["readout_cosine"] for m in readout_metrics
                                    }
                                    mean_readout_rel = sum(readout_rels.values()) / max(1, len(readout_rels))
                                    mean_readout_cos = sum(readout_coss.values()) / max(1, len(readout_coss))
                                    grad_items = [
                                        f"L{layer}={step_info[key]:.6f}"
                                        for key, layer in sorted(
                                            (
                                                (key, int(key.replace("grad_L", "")))
                                                for key in step_info
                                                if key.startswith("grad_L")
                                            ),
                                            key=lambda item: item[1],
                                        )
                                    ]
                                    grad_by_layer = ",".join(grad_items)
                                    update_items = [
                                        f"L{layer}={step_info[key]:.6f}"
                                        for key, layer in sorted(
                                            (
                                                (key, int(key.replace("update_L", "")))
                                                for key in step_info
                                                if key.startswith("update_L")
                                            ),
                                            key=lambda item: item[1],
                                        )
                                    ]
                                    update_by_layer = ",".join(update_items)
                                    update_rms_items = []
                                    for layer in sorted(target_layers):
                                        update_rms_items.append(
                                            f"L{layer}:E={step_info.get(f'update_rms_encoder_L{layer}', 0.0):.3e}"
                                            f"/D={step_info.get(f'update_rms_decoder_L{layer}', 0.0):.3e}"
                                        )
                                    update_rms_by_group = ",".join(update_rms_items)
                                    local_guard_ratios = ",".join(
                                        f"L{layer}={guard_metrics.get(f'local_ratio_L{layer}', 1.0):.6f}"
                                        for layer in sorted(target_layers)
                                    )
                                    recon_guard_ratios = ",".join(
                                        f"L{layer}={guard_metrics.get(f'recon_ratio_L{layer}', 1.0):.6f}"
                                        for layer in sorted(target_layers)
                                    )
                                    print(
                                        "[FunctionalBlockFT] "
                                        f"mode={functional_loss_mode} "
                                        f"ft_step={_functional_ft_total_step} "
                                        f"chunk={chunk_i} step={i} "
                                        f"loss={float(train_loss.detach().cpu()):.6f} "
                                        f"func_loss={float(functional_loss.detach().cpu()):.6f} "
                                        f"pred_rel_l2={prediction_rel_l2:.6f} "
                                        f"pred_cosine={prediction_cosine:.6f} "
                                        f"clean_preserve_rel={clean_preserve_rel_l2:.6f} "
                                        f"latent_rel_l2={latent_rel_l2:.6f} "
                                        f"trajectory_input_rel_l2={trajectory_input_rel_l2:.6f} "
                                        f"trajectory_grad_rel_mse={trajectory_grad_rel_mse:.6f} "
                                        f"block_rel_l2={mean_r:.6f} "
                                        f"endpoint_rel_l2={mean_endpoint_rel:.6f} "
                                        f"endpoint_loss={float(endpoint_metric.get('endpoint_loss', 0.0)):.6f} "
                                        f"endpoint_hidden_rms={float(endpoint_metric.get('endpoint_student_hidden_rms', 0.0)):.6f} "
                                        f"teacher_hidden_rms={float(endpoint_metric.get('endpoint_teacher_hidden_rms', 0.0)):.6f} "
                                        f"entry_hidden_rms={float(endpoint_metric.get('endpoint_entry_hidden_rms', 0.0)):.6f} "
                                        f"L13={l13_r:.6f} L40={l40_r:.6f} L53={l53_r:.6f} "
                                        f"local_by_layer={local_by_layer} "
                                        f"cosine={mean_c:.6f} "
                                        f"readout_rel={mean_readout_rel:.6f} "
                                        f"readout_cos={mean_readout_cos:.6f} "
                                        f"anchor_rel={anchor_metrics.get('anchor_rel_l2', 0.0):.6f} "
                                        f"anchor_cos={anchor_metrics.get('anchor_cosine', 0.0):.6f} "
                                        f"recon_by_layer={recon_by_layer} "
                                        f"grad={step_info.get('max_grad_norm', 0.0):.6f} "
                                        f"adapter_norm={step_info.get('adapter_weight_norm', 0.0):.6f} "
                                        f"grad_by_layer={grad_by_layer} "
                                        f"update_by_layer={update_by_layer} "
                                        f"update_rms_by_group={update_rms_by_group} "
                                        f"local_guard_ratios={local_guard_ratios} "
                                        f"recon_guard_ratios={recon_guard_ratios} "
                                        f"local_guard_trigger_rate={guard_metrics.get('local_trigger_rate', 0.0):.3f} "
                                        f"recon_guard_trigger_rate={guard_metrics.get('recon_trigger_rate', 0.0):.3f} "
                                        f"raw_guard_grad_ratio={guard_metrics.get('raw_guard_grad_ratio', 0.0):.6f} "
                                        f"guard_scale={guard_metrics.get('guard_scale', 1.0):.6f} "
                                        f"guard_grad_ratio={guard_metrics.get('guard_grad_ratio', 0.0):.6f} "
                                        f"cumulative_local_guard_rate={functional_guard_totals['local_triggers'] / max(1, functional_guard_totals['local_checks']):.6f} "
                                        f"cumulative_recon_guard_rate={functional_guard_totals['recon_triggers'] / max(1, functional_guard_totals['recon_checks']):.6f} "
                                        f"guard_active_steps={functional_guard_totals['active_steps']} "
                                        f"mean_active_guard_grad_ratio={functional_guard_totals['guard_ratio_sum'] / max(1, functional_guard_totals['active_steps']):.6f} "
                                        f"max_guard_grad_ratio={functional_guard_totals['guard_ratio_max']:.6f} "
                                        f"accum={int(step_info.get('accum_count', 0))}/"
                                        f"{int(step_info.get('accum_target', 0))} "
                                        f"optimizer_step={int(step_info.get('optimizer_step', 1))} "
                                        f"opt={step_info.get('num_optimizers', 0):.0f}",
                                        flush=True,
                                    )
                            # Drop references to the student graph before continuing rollout.
                            try:
                                del train_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del anchor_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del block_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del readout_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del latent_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del prediction_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del routing_loss
                            except UnboundLocalError:
                                pass
                            try:
                                del clean_adapter_noise_pred
                            except UnboundLocalError:
                                pass
                            try:
                                del student_latent_target
                            except UnboundLocalError:
                                pass
                            try:
                                del teacher_latent_dev
                            except UnboundLocalError:
                                pass
                            try:
                                del noise_pred
                            except UnboundLocalError:
                                pass
                        os.environ["HY_FUNCTIONAL_BLOCK_FT_PASS"] = ""
                        os.environ.pop("HY_FUNCTIONAL_BLOCK_FT_INNER_INDEX", None)
                        zero_k_functional_train_grads()
                        _zero_dirty_history_output_adapter_grads(self.transformer)
                        for layer_idx in target_layers:
                            if layer_idx < len(self._kv_cache):
                                if _functional_ft_target() in {"k", "kv"}:
                                    self._kv_cache[layer_idx] = detach_k_functional_cache_for_rollout(
                                        self._kv_cache[layer_idx], layer_idx
                                    )
                                if _functional_ft_target() in {"v", "kv"}:
                                    self._kv_cache[layer_idx] = detach_v_functional_cache_for_rollout(
                                        self._kv_cache[layer_idx], layer_idx
                                    )
                        # Restore targets for final cleanup (but they'll be cleared next epoch anyway)
                        _FUNCTIONAL_BLOCK_TARGETS.clear()
                        reset_functional_readout_losses()
                        reset_codec_native_routing_losses()
                        if _drift_env_enabled("HY_FUNCTIONAL_BLOCK_FT_TEACHER_FORCED") or teacher_force_all:
                            noise_pred = teacher_noise_pred
                        else:
                            if student_rollout_noise_pred is None:
                                raise RuntimeError(
                                    "Functional FT produced no Student noise prediction for rollout"
                                )
                            noise_pred = student_rollout_noise_pred
                        # Keep only the selected rollout tensor alive. The other branch can
                        # otherwise consume a full prediction buffer until the next step.
                        del teacher_noise_pred
                        del student_rollout_noise_pred
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self.target_dtype,
                            enabled=self.autocast_enabled,
                        ):
                            noise_pred = self.transformer(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                hidden_states=latents_concat,
                                timestep=timestep_input,
                                timestep_r=None,
                                mask_type=task_type,
                                return_dict=False,
                                viewmats=viewmats_input.to(self.target_dtype),
                                Ks=Ks_input.to(self.target_dtype),
                                action=action_input.to(self.target_dtype),
                                kv_cache=self._kv_cache,
                                cache_vision=False,
                                rope_temporal_size=latents_concat.shape[2]
                                + len(selected_frame_indices),
                                start_rope_start_idx=len(selected_frame_indices),
                            )[0]
                        if teacher_force_all:
                            clean_cache = _functional_ft_clean_kv_cache(self._kv_cache)
                            clean_cache = _functional_ft_clean_target_cache(clean_cache)
                            with (
                                torch.no_grad(),
                                torch.autocast(
                                    device_type="cuda",
                                    dtype=self.target_dtype,
                                    enabled=self.autocast_enabled,
                                ),
                            ):
                                noise_pred = self.transformer(
                                    bi_inference=False,
                                    ar_txt_inference=False,
                                    ar_vision_inference=True,
                                    hidden_states=latents_concat.detach(),
                                    timestep=timestep_input,
                                    timestep_r=None,
                                    mask_type=task_type,
                                    return_dict=False,
                                    viewmats=viewmats_input.to(self.target_dtype),
                                    Ks=Ks_input.to(self.target_dtype),
                                    action=action_input.to(self.target_dtype),
                                    kv_cache=clean_cache,
                                    cache_vision=False,
                                    rope_temporal_size=latents_concat.shape[2]
                                    + len(selected_frame_indices),
                                    start_rope_start_idx=len(selected_frame_indices),
                                )[0]
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond = self.transformer(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                hidden_states=latents_concat,
                                timestep=timestep_input,
                                timestep_r=None,
                                mask_type=task_type,
                                return_dict=False,
                                viewmats=viewmats_input.to(self.target_dtype),
                                Ks=Ks_input.to(self.target_dtype),
                                action=action_input.to(self.target_dtype),
                                kv_cache=self._kv_cache_neg,
                                cache_vision=False,
                                rope_temporal_size=latents_concat.shape[2]
                                + len(selected_frame_indices),
                                start_rope_start_idx=len(selected_frame_indices),
                            )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )
                    if (
                        prediction_oracle_cache is not None
                        and os.environ.get(
                            "HY_DIRTY_HISTORY_PRED_ORACLE_STATE_DIR", ""
                        ).strip()
                        and _prediction_oracle_step_enabled(i)
                    ):
                        with (
                            torch.no_grad(),
                            torch.autocast(
                                device_type="cuda",
                                dtype=self.target_dtype,
                                enabled=self.autocast_enabled,
                            ),
                        ):
                            teacher_prediction = self.transformer(
                                bi_inference=False,
                                ar_txt_inference=False,
                                ar_vision_inference=True,
                                hidden_states=latents_concat.detach(),
                                timestep=timestep_input,
                                timestep_r=None,
                                mask_type=task_type,
                                return_dict=False,
                                viewmats=viewmats_input.to(self.target_dtype),
                                Ks=Ks_input.to(self.target_dtype),
                                action=action_input.to(self.target_dtype),
                                kv_cache=prediction_oracle_cache,
                                cache_vision=False,
                                rope_temporal_size=latents_concat.shape[2]
                                + len(selected_frame_indices),
                                start_rope_start_idx=len(selected_frame_indices),
                            )[0]
                            if self.do_classifier_free_guidance:
                                teacher_prediction_uncond = self.transformer(
                                    bi_inference=False,
                                    ar_txt_inference=False,
                                    ar_vision_inference=True,
                                    hidden_states=latents_concat.detach(),
                                    timestep=timestep_input,
                                    timestep_r=None,
                                    mask_type=task_type,
                                    return_dict=False,
                                    viewmats=viewmats_input.to(self.target_dtype),
                                    Ks=Ks_input.to(self.target_dtype),
                                    action=action_input.to(self.target_dtype),
                                    kv_cache=prediction_oracle_cache_neg,
                                    cache_vision=False,
                                    rope_temporal_size=latents_concat.shape[2]
                                    + len(selected_frame_indices),
                                    start_rope_start_idx=len(selected_frame_indices),
                                )[0]
                                teacher_prediction = teacher_prediction_uncond + self.guidance_scale * (
                                    teacher_prediction - teacher_prediction_uncond
                                )
                        student_prediction = noise_pred
                        alpha = float(
                            os.environ.get("HY_DIRTY_HISTORY_PRED_ORACLE_ALPHA", "1.0")
                        )
                        student_float = student_prediction.float()
                        teacher_float = teacher_prediction.float()
                        delta = teacher_float - student_float
                        rel_l2 = delta.norm() / teacher_float.norm().clamp_min(1e-8)
                        cosine = F.cosine_similarity(
                            student_float.reshape(1, -1),
                            teacher_float.reshape(1, -1),
                            dim=1,
                        ).mean()
                        noise_pred = (
                            student_float + alpha * delta
                        ).to(dtype=student_prediction.dtype)
                        if get_rank() == 0:
                            print(
                                f"[DirtyHistoryPredOracle] boundary={chunk_i - 1} "
                                f"chunk={chunk_i} step={i} alpha={alpha:g} "
                                f"pred_rel_l2={rel_l2.item():.6f} "
                                f"pred_cos={cosine.item():.6f}",
                                flush=True,
                            )
                        del teacher_prediction
                        if self.do_classifier_free_guidance:
                            del teacher_prediction_uncond
                    _dump_rollout_latent_tensor("noise_pred", noise_pred, chunk_i, i)

                    latent_model_input = self.scheduler.step(
                        noise_pred, t, latent_model_input, return_dict=False
                    )[0]
                    _dump_rollout_latent_tensor("x_next", latent_model_input, chunk_i, i)
                    latents[:, :, start_idx:end_idx] = latent_model_input[
                        :, :, -self.chunk_latent_frames :
                    ]

                    if stop_after_functional_update:
                        if get_rank() == 0:
                            print(
                                "[FunctionalBlockFT] stopping denoise immediately "
                                f"after optimizer update at chunk={chunk_i} step={i}",
                                flush=True,
                            )
                        # Exact-clean K/V is a training-only teacher. Release both
                        # aliases before returning through the normal checkpoint path.
                        prediction_oracle_cache = None
                        prediction_oracle_cache_neg = None
                        try:
                            del clean_cache
                        except UnboundLocalError:
                            pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        break

                    if i == len(timesteps) - 1 or (
                        (i + 1) > self.num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()
            chunk_stats["denoise_time_s"] = _elapsed_since(denoise_start, device)
            if transient_cache_restore is not None:
                self._kv_cache = transient_cache_restore
                self._kv_cache_neg = transient_cache_restore_neg
                if get_rank() == 0:
                    print(
                        f"[RolloutStateOracle] discarded transient vision cache "
                        f"after chunk={chunk_i}",
                        flush=True,
                    )
            _maybe_dump_rollout_boundary_state(latents, end_idx, chunk_i)
            _maybe_reset_rollout_boundary_state(latents, start_idx, end_idx, chunk_i)
            perf_stats["chunks"].append(chunk_stats)
            stop_after_chunk = int(os.environ.get("HY_STOP_AFTER_AR_CHUNK", "-1"))
            if stop_after_chunk >= 0 and chunk_i >= stop_after_chunk:
                print(
                    f"[AR Rollout] HY_STOP_AFTER_AR_CHUNK={stop_after_chunk}; "
                    f"stopping after chunk {chunk_i}.",
                    flush=True,
                )
                break

        perf_stats["total_history_selection_time_s"] = sum(
            item["history_selection_time_s"] for item in perf_stats["chunks"]
        )
        perf_stats["total_recompute_time_s"] = sum(
            item["recompute_time_s"] for item in perf_stats["chunks"]
        )
        perf_stats["total_denoise_time_s"] = sum(
            item["denoise_time_s"] for item in perf_stats["chunks"]
        )
        perf_stats["total_dit_core_time_s"] = (
            perf_stats["total_recompute_time_s"]
            + perf_stats["total_denoise_time_s"]
        )
        perf_stats["total_ar_rollout_inner_time_s"] = _elapsed_since(
            rollout_start, device
        )
        if torch.cuda.is_available():
            perf_stats["peak_allocated_gb"] = torch.cuda.max_memory_allocated(
                device
            ) / (1024**3)
            perf_stats["peak_reserved_gb"] = torch.cuda.max_memory_reserved(
                device
            ) / (1024**3)
        functional_diagnostic_only = (
            _drift_env_enabled("HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY")
            or _drift_env_enabled("HY_K_P1_COMPONENT_GRAD_AUDIT")
            or bool(
                os.environ.get("HY_FUNCTIONAL_GRADIENT_SNAPSHOT", "").strip()
            )
        )
        if (
            _functional_ft_enabled()
            and not _drift_env_enabled("HY_K_T0_ENABLE")
            and not functional_diagnostic_only
        ):
            out_root = os.environ.get(
                "HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT",
                "kv_ae_outputs/ae16x_v2/k_spatial4x_block_functional_ft",
            )
            if _dirty_history_any_adapter_enabled():
                saved = {"adapter": _save_dirty_history_output_adapter(self.transformer, out_root)}
            elif _functional_ft_target() == "kv":
                saved = {
                    "k": save_k_functional_train_checkpoints(str(Path(out_root) / "k")),
                    "v": save_v_functional_train_checkpoints(str(Path(out_root) / "v")),
                }
            elif _functional_ft_target() == "v":
                saved = save_v_functional_train_checkpoints(out_root)
            else:
                saved = save_k_functional_train_checkpoints(out_root)
            if get_rank() == 0:
                print(f"[FunctionalBlockFT] saved={saved}", flush=True)
        elif (
            _functional_ft_enabled()
            and functional_diagnostic_only
            and get_rank() == 0
        ):
            print(
                "[FunctionalBlockFT] diagnostic-only run; no codec checkpoint was written",
                flush=True,
            )
        elif _drift_env_enabled("HY_K_T0_ENABLE") and get_rank() == 0:
            print(
                "[K-T0-Gate1] audit-only run; no codec checkpoint was written",
                flush=True,
            )
        self._last_ar_perf_stats = perf_stats
        return latents

    def bi_rollout(
        self,
        latents,
        timesteps,
        prompt_embeds,
        prompt_mask,
        vision_states,
        cond_latents,
        task_type,
        extra_kwargs,
        viewmats,
        Ks,
        action,
        device,
    ):
        stabilization_level = 15
        retrieval_temporal_context_size = int(
            os.environ.get("HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE", "12")
        )
        retrieval_memory_frames = int(
            os.environ.get("HY_WORLDPLAY_MEMORY_FRAMES", "20")
        )
        if retrieval_temporal_context_size < 0:
            raise ValueError("HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE must be non-negative")
        if retrieval_memory_frames < retrieval_temporal_context_size + 4:
            raise ValueError(
                "HY_WORLDPLAY_MEMORY_FRAMES must reserve the first 4 latent frames "
                "in addition to HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE; got "
                f"memory={retrieval_memory_frames}, "
                f"temporal={retrieval_temporal_context_size}"
            )
        for chunk_i in range(self.chunk_num):
            if chunk_i > 0:
                # the current frame index to generate (in latent space): 4, 8, 12, 16, 20
                current_frame_idx = chunk_i * self.chunk_latent_frames

                selected_frame_indices = []
                for chunk_start_idx in range(
                    current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4
                ):
                    selected_history_frame_id = select_aligned_memory_frames(
                        viewmats[0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=retrieval_memory_frames,
                        temporal_context_size=retrieval_temporal_context_size,
                        pred_latent_size=4,
                        points_local=self.points_local,
                        device=device,
                    )
                    selected_frame_indices = (
                        selected_frame_indices + selected_history_frame_id
                    )
                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                to_remove = list(
                    range(
                        current_frame_idx, current_frame_idx + self.chunk_latent_frames
                    )
                )
                selected_frame_indices = [
                    x for x in selected_frame_indices if x not in to_remove
                ]

                context_latents = latents[:, :, selected_frame_indices]
                context_w2c = viewmats[:, selected_frame_indices]
                context_Ks = Ks[:, selected_frame_indices]
                context_action = action[:, selected_frame_indices]

                self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            start_idx = chunk_i * self.chunk_latent_frames
            end_idx = chunk_i * self.chunk_latent_frames + self.chunk_latent_frames

            with (
                self.progress_bar(total=self.num_inference_steps) as progress_bar,
                auto_offload_model(
                    self.transformer,
                    self.execution_device,
                    enabled=self.enable_offloading,
                ),
            ):
                for i, t in enumerate(timesteps):
                    if chunk_i == 0:
                        timestep_input = torch.full(
                            (self.chunk_latent_frames,),
                            t,
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        latent_model_input = latents[:, :, : self.chunk_latent_frames]
                        cond_latents_input = cond_latents[
                            :, :, : self.chunk_latent_frames
                        ]
                    else:
                        t_now = torch.full(
                            (self.chunk_latent_frames,),
                            t,
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        t_ctx = torch.full(
                            (len(selected_frame_indices),),
                            stabilization_level - 1,
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        timestep_input = torch.cat([t_ctx, t_now], dim=0)

                        latents_model_now = latents[:, :, start_idx:end_idx]
                        latent_model_input = torch.cat(
                            [context_latents, latents_model_now], dim=2
                        )
                        cond_latents_input = cond_latents[
                            :, :, : latent_model_input.shape[2]
                        ]

                    viewmats_input = viewmats[:, start_idx:end_idx]
                    Ks_input = Ks[:, start_idx:end_idx]
                    action_input = action[:, start_idx:end_idx]

                    if chunk_i > 0:
                        viewmats_input = torch.cat([context_w2c, viewmats_input], dim=1)
                        Ks_input = torch.cat([context_Ks, Ks_input], dim=1)
                        action_input = torch.cat([context_action, action_input], dim=1)

                    latents_concat = torch.concat(
                        [latent_model_input, cond_latents_input], dim=1
                    )
                    if self.do_classifier_free_guidance:
                        latents_concat = torch.cat([latents_concat] * 2)
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    batch_size = latents_concat.shape[0]
                    t_expand_txt = t.repeat(batch_size)
                    t_expand = timestep_input.repeat(batch_size)
                    viewmats_input = repeat(
                        viewmats_input, "B L H W -> (B R) L H W", R=batch_size
                    ).to(device)
                    Ks_input = repeat(
                        Ks_input, "B L H W -> (B R) L H W", R=batch_size
                    ).to(device)
                    action_input = (
                        repeat(action_input, "B L -> (B R) L", R=batch_size)
                        .reshape(-1)
                        .to(device)
                    )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.target_dtype,
                        enabled=self.autocast_enabled,
                    ):
                        output = self.transformer(
                            bi_inference=True,
                            ar_txt_inference=False,
                            ar_vision_inference=False,
                            hidden_states=latents_concat,
                            timestep=t_expand,
                            timestep_txt=t_expand_txt,
                            text_states=prompt_embeds,
                            text_states_2=None,
                            encoder_attention_mask=prompt_mask,
                            timestep_r=None,
                            vision_states=vision_states,
                            mask_type=task_type,
                            guidance=None,
                            return_dict=False,
                            extra_kwargs=extra_kwargs,
                            viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype),
                            action=action_input.to(self.target_dtype),
                        )
                        noise_pred = output[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    latent_model_input = self.scheduler.step(
                        noise_pred, t, latent_model_input, return_dict=False
                    )[0]
                    latents[:, :, start_idx:end_idx] = latent_model_input[
                        :, :, -self.chunk_latent_frames :
                    ]

                    # Update progress bar
                    if i == len(timesteps) - 1 or (
                        (i + 1) > self.num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        aspect_ratio: str,
        video_length: int,
        prompt_rewrite: bool = True,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        enable_sr: bool = True,
        sr_num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = None,
        flow_shift: Optional[float] = None,
        embedded_guidance_scale: Optional[float] = None,
        reference_image=None,  # For i2v tasks: PIL Image or path to image file
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        return_pre_sr_video: bool = False,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        few_step: bool = False,
        chunk_latent_frames: int = 4,
        model_type: str = "ar",
        user_height: Optional[int] = None,
        user_width: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Generates a video (or videos) based on text (and optionally image) conditions.

        Args:
            prompt (`str` or `List[str]`):
                Text prompt(s) to guide video generation.
            aspect_ratio (`str`):
                Output video aspect ratio as a string formatted like "720:1280" or "16:9".
                Required for text-to-video tasks.
            video_length (`int`):
                Number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps during generation.
                Larger values may improve video quality at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to value in config):
                Scale to encourage the model to better follow the prompt.
                `guidance_scale > 1` enables classifier-free guidance.
            enable_sr (`bool`, *optional*, defaults to True):
                Whether to apply super-resolution to the generated video.
            sr_num_inference_steps (`int`, *optional*, defaults to 30):
                Number of inference steps in the super-resolution module (if enabled).
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) that describe what should NOT be shown in the generated video.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch random generator(s) for deterministic results.
            seed (`int`, *optional*):
                If specified, used to create the generator for reproducible sampling.
            flow_shift (`float`, *optional*):
                Flow shift parameter for the scheduler. Overrides the default pipeline configuration if provided.
            embedded_guidance_scale (`float`, *optional*):
                Additional control guidance scale, if supported.
            reference_image (PIL.Image or `str`, *optional*):
                Reference image for image-to-video (i2v) tasks.
                Can be a PIL image or a path to an image file. Set to `None` for text-to-video (t2v) generation.
            output_type (`str`, *optional*, defaults to "pt"):
                Output format of the returned video(s).
                Accepted values: `"pt"` for torch.Tensor or `"np"` for numpy.ndarray.
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return a [`HunyuanVideoPipelineOutput`] or a tuple.
            **kwargs:
                Additional keyword arguments.

        Returns:
            HunyuanVideoPipelineOutput or `tuple`:
                If `return_dict` is True, returns a [`HunyuanVideoPipelineOutput`] with fields:
                    - `videos`: Generated video(s) as a tensor or numpy array.
                    - `sr_videos`: Super-resolved video(s) if `enable_sr` is True, else None.
                Otherwise, returns a tuple containing the outputs as above.

        Example:
            ```python
            pipe = HunyuanVideoPipeline.from_pretrained("your_model_dir")
            # Text-to-video
            video = pipe(
                        prompt="A dog surfing on the beach",
                        aspect_ratio="9:16",
                        video_length=32
                        ).videos
            # Image-to-video
            video = pipe(
                prompt="Make this image move",
                reference_image="img.jpg",
                aspect_ratio="16:9",
                video_length=24).videos
            ```
        """
        num_videos_per_prompt = 1
        target_resolution = self.ideal_resolution

        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        if embedded_guidance_scale is None:
            embedded_guidance_scale = self.config.embedded_guidance_scale
        if flow_shift is None:
            flow_shift = self.config.flow_shift

        # enable few step inference
        if few_step:
            guidance_scale = 1.0

        if embedded_guidance_scale is not None:
            assert not self.do_classifier_free_guidance
            assert self.transformer.config.guidance_embed
        else:
            assert not self.transformer.config.guidance_embed

        user_reference_image = reference_image
        user_prompt = prompt

        if reference_image is not None:
            task_type = "i2v"
            if isinstance(reference_image, str):
                reference_image = Image.open(reference_image).convert("RGB")
            elif not isinstance(reference_image, Image.Image):
                raise ValueError(
                    "reference_image must be a PIL Image or path to image file"
                )
            semantic_images_np = np.array(reference_image)

        else:
            task_type = "t2v"
            semantic_images_np = None

        if prompt_rewrite:
            from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

            if not dist.is_initialized() or get_parallel_state().sp_rank == 0:
                try:
                    prompt = run_prompt_rewrite(user_prompt, reference_image, task_type)
                except Exception as e:
                    loguru.logger.warning(f"Failed to rewrite prompt: {e}")
                    prompt = user_prompt

            if dist.is_initialized() and get_parallel_state().sp_enabled:
                obj_list = [prompt]
                # not use group_src to support old PyTorch
                group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
                dist.broadcast_object_list(
                    obj_list, src=group_src_rank, group=get_parallel_state().sp_group
                )
                prompt = obj_list[0]

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded pipeline is trained for '{self.ideal_task}' task, "
                f"but received input for '{task_type}' task. "
                "Please load a pipeline trained for the correct task, "
                "or check and update your arguments accordingly."
            )

        if flow_shift is None:
            self.scheduler = self._create_scheduler(self.config.flow_shift)
        else:
            self.scheduler = self._create_scheduler(flow_shift)

        if get_parallel_state().sp_enabled:
            assert seed is not None

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.execution_device).manual_seed(seed)

        if reference_image is not None:
            if (
                self.ideal_resolution is not None
                and target_resolution != self.ideal_resolution
            ):
                raise ValueError(
                    f"The loaded pipeline is trained for {self.ideal_resolution} resolution, "
                    f"but received input for {target_resolution} resolution. "
                )
            height, width = self.get_closest_resolution_given_reference_image(
                reference_image, target_resolution
            )
        else:
            if self.ideal_resolution is not None:
                if ":" not in aspect_ratio:
                    raise ValueError("aspect_ratio must be separated by a colon")
                width, height = aspect_ratio.split(":")
                # check if width and height are integers
                if (
                    not width.isdigit()
                    or not height.isdigit()
                    or int(width) <= 0
                    or int(height) <= 0
                ):
                    raise ValueError(
                        "w and h must be positive and separated by a colon in aspect_ratio"
                    )
                width = int(width)
                height = int(height)
                height, width = self.get_closest_resolution_given_original_size(
                    (width, height), self.ideal_resolution
                )

        height = user_height if user_height is not None else height
        width = user_width if user_width is not None else width

        latent_target_length, latent_height, latent_width = self.get_latent_size(
            video_length, height, width
        )
        n_tokens = latent_target_length * latent_height * latent_width
        multitask_mask = self.get_task_mask(task_type, latent_target_length)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        self._clip_skip = kwargs.get("clip_skip", None)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        device = self.execution_device

        if get_rank() == 0:
            print(
                "\n"
                f"{'=' * 60}\n"
                f"🎬  HunyuanVideo Generation Task\n"
                f"{'-' * 60}\n"
                f"User Prompt:               {user_prompt}\n"
                f"Rewritten Prompt:          {prompt if prompt_rewrite else '<disabled>'}\n"
                f"Aspect Ratio:              {aspect_ratio}\n"
                f"Video Length:              {video_length}\n"
                f"Reference Image:           {reference_image}\n"
                f"Guidance Scale:            {guidance_scale}\n"
                f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
                f"Shift:                     {flow_shift}\n"
                f"Seed:                      {seed}\n"
                f"Video Resolution:          {width} x {height}\n"
                f"Attn mode:                 {self.transformer.config.attn_mode}\n"
                f"Transformer dtype:         {self.transformer.dtype}\n"
                f"Sampling Steps:            {num_inference_steps}\n"
                f"Use Meanflow:              {self.use_meanflow}\n"
                f"{'=' * 60}"
                "\n"
            )

        _sync_cuda(device)
        pipeline_start = time.perf_counter()
        encoding_start = time.perf_counter()

        # Keep text encoders transient even when AR/model offloading is disabled.
        # The text models are only needed for prompt embedding and should not
        # inflate the AR rollout peak memory measurement.
        with auto_offload_model(
            self.text_encoder, self.execution_device, enabled=True
        ):
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=self.clip_skip,
                data_type="video",
            )

        # Encode prompts with second encoder if available
        if self.text_encoder_2 is not None:
            with auto_offload_model(
                self.text_encoder_2,
                self.execution_device,
                enabled=True,
            ):
                (
                    prompt_embeds_2,
                    negative_prompt_embeds_2,
                    prompt_mask_2,
                    negative_prompt_mask_2,
                ) = self.encode_prompt(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=self.clip_skip,
                    text_encoder=self.text_encoder_2,
                    data_type="video",
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        extra_kwargs = {}
        if self.config.glyph_byT5_v2:
            with auto_offload_model(
                self.byt5_model, self.execution_device, enabled=True
            ):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        with auto_offload_model(
            self.vae, self.execution_device, enabled=self.enable_offloading
        ):
            image_cond = self.get_image_condition_latents(
                task_type, reference_image, height, width
            )

        cond_latents = self._prepare_cond_latents(
            task_type, image_cond, latents, multitask_mask
        )
        with auto_offload_model(
            self.vision_encoder, self.execution_device, enabled=self.enable_offloading
        ):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, latents, device
            )

        self.num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )
        self._num_timesteps = len(timesteps)

        latent_frames = latents.shape[2]
        self.points_local = generate_points_in_sphere(50000, 8.0).to(device)
        self.chunk_num = latent_frames // chunk_latent_frames
        self.chunk_latent_frames = chunk_latent_frames
        self.num_inference_steps = num_inference_steps

        encoding_time = _elapsed_since(encoding_start, device)
        ar_start = time.perf_counter()
        if model_type == "ar":
            latents = self.ar_rollout(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                vision_states=vision_states,
                cond_latents=cond_latents,
                task_type=task_type,
                extra_kwargs=extra_kwargs,
                viewmats=viewmats,
                Ks=Ks,
                action=action,
                device=device,
                transformer_resident=kwargs.get("transformer_resident_ar_rollout", False),
            )
        elif model_type == "bi":
            latents = self.bi_rollout(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                vision_states=vision_states,
                cond_latents=cond_latents,
                task_type=task_type,
                extra_kwargs=extra_kwargs,
                viewmats=viewmats,
                Ks=Ks,
                action=action,
                device=device,
            )
        ar_rollout_time = _elapsed_since(ar_start, device)

        vae_start = time.perf_counter()
        if enable_sr:
            assert hasattr(self, "sr_pipeline")
            sr_out = self.sr_pipeline(
                prompt=prompt,
                num_inference_steps=sr_num_inference_steps,
                video_length=video_length,
                negative_prompt="",
                num_videos_per_prompt=num_videos_per_prompt,
                seed=seed,
                output_type=output_type,
                lq_latents=latents,
                reference_image=user_reference_image,
            )

        if output_type == "latent":
            video_frames = latents
        else:
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)
            elif len(latents.shape) != 5:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            if get_infer_state() and get_infer_state().use_vae_parallel:
                self.vae.enable_spatial_tiling()
                self.vae.enable_tile_parallelism()

            if return_pre_sr_video or not enable_sr:
                with (
                    torch.autocast(
                        device_type="cuda",
                        dtype=self.vae_dtype,
                        enabled=self.vae_autocast_enabled,
                    ),
                    auto_offload_model(
                        self.vae, self.execution_device, enabled=self.enable_offloading
                    ),
                ):
                    video_frames = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

                if video_frames is not None:
                    video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()

            else:
                video_frames = sr_out.videos
        vae_decode_time = _elapsed_since(vae_start, device)

        # Offload all models
        self.maybe_free_model_hooks()
        if enable_sr:
            sr_video_frames = sr_out.videos

        total_e2e_time = _elapsed_since(pipeline_start, device)
        ar_perf_stats = getattr(self, "_last_ar_perf_stats", {})
        kv_recompute = ar_perf_stats.get("total_recompute_time_s", 0.0)
        denoise_loop = ar_perf_stats.get("total_denoise_time_s", 0.0)
        dit_core = ar_perf_stats.get(
            "total_dit_core_time_s", kv_recompute + denoise_loop
        )
        history_selection = ar_perf_stats.get("total_history_selection_time_s", 0.0)
        peak_allocated = ar_perf_stats.get("peak_allocated_gb", 0.0)
        peak_reserved = ar_perf_stats.get("peak_reserved_gb", 0.0)

        if get_rank() == 0:
            print("\n" + "=" * 60)
            print("[Inference Timing Summary]")
            print("=" * 60)
            print(f" Prompt & Vision Encode    : {encoding_time:>8.2f} s")
            print(f" History Selection         : {history_selection:>8.2f} s")
            print(f" KV Cache Recompute        : {kv_recompute:>8.2f} s")
            print(f" Denoise Loop              : {denoise_loop:>8.2f} s")
            print(f" DiT Core (KV + Denoise)   : {dit_core:>8.2f} s")
            print(f" Transformer AR Rollout    : {ar_rollout_time:>8.2f} s")
            if enable_sr:
                print(" Super Resolution (SR)     : included above")
            print(f" VAE Pixel Decoding        : {vae_decode_time:>8.2f} s")
            print("-" * 60)
            print(f" Total End-to-End Time     : {total_e2e_time:>8.2f} s")
            if peak_allocated:
                print(f" AR Peak Allocated         : {peak_allocated:>8.2f} GB")
            if peak_reserved:
                print(f" AR Peak Reserved          : {peak_reserved:>8.2f} GB")
            print("=" * 60 + "\n")
            write_kv_ae3d_summary()
            write_kv_codec_summary()
            write_qvg_summary()
            write_jensen_summary()
            write_turboquant_summary()

        if not return_dict:
            ret = video_frames
            if enable_sr:
                ret = (video_frames, sr_video_frames)
            return ret

        if enable_sr:
            return HunyuanVideoPipelineOutput(
                videos=video_frames, sr_videos=sr_video_frames
            )
        else:
            return HunyuanVideoPipelineOutput(videos=video_frames)

    @property
    def ideal_resolution(self):
        return self.transformer.config.ideal_resolution

    @property
    def ideal_task(self):
        return self.transformer.config.ideal_task

    @property
    def use_meanflow(self):
        return self.transformer.config.use_meanflow

    @classmethod
    def load_sr_transformer_upsampler(
        cls, cached_folder, sr_version, transformer_dtype=torch.bfloat16, device=None
    ):
        sr_path = os.path.join(cached_folder, "transformer", sr_version)
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            sr_path, torch_dtype=transformer_dtype
        ).to(device)
        upsampler_cls = (
            SRTo720pUpsampler if "720p" in sr_version else SRTo1080pUpsampler
        )
        upsampler = upsampler_cls.from_pretrained(
            os.path.join(cached_folder, "upsampler", sr_version)
        ).to(device)
        return transformer, upsampler

    def create_sr_pipeline(
        self, cached_folder, sr_version, transformer_dtype=torch.bfloat16, device=None
    ):
        from .hunyuan_video_sr_pipeline import HunyuanVideo_1_5_SR_Pipeline

        transformer, upsampler = self.load_sr_transformer_upsampler(
            cached_folder,
            sr_version,
            transformer_dtype=transformer_dtype,
            device=device,
        )

        return HunyuanVideo_1_5_SR_Pipeline(
            vae=self.vae,
            transformer=transformer,
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            upsampler=upsampler,
            text_encoder_2=self.text_encoder_2,
            progress_bar_config=None,
            glyph_byT5_v2=self.config.glyph_byT5_v2,
            byt5_model=self.byt5_model,
            byt5_tokenizer=self.byt5_tokenizer,
            byt5_max_length=self.byt5_max_length,
            prompt_format=self.prompt_format,
            execution_device="cuda",
            vision_encoder=self.vision_encoder,
            enable_offloading=self.enable_offloading,
            **SR_PIPELINE_CONFIGS[sr_version],
        )

    @classmethod
    def create_pipeline(
        cls,
        pretrained_model_name_or_path,
        transformer_version,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        enable_offloading=None,
        enable_group_offloading=None,
        overlap_group_offloading=True,
        device=None,
        action_ckpt=None,
        **kwargs,
    ):
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        if enable_offloading:
            # Assuming the user does not have sufficient GPU memory, we initialize the models on CPU
            device = torch.device("cpu")
        else:
            if device is None:
                device = torch.device("cuda")

        if enable_group_offloading:
            # Assuming the user does not have sufficient GPU memory, we initialize the models on CPU
            transformer_init_device = torch.device("cpu")
        else:
            transformer_init_device = device

        supported_transformer_version = os.listdir(
            os.path.join(cached_folder, "transformer")
        )
        if transformer_version not in supported_transformer_version:
            raise ValueError(
                f"Could not find {transformer_version} in {cached_folder}."
                f"Only {supported_transformer_version} are available."
            )

        vae_inference_config = cls.get_vae_inference_config()
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            os.path.join(cached_folder, "transformer", transformer_version),
            torch_dtype=transformer_dtype,
            low_cpu_mem_usage=True,
        )

        transformer.add_action_parameters()
        if action_ckpt is not None:
            from safetensors.torch import load_file

            safetensor_path = action_ckpt
            state_dict = load_file(safetensor_path)
            transformer.load_state_dict(state_dict, strict=True)
            print("HY-World 1.5 loading from: ", action_ckpt)

        transformer = transformer.to(transformer_dtype).to(transformer_init_device)

        infer_state = get_infer_state()
        if infer_state.use_fp8_gemm:
            from angelslim.compressor.diffusion import DynamicDiTQuantizer

            quant_type = infer_state.quant_type
            include_patterns = infer_state.include_patterns
            quantizer = DynamicDiTQuantizer(
                quant_type=quant_type, include_patterns=include_patterns
            )
            quantizer.convert_linear(transformer)

        vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
            os.path.join(cached_folder, "vae"),
            torch_dtype=vae_inference_config["dtype"],
        ).to(device)
        scheduler = FlowMatchDiscreteScheduler.from_pretrained(
            os.path.join(cached_folder, "scheduler")
        )

        if force_sparse_attn:
            if not is_sparse_attn_supported():
                raise RuntimeError(
                    f"Current GPU is {torch.cuda.get_device_properties(0).name}, "
                    f"which does not support sparse attention."
                )
            if transformer.config.attn_mode != "flex-block-attn":
                loguru.logger.warning(
                    f"The transformer loaded ({transformer_version}) is not trained with sparse attention."
                    f"Forcing to use sparse attention may lead to artifacts in the generated video."
                    f"To enable sparse attention, we recommend loading `{transformer_version}_distilled_sparse`."
                )
            transformer.set_attn_mode("flex-block-attn")

        byt5_kwargs, prompt_format = cls._load_byt5(
            cached_folder, True, 256, device=device
        )
        text_encoder, text_encoder_2 = cls._load_text_encoders(
            cached_folder, device=device
        )
        vision_encoder = cls._load_vision_encoder(cached_folder, device=device)

        group_offloading_kwargs = {
            "onload_device": torch.device("cuda"),
            "num_blocks_per_group": 4,
        }
        if overlap_group_offloading:
            # Using streams is only supported for num_blocks_per_group=1
            group_offloading_kwargs["num_blocks_per_group"] = 1
            group_offloading_kwargs["use_stream"] = True
            group_offloading_kwargs["record_stream"] = True

        if enable_group_offloading:
            assert enable_offloading
            transformer.enable_group_offload(**group_offloading_kwargs)

        pipeline = cls(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            progress_bar_config=None,
            byt5_model=byt5_kwargs["byt5_model"],
            byt5_tokenizer=byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=byt5_kwargs["byt5_max_length"],
            prompt_format=prompt_format,
            execution_device="cuda",
            vision_encoder=vision_encoder,
            enable_offloading=enable_offloading,
            **PIPELINE_CONFIGS[transformer_version],
        )

        if create_sr_pipeline:
            sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[transformer_version]
            sr_pipeline = pipeline.create_sr_pipeline(
                cached_folder,
                sr_version,
                transformer_dtype=transformer_dtype,
                device=device,
            )
            pipeline.sr_pipeline = sr_pipeline
            if enable_group_offloading:
                sr_pipeline.transformer.enable_group_offload(**group_offloading_kwargs)

        return pipeline

    @staticmethod
    def get_vae_inference_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 23 * GB:
            sample_size = 160
            tile_overlap_factor = 0.2
            dtype = torch.float16
        else:
            sample_size = 256
            tile_overlap_factor = 0.25
            dtype = torch.float32
        return {
            "sample_size": sample_size,
            "tile_overlap_factor": tile_overlap_factor,
            "dtype": dtype,
        }

    @classmethod
    def _load_text_encoders(cls, pretrained_model_path, device):
        text_encoder_path = f"{pretrained_model_path}/text_encoder/llm"
        if not os.path.exists(text_encoder_path):
            msg = f"{text_encoder_path} not found. Please refer to checkpoints-download.md."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=text_encoder_path,
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,
        )
        text_encoder_2 = None

        return text_encoder, text_encoder_2

    @classmethod
    def _load_vision_encoder(cls, pretrained_model_name_or_path, device):
        vision_encoder_path = f"{pretrained_model_name_or_path}/vision_encoder/siglip"
        if not os.path.exists(vision_encoder_path):
            msg = f"{vision_encoder_path} not found. Please refer to checkpoints-download.md."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=vision_encoder_path,
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=logger,
            device=device,
        )
        return vision_encoder
