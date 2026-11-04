"""Experimental direct attention over organized rank-512 INT4 codec payloads.

This module is dormant unless HY_DIRECT_CODEC_LATENT_ENABLE=1.  Selected layers
replace the formal full-decode history path with a learned codec coordinate
system whose read path is INT4 dequantization followed by reduced attention.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import weakref
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from hyvideo.models.transformers.modules.kv_codecs import (
    _joint_semantic_smooth,
    _operator_linear,
    _scale_storage_dtype,
    pack_quantized_codes,
    unpack_quantized_codes,
)
from hyvideo.models.transformers.modules.posemb_layers import apply_rotary_emb_one
from hyvideo.prope.camera_rope import (
    _prepare_apply_fns_all_dim,
    _prepare_projection_matrices,
)
from hyvideo.models.transformers.modules import (
    codec_native_history_adapter,
    codec_native_routing_distill,
    position_profile,
)


_MANIFEST = None
_CPU_LAYERS = {}
_CPU_OPERATORS = {}
_DEVICE_LAYERS = {}
_DEVICE_ROPE_INDICES = {}
_MERGED_BLOCKS = set()
_OPERATOR_SHA256_CACHE = {}

# These caches only hold small positional transforms / gathered frequencies.
# Entries never own the source tensors: weak references plus Tensor._version
# prevent a recycled Python address or an in-place camera/frequency update from
# reusing stale results.  LRU bounds prevent one entry per generated chunk from
# accumulating over a long video.
_PROPE_FN_CACHE = OrderedDict()
_PROPE_MATRIX_CACHE = OrderedDict()
_ROPE_DEVICE_CACHE = OrderedDict()
_SELECTED_ROPE_CACHE = OrderedDict()
_SELECTED_ROPE_SEEN = OrderedDict()
_PROPE_FN_CACHE_MAX = 32
_PROPE_MATRIX_CACHE_MAX = 64
_PROPE_MATRIX_CACHE_MAX_BYTES = 64 * 1024 * 1024
_ROPE_DEVICE_CACHE_MAX = 16
_ROPE_DEVICE_CACHE_MAX_BYTES = 64 * 1024 * 1024
_SELECTED_ROPE_CACHE_MAX = 32
_SELECTED_ROPE_SEEN_MAX = 256
_SELECTED_ROPE_CACHE_MAX_BYTES = 512 * 1024 * 1024
_COMPILED_SELECTED_ROPE = None
_COMPILED_PROPE_KV_PAIR = None
_COMPILED_PROPE_ONE = None
_COMPILED_CENTER_PAIR_ADD = None
_COMPILED_LSE_COMBINE = None
_PARALLEL_NATIVE_STREAMS = {}
_PARALLEL_REBUILD_STREAMS = {}
_FLASH_BACKEND_LAYER_IDX = None


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _triton_data_flag(name: str) -> bool:
    """Gate byte-preserving Triton data movement independently of math."""
    if name in os.environ:
        return _flag(name)
    return _flag("HY_DIRECT_CODEC_TRITON_POSITION")


def _scoped_flag(name: str, umbrella: str) -> bool:
    """Let a branch override a legacy umbrella without changing defaults."""
    if name in os.environ:
        return _flag(name)
    return _flag(umbrella)


def _any_fused_prope_scope() -> bool:
    return any(
        _scoped_flag(name, "HY_DIRECT_CODEC_STREAM_FUSED_PROPE")
        for name in (
            "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
            "HY_DIRECT_CODEC_LOW_QUERY_FUSED_PROPE",
            "HY_DIRECT_CODEC_NATIVE_FUSED_PROPE",
        )
    )


def _integer_env_set(name: str) -> set[int]:
    """Parse an optional comma-separated integer set used by runtime gates."""
    text = os.environ.get(name, "").strip()
    if not text:
        return set()
    try:
        return {int(value.strip()) for value in text.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of integers") from error


def _string_env_set(name: str) -> set[str]:
    """Parse an optional comma-separated lowercase string set."""
    return {
        value.strip().lower()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    }


def _effective_flash_backend() -> str:
    """Resolve the low-D Attention backend for the current layer/denoise step.

    The base policy remains ``HY_DIRECT_CODEC_FLASH_BACKEND``. Opt-in
    guards permit quality-sensitive calls to retain the ATen numerical path
    while other low-dimensional calls use a faster backend. Empty guards are
    exactly backward compatible.
    """
    backend = os.environ.get(
        "HY_DIRECT_CODEC_FLASH_BACKEND", "aten"
    ).strip().lower()
    aten_layers = _integer_env_set("HY_DIRECT_CODEC_FLASH_ATEN_LAYERS")
    if (
        _FLASH_BACKEND_LAYER_IDX is not None
        and int(_FLASH_BACKEND_LAYER_IDX) in aten_layers
    ):
        return "aten"
    aten_steps = _integer_env_set("HY_DIRECT_CODEC_FLASH_ATEN_STEPS")
    step_text = os.environ.get("HY_ROLLOUT_DRIFT_STEP", "").strip()
    if step_text and int(step_text) in aten_steps:
        return "aten"
    aten_stages = _string_env_set("HY_DIRECT_CODEC_FLASH_ATEN_STAGES")
    stage = os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "").strip().lower()
    if stage and stage in aten_stages:
        return "aten"
    return backend


def _profile_call(bucket: str, layer_idx: int, fn, *args, **kwargs):
    """CUDA-time one direct-codec component only in explicit profile runs."""
    return position_profile.cuda_call(bucket, layer_idx, fn, *args, **kwargs)


def _parallel_native_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _PARALLEL_NATIVE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _PARALLEL_NATIVE_STREAMS[index] = stream
    return stream


def _parallel_rebuild_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _PARALLEL_REBUILD_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _PARALLEL_REBUILD_STREAMS[index] = stream
    return stream


def _record_low_history_stream(
    old_groups: list[dict] | None,
    old_k: torch.Tensor | None,
    old_v: torch.Tensor | None,
    stream: torch.cuda.Stream,
) -> None:
    """Tell the caching allocator that side-stream results cross to ``stream``."""
    if old_groups is not None:
        for group in old_groups:
            group["k"].record_stream(stream)
            group["v"].record_stream(stream)
    else:
        old_k.record_stream(stream)
        old_v.record_stream(stream)


def _formal_teacher_needs_centers(layer_idx: int) -> bool:
    """Keep the direct encoder's C64 table for a training-only Teacher view.

    The ordinary raw-recent inference path deliberately drops this full table.
    A causal formal Teacher must instead use the *same* C64 table which produced
    the deployed direct payload, rather than trusting a second online K-Means
    execution to reproduce it.
    """
    if not _flag("HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER"):
        return False
    text = os.environ.get(
        "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS", ""
    ).strip()
    if not text:
        return False
    try:
        return int(layer_idx) in {
            int(value.strip()) for value in text.split(",") if value.strip()
        }
    except ValueError as error:
        raise ValueError(
            "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS must be a "
            "comma-separated integer list"
        ) from error


def _formal_source_identity(x: torch.Tensor) -> dict:
    """Identity of the raw cache tensor consumed by the sole numeric encode."""
    return {
        "data_ptr": int(x.data_ptr()),
        "version": int(x._version),
        "shape": tuple(int(value) for value in x.shape),
        "stride": tuple(int(value) for value in x.stride()),
        "storage_offset": int(x.storage_offset()),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def _operator_sha256(path: Path) -> str:
    resolved = str(path.resolve())
    cached = _OPERATOR_SHA256_CACHE.get(resolved)
    if cached is None:
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        cached = digest.hexdigest()
        _OPERATOR_SHA256_CACHE[resolved] = cached
    return cached


def _recent_native_frames() -> int:
    """Number of most-recent cached latent frames restored at native D128."""
    value = int(os.environ.get("HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES", "0") or 0)
    if value < 0:
        raise ValueError("HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES must be non-negative")
    return value


def _recent_native_frames_for_layer(layer_idx: int) -> int:
    """Return the effective recent-native horizon for one selected layer.

    An unset/empty layer list deliberately means all manifest-selected layers,
    preserving the original ``HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES`` canary.
    A layer excluded by the list is indistinguishable from recent-native=0:
    it uses the original device-cache key and owns no native-only state.
    """
    frames = _recent_native_frames()
    if frames == 0:
        return 0
    text = os.environ.get("HY_DIRECT_CODEC_RECENT_NATIVE_LAYERS", "").strip()
    if not text:
        return frames
    try:
        selected = {int(value.strip()) for value in text.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(
            "HY_DIRECT_CODEC_RECENT_NATIVE_LAYERS must be a comma-separated "
            "list of integer layer indices"
        ) from error
    return frames if int(layer_idx) in selected else 0


def _recent_capture_raw() -> bool:
    """Whether recent-native should capture its input BF16 K/V tail directly."""
    return _flag("HY_DIRECT_CODEC_RECENT_CAPTURE_RAW")


def _nonnegative_env_int(name: str) -> int:
    value = int(os.environ.get(name, "0") or 0)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _formal_anchor_policy_for_layer(layer_idx: int) -> tuple[int, int]:
    """Return ``(stride, identity_sink_frames)`` for one selected layer.

    A stride of ``N`` promotes the newest frame in every ``N``-frame bucket
    of the old (non-recent) history to a formal D128 codec decode.  The
    optional identity sink independently promotes the first requested cached
    latent frames.  Both controls default to zero, so the established
    recent-native path remains byte-for-byte dormant unless explicitly opted
    in.  The layer filter follows the recent-native convention: an empty list
    means every manifest-selected layer.
    """
    stride = _nonnegative_env_int("HY_DIRECT_CODEC_FORMAL_ANCHOR_STRIDE")
    sink = _nonnegative_env_int(
        "HY_DIRECT_CODEC_FORMAL_ANCHOR_IDENTITY_SINK_FRAMES"
    )
    # Optionally override the global stride for a small set of sensitive
    # layers.  This keeps the original single-stride interface as the default,
    # while allowing a compute-budgeted two-tier policy such as global stride
    # 16 plus stride 8 only on the layers where exact history anchors buy the
    # most closed-loop stability.  The compact ``layer:stride`` mapping is
    # intentionally explicit: an omitted layer inherits the global policy.
    overrides = os.environ.get(
        "HY_DIRECT_CODEC_FORMAL_ANCHOR_LAYER_STRIDES", ""
    ).strip()
    if overrides:
        try:
            parsed = {}
            for item in overrides.split(","):
                item = item.strip()
                if not item:
                    continue
                layer_text, stride_text = item.split(":", 1)
                override_stride = int(stride_text.strip())
                if override_stride < 0:
                    raise ValueError("stride must be non-negative")
                parsed[int(layer_text.strip())] = override_stride
        except (TypeError, ValueError) as error:
            raise ValueError(
                "HY_DIRECT_CODEC_FORMAL_ANCHOR_LAYER_STRIDES must be a "
                "comma-separated layer:stride mapping with non-negative "
                "integer values"
            ) from error
        stride = parsed.get(int(layer_idx), stride)
    if stride == 0 and sink == 0:
        return 0, 0
    text = os.environ.get("HY_DIRECT_CODEC_FORMAL_ANCHOR_LAYERS", "").strip()
    if text:
        try:
            selected = {
                int(value.strip()) for value in text.split(",") if value.strip()
            }
        except ValueError as error:
            raise ValueError(
                "HY_DIRECT_CODEC_FORMAL_ANCHOR_LAYERS must be a "
                "comma-separated list of integer layer indices"
            ) from error
        if int(layer_idx) not in selected:
            return 0, 0
    return stride, sink


def _periodic_refresh_policy() -> tuple[int, int]:
    """Return ``(interval, phase)`` for transient full-history refreshes.

    The feature is strictly dormant when ``interval == 0``.  For a positive
    interval, a cache written for autoregressive chunk ``c`` is refreshed iff
    ``c % interval == phase``.  Binding the decision when the cache is written
    (rather than re-reading a mutable process environment at every attention
    call) keeps all denoise steps in one generated chunk on the same path.
    """
    interval = _nonnegative_env_int(
        "HY_DIRECT_CODEC_PERIODIC_REFRESH_INTERVAL"
    )
    phase = _nonnegative_env_int("HY_DIRECT_CODEC_PERIODIC_REFRESH_PHASE")
    if interval == 0:
        if phase != 0:
            raise ValueError(
                "HY_DIRECT_CODEC_PERIODIC_REFRESH_PHASE must be zero when "
                "HY_DIRECT_CODEC_PERIODIC_REFRESH_INTERVAL is zero"
            )
        return 0, 0
    if phase >= interval:
        raise ValueError(
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_PHASE must be smaller than "
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_INTERVAL"
        )
    return interval, phase


def _periodic_refresh_steps() -> tuple[int, ...]:
    """Return optional denoise-step selectors for a refresh chunk.

    An empty tuple preserves the original behavior: every denoise read of a
    selected chunk uses full-history formal Attention.  A non-empty list makes
    the refresh transient at the selected denoise steps only.
    """
    text = os.environ.get(
        "HY_DIRECT_CODEC_PERIODIC_REFRESH_STEPS", ""
    ).strip()
    if not text:
        return ()
    try:
        steps = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    except ValueError as error:
        raise ValueError(
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_STEPS must be a comma-separated "
            "list of non-negative denoise-step indices"
        ) from error
    if not steps or any(step < 0 for step in steps):
        raise ValueError(
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_STEPS must contain non-negative "
            "denoise-step indices"
        )
    return steps


def _periodic_refresh_for_step(
    chunk_refresh: bool, steps: tuple[int, ...]
) -> tuple[bool, int]:
    """Resolve a cache-bound chunk decision at the current denoise step."""
    if not chunk_refresh or not steps:
        return chunk_refresh, -1
    text = os.environ.get("HY_ROLLOUT_DRIFT_STEP", "").strip()
    if not text:
        raise RuntimeError(
            "step-selective periodic refresh requires HY_ROLLOUT_DRIFT_STEP"
        )
    try:
        step = int(text)
    except ValueError as error:
        raise ValueError(
            "HY_ROLLOUT_DRIFT_STEP must be an integer for step-selective "
            "periodic refresh"
        ) from error
    return step in steps, step


def _periodic_refresh_for_chunk(
    chunk_id: int, interval: int, phase: int
) -> bool:
    """Whether one cache-write chunk receives a transient formal refresh."""
    chunk_id, interval, phase = int(chunk_id), int(interval), int(phase)
    if interval == 0:
        return False
    if chunk_id < 0 or interval < 0 or phase < 0 or phase >= interval:
        raise ValueError(
            f"invalid periodic refresh tuple chunk={chunk_id}, "
            f"interval={interval}, phase={phase}"
        )
    return chunk_id % interval == phase


_PERIODIC_REFRESH_CACHE_KEYS = (
    "_direct_codec_cache_write_chunk_id",
    "_direct_codec_periodic_refresh_interval",
    "_direct_codec_periodic_refresh_phase",
    "_direct_codec_periodic_refresh_active",
)


def _bind_periodic_refresh_cache_write(layer_cache: dict, runtime: dict) -> None:
    """Freeze a periodic-refresh decision into a newly written layer cache."""
    interval = int(runtime.get("periodic_refresh_interval", 0))
    phase = int(runtime.get("periodic_refresh_phase", 0))
    if interval == 0:
        # Cache dictionaries are normally newly allocated, but remove stale
        # opt-in metadata if an offline caller deliberately reuses one.
        for name in _PERIODIC_REFRESH_CACHE_KEYS:
            layer_cache.pop(name, None)
        return
    text = os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "").strip()
    if not text:
        raise RuntimeError(
            "periodic refresh requires HY_KV_MULTIPOS_AR_CHUNK_ID at the "
            "cache-write boundary"
        )
    try:
        chunk_id = int(text)
    except ValueError as error:
        raise ValueError(
            "HY_KV_MULTIPOS_AR_CHUNK_ID must be an integer for periodic refresh"
        ) from error
    active = _periodic_refresh_for_chunk(chunk_id, interval, phase)
    layer_cache["_direct_codec_cache_write_chunk_id"] = chunk_id
    layer_cache["_direct_codec_periodic_refresh_interval"] = interval
    layer_cache["_direct_codec_periodic_refresh_phase"] = phase
    layer_cache["_direct_codec_periodic_refresh_active"] = active


def _periodic_refresh_from_cache(
    layer_cache: dict, runtime: dict
) -> tuple[bool, int]:
    """Read and validate the refresh decision frozen at cache-write time."""
    interval = int(runtime.get("periodic_refresh_interval", 0))
    phase = int(runtime.get("periodic_refresh_phase", 0))
    has_metadata = any(name in layer_cache for name in _PERIODIC_REFRESH_CACHE_KEYS)
    if interval == 0:
        if has_metadata:
            raise RuntimeError(
                "periodic refresh environment changed after this cache was written"
            )
        return False, -1
    missing = [name for name in _PERIODIC_REFRESH_CACHE_KEYS if name not in layer_cache]
    if missing:
        raise RuntimeError(
            "periodic refresh cache-write metadata is missing: " + ",".join(missing)
        )
    stored_interval = int(
        layer_cache["_direct_codec_periodic_refresh_interval"]
    )
    stored_phase = int(layer_cache["_direct_codec_periodic_refresh_phase"])
    if (stored_interval, stored_phase) != (interval, phase):
        raise RuntimeError(
            "periodic refresh policy changed after this cache was written: "
            f"cache=({stored_interval},{stored_phase}) "
            f"runtime=({interval},{phase})"
        )
    chunk_id = int(layer_cache["_direct_codec_cache_write_chunk_id"])
    expected = _periodic_refresh_for_chunk(chunk_id, interval, phase)
    active = bool(layer_cache["_direct_codec_periodic_refresh_active"])
    if active != expected:
        raise RuntimeError(
            "periodic refresh cache-write decision is internally inconsistent"
        )
    return active, chunk_id


def _effective_formal_anchor_policy(
    layer_cache: dict, runtime: dict
) -> tuple[int, int, bool, int]:
    """Resolve static anchors and a transient all-old-history refresh.

    A refresh is represented as stride one over the old (non-recent) prefix,
    which reuses the exact formal-anchor decode/attention implementation and
    therefore guarantees ``low_tokens == 0`` without storing native K/V in the
    persistent codec payload.
    """
    chunk_refresh, chunk_id = _periodic_refresh_from_cache(layer_cache, runtime)
    refresh, _step = _periodic_refresh_for_step(
        chunk_refresh, tuple(runtime.get("periodic_refresh_steps", ()))
    )
    if refresh:
        return 1, 0, True, chunk_id
    return (
        int(runtime.get("formal_anchor_stride", 0)),
        int(runtime.get("formal_anchor_identity_sink_frames", 0)),
        False,
        chunk_id,
    )


def _manifest():
    global _MANIFEST
    if _MANIFEST is None:
        path = os.environ.get("HY_DIRECT_CODEC_LATENT_MANIFEST", "").strip()
        if not path:
            raise RuntimeError("HY_DIRECT_CODEC_LATENT_MANIFEST is required")
        with open(path, "r", encoding="utf-8") as handle:
            _MANIFEST = json.load(handle)
    return _MANIFEST


def enabled_for_layer(layer_idx: int) -> bool:
    if not _flag("HY_DIRECT_CODEC_LATENT_ENABLE"):
        return False
    layer_idx = int(layer_idx)
    if str(layer_idx) not in _manifest().get("layers", {}):
        return False
    include = os.environ.get("HY_DIRECT_CODEC_LATENT_INCLUDE_LAYERS", "").strip()
    if include:
        selected = {int(value) for value in include.split(",") if value.strip()}
        if layer_idx not in selected:
            return False
    exclude = os.environ.get("HY_DIRECT_CODEC_LATENT_EXCLUDE_LAYERS", "").strip()
    if exclude:
        skipped = {int(value) for value in exclude.split(",") if value.strip()}
        if layer_idx in skipped:
            return False
    return True


def ensure_history_adapter_runtime(
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Materialize the opt-in history-only adapter bank for configured layers.

    The low query width is taken from the active dictionary-lowkv manifest,
    so the adapter cannot silently target a different representation.  No
    codec checkpoint, operator, or manifest is modified.
    """

    if not codec_native_history_adapter.enabled():
        return codec_native_history_adapter.bank()
    # ``attend`` is called once per selected block.  Materializing the bank is
    # process-global, so do not re-parse and revalidate every selected layer on
    # every block call.  Keep the cache key on the bank itself: the test reset
    # replaces the bank, and a meaningful runtime/env change invalidates the
    # fast path without introducing a second process-global lifetime.
    adapter_bank = codec_native_history_adapter.bank()
    runtime_key = (
        "direct_history_adapter_v1",
        str(torch.device(device)),
        dtype,
        os.environ.get("HY_DIRECT_CODEC_LATENT_ENABLE", ""),
        os.environ.get("HY_DIRECT_CODEC_LATENT_MANIFEST", ""),
        os.environ.get("HY_DIRECT_CODEC_LATENT_INCLUDE_LAYERS", ""),
        os.environ.get("HY_DIRECT_CODEC_LATENT_EXCLUDE_LAYERS", ""),
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS", ""),
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_RANK", ""),
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_ALPHA", ""),
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_TARGETS", ""),
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_CKPT", ""),
    )
    if getattr(adapter_bank, "_direct_runtime_key", None) == runtime_key:
        return adapter_bank
    layer_dims = {}
    for layer_idx in sorted(codec_native_history_adapter.selected_layers()):
        if not enabled_for_layer(layer_idx):
            raise RuntimeError(
                f"history adapter requested L{layer_idx}, but that layer is not "
                "enabled by the active direct-codec manifest/include policy"
            )
        layer = _cpu_layer(layer_idx)
        if layer["payload_mode"] != "dictionary_lowkv":
            raise RuntimeError(
                f"history adapter requires dictionary_lowkv at L{layer_idx}, "
                f"got {layer['payload_mode']!r}"
            )
        layer_dims[layer_idx] = int(layer["shape"][1])
    adapter_bank = codec_native_history_adapter.initialize(
        layer_dims,
        device=torch.device(device),
        dtype=dtype,
        heads=16,
        output_dim=128,
    )
    adapter_bank._direct_runtime_key = runtime_key
    return adapter_bank


def _unpack_weight(entry: dict, name: str) -> torch.Tensor:
    packed = entry[name]
    qweight = packed["qweight"]
    scale = packed["scale"]
    group = int(packed["group_size"])
    return (qweight.reshape(qweight.shape[0], -1, group).to(torch.bfloat16)
            * scale[:, :, None]).reshape_as(qweight).float()


def _packed_weight_to_device(entry: dict, name: str, device: torch.device) -> dict:
    """Move one prepacked INT8 operator without expanding its resident weight."""
    packed = entry[name]
    result = {
        "storage": "int8",
        "qweight": packed["qweight"].to(device=device),
        "scale": packed["scale"].to(device=device, dtype=torch.bfloat16),
        "group_size": int(packed["group_size"]),
    }
    if _flag("HY_DIRECT_CODEC_PREDEQUANT_DECODER", "1"):
        # The formal decoder is invoked repeatedly for recent/anchor rows in
        # strict streaming.  Materializing its byte-identical BF16 weight once
        # costs 2 MiB per K/V decoder (216 MiB for all 54 layers), below the
        # 0.5-GiB contract, and removes the INT8->BF16 expansion on every read.
        qweight = result["qweight"]
        group_size = result["group_size"]
        result["bf16_weight"] = (
            qweight.reshape(qweight.shape[0], -1, group_size).to(torch.bfloat16)
            * result["scale"][:, :, None]
        ).reshape_as(qweight)
    return result


def _native_decoder_linear(
    rows: torch.Tensor,
    packed: dict,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Formal decoder GEMM with an optional persistent exact BF16 weight."""
    weight = packed.get("bf16_weight")
    if isinstance(weight, torch.Tensor):
        bias_bf16 = None if bias is None else bias.to(torch.bfloat16)
        return F.linear(rows.to(torch.bfloat16), weight, bias_bf16)
    return _operator_linear(rows, packed, bias)


def _cpu_layer(layer_idx: int):
    layer_idx = int(layer_idx)
    cached = _CPU_LAYERS.get(layer_idx)
    if cached is not None:
        return cached
    manifest = _manifest()
    checkpoint = torch.load(manifest["layers"][str(layer_idx)], map_location="cpu", weights_only=False)
    operator_path = str(manifest["operator"])
    operator = _CPU_OPERATORS.get(operator_path)
    if operator is None:
        operator = torch.load(operator_path, map_location="cpu", weights_only=False)
        _CPU_OPERATORS[operator_path] = operator
    grouped_dictionary_lowkv = checkpoint.get("method") == (
        "formal_z512_int4_c64_dictionary_learned_rope_grouped_head_width"
    )
    dictionary_lowkv = (
        checkpoint.get("method") == "low_c64_kv_anchor_structured_residual_attention"
        or grouped_dictionary_lowkv
    )
    if grouped_dictionary_lowkv:
        formal_rank = int(checkpoint.get("formal_rank", -1))
        operator_rank = int(operator.get("rank", -1))
        if formal_rank not in {256, 512} or formal_rank != operator_rank:
            raise RuntimeError(
                "grouped dictionary-lowkv checkpoint/operator rank mismatch: "
                f"checkpoint={formal_rank} operator={operator_rank}"
            )
        head_dims = tuple(int(value) for value in checkpoint.get("head_dims", ()))
        groups = checkpoint.get("groups")
        unique_dims = sorted(set(head_dims))
        if (
            len(head_dims) != 16
            or not set(unique_dims).issubset({32, 64, 96})
            or not isinstance(groups, list)
            or len(groups) != len(unique_dims)
            or not 1 <= len(groups) <= 3
        ):
            raise RuntimeError(
                "grouped dictionary-lowkv runtime requires a 16-head diagonal "
                "D32/D64/D96 allocation with 16 assigned heads"
            )
        assigned = []
        grouped_states = []
        required = {
            "k_main_read", "q_main_reduce", "k_main_anchor_reduce",
            "v_main_read", "v_main_lift", "k_prope_read", "q_prope_mix",
            "v_prope_read", "v_prope_mix", "v_main_anchor_reduce",
            "v_prope_anchor_mix", "log_residual_scale_main",
            "log_residual_scale_prope", "main_indices",
        }
        for group in groups:
            dim = int(group["dim"])
            heads = group["head_indices"].detach().to(
                device="cpu", dtype=torch.long
            ).contiguous()
            tensors = group.get("tensors", {})
            head_count = head_dims.count(dim)
            if dim not in {32, 64, 96} or not required.issubset(tensors):
                raise RuntimeError("invalid grouped dictionary-lowkv tensor set")
            if heads.numel() != head_count or len(set(heads.tolist())) != head_count:
                raise RuntimeError(
                    "grouped dictionary-lowkv width/head-count mismatch"
                )
            if any(
                head < 0 or head >= 16 or head_dims[head] != dim
                for head in heads.tolist()
            ):
                raise RuntimeError(
                    "grouped dictionary-lowkv head_dims disagree with group assignment"
                )
            expected_shapes = {
                "k_main_read": (head_count, dim, formal_rank),
                "q_main_reduce": (head_count, dim, 128),
                "k_main_anchor_reduce": (head_count, dim, 128),
                "v_main_read": (head_count, dim, formal_rank),
                "v_main_lift": (head_count, dim, 128),
                "k_prope_read": (head_count, dim, formal_rank),
                "q_prope_mix": (head_count, dim // 4, 32),
                "v_prope_read": (head_count, dim, formal_rank),
                "v_prope_mix": (head_count, dim // 4, 32),
                "v_main_anchor_reduce": (head_count, dim, 128),
                "v_prope_anchor_mix": (head_count, dim // 4, 32),
                "log_residual_scale_main": (head_count,),
                "log_residual_scale_prope": (head_count,),
                "main_indices": (head_count, dim),
            }
            if any(
                tuple(tensors[name].shape) != shape
                for name, shape in expected_shapes.items()
            ):
                raise RuntimeError(
                    "grouped dictionary-lowkv checkpoint tensor shape mismatch"
                )
            assigned.extend(heads.tolist())
            grouped_states.append({
                "dim": dim,
                "head_indices": heads,
                "attention": {
                    name: value.detach().cpu() for name, value in tensors.items()
                },
            })
        if sorted(assigned) != list(range(16)):
            raise RuntimeError("grouped dictionary-lowkv heads must partition H0-H15")
        grouped_states.sort(key=lambda item: item["dim"])
        state = {}
        result = {
            "main_k_mode": "learned_rope",
            "k_anchor_mode": "low",
            "v_anchor_mode": "low",
            "key_dim": 64,
            "value_dim": 64,
            "compress_current": False,
        }
    else:
        state = checkpoint["model"] if dictionary_lowkv else checkpoint["attention"]
        result = checkpoint["config"] if dictionary_lowkv else checkpoint["result"]["config"]
    if dictionary_lowkv and not grouped_dictionary_lowkv:
        main_mode = result.get("main_k_mode")
        if main_mode not in {"pair_select", "learned_rope"} or result.get("k_anchor_mode") != "low" or result.get("v_anchor_mode") != "low":
            raise RuntimeError("dictionary-lowkv runtime requires pair_select/learned_rope and low K/V anchors")
        if checkpoint.get("main_indices") is None:
            raise RuntimeError("dictionary-lowkv checkpoint is missing main_indices")
        if main_mode == "learned_rope" and not {"q_main_reduce", "k_main_anchor_reduce"}.issubset(state):
            raise RuntimeError("learned_rope checkpoint is missing Q/anchor reduction matrices")
        state = dict(state)
        state["main_indices"] = checkpoint["main_indices"]
    payload_mode = "dictionary_lowkv" if dictionary_lowkv else result.get("payload_mode", "qat_organized")
    compress_current = bool(result.get("compress_current", False))
    shape = ((16, int(result["key_dim"]), 16, int(result["value_dim"])) if dictionary_lowkv else
             tuple(int(result[name]) for name in ("k_heads", "k_dim", "v_heads", "v_dim")))
    k_heads, k_dim, v_heads, v_dim = shape
    if payload_mode == "dictionary_lowkv":
        pass
    elif payload_mode == "half_decoder":
        if k_heads != 16 or v_heads != 16:
            raise RuntimeError(f"half decoder requires independent 16-head K/V, got {shape}")
    elif k_heads * k_dim != 512 or v_heads * v_dim != 512:
        raise RuntimeError(f"legacy online direct codec gate requires 512-D K/V payloads, got {shape}")
    if 16 % k_heads or 16 % v_heads:
        raise RuntimeError(f"online direct codec gate requires K/V heads dividing 16, got {shape}")
    if payload_mode != "dictionary_lowkv" and k_dim != v_dim:
        raise RuntimeError(f"online FlashAttention requires equal K/V dimensions, got {shape}")
    cached = {
        "attention": {name: value.detach().cpu() for name, value in state.items()},
        "block_lora": None if checkpoint.get("block_lora") is None else {
            name: value.detach().cpu() for name, value in checkpoint["block_lora"].items()
        },
        "k_entry": operator["entries"][f"k:{layer_idx:02d}"],
        "v_entry": operator["entries"][f"v:{layer_idx:02d}"],
        "shape": shape,
        "payload_mode": payload_mode,
        "compress_current": compress_current,
        "main_k_mode": result.get("main_k_mode"),
        "operator_path": str(Path(operator_path).resolve()),
        "operator_format": operator.get("format"),
        "operator_rank": int(operator.get("rank", -1)),
        "operator_group_size": int(operator.get("group_size", -1)),
    }
    if grouped_dictionary_lowkv:
        cached["grouped_attention"] = grouped_states
        cached["grouped_head_dims"] = head_dims
        cached["grouped_allocation_key"] = tuple(
            (
                int(group["dim"]),
                tuple(int(head) for head in group["head_indices"].tolist()),
            )
            for group in grouped_states
        )
    _CPU_LAYERS[layer_idx] = cached
    return cached


def formal_teacher_metadata(
    layer_idx: int,
    target: str,
    *,
    shape,
    source_dtype: torch.dtype,
) -> dict:
    """Build a D128 Teacher descriptor for the direct encoder's operator.

    This function performs no K/V encode.  It hard-binds the formal decoder to
    the same prepacked rank-512 operator artifact used by the Student writer.
    """
    layer_idx = int(layer_idx)
    target = str(target)
    if target not in {"k", "v"}:
        raise ValueError(f"formal Teacher target must be k or v, got {target!r}")
    cpu = _cpu_layer(layer_idx)
    if "grouped_attention" in cpu and recent_native_frames == 0:
        raise RuntimeError(
            "grouped dictionary-lowkv runtime requires a positive "
            "HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES setting"
        )
    if cpu["payload_mode"] != "dictionary_lowkv":
        raise RuntimeError(
            f"formal Teacher requires dictionary_lowkv at L{layer_idx}"
        )
    operator_path = Path(cpu["operator_path"]).resolve()
    configured_path = os.environ.get(
        "HY_KV_CODEC_PREPACKED_OPERATOR_PATH", ""
    ).strip()
    if not configured_path or Path(configured_path).resolve() != operator_path:
        raise RuntimeError(
            "formal Teacher decoder must use the exact prepacked operator "
            f"from the Student manifest: configured={configured_path!r} "
            f"student={str(operator_path)!r}"
        )
    expected_sha256 = os.environ.get(
        "HY_CODEC_NATIVE_FORMAL_OPERATOR_SHA256", ""
    ).strip().lower()
    if len(expected_sha256) != 64 or _operator_sha256(operator_path) != expected_sha256:
        raise RuntimeError(
            "formal Teacher/Student prepacked operator SHA256 mismatch: "
            f"path={operator_path} expected={expected_sha256!r}"
        )
    operator_rank = int(cpu["operator_rank"])
    required_env = {
        "HY_KV_CODEC_MODE": "global_pca",
        "HY_KV_CODEC_GLOBAL_PCA_RANK": str(operator_rank),
        "HY_KV_CODEC_COEFF_QUANT": "int4",
        "HY_KV_CODEC_OPERATOR_WEIGHT_MODE": "int8",
        f"HY_KV_CODEC_SEMANTIC_SMOOTH_{target.upper()}": "1",
    }
    for name, expected in required_env.items():
        actual = os.environ.get(name, "").strip().lower()
        if actual != expected:
            raise RuntimeError(
                f"formal Teacher/Student codec config mismatch: "
                f"{name}={actual!r}, expected={expected!r}"
            )
    if os.environ.get("HY_KV_CODEC_RESIDUAL_MLP_DIR", "").strip():
        raise RuntimeError(
            "formal Teacher single-payload decoder forbids an extra residual MLP"
        )
    group_size = int(
        os.environ.get("HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE", "0") or 0
    )
    if (
        cpu["operator_format"] != "hy_kv_codec_operator_int8_v1"
        or cpu["operator_rank"] not in {256, 512}
        or cpu["operator_group_size"] != group_size
    ):
        raise RuntimeError(
            "formal Teacher/Student operator contract mismatch: "
            f"format={cpu['operator_format']!r} rank={cpu['operator_rank']} "
            f"artifact_group={cpu['operator_group_size']} runtime_group={group_size}"
        )
    entry = cpu[f"{target}_entry"]
    for name in ("mean", "encoder", "decoder"):
        if name not in entry:
            raise RuntimeError(
                f"Student operator {target}:L{layer_idx:02d} missing {name}"
            )
    # kv_codecs validates these provenance strings against the selected
    # prepacked entry before materializing its decoder.
    return {
        "kv_codec": True,
        "codec": "global_pca",
        "coeff_quant": "int4",
        "rank": operator_rank,
        "shape": tuple(int(value) for value in shape),
        "layer_idx": layer_idx,
        "tensor_name": target,
        "basis_root": str(Path(entry["basis_root"]).resolve()),
        "source_dtype": str(source_dtype).replace("torch.", ""),
        "semantic_smooth": True,
        "semantic_clusters": 64,
        "semantic_linear_root": str(Path(entry["linear_root"]).resolve()),
        "single_encode_operator_path": str(operator_path),
        "single_encode_operator_sha256": expected_sha256,
    }


def _prejoin_dictionary_read_weights(attention: dict) -> None:
    """Pre-join equal-width Main/PRoPE reads without retaining duplicates."""
    for target in ("k", "v"):
        main_name, prope_name = f"{target}_main_read", f"{target}_prope_read"
        main, prope = attention.get(main_name), attention.get(prope_name)
        if (
            isinstance(main, torch.Tensor) and isinstance(prope, torch.Tensor)
            and main.ndim == 3 and main.shape == prope.shape
            and main.dtype == prope.dtype and main.device == prope.device
        ):
            width = int(main.shape[1])
            joined = torch.cat((main, prope), dim=1).contiguous()
            attention[f"{target}_read_joined"] = joined
            attention[main_name] = joined[:, :width]
            attention[prope_name] = joined[:, width:]


def _intern_rope_indices(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Share identical, tiny learned-RoPE index maps across layer runtimes."""
    cpu = indices.detach().to(device="cpu", dtype=torch.long).contiguous()
    key = (str(device), tuple(cpu.shape), tuple(cpu.reshape(-1).tolist()))
    resident = _DEVICE_ROPE_INDICES.get(key)
    if resident is None:
        resident = cpu.to(device=device)
        _DEVICE_ROPE_INDICES[key] = resident
    return resident


def _device_layer(layer_idx: int, device: torch.device):
    recent_native_frames = _recent_native_frames_for_layer(layer_idx)
    recent_capture_raw = recent_native_frames > 0 and _recent_capture_raw()
    refresh_interval, refresh_phase = _periodic_refresh_policy()
    refresh_steps = _periodic_refresh_steps()
    if refresh_interval == 0 and refresh_steps:
        raise ValueError(
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_STEPS requires a positive "
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_INTERVAL"
        )
    if refresh_interval > 0 and recent_native_frames == 0:
        raise RuntimeError(
            "periodic refresh requires HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES "
            "for every selected low-dimensional layer"
        )
    anchor_stride, anchor_sink_frames = (
        _formal_anchor_policy_for_layer(layer_idx)
        if recent_native_frames > 0 else (0, 0)
    )
    formal_anchor = anchor_stride > 0 or anchor_sink_frames > 0
    formal_decode_support = formal_anchor or refresh_interval > 0
    # Preserve the original cache key and device state byte-for-byte when the
    # canary is disabled.  A separate key also makes an in-process env toggle
    # incapable of reusing a runtime which lacks (or unnecessarily owns) the
    # formal decoder.
    if recent_native_frames == 0:
        key = (int(layer_idx), str(device))
    elif not formal_decode_support:
        key = (
            (int(layer_idx), str(device), "recent_native", recent_native_frames)
            if not recent_capture_raw
            else (
                int(layer_idx), str(device), "recent_native", recent_native_frames,
                "capture_raw_v1",
            )
        )
    elif refresh_interval == 0:
        # Preserve the established static-anchor key exactly when periodic
        # refresh is disabled.
        key = (
            int(layer_idx), str(device), "recent_native", recent_native_frames,
            "capture_raw_v1" if recent_capture_raw else "decode_tail_v1",
            "formal_anchor_v1", anchor_stride, anchor_sink_frames,
        )
    else:
        key = (
            int(layer_idx), str(device), "recent_native", recent_native_frames,
            "capture_raw_v1" if recent_capture_raw else "decode_tail_v1",
            "formal_anchor_refresh_v2", anchor_stride, anchor_sink_frames,
            refresh_interval, refresh_phase, refresh_steps,
        )
    cached = _DEVICE_LAYERS.get(key)
    if cached is not None:
        return cached
    cpu = _cpu_layer(layer_idx)
    # Keep the device-loader boundary self-validating even when callers or
    # tests inject a pre-parsed CPU artifact.  The grouped runtime depends on
    # the recent/native segmented path and cannot execute the legacy branch.
    if "grouped_attention" in cpu and recent_native_frames == 0:
        raise RuntimeError(
            "grouped dictionary-lowkv runtime requires a positive "
            "HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES"
        )
    if recent_native_frames > 0 and cpu["payload_mode"] != "dictionary_lowkv":
        raise RuntimeError(
            "HY_DIRECT_CODEC_RECENT_NATIVE_FRAMES is supported only by the "
            "dictionary-lowkv payload path"
        )
    attention = {}
    for name, value in cpu["attention"].items():
        if name in {"k_decoder", "v_decoder"}:
            continue
        attention[name] = (
            _intern_rope_indices(value, device) if name == "main_indices"
            else value.to(device=device, dtype=torch.bfloat16)
        )
    # The dictionary path always consumes the Main and PRoPE rank-512 reads
    # together.  Pre-join equal-width maps once at device load rather than
    # allocating/copying a cat tensor at every cache miss.  Replace the two
    # original tensors with views of the joined storage, so this has no
    # persistent weight-memory duplication and the legacy keys remain valid.
    _prejoin_dictionary_read_weights(attention)
    grouped_attention = None
    grouped_read_all = None
    if "grouped_attention" in cpu:
        grouped_attention = []
        for group in cpu["grouped_attention"]:
            group_state = {}
            for name, value in group["attention"].items():
                group_state[name] = (
                    _intern_rope_indices(value, device)
                    if name == "main_indices"
                    else value.to(device=device, dtype=torch.bfloat16)
                )
            _prejoin_dictionary_read_weights(group_state)
            grouped_attention.append({
                "dim": int(group["dim"]),
                "head_indices": group["head_indices"].to(
                    device=device, dtype=torch.long
                ),
                "attention": group_state,
            })
        if _flag("HY_DIRECT_CODEC_GROUPED_CONSOLIDATED_READ"):
            grouped_read_all = {"slices": []}
            for target in ("k", "v"):
                parts = []
                offset = 0
                for group in grouped_attention:
                    state = group["attention"]
                    heads = int(group["head_indices"].numel())
                    width = int(group["dim"])
                    joined = state[f"{target}_read_joined"].flatten(0, 1)
                    count = int(joined.shape[0])
                    parts.append(joined)
                    if target == "k":
                        grouped_read_all["slices"].append(
                            (offset, count, heads, width)
                        )
                    offset += count
                master = torch.cat(parts, 0).contiguous()
                grouped_read_all[target] = master
                # Replace every per-group copy with views of the consolidated
                # resident matrix, so dispatch fusion adds no persistent HBM.
                for group, (offset, count, heads, width) in zip(
                    grouped_attention, grouped_read_all["slices"]
                ):
                    joined = master[offset : offset + count].reshape(
                        heads, 2 * width, master.shape[-1]
                    )
                    state = group["attention"]
                    state[f"{target}_read_joined"] = joined
                    state[f"{target}_main_read"] = joined[:, :width]
                    state[f"{target}_prope_read"] = joined[:, width:]
    operators = {}
    for target in ("k", "v"):
        entry = cpu[f"{target}_entry"]
        operators[target] = {
            "mean": entry["mean"].to(device=device, dtype=torch.float32),
            "encoder": _unpack_weight(entry, "encoder").to(device=device, dtype=torch.bfloat16),
        }
        if (
            cpu["payload_mode"] == "dictionary_lowkv"
            and recent_native_frames > 0
            and (not recent_capture_raw or formal_decode_support)
        ):
            # A decoded recent tail or sparse formal anchors own the formal
            # decoder.  The default low-K/V runtime and raw-tail-only runtime
            # therefore retain their previous HBM footprint.
            operators[target]["decoder"] = _packed_weight_to_device(
                entry, "decoder", device
            )
    cached = {
        "attention": attention,
        "operators": operators,
        "block_lora": cpu["block_lora"],
        "shape": cpu["shape"],
        "payload_mode": cpu["payload_mode"],
        "compress_current": cpu["compress_current"],
        "main_k_mode": cpu["main_k_mode"],
    }
    if grouped_attention is not None:
        cached["grouped_attention"] = grouped_attention
        if grouped_read_all is not None:
            cached["grouped_read_all"] = grouped_read_all
        cached["grouped_head_dims"] = cpu["grouped_head_dims"]
        # Built exclusively from CPU metadata.  The attention hot path must
        # never call .tolist() on resident CUDA head-index tensors merely to
        # construct its positioned-cache key.
        cached["grouped_allocation_key"] = cpu["grouped_allocation_key"]
    if recent_native_frames > 0:
        cached["recent_native_frames"] = recent_native_frames
        if recent_capture_raw:
            cached["recent_capture_raw"] = True
        if formal_anchor:
            cached["formal_anchor_stride"] = anchor_stride
            cached["formal_anchor_identity_sink_frames"] = anchor_sink_frames
        if refresh_interval > 0:
            cached["periodic_refresh_interval"] = refresh_interval
            cached["periodic_refresh_phase"] = refresh_phase
            if refresh_steps:
                cached["periodic_refresh_steps"] = refresh_steps
    _DEVICE_LAYERS[key] = cached
    return cached


@torch.no_grad()
def maybe_merge_block_lora(block, layer_idx: int) -> None:
    if not enabled_for_layer(layer_idx):
        return
    if not _flag("HY_DIRECT_CODEC_LATENT_BLOCK_LORA", "1"):
        return
    key = (id(block), int(layer_idx))
    if key in _MERGED_BLOCKS:
        return
    state = _cpu_layer(layer_idx)["block_lora"]
    if state is None:
        return
    targets = (
        (block.img_attn_proj.weight, "main"),
        (block.img_attn_prope_proj.weight, "prope"),
        (block.img_mlp.fc1.weight, "fc1"),
        (block.img_mlp.fc2.weight, "fc2"),
    )
    for weight, name in targets:
        delta = state[f"{name}.b"].float() @ state[f"{name}.a"].float()
        weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
    _MERGED_BLOCKS.add(key)
    rank = int(state["main.a"].shape[0])
    print(f"[DirectCodecLatent] merged rank{rank} block adapter at layer {layer_idx}", flush=True)


def _quartet_reduce_runtime(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    native = x.float().reshape(*x.shape[:-1], 32, 4)
    return torch.einsum("hro,bhnoc->bhnrc", mix.float(), native).flatten(-2)


def _quartet_lift_runtime(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    low = x.float().reshape(*x.shape[:-1], x.shape[-1] // 4, 4)
    return torch.einsum("hro,bhnrc->bhnoc", mix.float(), low).flatten(-2)


def _assigned_low(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(table, 2, ids[:, None, :, None].expand(-1, table.shape[1], -1, table.shape[-1]))


def _center_pair_add_fused_impl(
    first: torch.Tensor,
    second: torch.Tensor,
    first_table: torch.Tensor,
    second_table: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_index = ids[:, None, :, None].expand(
        -1, first_table.shape[1], -1, first_table.shape[-1]
    )
    second_index = ids[:, None, :, None].expand(
        -1, second_table.shape[1], -1, second_table.shape[-1]
    )
    return (
        first + torch.gather(first_table, 2, first_index),
        second + torch.gather(second_table, 2, second_index),
    )


def _center_pair_add_runtime(
    first: torch.Tensor,
    second: torch.Tensor,
    first_table: torch.Tensor,
    second_table: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _flag("HY_DIRECT_CODEC_TRITON_POSITION") and first.is_cuda:
        from .codec_native_position_triton import center_pair_add
        return center_pair_add(
            first, second, first_table, second_table, ids
        )
    global _COMPILED_CENTER_PAIR_ADD
    if _COMPILED_CENTER_PAIR_ADD is None:
        _COMPILED_CENTER_PAIR_ADD = torch.compile(
            _center_pair_add_fused_impl, dynamic=True
        )
    return _COMPILED_CENTER_PAIR_ADD(
        first, second, first_table, second_table, ids
    )


def _tensor_cache_key(tensor: torch.Tensor):
    """Cheap identity/version key; an accompanying weakref validates reuse."""
    return (
        id(tensor), int(tensor._version), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, str(tensor.device),
    )


def _cache_lookup(cache: OrderedDict, key, sources):
    entry = cache.get(key)
    if entry is None:
        return None
    refs, versions, value = entry
    valid = len(refs) == len(sources) and all(
        ref() is source and version == int(source._version)
        for ref, version, source in zip(refs, versions, sources)
    )
    if not valid:
        # In particular, this rejects an id/address that Python has recycled
        # after the original Tensor died.
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _cache_value_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_cache_value_nbytes(item) for item in value)
    return 0


def _cache_prune_invalid(cache: OrderedDict) -> None:
    # Remove entries whose identity guards have died before accounting toward
    # the memory budget.  This is common when a denoising forward created a new
    # current-view tensor for the next step.
    for old_key, (refs, versions, _) in list(cache.items()):
        if any(
            ref() is None or int(ref()._version) != version
            for ref, version in zip(refs, versions)
        ):
            cache.pop(old_key, None)


def _cache_insert(cache: OrderedDict, key, sources, value, limit: int,
                  byte_limit: int | None = None):
    _cache_prune_invalid(cache)
    cache[key] = (
        tuple(weakref.ref(source) for source in sources),
        tuple(int(source._version) for source in sources),
        value,
    )
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    if byte_limit is not None:
        total = sum(_cache_value_nbytes(entry[2]) for entry in cache.values())
        while cache and total > byte_limit:
            _, removed = cache.popitem(last=False)
            total -= _cache_value_nbytes(removed[2])
    return value


def _cache_try_admit(cache: OrderedDict, key, sources, value, limit: int,
                     byte_limit: int):
    """Admit without evicting live hot entries; avoids cyclic layer thrash."""
    _cache_prune_invalid(cache)
    if key not in cache and (
        len(cache) >= limit
        or sum(_cache_value_nbytes(entry[2]) for entry in cache.values())
            + _cache_value_nbytes(value) > byte_limit
    ):
        return value
    return _cache_insert(cache, key, sources, value, limit, byte_limit)


def _rope_device_pair_runtime(
    cos: torch.Tensor,
    sin: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the small base RoPE tables copied from CPU metadata.

    This cache contains only the native positional tables shared by all
    layers, not per-head selected tables and never K/V.  It prevents each
    strict denoising read from repeating the same host-to-device transfer.
    """
    if cos.device == device and sin.device == device:
        return cos, sin
    sources = (cos, sin)
    key = (str(device), *(_tensor_cache_key(item) for item in sources))
    cached = _cache_lookup(_ROPE_DEVICE_CACHE, key, sources)
    if cached is not None:
        return cached
    value = (cos.to(device=device), sin.to(device=device))
    return _cache_try_admit(
        _ROPE_DEVICE_CACHE, key, sources, value,
        _ROPE_DEVICE_CACHE_MAX, _ROPE_DEVICE_CACHE_MAX_BYTES,
    )


def _selected_rope_gather(freqs_cis, indices: torch.Tensor, device: torch.device):
    """Gather learned per-head RoPE channels once per immutable frequency set."""
    freq_tensors = freqs_cis if isinstance(freqs_cis, tuple) else (freqs_cis,)
    sources = tuple(freq_tensors) + (indices,)
    cacheable = not (torch.is_grad_enabled() and any(item.requires_grad for item in sources))
    key = (
        "tuple" if isinstance(freqs_cis, tuple) else "complex", str(device),
        *(_tensor_cache_key(item) for item in sources),
    )
    if cacheable:
        cached = _cache_lookup(_SELECTED_ROPE_CACHE, key, sources)
        if cached is not None:
            return cached
        # A selected [H,S,D] cos/sin table can be tens or hundreds of MiB.
        # Admit it only after the exact immutable source set repeats; one-shot
        # history/chunk gathers therefore consume no persistent HBM.
        repeated = _cache_lookup(_SELECTED_ROPE_SEEN, key, sources) is not None
        if not repeated:
            _cache_insert(
                _SELECTED_ROPE_SEEN, key, sources, True,
                _SELECTED_ROPE_SEEN_MAX,
            )
    else:
        repeated = False

    selected_indices = indices.to(device=device)
    selected_heads = int(selected_indices.shape[0])
    if isinstance(freqs_cis, tuple):
        cos, sin = (item.to(device=device) for item in freqs_cis)
        per_head = selected_indices[:, None, :].expand(-1, cos.shape[0], -1)
        value = (
            torch.gather(cos[None].expand(selected_heads, -1, -1), -1, per_head),
            torch.gather(sin[None].expand(selected_heads, -1, -1), -1, per_head),
        )
    else:
        freqs = freqs_cis.to(device=device)
        pair_indices = (selected_indices[:, ::2] // 2)[:, None].expand(-1, freqs.shape[0], -1)
        value = torch.gather(
            freqs[None].expand(selected_heads, -1, -1), -1, pair_indices
        )
    if cacheable and repeated:
        return _cache_try_admit(
            _SELECTED_ROPE_CACHE, key, sources, value, _SELECTED_ROPE_CACHE_MAX,
            _SELECTED_ROPE_CACHE_MAX_BYTES,
        )
    return value


def _selected_rope_fused_impl(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Fusible tuple-RoPE path without an expanded persistent frequency table."""
    selected_heads = indices.shape[0]
    per_head = indices[:, None, :].expand(-1, cos.shape[0], -1)
    selected_cos = torch.gather(
        cos[None].expand(selected_heads, -1, -1), -1, per_head
    )
    selected_sin = torch.gather(
        sin[None].expand(selected_heads, -1, -1), -1, per_head
    )
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return (
        x.float() * selected_cos[None] + rotated * selected_sin[None]
    ).to(x.dtype)


def _selected_rope_runtime(x: torch.Tensor, freqs_cis, indices: torch.Tensor) -> torch.Tensor:
    """Apply native 3D-RoPE frequencies to per-head complete-pair selections."""
    if isinstance(freqs_cis, tuple):
        cos, sin = _rope_device_pair_runtime(
            freqs_cis[0], freqs_cis[1], x.device
        )
        if (
            _triton_data_flag("HY_DIRECT_CODEC_TRITON_SELECTED_ROPE_ONLY")
            and x.is_cuda
        ):
            from .codec_native_position_triton import selected_rope
            return selected_rope(x, cos, sin, indices)
        if _flag("HY_DIRECT_CODEC_STREAM_FUSED_ROPE") and x.is_cuda:
            # Strict streaming rebuilds history on every denoise read.  Let
            # Inductor fuse per-head frequency gather, rotation, and cast so
            # the expanded [H,S,D] cos/sin tables never materialize.  The
            # compiled callable owns code only; no K/V or frequency output is
            # retained between calls.  Dynamic S avoids one graph per chunk.
            global _COMPILED_SELECTED_ROPE
            if _COMPILED_SELECTED_ROPE is None:
                _COMPILED_SELECTED_ROPE = torch.compile(
                    _selected_rope_fused_impl, dynamic=True
                )
            return _COMPILED_SELECTED_ROPE(
                x, cos, sin,
                indices.to(device=x.device),
            )
        cached = _selected_rope_gather((cos, sin), indices, x.device)
        cos, sin = cached
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
        return (x.float() * cos[None] + rotated * sin[None]).to(x.dtype)
    selected = _selected_rope_gather(freqs_cis, indices, x.device)
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * selected[None]).flatten(-2).to(x.dtype)


@torch.no_grad()
def _compress_one(x: torch.Tensor, layer_idx: int, target: str, runtime: dict):
    residual, centers, ids = _joint_semantic_smooth(x, layer_idx)
    batch, heads, sequence, dim = residual.shape
    op = runtime["operators"][target]
    rows = residual.permute(0, 2, 1, 3).reshape(-1, heads * dim)
    coeff = F.linear((rows.float() - op["mean"]).to(torch.bfloat16), op["encoder"])
    if runtime["payload_mode"] == "dictionary_lowkv":
        organized = coeff.float().reshape(batch, sequence, coeff.shape[-1])
        half_decoder = None
        organizer = None
    elif runtime["payload_mode"] == "half_decoder":
        # Preserve the exact formal 512-D codec cache.  The rectangular
        # 512->(16*D) half decoder runs only when attention reads the cache.
        organized = coeff.float().reshape(batch, sequence, coeff.shape[-1])
        half_decoder = runtime["attention"][f"{target}_half_decoder"]
    else:
        organizer = runtime["attention"][f"{target}_organizer"]
        organized = F.linear(coeff, organizer).float().reshape(batch, sequence, 512)
        half_decoder = None
    scale = organized.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
    codes = (organized / scale).round().clamp(-7, 7).to(torch.int8)
    if runtime["payload_mode"] == "dictionary_lowkv":
        native = (centers.float() + op["mean"][None, None]).reshape(batch, -1, 16, 128).permute(0, 2, 1, 3)
        centre_groups = None
        if runtime.get("grouped_attention") is not None:
            centre_groups = []
            for group in runtime["grouped_attention"]:
                head_indices = group["head_indices"]
                group_state = group["attention"]
                group_native = native.index_select(1, head_indices)
                if target == "k":
                    group_main = torch.einsum(
                        "bgnr,gdr->bgnd",
                        group_native,
                        group_state["k_main_anchor_reduce"],
                    )
                    group_prope = _quartet_reduce_runtime(
                        group_native,
                        group_state.get(
                            "k_prope_anchor_mix", group_state["q_prope_mix"]
                        ),
                    )
                else:
                    group_main = torch.einsum(
                        "bgco,gdo->bgcd",
                        group_native,
                        group_state["v_main_anchor_reduce"],
                    )
                    group_prope = _quartet_reduce_runtime(
                        group_native, group_state["v_prope_anchor_mix"]
                    )
                centre_groups.append({
                    "dim": int(group["dim"]),
                    "main": group_main.to(torch.bfloat16),
                    "prope": group_prope.to(torch.bfloat16),
                })
            centre_main = centre_prope = None
        elif target == "k":
            idx = runtime["attention"]["main_indices"]
            if runtime["main_k_mode"] == "learned_rope":
                centre_main = torch.einsum(
                    "bhnr,hdr->bhnd", native, runtime["attention"]["k_main_anchor_reduce"]
                )
            else:
                centre_main = torch.gather(native, -1, idx[None, :, None].expand(batch, -1, native.shape[2], -1))
            centre_prope = _quartet_reduce_runtime(
                native,
                runtime["attention"].get(
                    "k_prope_anchor_mix", runtime["attention"]["q_prope_mix"]
                ),
            )
        else:
            centre_main = torch.einsum("bhco,hdo->bhcd", native, runtime["attention"]["v_main_anchor_reduce"])
            centre_prope = _quartet_reduce_runtime(native, runtime["attention"]["v_prope_anchor_mix"])
        centre_latent = None
    else:
        centre_code = F.linear((centers.float() + op["mean"][None, None]).to(torch.bfloat16), op["encoder"])
        centre_latent = F.linear(centre_code, half_decoder if half_decoder is not None else organizer).to(torch.bfloat16)
    k_heads, k_dim, v_heads, v_dim = runtime["shape"]
    kv_heads, head_dim = (k_heads, k_dim) if target == "k" else (v_heads, v_dim)
    payload = {
        "direct_codec_latent": True,
        "layer_idx": int(layer_idx),
        "tensor_name": target,
        "shape": tuple(int(value) for value in x.shape),
        "latent_int4": pack_quantized_codes(codes, 4),
        "latent_shape": tuple(int(value) for value in codes.shape),
        "latent_numel": int(codes.numel()),
        "latent_scale": (
            scale.to(_scale_storage_dtype())
            if runtime["payload_mode"] == "dictionary_lowkv"
            and runtime.get("recent_native_frames", 0) > 0
            else scale.to(torch.float32)
        ),
        "centre_latent": centre_latent,
        "centre_main": (
            None
            if runtime["payload_mode"] != "dictionary_lowkv"
            or runtime.get("grouped_attention") is not None
            else centre_main.to(torch.bfloat16)
        ),
        "centre_prope": (
            None
            if runtime["payload_mode"] != "dictionary_lowkv"
            or runtime.get("grouped_attention") is not None
            else centre_prope.to(torch.bfloat16)
        ),
        "semantic_ids": ids.to(torch.uint8),
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "payload_mode": runtime["payload_mode"],
    }
    if (
        runtime["payload_mode"] == "dictionary_lowkv"
        and runtime.get("grouped_attention") is not None
    ):
        payload["centre_groups"] = centre_groups
    if (
        runtime["payload_mode"] == "dictionary_lowkv"
        and runtime.get("recent_native_frames", 0) > 0
        and (
            not runtime.get("recent_capture_raw", False)
            or runtime.get("formal_anchor_stride", 0) > 0
            or runtime.get("formal_anchor_identity_sink_frames", 0) > 0
            or runtime.get("periodic_refresh_interval", 0) > 0
            or _formal_teacher_needs_centers(layer_idx)
            or (
                codec_native_history_adapter.enabled()
                and "center" in codec_native_history_adapter.selected_targets()
            )
        )
    ):
        # This is the same stored BF16 C64 table used by the formal codec.  It
        # is cache-level metadata (64x2048), not a per-token D128 payload.  A
        # raw recent tail normally does not need it, but formal old-history
        # anchors do.
        payload["semantic_centers"] = centers
    if _formal_teacher_needs_centers(layer_idx):
        payload["_formal_source_identity"] = _formal_source_identity(x)
    return payload


def _recent_raw_tail_key(layer_idx: int, shape, recent_frames: int,
                         old_tokens: int, recent_tokens: int):
    return (
        "recent_raw_pre_v1", int(layer_idx), tuple(int(value) for value in shape),
        int(recent_frames), int(old_tokens), int(recent_tokens),
    )


@torch.no_grad()
def _capture_recent_native_tail(layer_cache: dict, layer_idx: int,
                                raw_k: torch.Tensor, raw_v: torch.Tensor,
                                runtime: dict) -> None:
    """Keep only the recent unpositioned BF16 K/V suffix for zero re-decode.

    The complete history is still encoded into the normal rank-512 INT4
    payload.  This sidecar is transient: once its positioned D128 tensors have
    been materialized into the read cache it can be released.
    """
    if not runtime.get("recent_capture_raw", False):
        return
    if raw_k.shape != raw_v.shape or raw_k.ndim != 4:
        raise RuntimeError(
            f"recent raw K/V must share [B,H,S,D], got "
            f"K={tuple(raw_k.shape)} V={tuple(raw_v.shape)}"
        )
    if raw_k.shape[1] != 16 or raw_k.shape[-1] != 128:
        raise RuntimeError(
            f"recent raw capture requires native H16D128, got {tuple(raw_k.shape)}"
        )
    recent_frames = int(runtime["recent_native_frames"])
    probe = {"shape": tuple(int(value) for value in raw_k.shape)}
    old_tokens, recent_tokens = _recent_history_partition(
        layer_cache, probe, recent_frames
    )
    layer_cache["_direct_codec_recent_native_pre"] = {
        "key": _recent_raw_tail_key(
            layer_idx, raw_k.shape, recent_frames, old_tokens, recent_tokens
        ),
        "k": raw_k[:, :, old_tokens:].detach().to(torch.bfloat16).contiguous(),
        "v": raw_v[:, :, old_tokens:].detach().to(torch.bfloat16).contiguous(),
    }


def _valid_recent_raw_tail(value, key, device: torch.device,
                           batch: int, recent_tokens: int) -> bool:
    if not isinstance(value, dict) or value.get("key") != key:
        return False
    expected = (int(batch), 16, int(recent_tokens), 128)
    return all(
        isinstance(value.get(name), torch.Tensor)
        and value[name].device == device
        and value[name].dtype == torch.bfloat16
        and tuple(value[name].shape) == expected
        for name in ("k", "v")
    )


@torch.no_grad()
def compress_layer_cache(layer_cache: dict, layer_idx: int) -> dict:
    if not enabled_for_layer(layer_idx):
        return layer_cache
    first = layer_cache.get("k_vision")
    if not isinstance(first, torch.Tensor):
        raise RuntimeError("direct codec-latent compression must run before the formal codec")
    runtime = _device_layer(layer_idx, first.device)
    second = layer_cache.get("v_vision")
    if not isinstance(second, torch.Tensor):
        raise RuntimeError("direct codec-latent compression requires raw V history")
    _bind_periodic_refresh_cache_write(layer_cache, runtime)
    # A newly written history invalidates both transient read representations.
    layer_cache.pop("_direct_codec_positioned_history", None)
    layer_cache.pop("_direct_codec_recent_native_pre", None)
    _capture_recent_native_tail(layer_cache, layer_idx, first, second, runtime)
    layer_cache["k_vision"] = _compress_one(first, layer_idx, "k", runtime)
    layer_cache["v_vision"] = _compress_one(second, layer_idx, "v", runtime)
    layer_cache["direct_codec_latent"] = True
    return layer_cache


def _decode_state(payload: dict, dtype: torch.dtype, device: torch.device, runtime: dict, target: str):
    codes = unpack_quantized_codes(
        payload["latent_int4"].to(device), 4,
        int(payload["latent_numel"]), tuple(payload["latent_shape"]),
    ).to(torch.float32)
    state = codes * payload["latent_scale"].to(device=device, dtype=torch.float32)
    if payload.get("payload_mode") == "half_decoder":
        state = F.linear(state.to(torch.bfloat16), runtime["attention"][f"{target}_half_decoder"]).float()
    centres = payload["centre_latent"].to(device=device, dtype=torch.float32)
    ids = payload["semantic_ids"].to(device=device, dtype=torch.long)
    assigned = torch.gather(centres, 1, ids[..., None].expand(-1, -1, centres.shape[-1]))
    kv_heads = int(payload["kv_heads"])
    head_dim = int(payload["head_dim"])
    if state.shape[-1] != kv_heads * head_dim or 16 % kv_heads:
        raise RuntimeError(f"invalid direct codec payload shape K/V{kv_heads}x{head_dim}")
    state = (state + assigned).reshape(state.shape[0], state.shape[1], kv_heads, head_dim)
    state = state.permute(0, 2, 1, 3).repeat_interleave(16 // kv_heads, dim=1)
    return state.to(dtype)


def _rope_indices(dim: int, device: torch.device):
    if dim <= 0 or dim > 128 or dim % 4:
        raise ValueError(f"unsupported reduced RoPE dimension {dim}")
    temporal = int(round((dim * 16 / 128) / 2) * 2)
    temporal = max(2, min(16, temporal))
    remaining = dim - temporal
    height = (remaining // 4) * 2
    targets = (temporal, height, remaining - height)
    channels = []
    for (start, end), target in zip(((0, 16), (16, 72), (72, 128)), targets):
        pair_ids = torch.linspace(0, (end - start) // 2 - 1, steps=target // 2, device=device).round().long()
        channels.append(torch.stack((2 * pair_ids, 2 * pair_ids + 1), -1).reshape(-1) + start)
    return torch.cat(channels)


def _reduced_rope(x: torch.Tensor, freqs_cis):
    indices = _rope_indices(x.shape[-1], x.device)
    reduced = tuple(item.to(x.device).index_select(-1, indices) for item in freqs_cis)
    return apply_rotary_emb_one(x.transpose(1, 2), reduced, head_first=False).transpose(1, 2)


def _prope_fns(head_dim: int, viewmats, Ks, device):
    sources = (viewmats,) if Ks is None else (viewmats, Ks)
    cacheable = not (torch.is_grad_enabled() and any(item.requires_grad for item in sources))
    key = (
        int(head_dim), str(device), Ks is None,
        *(_tensor_cache_key(item) for item in sources),
    )
    if cacheable:
        cached = _cache_lookup(_PROPE_FN_CACHE, key, sources)
        if cached is not None:
            return cached
    # Snapshot sources once.  In the Ks=None implementation the returned
    # closures otherwise capture `viewmats` itself, so a later in-place update
    # could silently mutate an already cached transform before its version is
    # checked again.
    prepared_viewmats = viewmats.to(device=device, dtype=torch.float32).clone()
    prepared_Ks = (
        Ks.to(device=device, dtype=torch.float32).clone() if Ks is not None else None
    )
    value = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=prepared_viewmats,
        Ks=prepared_Ks,
        patches_x=None, patches_y=None, image_width=None, image_height=None,
    )
    if cacheable:
        return _cache_insert(_PROPE_FN_CACHE, key, sources, value, _PROPE_FN_CACHE_MAX)
    return value


def _prope_matrix_runtime(
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    device: torch.device,
    role: str = "kv",
) -> torch.Tensor:
    """Build/cache the tokenwise 4x4 KV projection, never positioned K/V.

    Camera metadata is shared by every transformer layer and remains immutable
    during the denoising reads of one cache generation. Recomputing its SE(3)
    inverse in every selected layer was therefore pure duplicated work. The
    bounded weak-reference cache owns at most 64 MiB and is invalidated by
    tensor identity/version changes.
    """
    if role not in {"q", "kv"}:
        raise ValueError(f"invalid PRoPE role: {role}")
    # Explicit historical A/B replay only. Both variants retain the old matrix
    # precision; the sole numerical difference is Q's P_inv versus P_T.
    replay = os.environ.get("HY_DIRECT_CODEC_PROPE_HISTORICAL_REPLAY", "")
    if replay not in {"", "old", "qfix"}:
        raise ValueError(f"invalid historical PRoPE replay: {replay}")
    if replay == "old":
        role = "kv"
    sources = (viewmats,) if Ks is None else (viewmats, Ks)
    cacheable = not (
        torch.is_grad_enabled() and any(item.requires_grad for item in sources)
    )
    key = (
        str(device), Ks is None, role, replay, torch.is_autocast_enabled(),
        *(_tensor_cache_key(item) for item in sources),
    )
    if cacheable:
        cached = _cache_lookup(_PROPE_MATRIX_CACHE, key, sources)
        if cached is not None:
            return cached
    prepared_viewmats = viewmats.to(device=device, dtype=torch.float32)
    prepared_Ks = (
        None if Ks is None else Ks.to(device=device, dtype=torch.float32)
    )
    _, query_matrix, kv_matrix = _prepare_projection_matrices(prepared_viewmats, prepared_Ks)
    matrix = query_matrix if role == "q" else kv_matrix
    # Canonical einsum autocasts BOTH operands before its FP32 accumulation.
    # Match its matrix operand rather than silently using extra FP32 bits.
    if not replay and torch.is_autocast_enabled():
        matrix = matrix.to(torch.get_autocast_gpu_dtype()).float()
    if cacheable:
        return _cache_try_admit(
            _PROPE_MATRIX_CACHE, key, sources, matrix,
            _PROPE_MATRIX_CACHE_MAX, _PROPE_MATRIX_CACHE_MAX_BYTES,
        )
    return matrix


def _prope_kv_pair_fused_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one per-token 4x4 PRoPE matrix without small-N FP32 GEMMs."""
    batch, heads, sequence, width = key.shape

    def apply_one(x: torch.Tensor) -> torch.Tensor:
        q = x.float().reshape(batch, heads, sequence, width // 4, 4)
        # Spell out the fixed 4x4 product.  Inductor lowers each result to one
        # fused pointwise/reduction kernel instead of cuBLAS small-N GEMV.
        rows = []
        for i in range(4):
            rows.append(
                matrix[:, None, :, None, i, 0] * q[..., 0]
                + matrix[:, None, :, None, i, 1] * q[..., 1]
                + matrix[:, None, :, None, i, 2] * q[..., 2]
                + matrix[:, None, :, None, i, 3] * q[..., 3]
            )
        return torch.stack(rows, -1).reshape_as(x).to(x.dtype)

    return apply_one(key), apply_one(value)


def _prope_kv_pair_runtime(
    key: torch.Tensor,
    value: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    device: torch.device,
    matrix: torch.Tensor | None = None,
    triton_scope: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict-streaming PRoPE pair path; retains code but no transformed K/V."""
    global _COMPILED_PROPE_KV_PAIR
    if matrix is None:
        matrix = _prope_matrix_runtime(viewmats, Ks, device)
    use_triton = (
        _flag("HY_DIRECT_CODEC_TRITON_POSITION")
        if triton_scope is None
        else _scoped_flag(triton_scope, "HY_DIRECT_CODEC_TRITON_POSITION")
    )
    if use_triton and key.is_cuda:
        from .codec_native_position_triton import prope_pair
        return prope_pair(key, value, matrix)
    if _COMPILED_PROPE_KV_PAIR is None:
        _COMPILED_PROPE_KV_PAIR = torch.compile(
            _prope_kv_pair_fused_impl, dynamic=True
        )
    return _COMPILED_PROPE_KV_PAIR(key, value, matrix)


def _prope_one_fused_impl(
    x: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply one per-token 4x4 PRoPE matrix without small-N GEMMs."""
    batch, heads, sequence, width = x.shape
    q = x.float().reshape(batch, heads, sequence, width // 4, 4)
    rows = []
    for i in range(4):
        rows.append(
            matrix[:, None, :, None, i, 0] * q[..., 0]
            + matrix[:, None, :, None, i, 1] * q[..., 1]
            + matrix[:, None, :, None, i, 2] * q[..., 2]
            + matrix[:, None, :, None, i, 3] * q[..., 3]
        )
    return torch.stack(rows, -1).reshape_as(x).to(x.dtype)


def _prope_one_runtime(
    x: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    device: torch.device,
    matrix: torch.Tensor | None = None,
    triton_scope: str | None = None,
) -> torch.Tensor:
    """Strict-streaming one-tensor PRoPE path with no retained transform."""
    global _COMPILED_PROPE_ONE
    if matrix is None:
        matrix = _prope_matrix_runtime(viewmats, Ks, device)
    use_triton = (
        _flag("HY_DIRECT_CODEC_TRITON_POSITION")
        if triton_scope is None
        else _scoped_flag(triton_scope, "HY_DIRECT_CODEC_TRITON_POSITION")
    )
    if use_triton and x.is_cuda:
        from .codec_native_position_triton import prope_one
        return prope_one(x, matrix)
    if _COMPILED_PROPE_ONE is None:
        _COMPILED_PROPE_ONE = torch.compile(
            _prope_one_fused_impl, dynamic=True
        )
    return _COMPILED_PROPE_ONE(x, matrix)


def _flash_lse(q, k, v, scale=None):
    backend = _effective_flash_backend()
    use_flashinfer = backend == "flashinfer" or (
        backend in {
            "auto_flashinfer", "auto_flashinfer_low64",
            "auto_package_low64_flashinfer",
        }
        and int(q.shape[-1]) == 128
        and int(k.shape[-2]) <= 20000
    ) or (
        # Experimental policy: retain the established short-D128 selection
        # and additionally use FlashInfer for uniform D64 history.  Keep this
        # behind a distinct backend name because FlashInfer is a few percent
        # faster at the deployed shape but differs from ATen by ~3e-3 rel-L2.
        backend == "auto_flashinfer_low64"
        and int(q.shape[-1]) == 64
    )
    if use_flashinfer:
        # FlashInfer's single-request API has no explicit batch dimension.
        # Folding B into H is exact because heads are independent.  HND K/V is
        # a zero-copy reshape of our contiguous [B,H,N,D] tensors; only the
        # much shorter query needs an NHD reorder.  FlashInfer returns base-2
        # LSE, whereas ATen/flash-attn return natural-log LSE.
        import flashinfer

        batch, heads, query, width = q.shape
        keys = k.shape[-2]
        q_nhd = q.permute(2, 0, 1, 3).reshape(
            query, batch * heads, width
        ).contiguous()
        k_hnd = k.contiguous().reshape(batch * heads, keys, width)
        v_hnd = v.contiguous().reshape(batch * heads, keys, v.shape[-1])
        output, lse = flashinfer.single_prefill_with_kv_cache(
            q_nhd, k_hnd, v_hnd,
            causal=False,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=False,
            sm_scale=scale,
            return_lse=True,
            backend="fa2",
        )
        output = output.reshape(
            query, batch, heads, v.shape[-1]
        ).permute(1, 2, 0, 3)
        lse = lse.reshape(query, batch, heads).permute(1, 2, 0)
        return output, lse * math.log(2.0)
    use_package = backend == "package" or (
        # Isolated SM80 long-context build: its D32 path retains the stock
        # kernel while D64 uses the selected low-dimension tile.  Keep this
        # policy experimental so the installed FlashAttention and the formal
        # ``auto`` backend remain untouched.
        backend in {"auto_package_low64", "auto_package_low64_flashinfer"}
        and int(q.shape[-1]) in {32, 64, 96}
    ) or (
        backend in {"auto", "auto_flashinfer", "auto_flashinfer_low64"}
        # Exclusive A100 screens show package FA is materially faster for
        # D32/D96, but about 3% slower for the short native D128 branch.  Keep
        # D128 on ATen under ``auto``; explicit ``package`` remains available.
        and int(q.shape[-1]) in {32, 96}
    )
    if use_package:
        from flash_attn import flash_attn_func

        output, lse, _ = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            return_attn_probs=True,
        )
        return output.transpose(1, 2), lse
    if backend not in {
        "aten", "auto", "auto_flashinfer", "auto_flashinfer_low64",
        "auto_package_low64", "auto_package_low64_flashinfer",
    }:
        raise ValueError(
            "HY_DIRECT_CODEC_FLASH_BACKEND must be aten, auto, "
            "auto_flashinfer, auto_flashinfer_low64, auto_package_low64, "
            "auto_package_low64_flashinfer, "
            "package, or flashinfer"
        )
    if (
        _flag("HY_DIRECT_CODEC_ATEN_FOLD_BATCH_HEAD")
        and q.shape[0] == k.shape[0] == v.shape[0] == 2
    ):
        # Main and PRoPE branches are independent attention batches.  Folding
        # B=2 into H changes only launch geometry; every head keeps the exact
        # same Q/K/V rows and ATen produces bitwise-identical output and LSE.
        heads = int(q.shape[1])
        query = int(q.shape[2])
        keys = int(k.shape[2])
        width = int(q.shape[3])
        value_width = int(v.shape[3])
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q.reshape(1, 2 * heads, query, width).contiguous(),
            k.reshape(1, 2 * heads, keys, width).contiguous(),
            v.reshape(1, 2 * heads, keys, value_width).contiguous(),
            0.0, False, False, scale=scale,
        )
        return (
            result[0].reshape(2, heads, query, value_width),
            result[1].reshape(2, heads, query),
        )
    result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q.contiguous(), k.contiguous(), v.contiguous(), 0.0, False, False,
        scale=scale,
    )
    return result[0], result[1]


def _combine(native, native_lse, history, history_lse):
    if _flag("HY_DIRECT_CODEC_STREAM_FUSED_COMBINE") and native.is_cuda:
        if _flag("HY_DIRECT_CODEC_TRITON_POSITION"):
            from .codec_native_position_triton import lse_combine
            return lse_combine(native, native_lse, history, history_lse)
        global _COMPILED_LSE_COMBINE
        if _COMPILED_LSE_COMBINE is None:
            _COMPILED_LSE_COMBINE = torch.compile(
                _combine_fused_impl, dynamic=True
            )
        return _COMPILED_LSE_COMBINE(
            native, native_lse, history, history_lse
        )
    total = torch.logaddexp(native_lse, history_lse)
    native_weight = (native_lse - total).exp().unsqueeze(-1)
    history_weight = (history_lse - total).exp().unsqueeze(-1)
    return native.float() * native_weight + history.float() * history_weight


def _combine_fused_impl(native, native_lse, history, history_lse):
    """Exact LSE merge expression lowered to one output-dtype kernel."""
    total = torch.logaddexp(native_lse, history_lse)
    native_weight = (native_lse - total).exp().unsqueeze(-1)
    history_weight = (history_lse - total).exp().unsqueeze(-1)
    return (
        native.float() * native_weight + history.float() * history_weight
    ).to(native.dtype)


def _rank512(payload: dict, device: torch.device) -> torch.Tensor:
    if (
        _flag("HY_DIRECT_CODEC_TRITON_FP32_DEQUANT")
        and torch.device(device).type == "cuda"
    ):
        from .kv_codec_triton import int4_unpack_dequant_per_row

        # Preserve the strict FP32 latent boundary while removing the chain
        # of nibble extraction, stacking, conversion and scale kernels.
        return int4_unpack_dequant_per_row(
            payload["latent_int4"].to(device),
            payload["latent_scale"].to(device=device, dtype=torch.float32),
            tuple(payload["latent_shape"]), torch.float32,
        )
    if (
        _flag("HY_DIRECT_CODEC_STREAM_BF16_DEQUANT")
        and _flag("HY_DIRECT_CODEC_TRITON_RANK512", "1")
        and torch.device(device).type == "cuda"
    ):
        # Reuse the formal codec's byte-compatible INT4 kernel.  It combines
        # nibble unpack and token-scale multiplication, eliminating the full
        # temporary INT8 codes tensor on every strict-streaming read.
        from .kv_codec_triton import int4_unpack_dequant_per_row

        return int4_unpack_dequant_per_row(
            payload["latent_int4"].to(device),
            payload["latent_scale"].to(device=device, dtype=torch.bfloat16),
            tuple(payload["latent_shape"]),
            torch.bfloat16,
        )
    codes = unpack_quantized_codes(
        payload["latent_int4"].to(device), 4,
        int(payload["latent_numel"]), tuple(payload["latent_shape"]),
    )
    if _flag("HY_DIRECT_CODEC_STREAM_BF16_DEQUANT"):
        # Every deployed consumer immediately casts rank512 to its resident
        # BF16 read/decoder weight.  Multiplying the exact INT4 code by the
        # stored BF16 token scale directly in BF16 is bit-identical to the old
        # FP32-product-then-BF16 path, while halving this temporary tensor.
        return codes.to(torch.bfloat16) * payload["latent_scale"].to(
            device=device, dtype=torch.bfloat16
        )
    return codes.float() * payload["latent_scale"].to(
        device=device, dtype=torch.float32
    )


def _recent_history_partition(layer_cache: dict, payload: dict,
                              recent_native_frames: int) -> tuple[int, int]:
    """Return ``(old_tokens, recent_tokens)`` for a time-major vision cache."""
    thw = layer_cache.get("vision_thw")
    if not isinstance(thw, (tuple, list)) or len(thw) != 3:
        raise RuntimeError("recent-native history requires vision_thw=(T,H,W)")
    temporal, height, width = (int(value) for value in thw)
    if temporal <= 0 or height <= 0 or width <= 0:
        raise RuntimeError(f"invalid recent-native vision_thw={tuple(thw)}")
    shape = tuple(int(value) for value in payload["shape"])
    if len(shape) != 4:
        raise RuntimeError(f"invalid direct payload shape={shape}")
    sequence = int(shape[2])
    expected = temporal * height * width
    if sequence != expected:
        raise RuntimeError(
            "recent-native cache is not a dense time-major THW layout: "
            f"S={sequence} but T*H*W={expected}"
        )
    recent_tokens = min(int(recent_native_frames), temporal) * height * width
    return sequence - recent_tokens, recent_tokens


def _formal_anchor_frames(
    old_frames: int, stride: int, identity_sink_frames: int
) -> tuple[int, ...]:
    """Return sorted cached-frame indices selected by one anchor policy."""
    old_frames = int(old_frames)
    stride = int(stride)
    identity_sink_frames = int(identity_sink_frames)
    if old_frames < 0 or stride < 0 or identity_sink_frames < 0:
        raise ValueError("formal-anchor lengths and policy must be non-negative")
    selected = set()
    if stride > 0:
        selected.update(range(old_frames - 1, -1, -stride))
    selected.update(range(min(identity_sink_frames, old_frames)))
    return tuple(sorted(selected))


_LOSSLESS_ANCHOR_PARTITION_CACHE = OrderedDict()


def _formal_anchor_partition(
    layer_cache: dict,
    payload: dict,
    recent_native_frames: int,
    stride: int,
    identity_sink_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Partition tensors depend only on integer geometry/policy, never on KV
    # values. Cache the *unchanged* reference result to remove repeated CUDA
    # boolean-index nonzero synchronization and H2D copies on every layer.
    # The normal deployment path remains unchanged unless explicitly enabled.
    if not _flag("HY_DIRECT_CODEC_LOSSLESS_PARTITION_CACHE"):
        return _formal_anchor_partition_uncached(
            layer_cache, payload, recent_native_frames, stride,
            identity_sink_frames, device,
        )
    key = (tuple(layer_cache["vision_thw"]), tuple(payload["shape"]),
           int(recent_native_frames), int(stride), int(identity_sink_frames),
           str(torch.device(device)))
    cache = _LOSSLESS_ANCHOR_PARTITION_CACHE
    if key not in cache:
        cache[key] = _formal_anchor_partition_uncached(
            layer_cache, payload, recent_native_frames, stride,
            identity_sink_frames, device,
        )
        while len(cache) > 64:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def _formal_anchor_partition_uncached(
    layer_cache: dict,
    payload: dict,
    recent_native_frames: int,
    stride: int,
    identity_sink_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition old dense THW rows into low and formal-anchor token indices.

    Periodic buckets are aligned backwards from the recent-history boundary,
    so their newest frame is retained.  This makes the sparse anchors cover
    the entire old prefix even when its length is not divisible by ``stride``.
    ``identity_sink_frames`` independently promotes the first cached frames;
    overlaps are removed.  All indices address the original history payload,
    while recent rows are excluded from both returned token partitions.
    """
    old_tokens, _ = _recent_history_partition(
        layer_cache, payload, recent_native_frames
    )
    _temporal, height, width = (
        int(value) for value in layer_cache["vision_thw"]
    )
    spatial = height * width
    old_frames = old_tokens // spatial
    anchor_frame_indices = torch.tensor(
        _formal_anchor_frames(
            old_frames, stride, identity_sink_frames
        ),
        device=device, dtype=torch.long,
    )
    if anchor_frame_indices.numel() == 0:
        return (
            torch.arange(old_tokens, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            anchor_frame_indices,
        )
    offsets = torch.arange(spatial, device=device, dtype=torch.long)
    anchor_token_indices = (
        anchor_frame_indices[:, None] * spatial + offsets[None]
    ).reshape(-1)
    low_mask = torch.ones(old_tokens, device=device, dtype=torch.bool)
    low_mask[anchor_token_indices] = False
    low_token_indices = torch.arange(
        old_tokens, device=device, dtype=torch.long
    )[low_mask]
    return low_token_indices, anchor_token_indices, anchor_frame_indices


def _slice_freqs(freqs_cis, start: int, end: int | None = None):
    if isinstance(freqs_cis, tuple):
        return tuple(item[start:end] for item in freqs_cis)
    return freqs_cis[start:end]


def _index_freqs(freqs_cis, indices: torch.Tensor):
    """Gather arbitrary sequence rows from tuple or complex RoPE metadata."""
    if isinstance(freqs_cis, tuple):
        return tuple(
            item.index_select(0, indices.to(device=item.device))
            for item in freqs_cis
        )
    return freqs_cis.index_select(0, indices.to(device=freqs_cis.device))


def _native_rope_runtime(x: torch.Tensor, freqs_cis) -> torch.Tensor:
    """Apply all native D128 RoPE channels to a ``[B,H,S,D]`` tensor."""
    if isinstance(freqs_cis, tuple):
        positioned_freqs = _rope_device_pair_runtime(
            freqs_cis[0], freqs_cis[1], x.device
        )
    else:
        positioned_freqs = freqs_cis.to(x.device)
    return apply_rotary_emb_one(
        x.transpose(1, 2), positioned_freqs, head_first=False
    ).transpose(1, 2).contiguous()


def _decode_native_rows(z: torch.Tensor, payload: dict, operator: dict,
                        start: int, dtype: torch.dtype) -> torch.Tensor:
    """Formally decode only ``z[:, start:]`` to native per-head K or V.

    The operation order deliberately matches ``decompress_kv_tensor`` in the
    formal INT8-operator path: BF16 decoder GEMM with the BF16 PCA-mean bias,
    followed by an FP32 add of the stored BF16 C64 center and a final cast.
    """
    centers = payload.get("semantic_centers")
    decoder = operator.get("decoder")
    if not isinstance(centers, torch.Tensor):
        raise RuntimeError("recent-native payload is missing full BF16 C64 centers")
    if not isinstance(decoder, dict) or decoder.get("storage") != "int8":
        raise RuntimeError("recent-native runtime is missing the formal decoder")
    shape = tuple(int(value) for value in payload["shape"])
    batch, heads, sequence, head_dim = shape
    codec_rank = int(decoder["qweight"].shape[1])
    if heads != 16 or head_dim != 128 or z.shape != (batch, sequence, codec_rank):
        raise RuntimeError(
            f"recent-native expects z[B,S,{codec_rank}] and K/V[B,16,S,128], got "
            f"z={tuple(z.shape)} payload={shape}"
        )
    start = int(start)
    if start < 0 or start > sequence:
        raise IndexError(f"recent-native start={start} outside [0,{sequence}]")
    latent = z[:, start:]
    rows = _native_decoder_linear(latent, decoder, operator["mean"])
    ids = payload["semantic_ids"].to(device=z.device, dtype=torch.long)[:, start:]
    centers = centers.to(device=z.device)
    if (
        _triton_data_flag("HY_DIRECT_CODEC_TRITON_NATIVE_CENTER_UNPACK_ONLY")
        and _flag("HY_DIRECT_CODEC_TRITON_NATIVE_CENTER_UNPACK", "1")
        and rows.is_cuda
        and rows.dtype == dtype
    ):
        from .codec_native_position_triton import native_center_unpack

        return native_center_unpack(rows, centers, ids, heads=heads, width=head_dim)
    rows = rows.float()
    centers = centers.float()
    assigned = torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, centers.shape[-1])
    )
    decoded = rows + assigned
    return (
        decoded.reshape(batch, sequence - start, heads, head_dim)
        .permute(0, 2, 1, 3).contiguous().to(dtype)
    )


def _decode_native_indices(
    z: torch.Tensor,
    payload: dict,
    operator: dict,
    indices: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Formally decode arbitrary rank-512 payload rows to native H16D128."""
    centers = payload.get("semantic_centers")
    decoder = operator.get("decoder")
    if not isinstance(centers, torch.Tensor):
        raise RuntimeError("formal-anchor payload is missing full BF16 C64 centers")
    if not isinstance(decoder, dict) or decoder.get("storage") != "int8":
        raise RuntimeError("formal-anchor runtime is missing the formal decoder")
    shape = tuple(int(value) for value in payload["shape"])
    batch, heads, sequence, head_dim = shape
    codec_rank = int(decoder["qweight"].shape[1])
    if heads != 16 or head_dim != 128 or z.shape != (batch, sequence, codec_rank):
        raise RuntimeError(
            f"formal-anchor expects z[B,S,{codec_rank}] and K/V[B,16,S,128], got "
            f"z={tuple(z.shape)} payload={shape}"
        )
    indices = indices.to(device=z.device, dtype=torch.long)
    if indices.ndim != 1:
        raise ValueError("formal-anchor indices must be one-dimensional")
    if indices.numel() == 0:
        return torch.empty(
            batch, heads, 0, head_dim, device=z.device, dtype=dtype
        )
    if bool(((indices < 0) | (indices >= sequence)).any()):
        raise IndexError("formal-anchor index outside the history sequence")
    latent = z.index_select(1, indices)
    rows = _native_decoder_linear(latent, decoder, operator["mean"])
    ids = payload["semantic_ids"].to(
        device=z.device, dtype=torch.long
    ).index_select(1, indices)
    centers = centers.to(device=z.device)
    if (
        _triton_data_flag("HY_DIRECT_CODEC_TRITON_NATIVE_CENTER_UNPACK_ONLY")
        and _flag("HY_DIRECT_CODEC_TRITON_NATIVE_CENTER_UNPACK", "1")
        and rows.is_cuda
        and rows.dtype == dtype
    ):
        from .codec_native_position_triton import native_center_unpack

        return native_center_unpack(rows, centers, ids, heads=heads, width=head_dim)
    rows = rows.float()
    centers = centers.float()
    assigned = torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, centers.shape[-1])
    )
    decoded = rows + assigned
    return (
        decoded.reshape(batch, indices.numel(), heads, head_dim)
        .permute(0, 2, 1, 3).contiguous().to(dtype)
    )


def _dual_low_read(z: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                   joined: torch.Tensor | None = None):
    """Fuse equal-shaped z512->D maps, with a reliable two-GEMM fallback."""
    compatible = (
        z.ndim == 3 and first.ndim == 3 and second.ndim == 3
        and first.shape == second.shape and first.shape[-1] == z.shape[-1]
        and first.dtype == second.dtype and first.device == second.device
    )
    if not compatible:
        return (
            torch.einsum("bnr,hdr->bhnd", z.to(first.dtype), first),
            torch.einsum("bnr,hdr->bhnd", z.to(second.dtype), second),
        )
    width = first.shape[1]
    if joined is None:
        joined = torch.cat((first, second), dim=1)
    elif (
        joined.ndim != 3 or joined.shape[0] != first.shape[0]
        or joined.shape[1] != 2 * width or joined.shape[2] != first.shape[2]
        or joined.dtype != first.dtype or joined.device != first.device
    ):
        raise ValueError("prejoined read weight does not match its Main/PRoPE maps")
    # Flatten (head, output) so this is one dense GEMM over z512 rather than
    # two head-specific reads.  The reshape only reinterprets the output.
    flat = F.linear(z.to(joined.dtype), joined.flatten(0, 1))
    decoded = flat.reshape(z.shape[0], z.shape[1], joined.shape[0], joined.shape[1]).permute(0, 2, 1, 3)
    return decoded[..., :width], decoded[..., width:]


def _grouped_allocation_key(runtime: dict) -> tuple:
    key = runtime.get("grouped_allocation_key")
    if not isinstance(key, tuple):
        raise RuntimeError(
            "grouped dictionary-lowkv runtime is missing its CPU-built "
            "allocation cache key"
        )
    return key


def _grouped_payload_centres(
    payload: dict, group_index: int, group: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tables = payload.get("centre_groups")
    if not isinstance(tables, list):
        raise RuntimeError("grouped dictionary-lowkv payload is missing centre_groups")
    if group_index >= len(tables):
        raise RuntimeError("grouped dictionary-lowkv centre group count mismatch")
    entry = tables[group_index]
    dim = int(group["dim"])
    heads = int(group["head_indices"].numel())
    if int(entry.get("dim", -1)) != dim:
        raise RuntimeError("grouped dictionary-lowkv centre width mismatch")
    main = entry.get("main")
    prope = entry.get("prope")
    if (
        not isinstance(main, torch.Tensor)
        or not isinstance(prope, torch.Tensor)
        or main.shape[1] != heads
        or prope.shape[1] != heads
        or main.shape[-1] != dim
        or prope.shape[-1] != dim
    ):
        raise RuntimeError("invalid grouped dictionary-lowkv centre tensors")
    return main.to(device), prope.to(device)


def _valid_grouped_old_history(
    value, groups: list[dict], device: torch.device, dtype: torch.dtype,
    batch: int, tokens: int,
) -> bool:
    if not isinstance(value, list) or len(value) != len(groups):
        return False
    for cached, group in zip(value, groups):
        expected = (
            2 * int(batch),
            int(group["head_indices"].numel()),
            int(tokens),
            int(group["dim"]),
        )
        for name in ("k", "v"):
            tensor = cached.get(name) if isinstance(cached, dict) else None
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != expected
            ):
                return False
    return True


@torch.no_grad()
def _build_uniform_low_history(
    layer_idx: int,
    runtime: dict,
    kp: dict,
    vp: dict,
    kz: torch.Tensor,
    vz: torch.Tensor,
    k_ids: torch.Tensor,
    v_ids: torch.Tensor,
    freqs_cis,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    dtype: torch.dtype,
    prope_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one uniform-D old-history K/V view from the formal z512 payload."""
    state = runtime["attention"]
    indices = state["main_indices"]
    km, kq = _profile_call(
        "direct_uniform_read", layer_idx, _dual_low_read,
        kz, state["k_main_read"], state["k_prope_read"],
        state.get("k_read_joined"),
    )
    vm, vq = _profile_call(
        "direct_uniform_read", layer_idx, _dual_low_read,
        vz, state["v_main_read"], state["v_prope_read"],
        state.get("v_read_joined"),
    )
    device = kz.device
    if (
        _scoped_flag(
            "HY_DIRECT_CODEC_TRITON_LOW_HISTORY_POSITION",
            "HY_DIRECT_CODEC_TRITON_POSITION",
        )
        and _flag("HY_DIRECT_CODEC_TRITON_POSITION_PACK", "1")
        and _flag("HY_DIRECT_CODEC_STREAM_FUSED_CENTER")
        and _scoped_flag(
            "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
            "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
        )
        and km.is_cuda
        and isinstance(freqs_cis, tuple)
    ):
        from .codec_native_position_triton import center_position_pack

        cos, sin = _rope_device_pair_runtime(
            freqs_cis[0], freqs_cis[1], device
        )
        if prope_matrix is None:
            prope_matrix = _prope_matrix_runtime(viewmats, Ks, device)
        return _profile_call(
            "direct_uniform_position_pack", layer_idx,
            center_position_pack,
            km, kq, vm, vq,
            kp["centre_main"].to(device),
            kp["centre_prope"].to(device),
            vp["centre_main"].to(device),
            vp["centre_prope"].to(device),
            k_ids, v_ids, cos, sin, indices, prope_matrix,
        )
    if _flag("HY_DIRECT_CODEC_STREAM_FUSED_CENTER") and km.is_cuda:
        km, kq = _profile_call(
            "direct_uniform_center", layer_idx, _center_pair_add_runtime,
            km, kq, kp["centre_main"].to(device),
            kp["centre_prope"].to(device), k_ids,
        )
        vm, vq = _profile_call(
            "direct_uniform_center", layer_idx, _center_pair_add_runtime,
            vm, vq, vp["centre_main"].to(device),
            vp["centre_prope"].to(device), v_ids,
        )
    else:
        km = km + _assigned_low(kp["centre_main"].to(device), k_ids)
        kq = kq + _assigned_low(kp["centre_prope"].to(device), k_ids)
        vm = vm + _assigned_low(vp["centre_main"].to(device), v_ids)
        vq = vq + _assigned_low(vp["centre_prope"].to(device), v_ids)
    km = _profile_call(
        "direct_uniform_rope", layer_idx, _selected_rope_runtime,
        km.to(dtype), freqs_cis, indices,
    )
    if (
        _scoped_flag(
            "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
            "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
        )
        and kq.is_cuda
    ):
        kq, vq = _profile_call(
            "direct_uniform_prope", layer_idx, _prope_kv_pair_runtime,
            kq, vq, viewmats, Ks, device, prope_matrix,
            "HY_DIRECT_CODEC_TRITON_LOW_HISTORY_POSITION",
        )
    else:
        _, apply_old_k, _ = _prope_fns(
            kq.shape[-1], viewmats, Ks, device
        )
        _, apply_old_v, _ = _prope_fns(
            vq.shape[-1], viewmats, Ks, device
        )
        kq = apply_old_k(kq.float()).to(dtype)
        vq = apply_old_v(vq.float()).to(dtype)
    return (
        torch.cat((km, kq), 0),
        torch.cat((vm.to(dtype), vq), 0),
    )


def _position_latent_read_delta(
    history_adapter,
    state: dict,
    kz: torch.Tensor,
    vz: torch.Tensor,
    freqs_cis,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    dtype: torch.dtype,
    prope_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a foldable z512-read residual in the canonical low-D bases.

    This helper is used only by opt-in adapter training/evaluation.  Export
    folds its matrices into the four existing low read weights, so production
    keeps the same cache payload, FlashAttention width, and operator count.
    """

    km, kq = history_adapter.latent_read_delta(kz, target="key")
    vm, vq = history_adapter.latent_read_delta(vz, target="value")
    km = _selected_rope_runtime(
        km.to(dtype), freqs_cis, state["main_indices"]
    )
    if kq.is_cuda:
        kq, vq = _prope_kv_pair_runtime(
            kq, vq, viewmats, Ks, kq.device, prope_matrix
        )
    else:
        _, apply_k, _ = _prope_fns(kq.shape[-1], viewmats, Ks, kq.device)
        _, apply_v, _ = _prope_fns(vq.shape[-1], viewmats, Ks, vq.device)
        kq = apply_k(kq.float()).to(dtype)
        vq = apply_v(vq.float()).to(dtype)
    return (
        torch.cat((km, kq.to(dtype)), 0),
        torch.cat((vm.to(dtype), vq.to(dtype)), 0),
    )


def _position_center_read_delta(
    history_adapter,
    state: dict,
    k_centers: torch.Tensor,
    v_centers: torch.Tensor,
    k_ids: torch.Tensor,
    v_ids: torch.Tensor,
    freqs_cis,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    dtype: torch.dtype,
    prope_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a foldable C64-center residual in the canonical low-D bases."""

    km, kq = history_adapter.center_read_delta(
        k_centers, k_ids, target="key"
    )
    vm, vq = history_adapter.center_read_delta(
        v_centers, v_ids, target="value"
    )
    km = _selected_rope_runtime(
        km.to(dtype), freqs_cis, state["main_indices"]
    )
    if kq.is_cuda:
        kq, vq = _prope_kv_pair_runtime(
            kq, vq, viewmats, Ks, kq.device, prope_matrix
        )
    else:
        _, apply_k, _ = _prope_fns(kq.shape[-1], viewmats, Ks, kq.device)
        _, apply_v, _ = _prope_fns(vq.shape[-1], viewmats, Ks, vq.device)
        kq = apply_k(kq.float()).to(dtype)
        vq = apply_v(vq.float()).to(dtype)
    return (
        torch.cat((km, kq.to(dtype)), 0),
        torch.cat((vm.to(dtype), vq.to(dtype)), 0),
    )


@torch.no_grad()
def _build_grouped_low_history(
    runtime: dict,
    kp: dict,
    vp: dict,
    kz: torch.Tensor,
    vz: torch.Tensor,
    k_ids: torch.Tensor,
    v_ids: torch.Tensor,
    freqs_cis,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    dtype: torch.dtype,
    prope_matrix: torch.Tensor | None = None,
) -> list[dict]:
    groups = runtime["grouped_attention"]
    if any(
        not isinstance(payload.get("centre_groups"), list)
        or len(payload["centre_groups"]) != len(groups)
        for payload in (kp, vp)
    ):
        raise RuntimeError("grouped dictionary-lowkv centre group count mismatch")
    fused_prope_matrix = prope_matrix
    if (
        fused_prope_matrix is None
        and _scoped_flag(
            "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
            "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
        )
        and kz.is_cuda
    ):
        fused_prope_matrix = _prope_matrix_runtime(viewmats, Ks, kz.device)
    consolidated = runtime.get("grouped_read_all")
    all_k = all_v = None
    if consolidated is not None:
        all_k = F.linear(
            kz.to(consolidated["k"].dtype), consolidated["k"]
        )
        all_v = F.linear(
            vz.to(consolidated["v"].dtype), consolidated["v"]
        )
    positioned = []
    for group_index, group in enumerate(groups):
        state = group["attention"]
        indices = state["main_indices"]
        if consolidated is None:
            km, kq = _dual_low_read(
                kz, state["k_main_read"], state["k_prope_read"],
                state.get("k_read_joined"),
            )
            vm, vq = _dual_low_read(
                vz, state["v_main_read"], state["v_prope_read"],
                state.get("v_read_joined"),
            )
        else:
            offset, count, heads, width = consolidated["slices"][group_index]

            def split(value):
                decoded = value[..., offset : offset + count].reshape(
                    value.shape[0], value.shape[1], heads, 2 * width
                ).permute(0, 2, 1, 3)
                return decoded[..., :width], decoded[..., width:]

            km, kq = split(all_k)
            vm, vq = split(all_v)
        k_main_centres, k_prope_centres = _grouped_payload_centres(
            kp, group_index, group, kz.device
        )
        v_main_centres, v_prope_centres = _grouped_payload_centres(
            vp, group_index, group, vz.device
        )
        if (
            _scoped_flag(
                "HY_DIRECT_CODEC_TRITON_LOW_HISTORY_POSITION",
                "HY_DIRECT_CODEC_TRITON_POSITION",
            )
            and _flag("HY_DIRECT_CODEC_TRITON_POSITION_PACK", "1")
            and _flag("HY_DIRECT_CODEC_STREAM_FUSED_CENTER")
            and _scoped_flag(
                "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
                "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
            )
            and km.is_cuda
            and isinstance(freqs_cis, tuple)
        ):
            from .codec_native_position_triton import center_position_pack

            cos, sin = _rope_device_pair_runtime(
                freqs_cis[0], freqs_cis[1], kz.device
            )
            if fused_prope_matrix is None:
                fused_prope_matrix = _prope_matrix_runtime(
                    viewmats, Ks, kz.device
                )
            packed_k, packed_v = center_position_pack(
                km, kq, vm, vq,
                k_main_centres, k_prope_centres,
                v_main_centres, v_prope_centres,
                k_ids, v_ids, cos, sin, indices, fused_prope_matrix,
            )
            positioned.append({"k": packed_k, "v": packed_v})
            continue
        if _flag("HY_DIRECT_CODEC_STREAM_FUSED_CENTER") and km.is_cuda:
            km, kq = _center_pair_add_runtime(
                km, kq, k_main_centres, k_prope_centres, k_ids
            )
            vm, vq = _center_pair_add_runtime(
                vm, vq, v_main_centres, v_prope_centres, v_ids
            )
        else:
            km = km + _assigned_low(k_main_centres, k_ids)
            kq = kq + _assigned_low(k_prope_centres, k_ids)
            vm = vm + _assigned_low(v_main_centres, v_ids)
            vq = vq + _assigned_low(v_prope_centres, v_ids)
        km = _selected_rope_runtime(km.to(dtype), freqs_cis, indices)
        if (
            _scoped_flag(
                "HY_DIRECT_CODEC_LOW_HISTORY_FUSED_PROPE",
                "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
            )
            and kq.is_cuda
        ):
            kq, vq = _prope_kv_pair_runtime(
                kq, vq, viewmats, Ks, kz.device,
                matrix=fused_prope_matrix,
                triton_scope="HY_DIRECT_CODEC_TRITON_LOW_HISTORY_POSITION",
            )
        else:
            _, apply_k, _ = _prope_fns(
                int(group["dim"]), viewmats, Ks, kz.device
            )
            _, apply_v, _ = _prope_fns(
                int(group["dim"]), viewmats, Ks, vz.device
            )
            kq = apply_k(kq.float()).to(dtype)
            vq = apply_v(vq.float()).to(dtype)
        positioned.append({
            "k": torch.cat((km, kq), 0),
            "v": torch.cat((vm.to(dtype), vq), 0),
        })
    return positioned


@torch.no_grad()
def _attend_grouped_low_history(
    runtime: dict,
    q_pre_bshd: torch.Tensor,
    current_freqs_cis,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor | None,
    positioned_groups: list[dict],
    dtype: torch.dtype,
    native: torch.Tensor | None = None,
    native_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exactly one old-history FA per width group.

    Main and PRoPE are concatenated along batch before each call.  Group
    outputs/LSEs are then scattered back to the canonical H16 order, so the
    caller can perform the usual per-head exact LSE merge with the native
    text/current/recent/anchor branch.
    """
    q_pre = q_pre_bshd.transpose(1, 2)
    batch, _heads, query, _dim = q_pre.shape
    device = q_pre.device
    combine_scatter = native is not None
    if combine_scatter:
        if native_lse is None:
            raise ValueError("native_lse is required for grouped fused combine")
        full = torch.empty_like(native)
        full_lse = None
    else:
        full = torch.empty(
            2 * batch, 16, query, 128, device=device, dtype=torch.float32
        )
        full_lse = torch.empty(
            2 * batch, 16, query, device=device, dtype=torch.float32
        )
    groups = runtime["grouped_attention"]
    if len(positioned_groups) != len(groups):
        raise RuntimeError("grouped positioned history count mismatch")
    fused_prope_matrix = None
    if (
        _scoped_flag(
            "HY_DIRECT_CODEC_LOW_QUERY_FUSED_PROPE",
            "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
        )
        and q_pre.is_cuda
    ):
        fused_prope_matrix = _prope_matrix_runtime(
            current_viewmats, current_Ks, device, role="q"
        )
    triton_query_pack = (
        _scoped_flag(
            "HY_DIRECT_CODEC_TRITON_LOW_QUERY_POSITION",
            "HY_DIRECT_CODEC_TRITON_POSITION",
        )
        and _flag("HY_DIRECT_CODEC_TRITON_QUERY_PACK", "1")
        and isinstance(current_freqs_cis, tuple)
        and q_pre.is_cuda
    )
    query_cos = query_sin = None
    if triton_query_pack:
        query_cos, query_sin = _rope_device_pair_runtime(
            current_freqs_cis[0], current_freqs_cis[1], device
        )
        if fused_prope_matrix is None:
            fused_prope_matrix = _prope_matrix_runtime(
                current_viewmats, current_Ks, device, role="q"
            )
    for group, history in zip(groups, positioned_groups):
        heads = group["head_indices"]
        state = group["attention"]
        indices = state["main_indices"]
        group_q_pre = q_pre.index_select(1, heads)
        q_main = torch.einsum(
            "bgnr,gdr->bgnd", group_q_pre, state["q_main_reduce"]
        )
        if not triton_query_pack:
            q_main = _selected_rope_runtime(
                q_main.to(dtype), current_freqs_cis, indices
            )
        q_prope = _quartet_reduce_runtime(
            group_q_pre, state["q_prope_mix"]
        )
        if triton_query_pack:
            from .codec_native_position_triton import query_position_pack

            q_group = query_position_pack(
                q_main, q_prope, query_cos, query_sin, indices,
                fused_prope_matrix,
                state["log_residual_scale_main"].exp(),
                state["log_residual_scale_prope"].exp(),
            )
        else:
            if fused_prope_matrix is not None:
                q_prope = _prope_one_runtime(
                    q_prope, current_viewmats, current_Ks, device,
                    matrix=fused_prope_matrix,
                    triton_scope="HY_DIRECT_CODEC_TRITON_LOW_QUERY_POSITION",
                ).to(dtype)
            else:
                apply_q, _, _ = _prope_fns(
                    int(group["dim"]), current_viewmats, current_Ks, device
                )
                q_prope = apply_q(q_prope.float()).to(dtype)
            q_main = q_main * state["log_residual_scale_main"].exp()[
                None, :, None, None
            ]
            q_prope = q_prope * state["log_residual_scale_prope"].exp()[
                None, :, None, None
            ]
            q_group = torch.cat((q_main, q_prope), 0).to(dtype)
        low, lse = _flash_lse(
            q_group,
            history["k"],
            history["v"],
            scale=1.0 / math.sqrt(128.0),
        )
        main_full = torch.einsum(
            "bgnd,gdo->bgno", low[:batch], state["v_main_lift"]
        )
        prope_full = _quartet_lift_runtime(
            low[batch:], state["v_prope_mix"]
        )
        if combine_scatter:
            from .codec_native_position_triton import grouped_lse_combine_scatter

            grouped_lse_combine_scatter(
                native, native_lse, main_full, prope_full, lse, heads, full
            )
        elif (
            _triton_data_flag("HY_DIRECT_CODEC_TRITON_GROUPED_SCATTER_ONLY")
            and _flag("HY_DIRECT_CODEC_TRITON_GROUPED_SCATTER", "1")
            and low.is_cuda
        ):
            from .codec_native_position_triton import grouped_scatter_pair

            grouped_scatter_pair(
                main_full, prope_full, lse, heads, full, full_lse
            )
        else:
            full[:batch].index_copy_(1, heads, main_full.float())
            full[batch:].index_copy_(1, heads, prope_full.float())
            full_lse[:batch].index_copy_(1, heads, lse[:batch].float())
            full_lse[batch:].index_copy_(1, heads, lse[batch:].float())
    return full, full_lse


def _valid_positioned_cache(value, key, device, dtype, k_dim, v_dim) -> bool:
    return (
        isinstance(value, dict) and value.get("key") == key
        and isinstance(value.get("k"), torch.Tensor) and isinstance(value.get("v"), torch.Tensor)
        and value["k"].device == device and value["v"].device == device
        and value["k"].dtype == dtype and value["v"].dtype == dtype
        and value["k"].shape[-1] == k_dim and value["v"].shape[-1] == v_dim
    )


def _valid_recent_positioned_cache(value, key, device, dtype, k_dim, v_dim,
                                   old_tokens: int, recent_tokens: int) -> bool:
    if not isinstance(value, dict) or value.get("key") != key:
        return False

    def valid(name: str, width: int, tokens: int) -> bool:
        tensor = value.get(name)
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device == device and tensor.dtype == dtype
            and tensor.shape[-1] == width and tensor.shape[-2] == tokens
        )

    if not valid("recent_k", 128, recent_tokens) or not valid(
        "recent_v", 128, recent_tokens
    ):
        return False
    if old_tokens == 0:
        return value.get("old_k") is None and value.get("old_v") is None
    return valid("old_k", k_dim, old_tokens) and valid("old_v", v_dim, old_tokens)


def _valid_formal_anchor_positioned_cache(
    value,
    key,
    device,
    dtype,
    k_dim,
    v_dim,
    low_tokens: int,
    anchor_tokens: int,
    recent_tokens: int,
) -> bool:
    """Validate the three historical representations owned by anchor mode."""
    if not _valid_recent_positioned_cache(
        value, key, device, dtype, k_dim, v_dim, low_tokens, recent_tokens
    ):
        return False

    def valid(name: str) -> bool:
        tensor = value.get(name) if isinstance(value, dict) else None
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device == device and tensor.dtype == dtype
            and tensor.shape[-1] == 128
            and tensor.shape[-2] == anchor_tokens
        )

    if anchor_tokens == 0:
        return value.get("anchor_k") is None and value.get("anchor_v") is None
    return valid("anchor_k") and valid("anchor_v")


def _valid_grouped_recent_positioned_cache(
    value,
    key,
    runtime: dict,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    low_tokens: int,
    anchor_tokens: int,
    recent_tokens: int,
) -> bool:
    if not isinstance(value, dict) or value.get("key") != key:
        return False

    def native(name: str, tokens: int) -> bool:
        tensor = value.get(name)
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device == device
            and tensor.dtype == dtype
            and tuple(tensor.shape) == (2 * batch, 16, tokens, 128)
        )

    if not native("recent_k", recent_tokens) or not native(
        "recent_v", recent_tokens
    ):
        return False
    if low_tokens == 0:
        if value.get("old_groups") not in (None, []):
            return False
    elif not _valid_grouped_old_history(
        value.get("old_groups"), runtime["grouped_attention"],
        device, dtype, batch, low_tokens,
    ):
        return False
    if anchor_tokens == 0:
        return value.get("anchor_k") is None and value.get("anchor_v") is None
    return native("anchor_k", anchor_tokens) and native(
        "anchor_v", anchor_tokens
    )


@torch.no_grad()
def _attend_dictionary_lowkv_recent(
    layer_idx, q_pre_bshd, query_bhsd, current_key_bhsd,
    current_value_bhsd, encoder_key_bhsd, encoder_value_bhsd,
    layer_cache, current_freqs_cis, current_viewmats, current_Ks, runtime,
):
    """Attend to old-low, optional formal anchors, and recent-native history."""
    device, dtype = query_bhsd.device, query_bhsd.dtype
    state = runtime["attention"]
    grouped = runtime.get("grouped_attention") is not None
    batch = q_pre_bshd.shape[0]
    kp, vp = layer_cache["k_vision"], layer_cache["v_vision"]
    recent_frames = int(runtime["recent_native_frames"])
    old_tokens, recent_tokens = _recent_history_partition(
        layer_cache, kp, recent_frames
    )
    v_old_tokens, v_recent_tokens = _recent_history_partition(
        layer_cache, vp, recent_frames
    )
    if (v_old_tokens, v_recent_tokens) != (old_tokens, recent_tokens):
        raise RuntimeError("recent-native K/V history partitions differ")

    (
        anchor_stride,
        anchor_sink_frames,
        periodic_refresh,
        cache_write_chunk_id,
    ) = _effective_formal_anchor_policy(
        layer_cache, runtime
    )
    formal_anchor = anchor_stride > 0 or anchor_sink_frames > 0
    if formal_anchor:
        _temporal, height, width = (
            int(value) for value in layer_cache["vision_thw"]
        )
        spatial = height * width
        anchor_frame_count = len(_formal_anchor_frames(
            old_tokens // spatial, anchor_stride, anchor_sink_frames
        ))
        anchor_tokens = anchor_frame_count * spatial
        low_tokens = old_tokens - anchor_tokens
    else:
        anchor_tokens = 0
        low_tokens = old_tokens
    if os.environ.get("HY_DIRECT_CODEC_MAC_AUDIT_PATH"):
        from .attention_compute_stats import record
        record(
            layer_idx, runtime, int(query_bhsd.shape[2]),
            int(current_key_bhsd.shape[2]), int(encoder_key_bhsd.shape[2]),
            old_tokens + recent_tokens, low_tokens, anchor_tokens,
            recent_tokens, cache_write_chunk_id,
        )
    if periodic_refresh and (low_tokens != 0 or anchor_tokens != old_tokens):
        raise RuntimeError(
            "periodic full-history refresh did not route every old token to "
            f"formal attention: chunk={cache_write_chunk_id} layer={layer_idx} "
            f"old_tokens={old_tokens} low_tokens={low_tokens} "
            f"anchor_tokens={anchor_tokens}"
        )

    capture_raw = bool(runtime.get("recent_capture_raw", False))
    raw_key = _recent_raw_tail_key(
        int(kp["layer_idx"]), kp["shape"], recent_frames,
        old_tokens, recent_tokens,
    )
    raw_tail = layer_cache.get("_direct_codec_recent_native_pre")
    grouped_cache_suffix = (
        ("grouped_2width_v1", _grouped_allocation_key(runtime))
        if grouped else ()
    )

    cache_key = (
        (
            "recent_native_formal_anchor_v1", str(device), dtype,
            int(runtime["shape"][1]), int(runtime["shape"][3]),
            recent_frames, old_tokens, low_tokens, anchor_tokens, recent_tokens,
            anchor_stride, anchor_sink_frames, capture_raw,
            *grouped_cache_suffix,
            *(("periodic_refresh_v1", cache_write_chunk_id)
              if periodic_refresh else ()),
        )
        if formal_anchor
        else (
            (
                "recent_native_raw_v1", str(device), dtype,
                int(runtime["shape"][1]), int(runtime["shape"][3]),
                recent_frames, old_tokens, recent_tokens,
                *grouped_cache_suffix,
            )
            if capture_raw
            else (
                # Preserve the pre-existing canary's key and behavior exactly
                # when HY_DIRECT_CODEC_RECENT_CAPTURE_RAW is disabled.
                "recent_native_v1", str(device), dtype,
                int(runtime["shape"][1]), int(runtime["shape"][3]),
                recent_frames, old_tokens, recent_tokens,
                *grouped_cache_suffix,
            )
        )
    )
    positioned = (
        layer_cache.get("_direct_codec_positioned_history")
        if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1") else None
    )
    positioned_valid = (
        _valid_grouped_recent_positioned_cache(
            positioned, cache_key, runtime, device, dtype, batch,
            low_tokens, anchor_tokens, recent_tokens,
        )
        if grouped else (
        _valid_formal_anchor_positioned_cache(
            positioned, cache_key, device, dtype,
            int(runtime["shape"][1]), int(runtime["shape"][3]),
            low_tokens, anchor_tokens, recent_tokens,
        )
        if formal_anchor else _valid_recent_positioned_cache(
            positioned, cache_key, device, dtype,
            int(runtime["shape"][1]), int(runtime["shape"][3]),
            old_tokens, recent_tokens,
        )
        )
    )
    rebuild_stream = None
    reusable_recent = None
    if not positioned_valid:
        # Step-selective refresh alternates between a full-history positioned
        # cache and the ordinary low-D/static-anchor cache.  Drop the stale
        # mode before constructing the replacement so both representations
        # are never simultaneously live at the Python ownership level.
        if positioned is not None:
            stale_k = positioned.get("recent_k")
            stale_v = positioned.get("recent_v")
            expected_recent = (2 * batch, 16, recent_tokens, 128)
            if (
                isinstance(stale_k, torch.Tensor)
                and isinstance(stale_v, torch.Tensor)
                and stale_k.device == device
                and stale_v.device == device
                and stale_k.dtype == dtype
                and stale_v.dtype == dtype
                and tuple(stale_k.shape) == expected_recent
                and tuple(stale_v.shape) == expected_recent
            ):
                reusable_recent = (stale_k, stale_v)
            layer_cache.pop("_direct_codec_positioned_history", None)
            positioned = None
        if periodic_refresh and _flag(
            "HY_DIRECT_CODEC_PERIODIC_REFRESH_AUDIT"
        ):
            print(
                "[DirectCodecPeriodicRefresh] "
                f"chunk={cache_write_chunk_id} layer={int(layer_idx)} "
                f"old_tokens={old_tokens} low_tokens={low_tokens} "
                f"formal_tokens={anchor_tokens} recent_tokens={recent_tokens} "
                f"interval={int(runtime['periodic_refresh_interval'])} "
                f"phase={int(runtime['periodic_refresh_phase'])} "
                f"step={int(os.environ.get('HY_ROLLOUT_DRIFT_STEP', '-1'))}",
                flush=True,
            )
        if (
            capture_raw
            and reusable_recent is None
            and not _valid_recent_raw_tail(
                raw_tail, raw_key, device, batch, recent_tokens
            )
        ):
            raise RuntimeError(
                "recent-native raw capture is missing or stale; cache "
                "compression and attention must use the same "
                "recent/capture/layer settings"
            )
        old_k = old_v = anchor_k = anchor_v = None
        old_groups = None
        full_prope_matrix = old_prope_matrix = None
        decoded_anchor_k_pre = decoded_anchor_v_pre = None
        decoded_recent_k_pre = decoded_recent_v_pre = None
        low_token_indices = anchor_token_indices = None
        if formal_anchor:
            (
                low_token_indices,
                anchor_token_indices,
                anchor_frame_indices,
            ) = _formal_anchor_partition(
                layer_cache, kp, recent_frames, anchor_stride,
                anchor_sink_frames, device,
            )
            if (
                low_token_indices.numel() != low_tokens
                or anchor_token_indices.numel() != anchor_tokens
                or anchor_frame_indices.numel() != anchor_frame_count
            ):
                raise RuntimeError("formal-anchor partition count changed unexpectedly")

        if old_tokens > 0:
            kz = _profile_call(
                "direct_rank512", layer_idx, _rank512, kp, device
            )
            vz = _profile_call(
                "direct_rank512", layer_idx, _rank512, vp, device
            )

        vision_Ks = layer_cache.get("vision_Ks")
        full_history_freqs = layer_cache["vision_freqs_cis"]
        if isinstance(full_history_freqs, tuple) and query_bhsd.is_cuda:
            full_history_freqs = _rope_device_pair_runtime(
                full_history_freqs[0], full_history_freqs[1], device
            )
        if (
            (low_tokens > 0 or anchor_tokens > 0 or recent_tokens > 0)
            and _any_fused_prope_scope()
            and query_bhsd.is_cuda
        ):
            full_prope_matrix = _prope_matrix_runtime(
                layer_cache["vision_viewmats"], vision_Ks, device
            )

        if low_tokens > 0:
            if formal_anchor:
                old_kz = kz.index_select(1, low_token_indices)
                old_vz = vz.index_select(1, low_token_indices)
            else:
                old_kz, old_vz = kz[:, :old_tokens], vz[:, :old_tokens]
            if formal_anchor:
                k_ids = kp["semantic_ids"].to(
                    device=device, dtype=torch.long
                ).index_select(1, low_token_indices)
                v_ids = vp["semantic_ids"].to(
                    device=device, dtype=torch.long
                ).index_select(1, low_token_indices)
            else:
                k_ids = kp["semantic_ids"].to(
                    device=device, dtype=torch.long
                )[:, :old_tokens]
                v_ids = vp["semantic_ids"].to(
                    device=device, dtype=torch.long
                )[:, :old_tokens]
            old_freqs = (
                _index_freqs(
                    full_history_freqs, low_token_indices
                )
                if formal_anchor else _slice_freqs(
                    full_history_freqs, 0, old_tokens
                )
            )
            if formal_anchor:
                old_viewmats = layer_cache["vision_viewmats"].index_select(
                    1, low_token_indices
                )
                old_Ks = (
                    None if vision_Ks is None
                    else vision_Ks.index_select(1, low_token_indices)
                )
            else:
                old_viewmats = layer_cache["vision_viewmats"][:, :old_tokens]
                old_Ks = None if vision_Ks is None else vision_Ks[:, :old_tokens]
            if full_prope_matrix is not None:
                old_prope_matrix = (
                    full_prope_matrix.index_select(1, low_token_indices)
                    if formal_anchor else full_prope_matrix[:, :old_tokens]
                )
            parallel_rebuild = (
                _flag("HY_DIRECT_CODEC_PARALLEL_REBUILD")
                and old_kz.is_cuda
                and not position_profile.enabled()
                # Max-context screening keeps overlap only on measured-positive
                # schedules: uniform widths up to the configured bound, plus
                # grouped layers after their reads have been consolidated.
                and (
                    (
                        grouped
                        and runtime.get("grouped_read_all") is not None
                        and _flag(
                            "HY_DIRECT_CODEC_PARALLEL_REBUILD_GROUPED", "0"
                        )
                    )
                    or (
                        not grouped
                        and int(runtime["shape"][1]) <= int(os.environ.get(
                            "HY_DIRECT_CODEC_PARALLEL_REBUILD_MAX_DIM", "32"
                        ))
                    )
                )
            )
            if parallel_rebuild:
                caller_stream = torch.cuda.current_stream(device)
                rebuild_stream = _parallel_rebuild_stream(device)
                rebuild_stream.wait_stream(caller_stream)
                with torch.cuda.stream(rebuild_stream):
                    if grouped:
                        old_groups = _build_grouped_low_history(
                            runtime, kp, vp, old_kz, old_vz, k_ids, v_ids,
                            old_freqs, old_viewmats, old_Ks, dtype,
                            old_prope_matrix,
                        )
                    else:
                        old_k, old_v = _build_uniform_low_history(
                            layer_idx, runtime, kp, vp, old_kz, old_vz,
                            k_ids, v_ids, old_freqs, old_viewmats, old_Ks, dtype,
                            old_prope_matrix,
                        )
            elif grouped:
                old_groups = _profile_call(
                    "direct_grouped_rebuild", layer_idx,
                    _build_grouped_low_history,
                    runtime, kp, vp, old_kz, old_vz, k_ids, v_ids,
                    old_freqs, old_viewmats, old_Ks, dtype,
                    old_prope_matrix,
                )
            else:
                old_k, old_v = _profile_call(
                    "direct_uniform_rebuild", layer_idx,
                    _build_uniform_low_history,
                    layer_idx, runtime, kp, vp, old_kz, old_vz,
                    k_ids, v_ids, old_freqs, old_viewmats, old_Ks, dtype,
                    old_prope_matrix,
                )

        if (
            anchor_tokens > 0
            and recent_tokens > 0
            and not capture_raw
            and _flag("HY_DIRECT_CODEC_CONSOLIDATED_NATIVE_DECODE")
            and not grouped
            and int(runtime["shape"][1]) >= int(os.environ.get(
                "HY_DIRECT_CODEC_CONSOLIDATED_NATIVE_DECODE_MIN_DIM", "64"
            ))
        ):
            recent_indices = torch.arange(
                old_tokens, old_tokens + recent_tokens,
                device=device, dtype=torch.long,
            )
            native_decode_indices = torch.cat(
                (anchor_token_indices, recent_indices), 0
            )
            decoded_k_pre = _profile_call(
                "direct_native_decode_consolidated", layer_idx,
                _decode_native_indices,
                kz, kp, runtime["operators"]["k"],
                native_decode_indices, dtype,
            )
            decoded_v_pre = _profile_call(
                "direct_native_decode_consolidated", layer_idx,
                _decode_native_indices,
                vz, vp, runtime["operators"]["v"],
                native_decode_indices, dtype,
            )
            decoded_anchor_k_pre = decoded_k_pre[:, :, :anchor_tokens]
            decoded_anchor_v_pre = decoded_v_pre[:, :, :anchor_tokens]
            decoded_recent_k_pre = decoded_k_pre[:, :, anchor_tokens:]
            decoded_recent_v_pre = decoded_v_pre[:, :, anchor_tokens:]

        if anchor_tokens > 0:
            if decoded_anchor_k_pre is None:
                anchor_k_pre = _profile_call(
                    "direct_anchor_decode", layer_idx,
                    _decode_native_indices,
                    kz, kp, runtime["operators"]["k"],
                    anchor_token_indices, dtype,
                )
                anchor_v_pre = _profile_call(
                    "direct_anchor_decode", layer_idx,
                    _decode_native_indices,
                    vz, vp, runtime["operators"]["v"],
                    anchor_token_indices, dtype,
                )
            else:
                anchor_k_pre = decoded_anchor_k_pre
                anchor_v_pre = decoded_anchor_v_pre
            anchor_freqs = _index_freqs(
                full_history_freqs, anchor_token_indices
            )
            anchor_viewmats = layer_cache["vision_viewmats"].index_select(
                1, anchor_token_indices
            )
            anchor_Ks = (
                None if vision_Ks is None
                else vision_Ks.index_select(1, anchor_token_indices)
            )
            triton_native_pack = (
                _scoped_flag(
                    "HY_DIRECT_CODEC_TRITON_NATIVE_POSITION",
                    "HY_DIRECT_CODEC_TRITON_POSITION",
                )
                and _flag("HY_DIRECT_CODEC_TRITON_NATIVE_POSITION_PACK", "1")
                and isinstance(anchor_freqs, tuple)
                and anchor_k_pre.is_cuda
            )
            if triton_native_pack:
                anchor_matrix = (
                    full_prope_matrix.index_select(1, anchor_token_indices)
                    if full_prope_matrix is not None
                    else _prope_matrix_runtime(anchor_viewmats, anchor_Ks, device)
                )
                if _flag("HY_DIRECT_CODEC_TRITON_NATIVE_POSITION_EXACT", "1"):
                    from .codec_native_position_triton import native_prope_pack

                    anchor_k_main = _native_rope_runtime(
                        anchor_k_pre, anchor_freqs
                    )
                    anchor_k, anchor_v = _profile_call(
                        "direct_anchor_position_pack", layer_idx,
                        native_prope_pack,
                        anchor_k_main, anchor_k_pre, anchor_v_pre, anchor_matrix,
                    )
                else:
                    from .codec_native_position_triton import native_position_pack

                    anchor_cos, anchor_sin = _rope_device_pair_runtime(
                        anchor_freqs[0], anchor_freqs[1], device
                    )
                    anchor_k, anchor_v = _profile_call(
                        "direct_anchor_position_pack", layer_idx,
                        native_position_pack,
                        anchor_k_pre, anchor_v_pre,
                        anchor_cos, anchor_sin, anchor_matrix,
                    )
            elif (
                _scoped_flag(
                    "HY_DIRECT_CODEC_NATIVE_FUSED_PROPE",
                    "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
                )
                and anchor_k_pre.is_cuda
            ):
                anchor_k_main = _native_rope_runtime(anchor_k_pre, anchor_freqs)
                anchor_k_prope, anchor_v_prope = _profile_call(
                    "direct_anchor_prope", layer_idx,
                    _prope_kv_pair_runtime,
                    anchor_k_pre, anchor_v_pre,
                    anchor_viewmats, anchor_Ks, device,
                    None if full_prope_matrix is None else
                    full_prope_matrix.index_select(1, anchor_token_indices),
                    "HY_DIRECT_CODEC_TRITON_NATIVE_POSITION",
                )
                anchor_k = torch.cat((anchor_k_main, anchor_k_prope), 0)
                anchor_v = torch.cat((anchor_v_pre, anchor_v_prope), 0)
            else:
                anchor_k_main = _native_rope_runtime(anchor_k_pre, anchor_freqs)
                _, apply_anchor_kv, _ = _prope_fns(
                    128, anchor_viewmats, anchor_Ks, device
                )
                anchor_k_prope = apply_anchor_kv(anchor_k_pre.float()).to(dtype)
                anchor_v_prope = apply_anchor_kv(anchor_v_pre.float()).to(dtype)
                anchor_k = torch.cat((anchor_k_main, anchor_k_prope), 0)
                anchor_v = torch.cat((anchor_v_pre, anchor_v_prope), 0)

        if reusable_recent is not None:
            # Positioning of the recent-native suffix is independent of the
            # old-history low/full route.  Transfer those tensors into the
            # replacement cache instead of retaining a second raw sidecar or
            # decoding them again.
            recent_k, recent_v = reusable_recent
        else:
            if capture_raw:
                recent_k_pre = raw_tail["k"].to(dtype)
                recent_v_pre = raw_tail["v"].to(dtype)
            elif decoded_recent_k_pre is not None:
                recent_k_pre = decoded_recent_k_pre
                recent_v_pre = decoded_recent_v_pre
            else:
                # All-recent is uncommon but valid; decode then needs rank512
                # even though the old low-D branch did not consume it above.
                if old_tokens == 0:
                    kz = _profile_call(
                        "direct_rank512", layer_idx, _rank512, kp, device
                    )
                    vz = _profile_call(
                        "direct_rank512", layer_idx, _rank512, vp, device
                    )
                recent_k_pre = _profile_call(
                    "direct_recent_decode", layer_idx,
                    _decode_native_rows,
                    kz, kp, runtime["operators"]["k"], old_tokens, dtype
                )
                recent_v_pre = _profile_call(
                    "direct_recent_decode", layer_idx,
                    _decode_native_rows,
                    vz, vp, runtime["operators"]["v"], old_tokens, dtype
                )
            recent_freqs = _slice_freqs(
                full_history_freqs, old_tokens, None
            )
            recent_viewmats = layer_cache["vision_viewmats"][:, old_tokens:]
            recent_Ks = None if vision_Ks is None else vision_Ks[:, old_tokens:]
            triton_native_pack = (
                _scoped_flag(
                    "HY_DIRECT_CODEC_TRITON_NATIVE_POSITION",
                    "HY_DIRECT_CODEC_TRITON_POSITION",
                )
                and _flag("HY_DIRECT_CODEC_TRITON_NATIVE_POSITION_PACK", "1")
                and isinstance(recent_freqs, tuple)
                and recent_k_pre.is_cuda
            )
            if triton_native_pack:
                recent_matrix = (
                    full_prope_matrix[:, old_tokens:]
                    if full_prope_matrix is not None
                    else _prope_matrix_runtime(recent_viewmats, recent_Ks, device)
                )
                if _flag("HY_DIRECT_CODEC_TRITON_NATIVE_POSITION_EXACT", "1"):
                    from .codec_native_position_triton import native_prope_pack

                    recent_k_main = _native_rope_runtime(
                        recent_k_pre, recent_freqs
                    )
                    recent_k, recent_v = _profile_call(
                        "direct_recent_position_pack", layer_idx,
                        native_prope_pack,
                        recent_k_main, recent_k_pre, recent_v_pre, recent_matrix,
                    )
                else:
                    from .codec_native_position_triton import native_position_pack

                    recent_cos, recent_sin = _rope_device_pair_runtime(
                        recent_freqs[0], recent_freqs[1], device
                    )
                    recent_k, recent_v = _profile_call(
                        "direct_recent_position_pack", layer_idx,
                        native_position_pack,
                        recent_k_pre, recent_v_pre,
                        recent_cos, recent_sin, recent_matrix,
                    )
            elif (
                _scoped_flag(
                    "HY_DIRECT_CODEC_NATIVE_FUSED_PROPE",
                    "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
                )
                and recent_k_pre.is_cuda
            ):
                recent_k_main = _native_rope_runtime(recent_k_pre, recent_freqs)
                recent_k_prope, recent_v_prope = _profile_call(
                    "direct_recent_prope", layer_idx,
                    _prope_kv_pair_runtime,
                    recent_k_pre, recent_v_pre,
                    recent_viewmats, recent_Ks, device,
                    None if full_prope_matrix is None else
                    full_prope_matrix[:, old_tokens:],
                    "HY_DIRECT_CODEC_TRITON_NATIVE_POSITION",
                )
                recent_k = torch.cat((recent_k_main, recent_k_prope), 0)
                recent_v = torch.cat((recent_v_pre, recent_v_prope), 0)
            else:
                recent_k_main = _native_rope_runtime(recent_k_pre, recent_freqs)
                _, apply_recent_kv, _ = _prope_fns(
                    128, recent_viewmats, recent_Ks, device
                )
                recent_k_prope = apply_recent_kv(recent_k_pre.float()).to(dtype)
                recent_v_prope = apply_recent_kv(recent_v_pre.float()).to(dtype)
                recent_k = torch.cat((recent_k_main, recent_k_prope), 0)
                recent_v = torch.cat((recent_v_pre, recent_v_prope), 0)
        positioned = {
            "key": cache_key,
            "recent_k": recent_k,
            "recent_v": recent_v,
        }
        if grouped:
            positioned["old_groups"] = old_groups
        else:
            positioned["old_k"] = old_k
            positioned["old_v"] = old_v
        if formal_anchor:
            positioned["anchor_k"] = anchor_k
            positioned["anchor_v"] = anchor_v
        if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1"):
            layer_cache["_direct_codec_positioned_history"] = positioned
            if capture_raw:
                # Positioned K/V now own everything needed by all subsequent
                # denoise reads.  Drop the unpositioned native copy so the two
                # representations never remain resident together.
                layer_cache.pop("_direct_codec_recent_native_pre", None)

    if grouped:
        old_groups = positioned.get("old_groups")
        old_k = old_v = None
    else:
        old_k, old_v = positioned["old_k"], positioned["old_v"]
    recent_k, recent_v = positioned["recent_k"], positioned["recent_v"]
    anchor_k = positioned.get("anchor_k") if formal_anchor else None
    anchor_v = positioned.get("anchor_v") if formal_anchor else None

    # Recent and formal-anchor rows are already in the native position/output
    # basis used by text/current.  The logical four segments therefore still
    # require only one native FA plus one old-low FA; `_combine` supplies the
    # exact global denominator for the two physical calls.
    if formal_anchor:
        if (
            _triton_data_flag("HY_DIRECT_CODEC_TRITON_NATIVE_CAT_ONLY")
            and query_bhsd.is_cuda
        ):
            from .codec_native_position_triton import native_kv_cat
            native_k, native_v = _profile_call(
                "direct_native_cat", layer_idx, native_kv_cat,
                encoder_key_bhsd, encoder_value_bhsd,
                anchor_k, anchor_v, recent_k, recent_v,
                current_key_bhsd, current_value_bhsd,
            )
        else:
            native_k_parts = [encoder_key_bhsd]
            native_v_parts = [encoder_value_bhsd]
            if anchor_k is not None:
                native_k_parts.append(anchor_k)
                native_v_parts.append(anchor_v)
            native_k_parts.extend((recent_k, current_key_bhsd))
            native_v_parts.extend((recent_v, current_value_bhsd))
            native_k = _profile_call(
                "direct_native_cat", layer_idx, torch.cat, native_k_parts, 2
            )
            native_v = _profile_call(
                "direct_native_cat", layer_idx, torch.cat, native_v_parts, 2
            )
    else:
        # Keep the original no-anchor cat order and call site untouched.
        native_k = torch.cat((encoder_key_bhsd, recent_k, current_key_bhsd), 2)
        native_v = torch.cat((encoder_value_bhsd, recent_v, current_value_bhsd), 2)
    native_stream = None
    if (
        low_tokens > 0
        and _flag("HY_DIRECT_CODEC_PARALLEL_SEGMENTS")
        and not position_profile.enabled()
        and not codec_native_history_adapter.enabled()
    ):
        caller_stream = torch.cuda.current_stream(device)
        native_stream = _parallel_native_stream(device)
        native_stream.wait_stream(caller_stream)
        with torch.cuda.stream(native_stream):
            native, native_lse = _flash_lse(
                query_bhsd, native_k, native_v,
                scale=1.0 / math.sqrt(128.0),
            )
    else:
        native, native_lse = _profile_call(
            "direct_native_flash", layer_idx, _flash_lse,
            query_bhsd, native_k, native_v, scale=1.0 / math.sqrt(128.0)
        )
    if low_tokens == 0:
        combined = native
    else:
        if grouped:
            if (
                codec_native_history_adapter.enabled()
                and codec_native_history_adapter.get_layer(layer_idx) is not None
            ):
                raise RuntimeError(
                    "history adapter is not supported by the opt-in grouped "
                    "dictionary-lowkv runtime"
                )
            if rebuild_stream is not None:
                caller_stream = torch.cuda.current_stream(device)
                caller_stream.wait_stream(rebuild_stream)
                _record_low_history_stream(
                    old_groups, None, None, caller_stream
                )
            fused_grouped_combine = (
                native_stream is None
                and _triton_data_flag(
                    "HY_DIRECT_CODEC_TRITON_GROUPED_COMBINE_ONLY"
                )
                and _flag("HY_DIRECT_CODEC_TRITON_GROUPED_COMBINE", "1")
                and native.is_cuda
            )
            old_full, old_lse = _profile_call(
                "direct_grouped_flash_lift", layer_idx,
                _attend_grouped_low_history,
                runtime,
                q_pre_bshd,
                current_freqs_cis,
                current_viewmats,
                current_Ks,
                old_groups,
                dtype,
                native if fused_grouped_combine else None,
                native_lse if fused_grouped_combine else None,
            )
            if fused_grouped_combine:
                combined = old_full.to(dtype)
                combined = combined.transpose(1, 2).reshape(
                    combined.shape[0], combined.shape[2], -1
                )
                return torch.chunk(combined, 2, 0)
            if native_stream is not None:
                torch.cuda.current_stream(device).wait_stream(native_stream)
            combined = _profile_call(
                "direct_lse_combine", layer_idx, _combine,
                native, native_lse, old_full, old_lse,
            ).to(dtype)
            combined = combined.transpose(1, 2).reshape(
                combined.shape[0], combined.shape[2], -1
            )
            return torch.chunk(combined, 2, 0)
        indices = state["main_indices"]
        q_pre = q_pre_bshd.transpose(1, 2)
        triton_query_pack = (
            runtime["main_k_mode"] == "learned_rope"
            and _scoped_flag(
                "HY_DIRECT_CODEC_TRITON_LOW_QUERY_POSITION",
                "HY_DIRECT_CODEC_TRITON_POSITION",
            )
            and _flag("HY_DIRECT_CODEC_TRITON_QUERY_PACK", "1")
            and isinstance(current_freqs_cis, tuple)
            and q_pre.is_cuda
        )
        fused_query_project = (
            triton_query_pack
            and _flag("HY_DIRECT_CODEC_TRITON_FUSED_QUERY_PROJECT")
        )
        if fused_query_project:
            from .codec_native_position_triton import query_project_position_pack

            query_cos, query_sin = _rope_device_pair_runtime(
                current_freqs_cis[0], current_freqs_cis[1], device
            )
            query_matrix = _prope_matrix_runtime(
                current_viewmats, current_Ks, device, role="q"
            )
            q_low = _profile_call(
                "direct_uniform_q_project_position_pack", layer_idx,
                query_project_position_pack,
                q_pre, state["q_main_reduce"], state["q_prope_mix"],
                query_cos, query_sin, indices, query_matrix,
                state["log_residual_scale_main"].exp(),
                state["log_residual_scale_prope"].exp(),
            )
        elif runtime["main_k_mode"] == "learned_rope":
            q_main = _profile_call(
                "direct_uniform_q_main", layer_idx, torch.einsum,
                "bhnr,hdr->bhnd", q_pre, state["q_main_reduce"]
            )
            if not triton_query_pack:
                q_main = _profile_call(
                    "direct_uniform_q_rope", layer_idx,
                    _selected_rope_runtime,
                    q_main.to(dtype), current_freqs_cis, indices
                )
        else:
            q_main = torch.gather(
                query_bhsd[:batch], -1,
                indices[None, :, None].expand(
                    batch, -1, query_bhsd.shape[2], -1
                ),
            )
        if not fused_query_project:
            q_prope = _profile_call(
                "direct_uniform_q_prope_reduce", layer_idx,
                _quartet_reduce_runtime, q_pre, state["q_prope_mix"]
            )
        if triton_query_pack and not fused_query_project:
            from .codec_native_position_triton import query_position_pack

            query_cos, query_sin = _rope_device_pair_runtime(
                current_freqs_cis[0], current_freqs_cis[1], device
            )
            query_matrix = _prope_matrix_runtime(
                current_viewmats, current_Ks, device, role="q"
            )
            q_low = _profile_call(
                "direct_uniform_q_position_pack", layer_idx,
                query_position_pack,
                q_main, q_prope, query_cos, query_sin, indices, query_matrix,
                state["log_residual_scale_main"].exp(),
                state["log_residual_scale_prope"].exp(),
            )
        elif not fused_query_project:
            if (
                _scoped_flag(
                    "HY_DIRECT_CODEC_LOW_QUERY_FUSED_PROPE",
                    "HY_DIRECT_CODEC_STREAM_FUSED_PROPE",
                )
                and q_prope.is_cuda
            ):
                q_prope = _profile_call(
                    "direct_uniform_q_prope", layer_idx,
                    _prope_one_runtime,
                    q_prope, current_viewmats, current_Ks, device,
                    _prope_matrix_runtime(current_viewmats, current_Ks, device, role="q"),
                    "HY_DIRECT_CODEC_TRITON_LOW_QUERY_POSITION",
                ).to(dtype)
            else:
                apply_q, _, _ = _prope_fns(
                    q_prope.shape[-1], current_viewmats, current_Ks, device
                )
                q_prope = apply_q(q_prope.float()).to(dtype)
            q_main = q_main * state["log_residual_scale_main"].exp()[None, :, None, None]
            q_prope = q_prope * state["log_residual_scale_prope"].exp()[None, :, None, None]
            q_low = torch.cat((q_main, q_prope), 0).to(dtype)
        history_adapter = None
        if codec_native_history_adapter.enabled():
            # The public ``attend`` entry has already initialized the bank.
            history_adapter = codec_native_history_adapter.get_layer(layer_idx)
        if history_adapter is None:
            # Frozen production path: preserve the established operation order
            # and the surrounding @torch.no_grad behavior exactly.
            if rebuild_stream is not None:
                caller_stream = torch.cuda.current_stream(device)
                caller_stream.wait_stream(rebuild_stream)
                _record_low_history_stream(
                    None, old_k, old_v, caller_stream
                )
            old_low, old_lse = _profile_call(
                "direct_uniform_flash", layer_idx, _flash_lse,
                q_low, old_k, old_v, scale=1.0 / math.sqrt(128.0)
            )
            if native_stream is not None:
                torch.cuda.current_stream(device).wait_stream(native_stream)
            use_fused_lift = (
                _flag("HY_DIRECT_CODEC_TRITON_FUSED_LIFT_COMBINE")
                and _flag("HY_DIRECT_CODEC_TRITON_POSITION")
                and native.is_cuda
            )
            if use_fused_lift:
                from .codec_native_position_triton import lift_lse_combine

                combined = _profile_call(
                    "direct_uniform_lift_combine", layer_idx,
                    lift_lse_combine,
                    native, native_lse, old_low,
                    state["v_main_lift"], state["v_prope_mix"], old_lse,
                )
            else:
                main_low, prope_low = old_low[:batch], old_low[batch:]
                main_full = _profile_call(
                    "direct_uniform_main_lift", layer_idx, torch.einsum,
                    "bhnd,hdo->bhno", main_low, state["v_main_lift"]
                )
                # As in the original runtime, keep the PRoPE result transformed
                # here; worldplay_1_5_transformer performs the sole canonical
                # restore.
                prope_full = _profile_call(
                    "direct_uniform_prope_lift", layer_idx,
                    _quartet_lift_runtime, prope_low, state["v_prope_mix"]
                )
            if (
                not use_fused_lift
                and
                _triton_data_flag("HY_DIRECT_CODEC_TRITON_SPLIT_COMBINE_ONLY")
                and _flag("HY_DIRECT_CODEC_TRITON_SPLIT_COMBINE", "1")
                and native.is_cuda
            ):
                from .codec_native_position_triton import lse_combine_split

                combined = _profile_call(
                    "direct_lse_combine", layer_idx, lse_combine_split,
                    native, native_lse, main_full, prope_full, old_lse,
                )
            elif not use_fused_lift:
                old_full = torch.cat((main_full, prope_full), 0)
                combined = _profile_call(
                    "direct_lse_combine", layer_idx, _combine,
                    native, native_lse, old_full, old_lse,
                )
        else:
            # The outer runtime remains @torch.no_grad.  Only an explicit
            # training switch opens this narrow history branch to autograd;
            # text/current K/V and their native FlashAttention stay frozen.
            grad_context = (
                torch.enable_grad()
                if codec_native_history_adapter.autograd_enabled()
                else torch.no_grad()
            )
            with grad_context:
                if (
                    history_adapter.query_a is None
                    and history_adapter.query_main_a is None
                ):
                    adapted_q_low = q_low
                else:
                    # Query residuals act before position transforms.  Their
                    # linear map can therefore be composed into the low query
                    # read offline; placing it after RoPE/PRoPE would generally
                    # prevent an exact fold because the transforms need not
                    # commute with an arbitrary learned map.
                    if runtime["main_k_mode"] == "learned_rope":
                        adapted_q_main = torch.einsum(
                            "bhnr,hdr->bhnd", q_pre, state["q_main_reduce"]
                        )
                    else:
                        if history_adapter.query_main_a is not None:
                            raise RuntimeError(
                                "foldable query_read requires learned_rope Main mode"
                            )
                        adapted_q_main = torch.gather(
                            q_pre,
                            -1,
                            indices[None, :, None].expand(
                                batch, -1, q_pre.shape[2], -1
                            ),
                        )
                    query_main_delta, query_prope_delta = (
                        history_adapter.query_read_delta(q_pre)
                    )
                    adapted_q_main = adapted_q_main + query_main_delta
                    adapted_q_main = history_adapter.adapt_query(adapted_q_main)
                    adapted_q_main = _selected_rope_runtime(
                        adapted_q_main.to(dtype), current_freqs_cis, indices
                    )
                    adapted_q_prope = _quartet_reduce_runtime(
                        q_pre, state["q_prope_mix"]
                    )
                    adapted_q_prope = adapted_q_prope + query_prope_delta
                    adapted_q_prope = history_adapter.adapt_query(
                        adapted_q_prope
                    )
                    apply_adapted_q, _, _ = _prope_fns(
                        adapted_q_prope.shape[-1],
                        current_viewmats,
                        current_Ks,
                        device,
                    )
                    adapted_q_prope = apply_adapted_q(
                        adapted_q_prope.float()
                    ).to(dtype)
                    adapted_q_main = adapted_q_main * state[
                        "log_residual_scale_main"
                    ].exp()[None, :, None, None]
                    adapted_q_prope = adapted_q_prope * state[
                        "log_residual_scale_prope"
                    ].exp()[None, :, None, None]
                    adapted_q_low = torch.cat(
                        (adapted_q_main, adapted_q_prope), 0
                    ).to(dtype)
                # Adapt only the already-existing D-low coordinates.  This
                # leaves history QK/AV width and FlashAttention call count
                # unchanged while allowing the same codec latent readout to
                # move toward the same-state formal Teacher.
                adapted_old_k = old_k
                adapted_old_v = old_v
                if history_adapter.latent_k_a is not None:
                    # Re-read the immutable formal z512 payload during
                    # training so the learned residual may use latent
                    # directions outside the original D-low row-space.  The
                    # baseline positioned history remains cached; only this
                    # small rank-r delta changes between optimizer steps.
                    latent_kz = _rank512(kp, device)
                    latent_vz = _rank512(vp, device)
                    latent_freqs = layer_cache["vision_freqs_cis"]
                    latent_viewmats = layer_cache["vision_viewmats"]
                    latent_Ks = layer_cache.get("vision_Ks")
                    latent_prope_matrix = None
                    if formal_anchor:
                        (
                            latent_low_indices,
                            _latent_anchor_indices,
                            _latent_anchor_frames,
                        ) = _formal_anchor_partition(
                            layer_cache, kp, recent_frames, anchor_stride,
                            anchor_sink_frames, device,
                        )
                        latent_kz = latent_kz.index_select(1, latent_low_indices)
                        latent_vz = latent_vz.index_select(1, latent_low_indices)
                        latent_freqs = _index_freqs(
                            latent_freqs, latent_low_indices
                        )
                        latent_viewmats = latent_viewmats.index_select(
                            1, latent_low_indices
                        )
                        if latent_Ks is not None:
                            latent_Ks = latent_Ks.index_select(
                                1, latent_low_indices
                            )
                    else:
                        latent_kz = latent_kz[:, :old_tokens]
                        latent_vz = latent_vz[:, :old_tokens]
                        latent_freqs = _slice_freqs(
                            latent_freqs, 0, old_tokens
                        )
                        latent_viewmats = latent_viewmats[:, :old_tokens]
                        if latent_Ks is not None:
                            latent_Ks = latent_Ks[:, :old_tokens]
                    if isinstance(latent_freqs, tuple) and query_bhsd.is_cuda:
                        latent_freqs = _rope_device_pair_runtime(
                            latent_freqs[0], latent_freqs[1], device
                        )
                    if query_bhsd.is_cuda:
                        latent_prope_matrix = _prope_matrix_runtime(
                            latent_viewmats, latent_Ks, device
                        )
                    latent_delta_k, latent_delta_v = _position_latent_read_delta(
                        history_adapter,
                        state,
                        latent_kz,
                        latent_vz,
                        latent_freqs,
                        latent_viewmats,
                        latent_Ks,
                        dtype,
                        latent_prope_matrix,
                    )
                    adapted_old_k = adapted_old_k + latent_delta_k
                    adapted_old_v = adapted_old_v + latent_delta_v
                if history_adapter.center_k_main_a is not None:
                    # The semantic table/IDs are the second immutable half of
                    # the formal C64 codec payload.  Training re-reads them
                    # only to learn a rank-r delta; export folds that delta
                    # into the existing cache-write center projections.
                    k_centers = kp.get("semantic_centers")
                    v_centers = vp.get("semantic_centers")
                    if not isinstance(k_centers, torch.Tensor) or not isinstance(
                        v_centers, torch.Tensor
                    ):
                        # Same-state training deliberately moves the sole C64
                        # table reference out of the Student dict and into its
                        # metadata-only Teacher sidecar.  Reuse that exact
                        # reference; never retain a second center table.
                        from .codec_native_causal_window import sidecar_key

                        sidecars = layer_cache.get(sidecar_key())
                        if isinstance(sidecars, dict):
                            k_meta = sidecars.get("k", {})
                            v_meta = sidecars.get("v", {})
                            k_centers = k_meta.get("semantic_centers")
                            v_centers = v_meta.get("semantic_centers")
                    if not isinstance(k_centers, torch.Tensor) or not isinstance(
                        v_centers, torch.Tensor
                    ):
                        raise RuntimeError(
                            "center-read adapter requires formal C64 center tables"
                        )
                    center_k_ids = kp["semantic_ids"].to(
                        device=device, dtype=torch.long
                    )
                    center_v_ids = vp["semantic_ids"].to(
                        device=device, dtype=torch.long
                    )
                    center_freqs = layer_cache["vision_freqs_cis"]
                    center_viewmats = layer_cache["vision_viewmats"]
                    center_Ks = layer_cache.get("vision_Ks")
                    center_prope_matrix = None
                    if formal_anchor:
                        (
                            center_low_indices,
                            _center_anchor_indices,
                            _center_anchor_frames,
                        ) = _formal_anchor_partition(
                            layer_cache, kp, recent_frames, anchor_stride,
                            anchor_sink_frames, device,
                        )
                        center_k_ids = center_k_ids.index_select(
                            1, center_low_indices
                        )
                        center_v_ids = center_v_ids.index_select(
                            1, center_low_indices
                        )
                        center_freqs = _index_freqs(
                            center_freqs, center_low_indices
                        )
                        center_viewmats = center_viewmats.index_select(
                            1, center_low_indices
                        )
                        if center_Ks is not None:
                            center_Ks = center_Ks.index_select(
                                1, center_low_indices
                            )
                    else:
                        center_k_ids = center_k_ids[:, :old_tokens]
                        center_v_ids = center_v_ids[:, :old_tokens]
                        center_freqs = _slice_freqs(
                            center_freqs, 0, old_tokens
                        )
                        center_viewmats = center_viewmats[:, :old_tokens]
                        if center_Ks is not None:
                            center_Ks = center_Ks[:, :old_tokens]
                    if isinstance(center_freqs, tuple) and query_bhsd.is_cuda:
                        center_freqs = _rope_device_pair_runtime(
                            center_freqs[0], center_freqs[1], device
                        )
                    if query_bhsd.is_cuda:
                        center_prope_matrix = _prope_matrix_runtime(
                            center_viewmats, center_Ks, device
                        )
                    center_delta_k, center_delta_v = _position_center_read_delta(
                        history_adapter,
                        state,
                        k_centers,
                        v_centers,
                        center_k_ids,
                        center_v_ids,
                        center_freqs,
                        center_viewmats,
                        center_Ks,
                        dtype,
                        center_prope_matrix,
                    )
                    adapted_old_k = adapted_old_k + center_delta_k
                    adapted_old_v = adapted_old_v + center_delta_v
                adapted_old_k = history_adapter.adapt_history_key(adapted_old_k)
                adapted_old_v = history_adapter.adapt_history_value(adapted_old_v)
                # A routing-only run owns a small sampled autograd graph here.
                # The production all-query FA is still executed to generate
                # the causal rollout, but detached: no full-DiT prediction
                # graph is needed to train the query consumer.
                routing_active = codec_native_routing_distill.record_student(
                    int(layer_idx),
                    adapted_q_low,
                    adapted_old_k,
                    adapted_old_v,
                    native_lse,
                    layer_cache,
                    flash_lse_fn=_flash_lse,
                )
                if routing_active:
                    with torch.no_grad():
                        old_low, old_lse = _flash_lse(
                            adapted_q_low.detach(),
                            adapted_old_k.detach(),
                            adapted_old_v.detach(),
                            scale=1.0 / math.sqrt(128.0),
                        )
                else:
                    old_low, old_lse = _flash_lse(
                        adapted_q_low,
                        adapted_old_k,
                        adapted_old_v,
                        scale=1.0 / math.sqrt(128.0),
                    )
                main_low, prope_low = old_low[:batch], old_low[batch:]
                main_full = torch.einsum(
                    "bhnd,hdo->bhno", main_low, state["v_main_lift"]
                )
                prope_full = _quartet_lift_runtime(
                    prope_low, state["v_prope_mix"]
                )
                lift_main_delta, lift_prope_delta = history_adapter.lift_delta(
                    main_low, prope_low
                )
                main_full = main_full + lift_main_delta
                prope_full = prope_full + lift_prope_delta
                old_full = torch.cat((main_full, prope_full), 0)
                old_full = history_adapter.adapt_lifted_output(old_full)
                old_lse = history_adapter.adapt_history_lse(old_lse)
                adapted_combined = _combine(
                    native, native_lse, old_full, old_lse
                )
                # Return from inside enable_grad: reshaping outside this scope
                # would detach the adapter graph under the outer no_grad guard.
                adapted_combined = adapted_combined.to(dtype).transpose(1, 2)
                adapted_combined = adapted_combined.reshape(
                    adapted_combined.shape[0],
                    adapted_combined.shape[1],
                    -1,
                )
                return torch.chunk(adapted_combined, 2, 0)

    combined = combined.to(dtype).transpose(1, 2).reshape(
        combined.shape[0], combined.shape[2], -1
    )
    return torch.chunk(combined, 2, 0)


@torch.no_grad()
def _attend_dictionary_lowkv(layer_idx, q_pre_bshd, query_bhsd, current_key_bhsd,
                             current_value_bhsd, encoder_key_bhsd, encoder_value_bhsd,
                             layer_cache, current_freqs_cis, current_viewmats, current_Ks,
                             runtime):
    if runtime.get("recent_native_frames", 0) > 0:
        return _attend_dictionary_lowkv_recent(
            layer_idx, q_pre_bshd, query_bhsd, current_key_bhsd,
            current_value_bhsd, encoder_key_bhsd, encoder_value_bhsd,
            layer_cache, current_freqs_cis, current_viewmats, current_Ks, runtime,
        )
    device, dtype = query_bhsd.device, query_bhsd.dtype
    state = runtime["attention"]; batch = q_pre_bshd.shape[0]
    kp, vp = layer_cache["k_vision"], layer_cache["v_vision"]
    indices = state["main_indices"]
    q_pre = q_pre_bshd.transpose(1, 2)
    if runtime["main_k_mode"] == "learned_rope":
        q_main = torch.einsum("bhnr,hdr->bhnd", q_pre, state["q_main_reduce"])
        q_main = _selected_rope_runtime(q_main.to(dtype), current_freqs_cis, indices)
    else:
        q_main = torch.gather(query_bhsd[:batch], -1,
            indices[None, :, None].expand(batch, -1, query_bhsd.shape[2], -1))
    q_prope = _quartet_reduce_runtime(q_pre, state["q_prope_mix"])
    apply_q, _, _ = _prope_fns(q_prope.shape[-1], current_viewmats, current_Ks, device)
    q_prope = apply_q(q_prope.float()).to(dtype)
    q_main = q_main * state["log_residual_scale_main"].exp()[None, :, None, None]
    q_prope = q_prope * state["log_residual_scale_prope"].exp()[None, :, None, None]
    qh = torch.cat((q_main, q_prope), 0).to(dtype)
    cache_key = (str(device), dtype, int(runtime["shape"][1]), int(runtime["shape"][3]))
    positioned = (
        layer_cache.get("_direct_codec_positioned_history")
        if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1") else None
    )
    if not _valid_positioned_cache(positioned, cache_key, device, dtype, runtime["shape"][1], runtime["shape"][3]):
        kz, vz = _rank512(kp, device), _rank512(vp, device)
        # No 2048-D reconstruction: every residual read is 512 -> per-head D.
        # Do not rely on the outer generation autocast context: the formal
        # INT4 unpack is FP32 while deployment read maps are resident BF16.
        km, kq = _dual_low_read(
            kz, state["k_main_read"], state["k_prope_read"], state.get("k_read_joined")
        )
        vm, vq = _dual_low_read(
            vz, state["v_main_read"], state["v_prope_read"], state.get("v_read_joined")
        )
        km = km + _assigned_low(kp["centre_main"].to(device), kp["semantic_ids"].to(device).long())
        kq = kq + _assigned_low(kp["centre_prope"].to(device), kp["semantic_ids"].to(device).long())
        vm = vm + _assigned_low(vp["centre_main"].to(device), vp["semantic_ids"].to(device).long())
        vq = vq + _assigned_low(vp["centre_prope"].to(device), vp["semantic_ids"].to(device).long())
        km = _selected_rope_runtime(km.to(dtype), layer_cache["vision_freqs_cis"], indices)
        _, apply_k, _ = _prope_fns(
            kq.shape[-1], layer_cache["vision_viewmats"],
            layer_cache.get("vision_Ks"), device,
        )
        _, apply_v, _ = _prope_fns(
            vq.shape[-1], layer_cache["vision_viewmats"],
            layer_cache.get("vision_Ks"), device,
        )
        kq = apply_k(kq.float()).to(dtype)
        vq = apply_v(vq.float()).to(dtype)
        kh = torch.cat((km, kq), 0); vh = torch.cat((vm.to(dtype), vq), 0)
        if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1"):
            layer_cache["_direct_codec_positioned_history"] = {"key": cache_key, "k": kh, "v": vh}
    else:
        kh, vh = positioned["k"], positioned["v"]
    history_low, history_lse = _flash_lse(qh, kh, vh, scale=1.0 / math.sqrt(128.0))

    native_k = torch.cat((encoder_key_bhsd, current_key_bhsd), 2)
    native_v = torch.cat((encoder_value_bhsd, current_value_bhsd), 2)
    native, native_lse = _flash_lse(query_bhsd, native_k, native_v, scale=1.0 / math.sqrt(128.0))
    main_low, prope_low = history_low[:batch], history_low[batch:]
    main_full = torch.einsum("bhnd,hdo->bhno", main_low, state["v_main_lift"])
    # Keep the PRoPE branch in its transformed basis until the caller's one
    # canonical output restore.  The structured quartet lift has the form
    # L=A⊗I_4, while PRoPE applies T=I⊗P independently to every quartet.
    # Hence TL=(A⊗P)=LT: lifting here and restoring once in
    # worldplay_1_5_transformer is exactly equivalent to the trainer's
    # restore-then-lift semantics.  Restoring both here and in the caller
    # would apply P twice and also add two redundant per-token transforms.
    prope_full = _quartet_lift_runtime(prope_low, state["v_prope_mix"])
    history_full = torch.cat((main_full, prope_full), 0)
    combined = _combine(native, native_lse, history_full, history_lse).to(dtype)
    combined = combined.transpose(1, 2).reshape(combined.shape[0], combined.shape[2], -1)
    return torch.chunk(combined, 2, 0)


@torch.no_grad()
def attend(
    layer_idx: int,
    q_pre_bshd: torch.Tensor,
    query_bhsd: torch.Tensor,
    current_key_bhsd: torch.Tensor,
    current_value_bhsd: torch.Tensor,
    encoder_key_bhsd: torch.Tensor,
    encoder_value_bhsd: torch.Tensor,
    layer_cache: dict,
    current_freqs_cis,
    current_viewmats,
    current_Ks,
):
    global _FLASH_BACKEND_LAYER_IDX
    _FLASH_BACKEND_LAYER_IDX = int(layer_idx)
    if not layer_cache.get("direct_codec_latent"):
        raise RuntimeError("selected direct layer is missing a direct latent payload")
    if query_bhsd.shape[0] != 2 * q_pre_bshd.shape[0]:
        raise RuntimeError("direct codec attention expects stacked main/PRoPE batches")
    device, dtype = query_bhsd.device, query_bhsd.dtype
    if codec_native_history_adapter.enabled():
        ensure_history_adapter_runtime(device, dtype)
    runtime = _device_layer(layer_idx, device)
    if runtime["payload_mode"] == "dictionary_lowkv":
        return _attend_dictionary_lowkv(
            layer_idx, q_pre_bshd, query_bhsd, current_key_bhsd, current_value_bhsd,
            encoder_key_bhsd, encoder_value_bhsd, layer_cache, current_freqs_cis,
            current_viewmats, current_Ks, runtime,
        )
    state = runtime["attention"]
    _, k_dim, _, v_dim = runtime["shape"]
    batch = q_pre_bshd.shape[0]

    q_pre = q_pre_bshd.transpose(1, 2).contiguous()
    q_low = torch.einsum("bhsd,hkd->bhsk", q_pre.to(torch.bfloat16), state["q_weight"])
    q_main = _reduced_rope(q_low, current_freqs_cis)
    apply_q, _, _ = _prope_fns(k_dim, current_viewmats, current_Ks, device)
    q_prope = apply_q(q_low.float()).to(dtype)
    q_history = torch.cat((q_main, q_prope), dim=0)
    scales = torch.cat((
        state["logit_scale_main"].float().exp()[None].expand(batch, -1),
        state["logit_scale_prope"].float().exp()[None].expand(batch, -1),
    ), dim=0).reshape(2 * batch, 16, 1, 1)
    q_history = (q_history.float() * scales).to(dtype)

    positioned = None
    if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1"):
        positioned = layer_cache.get("_direct_codec_positioned_history")
        if positioned is not None and (
            positioned[0].device != device or positioned[0].dtype != dtype
        ):
            positioned = None
    if positioned is None:
        history_k = _decode_state(layer_cache["k_vision"], dtype, device, runtime, "k")
        history_v = _decode_state(layer_cache["v_vision"], dtype, device, runtime, "v")
        history_k_main = _reduced_rope(history_k, layer_cache["vision_freqs_cis"])
        _, apply_history_k, _ = _prope_fns(
            k_dim, layer_cache["vision_viewmats"], layer_cache.get("vision_Ks"), device
        )
        _, apply_history_v, _ = _prope_fns(
            v_dim, layer_cache["vision_viewmats"], layer_cache.get("vision_Ks"), device
        )
        history_k_prope = apply_history_k(history_k.float()).to(dtype)
        history_v_prope = apply_history_v(history_v.float()).to(dtype)
        history_key = torch.cat((history_k_main, history_k_prope), dim=0)
        history_value = torch.cat((history_v, history_v_prope), dim=0)
        if _flag("HY_DIRECT_CODEC_LATENT_READ_CACHE", "1"):
            layer_cache["_direct_codec_positioned_history"] = (history_key, history_value)
    else:
        history_key, history_value = positioned

    history_out, history_lse = _flash_lse(q_history, history_key, history_value)
    debug_mass = layer_cache.get("_direct_codec_segment_log_mass")
    if runtime["compress_current"]:
        q_current = torch.einsum(
            "bhsd,hkd->bhsk", query_bhsd.to(torch.bfloat16), state["current_q_weight"]
        )
        k_current = torch.einsum(
            "bhsd,hkd->bhsk", current_key_bhsd.to(torch.bfloat16), state["current_k_weight"]
        )
        v_current = torch.einsum(
            "bhsd,hkd->bhsk", current_value_bhsd.to(torch.bfloat16), state["current_v_weight"]
        )
        text_out, text_lse = _flash_lse(query_bhsd, encoder_key_bhsd, encoder_value_bhsd)
        current_out, current_lse = _flash_lse(q_current, k_current, v_current)
        if debug_mass is not None:
            text_mass, history_mass, current_mass = (float(value) for value in debug_mass)
            text_lse = text_lse + text_mass
            current_lse = current_lse + current_mass
            history_lse = history_lse + history_mass
        native_lse = torch.logaddexp(text_lse, current_lse)
    elif debug_mass is None:
        native_key = torch.cat((encoder_key_bhsd, current_key_bhsd), dim=2)
        native_value = torch.cat((encoder_value_bhsd, current_value_bhsd), dim=2)
        native_out, native_lse = _flash_lse(query_bhsd, native_key, native_value)
    else:
        # Offline mixed gates uniformly subsample history/current tokens.  The
        # production path never enters this branch because it has all tokens;
        # this only recreates each segment's population measure for a strict
        # runtime-vs-training equivalence test.
        text_mass, history_mass, current_mass = (float(value) for value in debug_mass)
        text_out, text_lse = _flash_lse(query_bhsd, encoder_key_bhsd, encoder_value_bhsd)
        current_out, current_lse = _flash_lse(query_bhsd, current_key_bhsd, current_value_bhsd)
        text_lse = text_lse + text_mass
        current_lse = current_lse + current_mass
        native_out = _combine(text_out, text_lse, current_out, current_lse)
        native_lse = torch.logaddexp(text_lse, current_lse)
        history_lse = history_lse + history_mass

    main_low, prope_low = history_out[:batch], history_out[batch:]
    _, _, apply_history_o = _prope_fns(v_dim, current_viewmats, current_Ks, device)
    prope_low = apply_history_o(prope_low.float()).to(torch.bfloat16)
    main_history = torch.einsum("bhsd,hod->bhso", main_low, state["o_main"])
    prope_history = torch.einsum("bhsd,hod->bhso", prope_low, state["o_prope"])

    # Native PRoPE output is still in its transformed basis; restore it before
    # combining with the already-restored learned history branch.
    _, _, apply_o_native = _prope_fns(128, current_viewmats, current_Ks, device)
    if runtime["compress_current"]:
        current_main_low, current_prope_low = current_out[:batch], current_out[batch:]
        current_main = torch.einsum("bhsd,hod->bhso", current_main_low, state["o_main"])
        current_prope = torch.einsum("bhsd,hod->bhso", current_prope_low, state["o_prope"])
        current_prope = apply_o_native(current_prope.float()).to(dtype)
        current_out = torch.cat((current_main, current_prope), dim=0)
        text_main, text_prope = text_out[:batch], text_out[batch:]
        text_prope = apply_o_native(text_prope.float()).to(dtype)
        text_out = torch.cat((text_main, text_prope), dim=0)
        native_out = _combine(text_out, text_lse, current_out, current_lse)
    else:
        native_main, native_prope = native_out[:batch], native_out[batch:]
        native_prope = apply_o_native(native_prope.float()).to(dtype)
        native_out = torch.cat((native_main, native_prope), dim=0)
    history_out = torch.cat((main_history, prope_history), dim=0)
    combined = _combine(native_out, native_lse, history_out, history_lse).to(dtype)
    combined = combined.transpose(1, 2).reshape(combined.shape[0], combined.shape[2], -1)
    return torch.chunk(combined, chunks=2, dim=0)
