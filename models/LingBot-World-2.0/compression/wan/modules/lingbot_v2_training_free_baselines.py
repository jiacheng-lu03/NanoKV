"""QVG and TurboQuant baselines for LingBot 2.0 pre-RoPE rolling KV.

QVG calls the official Quant-VideoGen checkout.  TurboQuant reuses the frozen
WorldPlay Algorithms 1/2 implementation and its bit-exact Triton decoder.  This
module only adapts LingBot's [B,S,H,D] layout and immutable rolling lifecycle.
"""
from __future__ import annotations

import logging
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


QVG_CODEC = "lingbot_v2_qvg_official_pre_v1"
TURBO_CODEC = "turboquant_uniform3_pre_v1"
_QVG_API = None
_TURBO_MODULE = None


@torch.no_grad()
def preload(
    mode: str, *, device: torch.device | None = None,
    layers: int | None = None, head_dim: int | None = None,
) -> None:
    """Import the frozen baseline backend before timed generation.

    Dynamic, content-dependent codes are intentionally not created here.
    TurboQuant's immutable matrices/codebooks are materialized before timing
    and remain fully counted as resident state; QVG has no static operator.
    """
    if mode == "qvg":
        _official_qvg_api()
    elif mode == "turboquant":
        turbo = _worldplay_turbo()
        if device is None or layers is None or head_dim is None:
            raise ValueError("TurboQuant preload requires device/layers/head_dim")
        # Pi/S and the Lloyd-Max tables are immutable model-side assets.  As
        # with codec/KVTC operators, initialize them in the model-ready phase;
        # their GPU bytes remain part of strict system-KV accounting.
        for layer in range(int(layers)):
            turbo._random_matrices(layer, int(head_dim), device)
        turbo._codebook(int(head_dim), 2, device)
        turbo._codebook(int(head_dim), 3, device)
    else:
        raise ValueError(mode)


def tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item, seen) for item in value)
    return 0


def _qvg_config() -> SimpleNamespace:
    config = SimpleNamespace(
        quant_type=os.environ.get(
            "LINGBOT_V2_QVG_QUANT_TYPE", "triton-nstages-kmeans-int2"
        ),
        quant_block_size=int(os.environ.get("LINGBOT_V2_QVG_BLOCK_SIZE", "64")),
        cache_num_k_centroids=int(
            os.environ.get("LINGBOT_V2_QVG_K_CENTROIDS", "256")
        ),
        cache_num_v_centroids=int(
            os.environ.get("LINGBOT_V2_QVG_V_CENTROIDS", "256")
        ),
        kmeans_max_iters=int(os.environ.get("LINGBOT_V2_QVG_KMEANS_ITERS", "2")),
        num_prq_stages=int(os.environ.get("LINGBOT_V2_QVG_PRQ_STAGES", "1")),
    )
    if config.quant_type != "triton-nstages-kmeans-int2":
        raise ValueError("formal LingBot QVG is frozen to official INT2")
    if max(config.cache_num_k_centroids, config.cache_num_v_centroids) > 256:
        raise ValueError("official QVG assignment storage is uint8")
    return config


def _official_qvg_api():
    global _QVG_API
    if _QVG_API is not None:
        return _QVG_API
    checkout = os.environ.get("LINGBOT_V2_QVG_ROOT", "external/Quant-VideoGen")
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    import triton

    original = triton.autotune

    def compatible_autotune(*args, **kwargs):
        if args:
            return original(*args, **kwargs)
        configs, keys = kwargs.pop("configs"), list(kwargs.pop("key"))

        def decorate(function):
            return original(
                configs=configs,
                key=[key for key in keys if key in function.arg_names],
                **kwargs,
            )(function)

        return decorate

    triton.autotune = compatible_autotune
    try:
        if "loguru" not in sys.modules:
            try:
                __import__("loguru")
            except ModuleNotFoundError:
                shim = types.ModuleType("loguru")
                shim.logger = logging.getLogger("quant_videogen")
                sys.modules["loguru"] = shim
        from quant_videogen.compress import compress_kv_cache, get_quantize_fn
        from quant_videogen.uncompress import uncompress_kv_cache
    finally:
        triton.autotune = original
    _QVG_API = compress_kv_cache, get_quantize_fn, uncompress_kv_cache
    return _QVG_API


def _worldplay_turbo():
    global _TURBO_MODULE
    if _TURBO_MODULE is not None:
        return _TURBO_MODULE
    root = os.environ.get(
        "LINGBOT_V2_WORLDPLAY_ROOT",
        str(Path(__file__).resolve().parents[4] / "HY-WorldPlay/compression"),
    )
    # Match the frozen WorldPlay/MG3.5 efficiency setting: exact PyTorch
    # encoder and bit-identical Triton decoder.
    os.environ.setdefault("HY_TURBOQUANT_TRITON", "0")
    os.environ.setdefault("HY_TURBOQUANT_TRITON_ENCODE", "0")
    os.environ.setdefault(
        "HY_TURBOQUANT_TRITON_DECODE", "1" if torch.cuda.is_available() else "0"
    )
    module_root = Path(root) / "hyvideo/models/transformers/modules"
    # Load the frozen files directly.  Importing HY-WorldPlay's top-level
    # package would mutate distributed rendezvous variables in the LingBot
    # process, which is unrelated to the baseline algorithm.
    packages = (
        ("hyvideo", Path(root) / "hyvideo"),
        ("hyvideo.models", Path(root) / "hyvideo/models"),
        ("hyvideo.models.transformers", Path(root) / "hyvideo/models/transformers"),
        ("hyvideo.models.transformers.modules", module_root),
    )
    for name, path in packages:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    for short_name in ("turboquant_triton", "turboquant_adapter"):
        name = f"hyvideo.models.transformers.modules.{short_name}"
        if name not in sys.modules:
            path = module_root / f"{short_name}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    _TURBO_MODULE = sys.modules[
        "hyvideo.models.transformers.modules.turboquant_adapter"
    ]
    return _TURBO_MODULE


@torch.no_grad()
def encode_pair(
    mode: str, key: torch.Tensor, value: torch.Tensor, *, layer: int
) -> dict[str, Any]:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("baseline expects matched [B,S,H,D] K/V")
    key_h = key.permute(0, 2, 1, 3).contiguous()
    value_h = value.permute(0, 2, 1, 3).contiguous()
    if mode == "qvg":
        config = _qvg_config()
        compress, get_quantize_fn, _ = _official_qvg_api()
        key_q, value_q = compress(
            key_h,
            value_h,
            config.quant_type,
            config,
            get_quantize_fn(config.quant_type, config),
        )
        if not isinstance(key_q, dict) or not isinstance(value_q, dict):
            raise RuntimeError("official QVG did not return packed dictionaries")
        key_q["info"] = {"output_dtype": key.dtype, "quant_config": config}
        value_q["info"] = {"output_dtype": value.dtype, "quant_config": config}
        codec = QVG_CODEC
    elif mode == "turboquant":
        turbo = _worldplay_turbo()
        key_q = turbo.quantize_tensor(key_h, target="k", layer_idx=int(layer))
        value_q = turbo.quantize_tensor(value_h, target="v", layer_idx=int(layer))
        codec = TURBO_CODEC
    else:
        raise ValueError(mode)
    return {
        "codec": codec,
        "mode": mode,
        "layer": int(layer),
        "shape": tuple(int(item) for item in key.shape),
        "k": key_q,
        "v": value_q,
    }


def _decode_qvg(
    segments: list[dict[str, Any]], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, uncompress = _official_qvg_api()
    keys, values = [], []
    for segment in segments:
        key, value = uncompress(segment["k"], segment["v"])
        keys.append(key.permute(0, 2, 1, 3).to(device=device, dtype=dtype))
        values.append(value.permute(0, 2, 1, 3).to(device=device, dtype=dtype))
    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


def _merge_turbo(parts: list[dict[str, Any]], target: str) -> dict[str, Any]:
    first = parts[0][target]
    merged = dict(first)
    shape = list(first["shape"])
    shape[2] = sum(int(part[target]["shape"][2]) for part in parts)
    merged["shape"] = tuple(shape)
    for name in ("indices", "norms"):
        merged[name] = torch.cat([part[target][name] for part in parts], dim=2)
    if target == "k":
        for name in ("qjl_signs", "residual_norms"):
            merged[name] = torch.cat([part[target][name] for part in parts], dim=2)
    return merged


def _decode_turbo(
    segments: list[dict[str, Any]], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    turbo = _worldplay_turbo()
    key = turbo.dequantize_tensor(_merge_turbo(segments, "k"), dtype, device)
    value = turbo.dequantize_tensor(_merge_turbo(segments, "v"), dtype, device)
    return key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)


def decode_segments(
    segments: list[dict[str, Any]], *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not segments:
        raise ValueError("empty baseline segment list")
    mode = str(segments[0]["mode"])
    if any(segment.get("mode") != mode for segment in segments):
        raise ValueError("mixed baseline segment formats")
    if mode == "qvg":
        return _decode_qvg(segments, dtype, device)
    if mode == "turboquant":
        return _decode_turbo(segments, dtype, device)
    raise ValueError(mode)


def _intersections(start: int, end: int, intervals: list[tuple[int, int]]):
    for left, right in intervals:
        begin, stop = max(start, left), min(end, right)
        if begin < stop:
            yield begin, stop


def _prune(
    segments: list[dict[str, Any]], *, committed_end: int, capacity: int, sink: int
) -> list[dict[str, Any]]:
    sink_stop = min(sink, committed_end)
    recent_start = max(sink_stop, committed_end - max(capacity - sink_stop, 0))
    return [
        segment
        for segment in segments
        if int(segment["global_start"]) < sink_stop
        or int(segment["global_end"]) > recent_start
    ]


def attention_inputs(
    cache: dict[str, Any], key: torch.Tensor, value: torch.Tensor, *,
    current_start: int, capacity: int, sink_tokens: int, max_attention_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segments = cache.setdefault("baseline_segments", [])
    committed_end = int(cache.get("baseline_committed_end", 0))
    sink_stop = min(sink_tokens, committed_end)
    recent_start = max(sink_stop, committed_end - max(capacity - sink_stop, 0))
    intervals = [(0, sink_stop)]
    if recent_start < committed_end:
        intervals.append((recent_start, committed_end))
    active = []
    for segment in segments:
        pieces = list(
            _intersections(
                int(segment["global_start"]), int(segment["global_end"]), intervals
            )
        )
        if pieces:
            active.append((segment, pieces))
    if active:
        decoded_k, decoded_v = decode_segments(
            [item[0] for item in active], dtype=value.dtype, device=value.device
        )
        k_parts, v_parts, position_parts = [], [], []
        offset = 0
        for segment, pieces in active:
            start, end = int(segment["global_start"]), int(segment["global_end"])
            for begin, stop in pieces:
                local = slice(offset + begin - start, offset + stop - start)
                k_parts.append(decoded_k[:, local])
                v_parts.append(decoded_v[:, local])
                position_parts.append(torch.arange(begin, stop, device=key.device))
            offset += end - start
        key_cache = torch.cat((*k_parts, key), dim=1)
        value_cache = torch.cat((*v_parts, value), dim=1)
        positions = torch.cat(
            (*position_parts,
             torch.arange(current_start, current_start + key.shape[1], device=key.device))
        )
    else:
        key_cache, value_cache = key, value
        positions = torch.arange(
            current_start, current_start + key.shape[1], device=key.device
        )
    if key_cache.shape[1] > max_attention_size:
        protected = min(sink_tokens, committed_end, max_attention_size)
        tail = max_attention_size - protected
        if tail:
            key_cache = torch.cat((key_cache[:, :protected], key_cache[:, -tail:]), 1)
            value_cache = torch.cat((value_cache[:, :protected], value_cache[:, -tail:]), 1)
            positions = torch.cat((positions[:protected], positions[-tail:]))
        else:
            key_cache = key_cache[:, :protected]
            value_cache = value_cache[:, :protected]
            positions = positions[:protected]

    current_end = current_start + int(key.shape[1])
    if cache.get("codec_commit", False):
        if current_start != committed_end:
            raise RuntimeError(
                f"non-causal baseline commit: {current_start} != {committed_end}"
            )
        segment = encode_pair(
            str(cache["baseline_mode"]), key, value, layer=int(cache["codec_layer"])
        )
        segment.update({"global_start": current_start, "global_end": current_end})
        segments.append(segment)
        cache["baseline_committed_end"] = current_end
        cache["baseline_segments"] = _prune(
            segments,
            committed_end=current_end,
            capacity=capacity,
            sink=sink_tokens,
        )
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(min(current_end, capacity))
    return key_cache, value_cache, positions


def static_operator_bytes(mode: str) -> int:
    if mode != "turboquant":
        return 0
    turbo = _worldplay_turbo()
    return tensor_bytes(turbo._MATRIX_CACHE, set()) + tensor_bytes(
        turbo._DEVICE_CODEBOOK_CACHE, set()
    )


def metadata(mode: str) -> dict[str, Any]:
    if mode == "qvg":
        config = _qvg_config()
        return {
            "backend": "official_quant_videogen",
            "cache_position": "pre_rope_k_raw_v",
            "configuration": "INT2,S1,B64,K256",
            "quant_type": config.quant_type,
            "quant_block_size": config.quant_block_size,
            "k_centroids": config.cache_num_k_centroids,
            "v_centroids": config.cache_num_v_centroids,
            "kmeans_max_iters": config.kmeans_max_iters,
            "prq_stages": config.num_prq_stages,
            # Upstream defaults to FP8 E4M3 scales.  That Triton dtype is not
            # supported on A100, so every paper port uses and counts BF16
            # scales instead of silently falling back to another algorithm.
            "scale_dtype": "bf16_a100_compat",
            "upstream_commit": "d3219e2ef6f8d196c2634df0a27953399860fb28",
            "history_lifecycle": "immutable_incremental_segments",
            "historical_fp_buffer_tokens": 0,
            "fallback_count": 0,
        }
    if mode == "turboquant":
        return {
            "backend": "frozen_worldplay_algorithms_1_2",
            "source_policy": "direct_source_reuse_without_algorithm_rewrite",
            "cache_position": "pre_rope_k_raw_v",
            "configuration": "uniform_3bit",
            "key_quantizer": "TurboQuant_prod",
            "value_quantizer": "TurboQuant_mse",
            "exact_int3_packing": True,
            "operator_backend": "triton_decode_only",
            "history_lifecycle": "immutable_incremental_segments",
            "historical_fp_buffer_tokens": 0,
            "fallback_count": 0,
        }
    raise ValueError(mode)
