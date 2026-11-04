"""Optional low-bit codecs for resident pre-position vision KV caches."""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


_BASIS_CACHE: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
_DEVICE_BASIS_CACHE: Dict[Tuple[str, int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
_RESIDUAL_CACHE: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
_RESIDUAL_DEVICE_CACHE: Dict[Tuple[str, int, str, str], Dict[str, torch.Tensor]] = {}
_LINEAR_DELTA_CACHE: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
_LINEAR_DELTA_DEVICE_CACHE: Dict[
    Tuple[str, int, str, str], torch.Tensor
] = {}
_LINEAR_ENCODER_DELTA_DEVICE_CACHE: Dict[
    Tuple[str, int, str, str], torch.Tensor | None
] = {}
_GLOBAL_OPERATOR_DEVICE_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_PREPACKED_OPERATOR_FILE_CACHE: Dict[str, Dict[str, Any]] = {}
_WARNED = set()
_CUDA_TIMINGS = {"encode": [], "decode": []}
_SKETCH_STORAGE = {}  # (layer_idx, tensor_name) -> {"codes": ..., "scale": ..., "n_blocks": int}
_STATS = {
    "enabled": False,
    "mode": "none",
    "target": "both",
    "granularity": "per_token",
    "rank_k": 32,
    "rank_v": 64,
    "coeff_quant": "none",
    "bits_k": 0,
    "bits_v": 0,
    "calls": 0,
    "k_calls": 0,
    "v_calls": 0,
    "fallback_count": 0,
    "fallback_reasons": {},
    "original_bytes": 0,
    "estimated_compressed_bytes": 0.0,
    "total_cache_original_bytes": 0,
    "estimated_effective_cache_bytes": 0.0,
    "encode_time_s": 0.0,
    "decode_time_s": 0.0,
    "packed_decode_calls": 0,
    "selected_decode_calls": 0,
    "selected_decode_rows": 0,
    "actual_resident_memory_reduced": False,
    "storage_mode": "packed",
    "actual_compressed_bytes": 0,
    "actual_effective_cache_bytes": 0,
    "residual_mlp_calls": 0,
    "linear_delta_calls": 0,
    "linear_encoder_delta_calls": 0,
    "operator_weight_mode": "legacy",
    "operator_weight_resident_bytes": 0,
    "operator_int8_group_size": 0,
}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _env_flag("HY_KV_CODEC_ENABLE", "0") and mode() != "none"


def mode() -> str:
    value = os.environ.get("HY_KV_CODEC_MODE", "none").strip().lower()
    if value not in {
        "none",
        "int8",
        "int4",
        "int3",
        "int2",
        "pca",
        "global_pca",
    }:
        raise ValueError(f"unsupported HY_KV_CODEC_MODE={value}")
    return value


def _granularity() -> str:
    value = os.environ.get("HY_KV_CODEC_GRANULARITY", "per_token").strip().lower()
    value = value.replace("-", "_")
    if value not in {"per_token", "per_channel", "per_tensor"}:
        raise ValueError(f"unsupported HY_KV_CODEC_GRANULARITY={value}")
    return value


def _target_enabled(tensor_name: str) -> bool:
    target = os.environ.get("HY_KV_CODEC_TARGET", "both").strip().lower()
    if target not in {"k", "v", "both"}:
        raise ValueError(f"unsupported HY_KV_CODEC_TARGET={target}")
    return target == "both" or target == tensor_name


def _semantic_smooth_enabled(layer_idx: int, tensor_name: str) -> bool:
    if tensor_name not in {"k", "v"}:
        return False
    if not _env_flag(f"HY_KV_CODEC_SEMANTIC_SMOOTH_{tensor_name.upper()}", "0"):
        return False
    value = os.environ.get("HY_KV_CODEC_SEMANTIC_SMOOTH_LAYERS", "").strip()
    if not value or value.lower() == "all":
        return True
    return int(layer_idx) in {int(item.strip()) for item in value.split(",") if item.strip()}


@torch.no_grad()
def _joint_semantic_smooth(x: torch.Tensor, layer_idx: int):
    """Online joint-head K-Means used before the residual PCA codec.

    Input is [B,H,S,D].  A single cluster id per token preserves the cross-head
    covariance exploited by the global-head PCA, unlike QVG's original
    independent per-head assignments.
    """
    batch, heads, sequence, dim = x.shape

    def prepare_rows():
        prepared = x.permute(0, 2, 1, 3).reshape(
            batch, sequence, heads * dim
        ).float()
        # The token norm is invariant across K-Means iterations.  The old
        # implementation recomputed this identical 2048-D reduction in every
        # iteration; retaining it once is bitwise-equivalent and removes one
        # full reduction for the default two-iteration C64 path.
        return prepared, prepared.square().sum(-1, keepdim=True)

    rows, row_norm = _time_call("c64_prepare", x, prepare_rows)
    clusters = int(os.environ.get("HY_KV_CODEC_SEMANTIC_CLUSTERS", "64"))
    iters = int(os.environ.get("HY_KV_CODEC_SEMANTIC_ITERS", "2"))
    if clusters <= 0 or clusters > sequence or clusters > 256:
        raise ValueError(f"semantic clusters must be in [1,min(S,256)], got {clusters}")
    generator = torch.Generator(device="cpu").manual_seed(20260804 + int(layer_idx))
    initial = torch.randperm(sequence, generator=generator)[:clusters].to(rows.device)
    centers = rows.index_select(1, initial).clone()
    labels = torch.zeros((batch, sequence), dtype=torch.long, device=rows.device)
    if _env_flag("HY_KV_CODEC_TRITON_KMEANS", "0"):
        from .kv_codec_triton import joint_kmeans_iteration
        for _ in range(iters):
            centers, labels = joint_kmeans_iteration(rows, centers)
    else:
        for _ in range(iters):
            def assign_clusters():
                distances = (
                    row_norm
                    + centers.square().sum(-1).unsqueeze(1)
                    - 2.0 * torch.bmm(rows, centers.transpose(1, 2))
                )
                return distances.argmin(-1)

            labels = _time_call("c64_assign", x, assign_clusters)

            def update_centers():
                if _env_flag("HY_KV_CODEC_DETERMINISTIC_CENTERS", "0"):
                    # A fixed GEMM reduction removes CUDA atomic-add ordering
                    # from online C64.  [B,K,S] stays small for K=64.
                    assignment = torch.nn.functional.one_hot(
                        labels, num_classes=clusters
                    ).transpose(1, 2).to(dtype=torch.float32)
                    counts = assignment.sum(dim=-1)
                    updated = torch.bmm(assignment, rows)
                else:
                    updated = torch.zeros_like(centers)
                    counts = torch.zeros(
                        (batch, clusters), dtype=torch.float32, device=rows.device
                    )
                    ones = torch.ones(
                        (sequence,), dtype=torch.float32, device=rows.device
                    )
                    for item in range(batch):
                        updated[item].scatter_add_(
                            0,
                            labels[item, :, None].expand(-1, heads * dim),
                            rows[item],
                        )
                        counts[item].scatter_add_(0, labels[item], ones)
                updated = updated / counts.clamp_min(1.0).unsqueeze(-1)
                return torch.where((counts > 0).unsqueeze(-1), updated, centers)

            centers = _time_call("c64_update", x, update_centers)
    # Store BF16 centers, and use exactly those stored values for encoding so
    # encode-time and later decode-time reconstructions are identical.
    def finalize_centers():
        stored = centers.to(torch.bfloat16)
        if _env_flag("HY_KV_CODEC_TRITON_CENTER", "0"):
            from .kv_codec_triton import subtract_joint_centers_from_rows
            residual_value = subtract_joint_centers_from_rows(
                rows, stored, labels, heads, dim, x.dtype
            )
        else:
            assigned = torch.gather(
                stored.float(), 1,
                labels[..., None].expand(-1, -1, heads * dim),
            )
            residual_value = (rows - assigned).reshape(
                batch, sequence, heads, dim
            ).permute(0, 2, 1, 3)
        return residual_value.to(x.dtype), stored

    residual, stored_centers = _time_call("c64_finalize", x, finalize_centers)
    return residual, stored_centers, labels.to(torch.uint8)


def _record_fallback(reason: str, message: str = "") -> None:
    _STATS["fallback_count"] += 1
    reasons = _STATS["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    if message and reason not in _WARNED:
        warnings.warn(message)
        _WARNED.add(reason)


def _time_call(bucket: str, tensor: torch.Tensor, fn):
    if not _env_flag("HY_KV_CODEC_PROFILE", "1"):
        return fn()
    if tensor.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        _CUDA_TIMINGS.setdefault(bucket, []).append((start, end))
        return output
    start_time = time.perf_counter()
    output = fn()
    _STATS[f"{bucket}_time_s"] += time.perf_counter() - start_time
    return output


def quantize_kv_tensor(x: torch.Tensor, num_bits: int, granularity: str):
    if num_bits not in {2, 3, 4, 8}:
        raise ValueError(f"supported KV quantization bits are 2,3,4,8; got {num_bits}")
    granularity = granularity.replace("-", "_")
    if granularity not in {"per_token", "per_channel", "per_tensor"}:
        raise ValueError(f"unsupported quantization granularity={granularity}")
    qmax = (1 << (num_bits - 1)) - 1
    eps = torch.finfo(torch.float32).eps
    work = x.float()
    if granularity == "per_tensor":
        scale = work.abs().amax().clamp_min(eps) / qmax
    elif granularity == "per_channel":
        # Vision KV is [B,H,S,D]. Keep independent D-channel scales per head.
        dims = (0, 2) if work.ndim == 4 else tuple(range(work.ndim - 1))
        scale = work.abs().amax(dim=dims, keepdim=True).clamp_min(eps) / qmax
    else:
        scale = work.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    codes = (work / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return codes, scale


def dequantize_kv_tensor(
    codes: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (codes.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def pack_quantized_codes(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack signed quantization codes into uint8 storage.

    INT2 and INT4 use four and two values per byte respectively. INT8 is
    represented directly as one byte per value. The symmetric quantizer uses
    one fewer than all available unsigned symbols; the spare symbol is kept
    unused so pack/unpack is exactly equivalent to the existing fake codec.
    """
    if num_bits not in {2, 3, 4, 8}:
        raise ValueError(f"real packing supports INT2/INT3/INT4/INT8, got INT{num_bits}")
    flat = codes.reshape(-1).to(torch.int16)
    qmax = (1 << (num_bits - 1)) - 1
    unsigned = flat + qmax
    if torch.any(unsigned < 0) or torch.any(unsigned >= (1 << num_bits)):
        raise ValueError(f"codes are outside the representable INT{num_bits} range")
    if num_bits == 8:
        return unsigned.to(torch.uint8).contiguous()
    if num_bits == 3:
        bit_offsets = torch.arange(
            unsigned.numel(), dtype=torch.int64, device=codes.device
        ) * num_bits
        byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
        shifts = bit_offsets.remainder(8)
        packed_i64 = torch.zeros(
            (unsigned.numel() * num_bits + 7) // 8,
            dtype=torch.int64,
            device=codes.device,
        )
        values = unsigned.to(torch.int64)
        packed_i64.scatter_add_(0, byte_indices, values << shifts)
        spills = shifts + num_bits > 8
        if spills.any():
            packed_i64.scatter_add_(
                0,
                byte_indices[spills] + 1,
                values[spills] >> (8 - shifts[spills]),
            )
        return packed_i64.bitwise_and_(0xFF).to(torch.uint8).contiguous()
    values_per_byte = 8 // num_bits
    padding = (-unsigned.numel()) % values_per_byte
    if padding:
        unsigned = torch.cat(
            [unsigned, torch.zeros(padding, dtype=unsigned.dtype, device=unsigned.device)]
        )
    grouped = unsigned.reshape(-1, values_per_byte)
    packed = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=codes.device)
    for index in range(values_per_byte):
        packed |= (grouped[:, index].to(torch.uint8) << (index * num_bits))
    return packed.contiguous()


def unpack_quantized_codes(
    packed: torch.Tensor, num_bits: int, numel: int, shape=None
) -> torch.Tensor:
    """Restore packed low-bit values as signed int8 quantization codes."""
    if num_bits not in {2, 3, 4, 8}:
        raise ValueError(f"real unpacking supports INT2/INT3/INT4/INT8, got INT{num_bits}")
    packed = packed.reshape(-1).to(torch.uint8)
    qmax = (1 << (num_bits - 1)) - 1
    if num_bits == 8:
        unsigned = packed
    elif num_bits == 3:
        bit_offsets = torch.arange(numel, dtype=torch.int64, device=packed.device) * num_bits
        byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
        shifts = bit_offsets.remainder(8)
        low = packed[byte_indices].to(torch.int64) >> shifts
        spills = shifts + num_bits > 8
        if spills.any():
            low[spills] |= packed[byte_indices[spills] + 1].to(torch.int64) << (
                8 - shifts[spills]
            )
        unsigned = low.bitwise_and_((1 << num_bits) - 1).to(torch.uint8)
    else:
        values_per_byte = 8 // num_bits
        mask = (1 << num_bits) - 1
        parts = [((packed >> (index * num_bits)) & mask) for index in range(values_per_byte)]
        unsigned = torch.stack(parts, dim=1).reshape(-1)
    codes = unsigned[: int(numel)].to(torch.int16).sub_(qmax).to(torch.int8)
    return codes.reshape(shape).contiguous() if shape is not None else codes


def _scale_storage_dtype() -> torch.dtype:
    value = os.environ.get("HY_KV_CODEC_SCALE_DTYPE", "bf16").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported HY_KV_CODEC_SCALE_DTYPE={value}")


@torch.no_grad()
def quantize_and_pack_kv_tensor(
    x: torch.Tensor, num_bits: int, granularity: str
) -> Dict[str, Any]:
    codes, scale = quantize_kv_tensor(x, num_bits, granularity)
    packed = pack_quantized_codes(codes, num_bits)
    scale = scale.to(dtype=_scale_storage_dtype())
    return {
        "kv_codec": True,
        "codec": "uniform_packed",
        "packed_codes": packed,
        "scale": scale,
        "shape": tuple(int(value) for value in x.shape),
        "numel": int(x.numel()),
        "num_bits": int(num_bits),
        "granularity": granularity.replace("-", "_"),
        "source_dtype": str(x.dtype).replace("torch.", ""),
    }


def _fake_quant_parts(x: torch.Tensor, num_bits: int, granularity: str):
    return quantize_kv_tensor(x, num_bits, granularity)


@torch.no_grad()
def fake_quant_kv_tensor(
    x: torch.Tensor, num_bits: int = 4, granularity: str = "per_token"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    granularity = granularity.replace("-", "_")
    holder = {}

    def encode():
        holder["parts"] = _fake_quant_parts(x, num_bits, granularity)
        return holder["parts"]

    quantized, scale = _time_call("encode", x, encode)
    reconstructed = _time_call(
        "decode", x, lambda: dequantize_kv_tensor(quantized, scale, x.dtype)
    )
    scale_count = scale.numel()
    estimated_bits = x.numel() * num_bits + scale_count * 16
    return reconstructed, {
        "num_bits": num_bits,
        "granularity": granularity,
        "scale_count": scale_count,
        "estimated_bytes": estimated_bits / 8,
    }


def _basis_path(root: Path, layer_idx: int, tensor_name: str) -> Path:
    return root / f"layer{layer_idx}_{tensor_name}.pt"


def _load_basis(root: Path, layer_idx: int, tensor_name: str):
    cache_key = (str(root.resolve()), int(layer_idx), tensor_name)
    if cache_key not in _BASIS_CACHE:
        path = _basis_path(root, layer_idx, tensor_name)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        if "mean" not in payload or "basis" not in payload:
            raise ValueError(f"invalid PCA basis payload: {path}")
        _BASIS_CACHE[cache_key] = payload
    return _BASIS_CACHE[cache_key]


def _global_basis_path(root: Path, layer_idx: int) -> Path:
    return root / f"layer_{int(layer_idx):02d}" / "global_head_pca.pt"


def _load_global_basis(root: Path, layer_idx: int, tensor_name: str = "v"):
    cache_key = (str(root.resolve()), int(layer_idx), f"{tensor_name}_global")
    if cache_key not in _BASIS_CACHE:
        path = _global_basis_path(root, layer_idx)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        mean = payload.get("mean")
        components = payload.get("components")
        payload_tensor_name = str(payload.get("tensor_name", "v"))
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(components, torch.Tensor)
            or tuple(mean.shape) != (2048,)
            or components.ndim != 2
            # Rank-specific resident artifacts (for example rank128/rank256)
            # intentionally store only the retained PCA rows.  The requested
            # runtime rank is checked against this dimension in
            # global_pca_compress_v, so requiring a rank512 container here
            # incorrectly rejects otherwise complete lower-rank codecs.
            or int(components.shape[0]) < 1
            or int(components.shape[1]) != 2048
        ):
            raise ValueError(f"invalid global PCA basis payload: {path}")
        if payload_tensor_name != tensor_name:
            raise ValueError(
                f"global PCA basis target mismatch: requested={tensor_name} "
                f"artifact={payload_tensor_name} path={path}"
            )
        _BASIS_CACHE[cache_key] = payload
    return _BASIS_CACHE[cache_key]


def _global_device_basis(root: Path, layer_idx: int, device, tensor_name: str = "v"):
    device_key = (
        str(root.resolve()),
        int(layer_idx),
        f"{tensor_name}_global",
        str(device),
    )
    cached = _DEVICE_BASIS_CACHE.get(device_key)
    if cached is None:
        payload = _load_global_basis(root, layer_idx, tensor_name)
        cached = (
            payload["mean"].to(device=device, dtype=torch.float32),
            payload["components"].to(device=device, dtype=torch.float32),
        )
        _DEVICE_BASIS_CACHE[device_key] = cached
    return cached


def _global_residual_path(root: Path, layer_idx: int, tensor_name: str) -> Path:
    return root / tensor_name / f"layer_{int(layer_idx):02d}" / "checkpoint_best.pt"


def _load_global_residual(root: Path, layer_idx: int, tensor_name: str):
    cache_key = (str(root.resolve()), int(layer_idx), tensor_name)
    if cache_key not in _RESIDUAL_CACHE:
        path = _global_residual_path(root, layer_idx, tensor_name)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        if payload.get("model_kind") != "pca512_int4_residual_decoder":
            raise ValueError(f"invalid PCA16 residual checkpoint: {path}")
        if payload.get("target") != tensor_name or int(payload.get("layer", -1)) != int(layer_idx):
            raise ValueError(
                f"PCA16 residual target/layer mismatch for {path}: "
                f"{payload.get('target')} L{payload.get('layer')}"
            )
        if int(payload.get("rank", -1)) != 512:
            raise ValueError(f"PCA16 residual rank must be 512: {path}")
        state = payload.get("model")
        required = (
            "norm.weight",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
        )
        if not isinstance(state, dict) or any(name not in state for name in required):
            raise ValueError(f"incomplete PCA16 residual state: {path}")
        _RESIDUAL_CACHE[cache_key] = payload
    return _RESIDUAL_CACHE[cache_key]


def _global_device_residual(root: Path, layer_idx: int, tensor_name: str, device):
    device_key = (str(root.resolve()), int(layer_idx), tensor_name, str(device))
    cached = _RESIDUAL_DEVICE_CACHE.get(device_key)
    if cached is None:
        payload = _load_global_residual(root, layer_idx, tensor_name)
        state = payload["model"]
        cached = {
            name: state[name].to(device=device, dtype=torch.float32)
            for name in (
                "norm.weight",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
            )
        }
        _RESIDUAL_DEVICE_CACHE[device_key] = cached
    return cached


def _global_linear_delta_path(root: Path, layer_idx: int) -> Path:
    return (
        root
        / f"layer_{int(layer_idx):02d}"
        / "checkpoints"
        / "checkpoint_functional_ft.pt"
    )


def _load_global_linear_delta(
    root: Path, layer_idx: int, tensor_name: str, basis_root: Path
):
    cache_key = (str(root.resolve()), int(layer_idx), tensor_name)
    if cache_key not in _LINEAR_DELTA_CACHE:
        path = _global_linear_delta_path(root, layer_idx)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        if payload.get("model_kind") != "global_pca_int4_linear_a1":
            raise ValueError(f"invalid PCA A1 linear checkpoint: {path}")
        metadata = payload.get("codec_metadata", {})
        if metadata.get("target") != tensor_name:
            raise ValueError(
                f"PCA A1 target mismatch: requested={tensor_name} "
                f"checkpoint={metadata.get('target')} path={path}"
            )
        if int(metadata.get("layer", -1)) != int(layer_idx):
            raise ValueError(f"PCA A1 layer mismatch for {path}")
        rank = int(metadata.get("rank", -1))
        if rank <= 0:
            raise ValueError(f"invalid PCA A1 rank={rank}: {path}")
        expected_basis = str(Path(metadata.get("basis_root", "")).resolve())
        actual_basis = str(basis_root.resolve())
        if expected_basis != actual_basis:
            raise ValueError(
                f"PCA A1 basis mismatch: checkpoint={expected_basis} "
                f"runtime={actual_basis}"
            )
        state = payload.get("model", {})
        weight = state.get("decoder_delta.weight")
        if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (2048, rank):
            raise ValueError(f"invalid PCA A1 decoder delta in {path}")
        encoder_delta = state.get("encoder_delta")
        if encoder_delta is not None and (
            not isinstance(encoder_delta, torch.Tensor)
            or tuple(encoder_delta.shape) != (rank, 2048)
        ):
            raise ValueError(f"invalid PCA A1 encoder delta in {path}")
        _LINEAR_DELTA_CACHE[cache_key] = payload
    return _LINEAR_DELTA_CACHE[cache_key]


def _global_device_linear_delta(
    root: Path, layer_idx: int, tensor_name: str, basis_root: Path, device
):
    device_key = (str(root.resolve()), int(layer_idx), tensor_name, str(device))
    cached = _LINEAR_DELTA_DEVICE_CACHE.get(device_key)
    if cached is None:
        payload = _load_global_linear_delta(
            root, layer_idx, tensor_name, basis_root
        )
        cached = payload["model"]["decoder_delta.weight"].to(
            device=device, dtype=torch.float32
        )
        _LINEAR_DELTA_DEVICE_CACHE[device_key] = cached
    return cached


def _global_device_encoder_delta(
    root: Path, layer_idx: int, tensor_name: str, basis_root: Path, device
):
    """Load the optional encoder update from encoder+decoder A1 checkpoints.

    Decoder-only A1 and older V-QAT artifacts have no ``encoder_delta``; in
    that common case return ``None`` instead of materializing and retaining a
    full all-zero matrix on every layer.  Keeping this separate from the
    decoder basis is essential: the trained decoder is
    ``PCA_basis.T + decoder_delta``, not the transpose of the updated encoder.
    """
    device_key = (str(root.resolve()), int(layer_idx), tensor_name, str(device))
    if device_key not in _LINEAR_ENCODER_DELTA_DEVICE_CACHE:
        payload = _load_global_linear_delta(
            root, layer_idx, tensor_name, basis_root
        )
        value = payload["model"].get("encoder_delta")
        cached = (
            None
            if value is None
            else value.to(device=device, dtype=torch.float32)
        )
        _LINEAR_ENCODER_DELTA_DEVICE_CACHE[device_key] = cached
    return _LINEAR_ENCODER_DELTA_DEVICE_CACHE[device_key]


def _codec_operator_weight_mode() -> str:
    """Execution/storage mode for the two PCA linear operators.

    ``legacy`` preserves the original two FP32 decoder GEMMs exactly.  The
    other modes fuse the frozen PCA decoder and its trained delta first.
    INT8 is true resident weight quantization; it is dequantized into a short-
    lived BF16 tensor at each GEMM until a native A100 kernel is selected.
    """
    value = os.environ.get("HY_KV_CODEC_OPERATOR_WEIGHT_MODE", "legacy").strip().lower()
    aliases = {"none": "legacy", "fp32": "fp32_fused", "bf16": "bf16_fused"}
    value = aliases.get(value, value)
    allowed = {"legacy", "fp32_fused", "bf16_fused", "int8_decoder", "int8"}
    if value not in allowed:
        raise ValueError(
            "HY_KV_CODEC_OPERATOR_WEIGHT_MODE must be one of "
            f"{sorted(allowed)}, got {value}"
        )
    return value


def _pack_operator_weight(weight: torch.Tensor, storage: str, device) -> Dict[str, Any]:
    weight = weight.detach().float().cpu().contiguous()
    if storage == "fp32":
        tensor = weight.to(device=device, dtype=torch.float32)
        return {"storage": storage, "weight": tensor, "resident_bytes": tensor.numel() * tensor.element_size()}
    if storage == "bf16":
        tensor = weight.to(device=device, dtype=torch.bfloat16)
        return {"storage": storage, "weight": tensor, "resident_bytes": tensor.numel() * tensor.element_size()}
    if storage == "int8":
        in_features = int(weight.shape[1])
        group_size = int(os.environ.get("HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE", "0") or 0)
        if group_size <= 0:
            group_size = in_features
        if group_size > in_features or in_features % group_size:
            raise ValueError(
                "HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE must divide the input "
                f"dimension {in_features}, got {group_size}"
            )
        grouped = weight.reshape(weight.shape[0], -1, group_size)
        scale = grouped.abs().amax(dim=2).clamp_min(1e-12) / 127.0
        qweight = (
            grouped / scale[:, :, None]
        ).round().clamp(-127, 127).to(torch.int8).reshape_as(weight)
        qweight = qweight.to(device=device)
        scale = scale.to(device=device, dtype=torch.bfloat16)
        return {
            "storage": storage,
            "qweight": qweight,
            "scale": scale,
            "group_size": group_size,
            "resident_bytes": qweight.numel() * qweight.element_size() + scale.numel() * scale.element_size(),
        }
    raise ValueError(f"unsupported operator weight storage={storage}")


def _operator_linear(rows: torch.Tensor, packed: Dict[str, Any], bias=None) -> torch.Tensor:
    storage = packed["storage"]
    if storage == "fp32":
        return torch.nn.functional.linear(rows.float(), packed["weight"], bias)
    input_bf16 = rows.to(torch.bfloat16)
    bias_bf16 = None if bias is None else bias.to(torch.bfloat16)
    if storage == "bf16":
        return torch.nn.functional.linear(input_bf16, packed["weight"], bias_bf16)
    if storage == "int8":
        # A100's stock PyTorch build has no CUDA weight-only INT8 GEMM.  Keep
        # the resident matrix genuinely INT8 and materialize only this one
        # layer's BF16 weight for the duration of the GEMM.
        qweight = packed["qweight"]
        group_size = int(packed["group_size"])
        weight = (
            qweight.reshape(qweight.shape[0], -1, group_size).to(torch.bfloat16)
            * packed["scale"][:, :, None]
        ).reshape_as(qweight)
        return torch.nn.functional.linear(input_bf16, weight, bias_bf16)
    raise ValueError(f"unsupported operator weight storage={storage}")


def _prepacked_operator_file() -> Dict[str, Any] | None:
    value = os.environ.get("HY_KV_CODEC_PREPACKED_OPERATOR_PATH", "").strip()
    if not value:
        return None
    path = str(Path(value).resolve())
    cached = _PREPACKED_OPERATOR_FILE_CACHE.get(path)
    if cached is None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        cached = torch.load(source, map_location="cpu")
        if cached.get("format") != "hy_kv_codec_operator_int8_v1":
            raise ValueError(f"invalid prepacked operator artifact: {source}")
        if not isinstance(cached.get("entries"), dict):
            raise ValueError(f"prepacked operator has no entries: {source}")
        _PREPACKED_OPERATOR_FILE_CACHE[path] = cached
    return cached


def _prepacked_operator_entry(
    root: Path,
    linear_root: Path,
    layer_idx: int,
    tensor_name: str,
    rank: int,
) -> Dict[str, Any] | None:
    payload = _prepacked_operator_file()
    if payload is None:
        return None
    if _codec_operator_weight_mode() != "int8":
        raise ValueError("prepacked operator artifact currently requires full int8 mode")
    if int(payload.get("rank", -1)) != int(rank):
        raise ValueError(
            f"prepacked operator rank mismatch: artifact={payload.get('rank')} runtime={rank}"
        )
    group_size = int(os.environ.get("HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE", "0") or 0)
    if int(payload.get("group_size", -1)) != group_size:
        raise ValueError(
            "prepacked operator group-size mismatch: "
            f"artifact={payload.get('group_size')} runtime={group_size}"
        )
    key = f"{tensor_name}:{int(layer_idx):02d}"
    entry = payload["entries"].get(key)
    if not isinstance(entry, dict):
        raise KeyError(f"prepacked operator missing entry {key}")
    expected_basis = str(root.resolve())
    expected_linear = str(linear_root.resolve())
    # Prepacked artifacts may outlive a repository relocation.  Compare the
    # canonical targets rather than the serialized spelling of paths so a
    # legacy symlink to the exact same frozen artifact remains valid.  This
    # preserves strict provenance checking: different canonical roots still
    # fail below.
    artifact_basis = str(Path(str(entry.get("basis_root", ""))).resolve())
    artifact_linear = str(Path(str(entry.get("linear_root", ""))).resolve())
    from nanokv.artifacts import portable_codec_verified
    portable = portable_codec_verified(os.environ["HY_KV_CODEC_PREPACKED_OPERATOR_PATH"])
    if not portable and (artifact_basis != expected_basis or artifact_linear != expected_linear):
        raise ValueError(
            f"prepacked operator provenance mismatch for {key}: "
            f"basis={entry.get('basis_root')} linear={entry.get('linear_root')}"
        )
    return entry


def _prepacked_to_device(entry: Dict[str, Any], device) -> Dict[str, Any]:
    def move(name: str):
        packed = entry[name]
        qweight = packed["qweight"].to(device=device)
        scale = packed["scale"].to(device=device)
        return {
            "storage": "int8",
            "qweight": qweight,
            "scale": scale,
            "group_size": int(packed["group_size"]),
            "resident_bytes": qweight.numel() * qweight.element_size()
            + scale.numel() * scale.element_size(),
        }

    mean = entry["mean"].to(device=device, dtype=torch.float32)
    encoder = move("encoder")
    decoder = move("decoder")
    return {
        "mean": mean,
        "encoder": encoder,
        "decoder": decoder,
        "mode": "int8",
        "prepacked": True,
        "resident_bytes": mean.numel() * mean.element_size()
        + encoder["resident_bytes"] + decoder["resident_bytes"],
    }


def preload_operator_weights(device=None) -> None:
    """Load a consolidated INT8 operator artifact before timed inference.

    Codec operators are static model parameters, so loading them alongside the
    transformer is the deployment-equivalent behavior.  Lazy per-layer loading
    incorrectly charges model-weight I/O to the first cache write.
    """
    payload = _prepacked_operator_file()
    if payload is None:
        return
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    mode = _codec_operator_weight_mode()
    if mode != "int8":
        raise ValueError("prepacked operator preload requires full int8 mode")
    rank = int(payload["rank"])
    group_size = int(payload["group_size"])
    path = str(
        Path(os.environ["HY_KV_CODEC_PREPACKED_OPERATOR_PATH"]).resolve()
    )
    for name, entry in payload["entries"].items():
        target, layer_text = name.split(":", 1)
        key = (
            entry["basis_root"], entry["linear_root"], int(layer_text),
            target, rank, str(device), mode, group_size, path,
        )
        if key not in _GLOBAL_OPERATOR_DEVICE_CACHE:
            _GLOBAL_OPERATOR_DEVICE_CACHE[key] = _prepacked_to_device(entry, device)
    _STATS["operator_weight_mode"] = mode
    _STATS["operator_int8_group_size"] = group_size
    _STATS["operator_prepacked"] = True
    _STATS["operator_preloaded"] = True
    _STATS["operator_weight_resident_bytes"] = sum(
        int(item["resident_bytes"])
        for item in _GLOBAL_OPERATOR_DEVICE_CACHE.values()
    )


def _global_device_operator_weights(
    root: Path,
    linear_root: Path,
    layer_idx: int,
    tensor_name: str,
    rank: int,
    device,
) -> Dict[str, Any]:
    mode = _codec_operator_weight_mode()
    if mode == "legacy":
        raise ValueError("legacy mode does not use fused operator weights")
    int8_group_size = int(
        os.environ.get("HY_KV_CODEC_INT8_WEIGHT_GROUP_SIZE", "0") or 0
    )
    prepacked_path = os.environ.get(
        "HY_KV_CODEC_PREPACKED_OPERATOR_PATH", ""
    ).strip()
    key = (
        str(root.resolve()), str(linear_root.resolve()), int(layer_idx),
        tensor_name, int(rank), str(device), mode, int8_group_size,
        str(Path(prepacked_path).resolve()) if prepacked_path else "",
    )
    cached = _GLOBAL_OPERATOR_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached

    prepacked = _prepacked_operator_entry(
        root, linear_root, layer_idx, tensor_name, rank
    )
    if prepacked is not None:
        cached = _prepacked_to_device(prepacked, device)
        _GLOBAL_OPERATOR_DEVICE_CACHE[key] = cached
        _STATS["operator_weight_mode"] = mode
        _STATS["operator_int8_group_size"] = int8_group_size
        _STATS["operator_prepacked"] = True
        _STATS["operator_weight_resident_bytes"] = sum(
            int(item["resident_bytes"])
            for item in _GLOBAL_OPERATOR_DEVICE_CACHE.values()
        )
        return cached

    basis_payload = _load_global_basis(root, layer_idx, tensor_name)
    components = basis_payload["components"][-rank:].float()
    delta_payload = _load_global_linear_delta(
        linear_root, layer_idx, tensor_name, root
    )
    state = delta_payload["model"]
    decoder = components.T.contiguous() + state["decoder_delta.weight"].float()
    encoder_delta = state.get("encoder_delta")
    encoder = components if encoder_delta is None else components + encoder_delta.float()

    if mode == "fp32_fused":
        encoder_storage = decoder_storage = "fp32"
    elif mode == "bf16_fused":
        encoder_storage = decoder_storage = "bf16"
    elif mode == "int8_decoder":
        encoder_storage, decoder_storage = "fp32", "int8"
    else:
        encoder_storage = decoder_storage = "int8"
    cached = {
        "mean": basis_payload["mean"].to(device=device, dtype=torch.float32),
        "encoder": _pack_operator_weight(encoder, encoder_storage, device),
        "decoder": _pack_operator_weight(decoder, decoder_storage, device),
        "mode": mode,
    }
    cached["resident_bytes"] = (
        cached["mean"].numel() * cached["mean"].element_size()
        + cached["encoder"]["resident_bytes"]
        + cached["decoder"]["resident_bytes"]
    )
    _GLOBAL_OPERATOR_DEVICE_CACHE[key] = cached
    _STATS["operator_weight_mode"] = mode
    _STATS["operator_int8_group_size"] = int8_group_size
    _STATS["operator_weight_resident_bytes"] = sum(
        int(item["resident_bytes"]) for item in _GLOBAL_OPERATOR_DEVICE_CACHE.values()
    )
    return cached


@torch.no_grad()
def _sketch_enabled(layer_idx: int | None = None) -> bool:
    if not _env_flag("HY_KV_CODEC_SKETCH_ENABLE", "0"):
        return False
    layers = os.environ.get("HY_KV_CODEC_SKETCH_LAYERS", "").strip()
    if not layers or layer_idx is None:
        return True
    selected = {int(value.strip()) for value in layers.split(",") if value.strip()}
    return int(layer_idx) in selected


def _sketch_block_size() -> int:
    return int(os.environ.get("HY_KV_CODEC_SKETCH_BLOCK_SIZE", "128"))


def _sketch_target_enabled(tensor_name: str) -> bool:
    target = os.environ.get("HY_KV_CODEC_SKETCH_TARGET", "both").strip().lower()
    if target not in {"k", "v", "both"}:
        raise ValueError(f"unsupported HY_KV_CODEC_SKETCH_TARGET={target}")
    return target == "both" or target == str(tensor_name).lower()


def _sketch_alpha() -> float:
    alpha = float(os.environ.get("HY_KV_CODEC_SKETCH_ALPHA", "1.0"))
    if not 0.0 <= alpha <= 2.0:
        raise ValueError(f"HY_KV_CODEC_SKETCH_ALPHA must be in [0,2], got {alpha}")
    return alpha


def _compute_block_sketch(residual: torch.Tensor, block_size: int):
    """INT4 block-mean sketch with per-block shared scale.

    residual: [B, H, N, D]
    Returns dict with packed INT4 codes and BF16 scale per block.
    Storage: (N/blk) * (2048*4 + 16) bits → ~3.1% of rank512 INT4 body.
    """
    B, H, N, D = residual.shape
    n_blocks = (N + block_size - 1) // block_size
    all_codes = []
    all_scales = []
    for b in range(n_blocks):
        start = b * block_size
        end = min(N, start + block_size)
        block_mean = residual[:, :, start:end].mean(2).float()  # [B,H,D]
        # Per-block shared scale across all D channels
        s = block_mean.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7  # [B,H,1]
        codes = (block_mean / s).round().clamp(-7, 7).to(torch.int8)  # [B,H,D]
        all_codes.append(codes)
        all_scales.append(s.squeeze(-1).to(torch.bfloat16))  # [B,H]
    # Stack: [B, H, n_blocks, D]
    stacked_codes = torch.stack(all_codes, dim=2)  # [B, H, n_blocks, D]
    stacked_scales = torch.stack(all_scales, dim=2)  # [B, H, n_blocks]
    return {
        "codes_packed": pack_quantized_codes(stacked_codes, 4),
        "codes_shape": tuple(stacked_codes.shape),
        "codes_numel": stacked_codes.numel(),
        "scales": stacked_scales,
        "n_blocks": n_blocks,
        "alpha": _sketch_alpha(),
    }


def _apply_block_sketch(x: torch.Tensor, sketch: dict, block_size: int) -> torch.Tensor:
    """Add INT4 block-mean sketch to decoded tensor in-place."""
    if sketch is None:
        return x
    B, H, N, D = x.shape
    if "codes_packed" in sketch:
        packed = sketch["codes_packed"].to(device=x.device)
        codes = unpack_quantized_codes(
            packed, 4, int(sketch["codes_numel"]), tuple(sketch["codes_shape"])
        ).float()
    else:
        # Backward compatibility for diagnostic payloads written before real packing.
        codes = sketch["codes"].to(device=x.device, dtype=torch.float32)
    scales = sketch["scales"].to(device=x.device, dtype=torch.float32)  # [B,H,n_blk]
    # Dequantize: codes * scales.unsqueeze(-1)
    means = codes * scales.unsqueeze(-1)  # [B, H, n_blocks, D]
    means = means * float(sketch.get("alpha", 1.0))
    n_blocks = means.shape[2]
    for b in range(n_blocks):
        start = b * block_size
        end = min(N, start + block_size)
        x[:, :, start:end] += means[:, :, b:b + 1]
    return x


def global_pca_compress_v(
    x: torch.Tensor, layer_idx: int, tensor_name: str = "v"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if x.ndim != 4:
        raise ValueError(f"global PCA expects [B,H,S,D], got {tuple(x.shape)}")
    batch, heads, sequence, dim = x.shape
    if heads * dim != 2048:
        raise ValueError(f"global PCA expects H*D=2048, got {heads}*{dim}")
    semantic_smooth = _semantic_smooth_enabled(layer_idx, tensor_name)
    if semantic_smooth:
        root_value = os.environ.get(
            f"HY_KV_CODEC_SEMANTIC_BASIS_DIR_{tensor_name.upper()}", ""
        ).strip()
        if not root_value:
            raise ValueError(
                f"HY_KV_CODEC_SEMANTIC_BASIS_DIR_{tensor_name.upper()} is required"
            )
        x, semantic_centers, semantic_ids = _joint_semantic_smooth(x, layer_idx)
    else:
        root_value = os.environ.get(
            f"HY_KV_CODEC_BASIS_DIR_{tensor_name.upper()}",
            os.environ.get("HY_KV_CODEC_BASIS_DIR", ""),
        ).strip()
    if not root_value:
        raise ValueError("HY_KV_CODEC_BASIS_DIR is required for global_pca")
    root = Path(root_value)
    rank = int(os.environ.get("HY_KV_CODEC_GLOBAL_PCA_RANK", "1024"))
    prepacked_file = (
        _prepacked_operator_file()
        if _codec_operator_weight_mode() == "int8"
        else None
    )
    if prepacked_file is not None:
        available_rank = int(prepacked_file.get("rank", -1))
    else:
        basis_payload = _load_global_basis(root, layer_idx, tensor_name)
        available_rank = int(basis_payload["components"].shape[0])
    if rank <= 0 or rank > available_rank:
        raise ValueError(
            f"global PCA rank must be in [1,{available_rank}], got {rank}"
        )
    linear_root_value = (
        os.environ.get(
            f"HY_KV_CODEC_SEMANTIC_LINEAR_DIR_{tensor_name.upper()}", ""
        ).strip()
        if semantic_smooth else
        os.environ.get(
            f"HY_KV_CODEC_LINEAR_DECODER_DIR_{tensor_name.upper()}",
            os.environ.get("HY_KV_CODEC_LINEAR_DECODER_DIR", ""),
        ).strip()
    )
    operator_mode = _codec_operator_weight_mode()
    if operator_mode == "legacy":
        mean, components = _global_device_basis(root, layer_idx, x.device, tensor_name)
        # torch.linalg.eigh returns eigenvectors in ascending eigenvalue order.
        # The saved basis preserves that order, so retain its suffix.
        components = components[-rank:]
        operator_weights = None
    else:
        if not linear_root_value:
            raise ValueError(
                f"{operator_mode} requires a linear decoder checkpoint for {tensor_name}"
            )
        operator_weights = _global_device_operator_weights(
            root, Path(linear_root_value), layer_idx, tensor_name, rank, x.device
        )
        mean = operator_weights["mean"]
        components = None
    coeff_quant = os.environ.get(
        f"HY_KV_CODEC_COEFF_QUANT_{tensor_name.upper()}",
        os.environ.get("HY_KV_CODEC_COEFF_QUANT", "none"),
    ).strip().lower()
    if coeff_quant not in {"none", "int8", "int4"}:
        raise ValueError(
            f"global PCA supports coefficient quantization none|int8|int4, got {coeff_quant}"
        )

    def encode():
        rows = x.permute(0, 2, 1, 3).reshape(-1, heads * dim).float()
        centered = rows - mean
        if operator_weights is not None:
            coeff = _operator_linear(centered, operator_weights["encoder"])
            # Packed operators already contain the fitted encoder; raw training
            # checkpoints are unnecessary for inference or its accounting.
            if prepacked_file is None:
                delta_payload = _load_global_linear_delta(
                    Path(linear_root_value), layer_idx, tensor_name, root
                )
                if delta_payload["model"].get("encoder_delta") is not None:
                    _STATS["linear_encoder_delta_calls"] += 1
        else:
            encoder_components = components
            if linear_root_value:
                encoder_delta = _global_device_encoder_delta(
                    Path(linear_root_value), layer_idx, tensor_name, root, x.device
                )
                if encoder_delta is not None:
                    encoder_components = components + encoder_delta
                    _STATS["linear_encoder_delta_calls"] += 1
            coeff = torch.nn.functional.linear(centered, encoder_components)
        coeff = coeff.reshape(batch, sequence, rank)
        if coeff_quant in {"int8", "int4"}:
            bits = int(coeff_quant.removeprefix("int"))
            if bits == 4 and _env_flag("HY_KV_CODEC_TRITON_INT4", "0"):
                from .kv_codec_triton import int4_quant_pack_per_row
                return int4_quant_pack_per_row(coeff, _scale_storage_dtype())
            codes, scale = quantize_kv_tensor(coeff, bits, "per_token")
            if bits == 4:
                codes = pack_quantized_codes(codes, bits)
            return codes, scale.to(dtype=_scale_storage_dtype())
        return coeff.to(torch.bfloat16)

    encoded = _time_call("encode", x, encode)
    payload = {
        "kv_codec": True,
        "codec": "global_pca",
        "coeff_quant": coeff_quant,
        "rank": rank,
        "shape": tuple(int(value) for value in x.shape),
        "layer_idx": int(layer_idx),
        "tensor_name": tensor_name,
        "basis_root": str(root),
        "source_dtype": str(x.dtype).replace("torch.", ""),
        "semantic_smooth": bool(semantic_smooth),
    }
    if semantic_smooth:
        payload["semantic_centers"] = semantic_centers
        payload["semantic_ids"] = semantic_ids
        payload["semantic_clusters"] = int(semantic_centers.shape[1])
        payload["semantic_linear_root"] = os.environ.get(
            f"HY_KV_CODEC_SEMANTIC_LINEAR_DIR_{tensor_name.upper()}", ""
        ).strip()
    if coeff_quant in {"int8", "int4"}:
        payload["latent_int8"], payload["latent_scale"] = encoded
        if coeff_quant == "int4":
            payload["latent_shape"] = (batch, sequence, rank)
            payload["latent_numel"] = batch * sequence * rank
    else:
        payload["latent"] = encoded
    actual_bytes = _payload_tensor_bytes(payload)
    return payload, {
        "rank": rank,
        "coeff_quant": coeff_quant,
        "estimated_bytes": actual_bytes,
        "actual_bytes": actual_bytes,
        "basis_file": str(_global_basis_path(root, layer_idx)),
    }


@torch.no_grad()
def pca_reconstruct_kv_tensor(
    x: torch.Tensor, layer_idx: int, tensor_name: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    root_value = os.environ.get("HY_KV_CODEC_BASIS_DIR", "").strip()
    if not root_value:
        raise ValueError("HY_KV_CODEC_BASIS_DIR is required for PCA mode")
    payload = _load_basis(Path(root_value), layer_idx, tensor_name)
    mean_cpu = payload["mean"]
    basis_cpu = payload["basis"]
    if x.ndim != 4:
        raise ValueError(f"PCA codec expects [B,H,S,D], got {tuple(x.shape)}")
    if tuple(mean_cpu.shape) != (x.shape[1], x.shape[-1]):
        raise ValueError(
            f"PCA mean shape {tuple(mean_cpu.shape)} does not match H,D={(x.shape[1], x.shape[-1])}"
        )
    rank_env = "HY_KV_CODEC_RANK_K" if tensor_name == "k" else "HY_KV_CODEC_RANK_V"
    rank_default = "32" if tensor_name == "k" else "64"
    requested_rank = int(os.environ.get(rank_env, rank_default))
    rank = min(requested_rank, basis_cpu.shape[-1])
    if rank <= 0:
        raise ValueError(f"invalid PCA rank {requested_rank}")
    device_key = (str(Path(root_value).resolve()), layer_idx, tensor_name, str(x.device))
    cached_device_basis = _DEVICE_BASIS_CACHE.get(device_key)
    if cached_device_basis is None:
        cached_device_basis = (
            mean_cpu.to(device=x.device, dtype=torch.float32),
            basis_cpu.to(device=x.device, dtype=torch.float32),
        )
        _DEVICE_BASIS_CACHE[device_key] = cached_device_basis
    mean, basis_full = cached_device_basis
    basis = basis_full[:, :, :rank]
    holder = {}

    def encode():
        centered = x.float() - mean[None, :, None, :]
        coeff = torch.einsum("bhsd,hdr->bhsr", centered, basis)
        coeff_quant = os.environ.get("HY_KV_CODEC_COEFF_QUANT", "none").strip().lower()
        if coeff_quant in {"int8", "int4"}:
            bits = 8 if coeff_quant == "int8" else 4
            quantized, scale = _fake_quant_parts(coeff, bits, "per_token")
            holder["coeff_parts"] = (quantized, scale)
            holder["coeff"] = None
        elif coeff_quant == "none":
            holder["coeff"] = coeff
            holder["coeff_parts"] = None
        else:
            raise ValueError(f"unsupported HY_KV_CODEC_COEFF_QUANT={coeff_quant}")
        return coeff_quant

    coeff_quant = _time_call("encode", x, encode)

    def decode():
        if holder["coeff_parts"] is not None:
            quantized, scale = holder["coeff_parts"]
            coeff = quantized * scale
        else:
            coeff = holder["coeff"]
        restored = torch.einsum("bhsr,hdr->bhsd", coeff, basis)
        return (restored + mean[None, :, None, :]).to(x.dtype)

    reconstructed = _time_call("decode", x, decode)
    coeff_bits = 16 if coeff_quant == "none" else (8 if coeff_quant == "int8" else 4)
    token_count = x.numel() // x.shape[-1]
    scale_bits = token_count * 16 if coeff_quant != "none" else 0
    estimated_bits = token_count * rank * coeff_bits + scale_bits
    return reconstructed, {
        "rank": rank,
        "requested_rank": requested_rank,
        "coeff_quant": coeff_quant,
        "estimated_bytes": estimated_bits / 8,
        "basis_file": str(_basis_path(Path(root_value), layer_idx, tensor_name)),
    }


def compress_kv_tensor(
    x: torch.Tensor, layer_idx: int, tensor_name: str
) -> Tuple[Any, Dict[str, Any]]:
    codec_mode = mode()
    if codec_mode == "global_pca":
        return global_pca_compress_v(x, layer_idx, tensor_name)
    if codec_mode in {"int8", "int4", "int3", "int2"}:
        bits = int(codec_mode.removeprefix("int"))
        override = os.environ.get(f"HY_KV_CODEC_BITS_{tensor_name.upper()}", "").strip()
        if override:
            bits = int(override)
        storage_mode = os.environ.get("HY_KV_CODEC_STORAGE_MODE", "packed").strip().lower()
        if storage_mode == "fake":
            return fake_quant_kv_tensor(x, bits, _granularity())
        if storage_mode != "packed":
            raise ValueError(f"unsupported HY_KV_CODEC_STORAGE_MODE={storage_mode}")
        payload = _time_call(
            "encode", x, lambda: quantize_and_pack_kv_tensor(x, bits, _granularity())
        )
        actual_bytes = _payload_tensor_bytes(payload)
        return payload, {
            "num_bits": bits,
            "granularity": _granularity(),
            "estimated_bytes": actual_bytes,
            "actual_bytes": actual_bytes,
            "storage_mode": "packed",
        }
    if codec_mode == "pca":
        return pca_reconstruct_kv_tensor(x, layer_idx, tensor_name)
    return x, {"estimated_bytes": x.numel() * x.element_size()}


def decompress_kv_tensor(
    obj_or_tensor: Any,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Any:
    """Decode a real resident codec payload; tensors/fake outputs pass through."""
    if isinstance(obj_or_tensor, dict) and obj_or_tensor.get("kv_codec"):
        payload = obj_or_tensor
        if payload.get("codec") == "global_pca":
            coeff_quant = payload.get("coeff_quant", "none")
            if coeff_quant in {"int8", "int4"}:
                codes = payload["latent_int8"]
                scale = payload["latent_scale"]
                if device is not None:
                    codes = codes.to(device=device)
                    scale = scale.to(device=device)
                if coeff_quant == "int4":
                    if _env_flag("HY_KV_CODEC_TRITON_INT4", "0"):
                        from .kv_codec_triton import int4_unpack_dequant_per_row
                        latent = int4_unpack_dequant_per_row(
                            codes, scale, tuple(payload["latent_shape"]), torch.float32
                        )
                    else:
                        codes = unpack_quantized_codes(
                            codes,
                            4,
                            int(payload["latent_numel"]),
                            tuple(payload["latent_shape"]),
                        )
                        latent = dequantize_kv_tensor(codes, scale, torch.float32)
                else:
                    latent = dequantize_kv_tensor(codes, scale, torch.float32)
            elif coeff_quant == "none":
                latent = payload["latent"]
                if device is not None:
                    latent = latent.to(device=device)
            else:
                raise ValueError(
                    f"unsupported global PCA coefficient quantization={coeff_quant}"
                )
            rank = int(payload.get("rank", latent.shape[-1]))
            if latent.shape[-1] != rank:
                raise ValueError(
                    f"global PCA payload rank mismatch: latent={latent.shape[-1]} rank={rank}"
                )
            shape = tuple(int(value) for value in payload["shape"])
            batch, heads, sequence, dim = shape
            root = Path(payload["basis_root"])
            tensor_name = str(payload.get("tensor_name", "v"))
            layer_idx = int(payload["layer_idx"])
            linear_root_value = (
                str(payload.get("semantic_linear_root", "")).strip()
                if payload.get("semantic_smooth") else
                os.environ.get(
                    f"HY_KV_CODEC_LINEAR_DECODER_DIR_{tensor_name.upper()}",
                    os.environ.get("HY_KV_CODEC_LINEAR_DECODER_DIR", ""),
                ).strip()
            )
            operator_mode = _codec_operator_weight_mode()
            if operator_mode == "legacy":
                mean, components = _global_device_basis(
                    root, layer_idx, latent.device, tensor_name
                )
                components = components[-rank:]
                operator_weights = None
            else:
                if not linear_root_value:
                    raise ValueError(
                        f"{operator_mode} requires a linear decoder checkpoint for {tensor_name}"
                    )
                operator_weights = _global_device_operator_weights(
                    root, Path(linear_root_value), layer_idx, tensor_name, rank,
                    latent.device,
                )
                mean = operator_weights["mean"]
                components = None

            def decode_global():
                latent_rows = latent.reshape(-1, rank)
                if operator_weights is not None:
                    rows = _operator_linear(
                        latent_rows, operator_weights["decoder"], mean
                    )
                    _STATS["linear_delta_calls"] += 1
                else:
                    latent_rows = latent_rows.float()
                    rows = torch.nn.functional.linear(
                        latent_rows,
                        components.T,
                        mean,
                    )
                if linear_root_value and operator_weights is None:
                    delta_weight = _global_device_linear_delta(
                        Path(linear_root_value),
                        layer_idx,
                        tensor_name,
                        root,
                        latent.device,
                    )
                    rows = rows + torch.nn.functional.linear(
                        latent_rows, delta_weight
                    )
                    _STATS["linear_delta_calls"] += 1
                residual_root_value = os.environ.get(
                    "HY_KV_CODEC_RESIDUAL_MLP_DIR", ""
                ).strip()
                if residual_root_value:
                    if rank != 512 or payload.get("coeff_quant") != "int4":
                        raise ValueError(
                            "PCA residual MLP requires rank512 INT4 coefficients"
                        )
                    weights = _global_device_residual(
                        Path(residual_root_value),
                        int(payload["layer_idx"]),
                        str(payload.get("tensor_name", "v")),
                        latent.device,
                    )
                    normalized = torch.nn.functional.rms_norm(
                        latent_rows,
                        (rank,),
                        weights["norm.weight"],
                    )
                    hidden = torch.nn.functional.silu(
                        torch.nn.functional.linear(
                            normalized,
                            weights["fc1.weight"],
                            weights["fc1.bias"],
                        )
                    )
                    rows = rows + torch.nn.functional.linear(
                        hidden,
                        weights["fc2.weight"],
                        weights["fc2.bias"],
                    )
                    _STATS["residual_mlp_calls"] += 1
                if payload.get("semantic_smooth"):
                    if _env_flag("HY_KV_CODEC_TRITON_CENTER", "0"):
                        from .kv_codec_triton import add_joint_centers_and_layout
                        result = add_joint_centers_and_layout(
                            rows,
                            payload["semantic_centers"].to(device=rows.device),
                            payload["semantic_ids"].to(device=rows.device),
                            shape,
                            dtype,
                        )
                    else:
                        centers = payload["semantic_centers"].to(
                            device=rows.device, dtype=torch.float32
                        )
                        ids = payload["semantic_ids"].to(
                            device=rows.device, dtype=torch.long
                        )
                        assigned = torch.gather(
                            centers, 1,
                            ids[..., None].expand(-1, -1, heads * dim),
                        ).reshape(-1, heads * dim)
                        rows = rows + assigned
                        result = (
                            rows.reshape(batch, sequence, heads, dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                            .to(dtype)
                        )
                    _STATS["semantic_smooth_decode_calls"] = (
                        _STATS.get("semantic_smooth_decode_calls", 0) + 1
                    )
                else:
                    result = (
                        rows.reshape(batch, sequence, heads, dim)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                        .to(dtype)
                    )
                # ---- Apply INT4 block-mean residual sketch if present ----
                sketch = payload.get("kv_sketch")
                if sketch is not None:
                    result = _apply_block_sketch(result, sketch, _sketch_block_size())
                    _STATS["sketch_applied"] = _STATS.get("sketch_applied", 0) + 1
                return result

            _STATS["packed_decode_calls"] += 1
            return _time_call("decode", latent, decode_global)
        if payload.get("codec") != "uniform_packed":
            raise ValueError(f"unsupported resident KV codec={payload.get('codec')}")
        packed = payload["packed_codes"]
        scale = payload["scale"]
        if device is not None:
            packed = packed.to(device=device)
            scale = scale.to(device=device)
        def decode():
            codes = unpack_quantized_codes(
                packed,
                int(payload["num_bits"]),
                int(payload["numel"]),
                tuple(payload["shape"]),
            )
            return dequantize_kv_tensor(codes, scale, dtype)

        _STATS["packed_decode_calls"] += 1
        return _time_call("decode", packed, decode)
    return obj_or_tensor


def _global_pca_selected_latent(
    payload: Dict[str, Any],
    token_ids: torch.Tensor,
    source_batch: int,
    device,
) -> torch.Tensor:
    """Load only selected [sequence, rank] rows from a global-PCA payload."""
    if payload.get("codec") != "global_pca":
        raise ValueError("selective decode requires a global_pca payload")
    shape = tuple(int(value) for value in payload["shape"])
    batch, _, sequence, _ = shape
    rank = int(payload["rank"])
    source_batch = int(source_batch)
    if source_batch < 0 or source_batch >= batch:
        raise IndexError(f"source_batch={source_batch} outside [0,{batch})")

    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).reshape(-1)
    if token_ids.numel() and (
        int(token_ids.min()) < 0 or int(token_ids.max()) >= sequence
    ):
        raise IndexError(f"selected token id outside [0,{sequence})")

    coeff_quant = payload.get("coeff_quant", "none")
    if coeff_quant == "none":
        latent = payload["latent"].to(device=device)
        return latent[source_batch].index_select(0, token_ids).float()

    scale = payload["latent_scale"].to(device=device)
    selected_scale = scale[source_batch].index_select(0, token_ids)
    stored = payload["latent_int8"].to(device=device)
    if coeff_quant == "int8":
        codes = stored.reshape(batch, sequence, rank)[source_batch].index_select(
            0, token_ids
        )
    elif coeff_quant == "int4":
        if rank % 2:
            raise ValueError("row-selective INT4 decode requires an even PCA rank")
        bytes_per_row = rank // 2
        packed_rows = stored.reshape(batch, sequence, bytes_per_row)[source_batch]
        selected_packed = packed_rows.index_select(0, token_ids)
        codes = unpack_quantized_codes(
            selected_packed,
            4,
            int(token_ids.numel()) * rank,
            (int(token_ids.numel()), rank),
        )
    else:
        raise ValueError(
            f"unsupported global PCA coefficient quantization={coeff_quant}"
        )
    return dequantize_kv_tensor(codes, selected_scale, torch.float32)


def decode_global_pca_selected(
    payload: Dict[str, Any],
    token_ids: torch.Tensor,
    source_batch: int = 0,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Decode selected sequence rows without materializing the full KV history.

    Returns ``[H, selected_tokens, D]`` in the same head-major layout as a
    gather from :func:`decompress_kv_tensor`.
    """
    if device is None:
        storage = payload.get("latent_int8", payload.get("latent"))
        device = storage.device
    latent_rows = _global_pca_selected_latent(
        payload, token_ids, source_batch, device
    )
    shape = tuple(int(value) for value in payload["shape"])
    _, heads, _, dim = shape
    rank = int(payload["rank"])
    root = Path(payload["basis_root"])
    tensor_name = str(payload.get("tensor_name", "v"))
    layer_idx = int(payload["layer_idx"])
    mean, components = _global_device_basis(
        root, layer_idx, latent_rows.device, tensor_name
    )
    components = components[-rank:]
    rows = torch.nn.functional.linear(latent_rows, components.T, mean)

    linear_root_value = os.environ.get(
        f"HY_KV_CODEC_LINEAR_DECODER_DIR_{tensor_name.upper()}",
        os.environ.get("HY_KV_CODEC_LINEAR_DECODER_DIR", ""),
    ).strip()
    if linear_root_value:
        delta_weight = _global_device_linear_delta(
            Path(linear_root_value),
            layer_idx,
            tensor_name,
            root,
            latent_rows.device,
        )
        rows = rows + torch.nn.functional.linear(latent_rows, delta_weight)
        _STATS["linear_delta_calls"] += 1

    residual_root_value = os.environ.get(
        "HY_KV_CODEC_RESIDUAL_MLP_DIR", ""
    ).strip()
    if residual_root_value:
        if rank != 512 or payload.get("coeff_quant") != "int4":
            raise ValueError(
                "PCA residual MLP requires rank512 INT4 coefficients"
            )
        weights = _global_device_residual(
            Path(residual_root_value), layer_idx, tensor_name, latent_rows.device
        )
        normalized = torch.nn.functional.rms_norm(
            latent_rows, (rank,), weights["norm.weight"]
        )
        hidden = torch.nn.functional.silu(
            torch.nn.functional.linear(
                normalized, weights["fc1.weight"], weights["fc1.bias"]
            )
        )
        rows = rows + torch.nn.functional.linear(
            hidden, weights["fc2.weight"], weights["fc2.bias"]
        )
        _STATS["residual_mlp_calls"] += 1

    _STATS["selected_decode_calls"] += 1
    _STATS["selected_decode_rows"] += int(latent_rows.shape[0])
    return (
        rows.reshape(-1, heads, dim)
        .permute(1, 0, 2)
        .contiguous()
        .to(dtype)
    )


def _payload_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_payload_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_tensor_bytes(item) for item in value)
    return 0


def online_sketch_encode_decode(
    x: torch.Tensor,
    layer_idx: int,
    tensor_name: str,
    dtype: torch.dtype = torch.float32,
    device=None,
):
    """Encode raw pre-position KV with the final codec, compute the INT4
    block-mean residual sketch, store it in the payload, and decode back.

    Returns (corrected_tensor, payload):
      corrected = FinalCodec(x) + dequant(sketch(x - FinalCodec(x)))

    The payload carries the sketch, so a later decompress_kv_tensor of the
    same payload returns the same corrected tensor — encode-time and
    decode-time corrections are consistent by construction.
    """
    payload, _ = compress_kv_tensor(x, layer_idx, tensor_name)
    if (_sketch_enabled(layer_idx) and _sketch_target_enabled(tensor_name)
            and isinstance(payload, dict) and payload.get("codec") == "global_pca"):
        with torch.no_grad():
            decoded = decompress_kv_tensor(payload, dtype, x.device if device is None else device)
            # DEBUG: no-sketch passthrough smoke test
            noop = os.environ.get("HY_KV_CODEC_SKETCH_NOOP", "0").strip()
            if noop in ("1", "true", "yes"):
                return decoded, payload
            residual = x - decoded
            payload["kv_sketch"] = _compute_block_sketch(residual, _sketch_block_size())
            corrected = _apply_block_sketch(decoded, payload["kv_sketch"], _sketch_block_size())
            _STATS["kv_sketch_bytes"] = _STATS.get("kv_sketch_bytes", 0) + sum(
                t.numel() * t.element_size()
                for t in (payload["kv_sketch"]["codes_packed"], payload["kv_sketch"]["scales"])
            )
            return corrected, payload
    return decompress_kv_tensor(payload, dtype, x.device if device is None else device), payload


def apply_codec_to_layer_cache(
    layer_cache: Dict[str, Any], layer_idx: int
) -> Dict[str, Any]:
    if not enabled():
        return layer_cache
    codec_mode = mode()
    global_rank = int(os.environ.get("HY_KV_CODEC_GLOBAL_PCA_RANK", "1024"))
    coeff_quant = os.environ.get("HY_KV_CODEC_COEFF_QUANT", "none")
    global_bits = (
        int(coeff_quant.removeprefix("int"))
        if codec_mode == "global_pca" and coeff_quant in {"int4", "int8"}
        else 0
    )
    _STATS.update(
        {
            "enabled": True,
            "mode": codec_mode,
            "target": os.environ.get("HY_KV_CODEC_TARGET", "both"),
            "granularity": _granularity(),
            "rank_k": (
                global_rank
                if codec_mode == "global_pca"
                else int(os.environ.get("HY_KV_CODEC_RANK_K", "32"))
            ),
            "rank_v": (
                global_rank
                if codec_mode == "global_pca"
                else int(os.environ.get("HY_KV_CODEC_RANK_V", "64"))
            ),
            "coeff_quant": coeff_quant,
            "bits_k": (
                global_bits
                if codec_mode == "global_pca"
                else int(os.environ.get("HY_KV_CODEC_BITS_K", "0") or 0)
            ),
            "bits_v": (
                global_bits
                if codec_mode == "global_pca"
                else int(os.environ.get("HY_KV_CODEC_BITS_V", "0") or 0)
            ),
            "storage_mode": os.environ.get("HY_KV_CODEC_STORAGE_MODE", "packed"),
        }
    )
    strict = _env_flag("HY_KV_CODEC_STRICT", "0")
    for cache_name, short_name in (("k_vision", "k"), ("v_vision", "v")):
        tensor = layer_cache.get(cache_name)
        if not isinstance(tensor, torch.Tensor):
            if _target_enabled(short_name):
                _record_fallback(
                    "non_tensor_cache",
                    "KV codec skipped a non-tensor cache payload. Disable HY_KV_AE3D_ENABLE "
                    "when running codec sensitivity experiments.",
                )
            continue
        tensor_bytes = tensor.numel() * tensor.element_size()
        _STATS["total_cache_original_bytes"] += tensor_bytes
        if not _target_enabled(short_name):
            _STATS["estimated_effective_cache_bytes"] += tensor_bytes
            _STATS["actual_effective_cache_bytes"] += tensor_bytes
            continue
        try:
            original_bytes = tensor.numel() * tensor.element_size()
            reconstructed, metadata = compress_kv_tensor(tensor, layer_idx, short_name)

            # ---- Sketch: compute once during encoding, store in payload ----
            if (_sketch_enabled(layer_idx) and _sketch_target_enabled(short_name)
                    and isinstance(reconstructed, dict)
                    and reconstructed.get("codec") == "global_pca"
                    and "kv_sketch" not in reconstructed):
                with torch.no_grad():
                    decoded = decompress_kv_tensor(reconstructed, tensor.dtype, tensor.device)
                    residual = tensor - decoded
                    reconstructed["kv_sketch"] = _compute_block_sketch(residual, _sketch_block_size())
                    _STATS["kv_sketch_bytes"] = _STATS.get("kv_sketch_bytes", 0) + sum(
                        s.numel() * s.element_size()
                        for s in (reconstructed["kv_sketch"]["codes_packed"],
                                  reconstructed["kv_sketch"]["scales"])
                    )

            layer_cache[cache_name] = reconstructed
            _STATS["calls"] += 1
            _STATS[f"{short_name}_calls"] += 1
            _STATS["original_bytes"] += original_bytes
            _STATS["estimated_compressed_bytes"] += float(metadata["estimated_bytes"])
            _STATS["estimated_effective_cache_bytes"] += float(metadata["estimated_bytes"])
            actual_bytes = _payload_tensor_bytes(reconstructed)
            _STATS["actual_compressed_bytes"] += actual_bytes
            _STATS["actual_effective_cache_bytes"] += actual_bytes
        except Exception as error:
            _STATS["estimated_effective_cache_bytes"] += tensor_bytes
            _STATS["actual_effective_cache_bytes"] += tensor_bytes
            _record_fallback(type(error).__name__, f"KV codec fallback at layer {layer_idx}: {error}")
            if strict:
                raise
    return layer_cache


def _finalize_cuda_timings() -> None:
    for bucket, events in _CUDA_TIMINGS.items():
        total_ms = 0.0
        for start, end in events:
            end.synchronize()
            total_ms += start.elapsed_time(end)
        key = f"{bucket}_time_s"
        _STATS[key] = _STATS.get(key, 0.0) + total_ms / 1000.0
        events.clear()


def write_summary() -> None:
    if not (enabled() or _env_flag("HY_KV_CODEC_PROFILE", "0")):
        return
    _finalize_cuda_timings()
    output = dict(_STATS)
    output["residual_mlp_dir"] = os.environ.get(
        "HY_KV_CODEC_RESIDUAL_MLP_DIR", ""
    )
    output["linear_decoder_dir_k"] = os.environ.get(
        "HY_KV_CODEC_LINEAR_DECODER_DIR_K", ""
    )
    output["linear_decoder_dir_v"] = os.environ.get(
        "HY_KV_CODEC_LINEAR_DECODER_DIR_V", ""
    )
    output["sketch_target"] = os.environ.get("HY_KV_CODEC_SKETCH_TARGET", "both")
    output["sketch_alpha"] = _sketch_alpha() if _sketch_enabled() else 0.0
    output["sketch_layers"] = os.environ.get("HY_KV_CODEC_SKETCH_LAYERS", "")
    compressed = float(output["estimated_compressed_bytes"])
    output["estimated_compression_ratio"] = (
        output["original_bytes"] / compressed if compressed > 0 else 1.0
    )
    effective = float(output["estimated_effective_cache_bytes"])
    output["estimated_effective_cache_ratio"] = (
        output["total_cache_original_bytes"] / effective if effective > 0 else 1.0
    )
    actual_compressed = float(output["actual_compressed_bytes"])
    output["actual_compression_ratio"] = (
        output["original_bytes"] / actual_compressed if actual_compressed > 0 else 1.0
    )
    output["compression_ratio_scope"] = "cumulative_cache_writes"
    output["compression_ratio_note"] = (
        "This ratio is cumulative over codec calls; use a final resident snapshot "
        "for cross-method KV storage comparisons."
    )
    actual_effective = float(output["actual_effective_cache_bytes"])
    output["actual_effective_cache_ratio"] = (
        output["total_cache_original_bytes"] / actual_effective
        if actual_effective > 0
        else 1.0
    )
    output["actual_resident_memory_reduced"] = output["actual_compression_ratio"] > 1.0
    path = Path(
        os.environ.get(
            "HY_KV_CODEC_SUMMARY",
            "kv_ae_outputs/ae16x_v2/online_kv_codec_summary.json",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
