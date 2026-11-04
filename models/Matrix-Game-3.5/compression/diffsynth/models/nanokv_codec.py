"""Packed resident HY-WorldPlay method codec adapted to MG3.5 causal chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


_OPERATORS = {}
_PREPACKED = None


def enabled() -> bool:
    return os.environ.get("MG35_KVCODEC_MODE", "").strip().lower() == "worldplay_method"


def _rung() -> str:
    value = os.environ.get("MG35_WORLDPLAY_RUNG", "full").strip().lower()
    if value not in {"pca", "c64", "c64_vqat", "full"}:
        raise ValueError(f"unsupported MG35_WORLDPLAY_RUNG={value!r}")
    return value


def _target_enabled(kind: str) -> bool:
    target = os.environ.get("MG35_KVCODEC_TARGET", "both").strip().lower()
    if target not in {"k", "v", "both"}:
        raise ValueError(f"unsupported MG35_KVCODEC_TARGET={target!r}")
    return target == "both" or target == kind


def _center_storage_mode() -> str:
    """Storage for dynamic C64 centers.

    ``bf16`` is the frozen HY-WorldPlay method. ``int8_block`` is an explicit
    MG-only side-information experiment and must never be reported as exact
    WorldPlay parity.
    """
    mode = os.environ.get("MG35_KVCODEC_CENTER_STORAGE", "bf16").strip().lower()
    if mode not in {"bf16", "int8_block"}:
        raise ValueError(f"unsupported MG35 center storage mode={mode!r}")
    return mode


def _expected_center_provenance() -> tuple[str, int]:
    if _rung() == "pca":
        return "none", 0
    return _center_storage_mode(), int(
        os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64")
    )


def _check_center_provenance(payload: dict, *, context: str) -> None:
    expected_storage, expected_block = _expected_center_provenance()
    if payload.get("center_storage") != expected_storage or int(
        payload.get("center_block_size", -1)
    ) != expected_block:
        raise RuntimeError(
            f"{context} center provenance mismatch: "
            f"artifact={payload.get('center_storage')}/"
            f"{payload.get('center_block_size')} runtime="
            f"{expected_storage}/{expected_block}"
        )


@torch.no_grad()
def store_centers(centers: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Return resident center payload and the exact FP32 values it decodes."""
    stored = centers.to(torch.bfloat16)
    if _center_storage_mode() == "bf16":
        return stored, stored.float()
    block_size = int(os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64"))
    dim = int(stored.shape[-1])
    if block_size <= 0 or dim % block_size:
        raise ValueError(
            f"center INT8 block size {block_size} must divide center width {dim}"
        )
    grouped = stored.float().reshape(*stored.shape[:-1], dim // block_size, block_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    stored_scale = scale.to(torch.bfloat16)
    codes = (grouped / scale).round().clamp(-127, 127).to(torch.int8)
    payload = {
        "storage": "int8_block",
        "qvalues": codes,
        "scale": stored_scale,
        "shape": tuple(int(v) for v in stored.shape),
        "block_size": block_size,
    }
    decoded = (codes.float() * stored_scale.float()).reshape_as(stored).float()
    return payload, decoded


def restore_centers(payload: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload.to(device=device, dtype=torch.float32)
    if not isinstance(payload, dict) or payload.get("storage") != "int8_block":
        raise RuntimeError("unsupported C64 center payload")
    shape = tuple(int(v) for v in payload["shape"])
    return (
        payload["qvalues"].to(device=device, dtype=torch.float32)
        * payload["scale"].to(device=device, dtype=torch.float32)
    ).reshape(shape)


def _pack_operator(weight: torch.Tensor, device: torch.device) -> dict:
    mode = os.environ.get("MG35_KVCODEC_OPERATOR_WEIGHT_MODE", "int8").strip().lower()
    weight = weight.detach().float().cpu().contiguous()
    if mode == "fp32":
        return {"storage": "fp32", "weight": weight.to(device=device)}
    if mode == "bf16":
        return {"storage": "bf16", "weight": weight.to(device=device, dtype=torch.bfloat16)}
    if mode != "int8":
        raise ValueError(f"unsupported MG35 operator mode={mode}")
    in_features = int(weight.shape[1])
    group_size = int(os.environ.get("MG35_KVCODEC_INT8_WEIGHT_GROUP_SIZE", "0") or 0)
    if group_size <= 0:
        group_size = in_features
    if group_size > in_features or in_features % group_size:
        raise ValueError(f"INT8 group size {group_size} must divide {in_features}")
    grouped = weight.reshape(weight.shape[0], -1, group_size)
    scale = grouped.abs().amax(dim=2).clamp_min(1e-12) / 127.0
    qweight = (grouped / scale[:, :, None]).round().clamp(-127, 127).to(torch.int8)
    return {
        "storage": "int8",
        "qweight": qweight.reshape_as(weight).to(device=device),
        "scale": scale.to(device=device, dtype=torch.bfloat16),
        "group_size": group_size,
    }


def _operator_linear(rows: torch.Tensor, packed: dict, bias=None) -> torch.Tensor:
    storage = packed["storage"]
    if storage == "fp32":
        return F.linear(rows.float(), packed["weight"], bias)
    input_bf16 = rows.to(torch.bfloat16)
    bias_bf16 = None if bias is None else bias.to(torch.bfloat16)
    if storage == "bf16":
        return F.linear(input_bf16, packed["weight"], bias_bf16)
    qweight = packed["qweight"]
    group_size = int(packed["group_size"])
    weight = (
        qweight.reshape(qweight.shape[0], -1, group_size).to(torch.bfloat16)
        * packed["scale"][:, :, None]
    ).reshape_as(qweight)
    return F.linear(input_bf16, weight, bias_bf16)


def _operator(kind: str, layer: int, device: torch.device) -> dict:
    global _PREPACKED
    key = (_rung(), kind, int(layer), str(device))
    if key in _OPERATORS:
        return _OPERATORS[key]
    prepacked_path = os.environ.get("MG35_KVCODEC_PREPACKED_OPERATOR_PATH", "").strip()
    if prepacked_path:
        if _PREPACKED is None:
            _PREPACKED = torch.load(prepacked_path, map_location="cpu", weights_only=True)
            if _PREPACKED.get("format") != "mg35_worldplay_operator_int8_v1":
                raise RuntimeError(f"bad prepacked WorldPlay operator: {prepacked_path}")
            _check_center_provenance(_PREPACKED, context="prepacked operator")
        if _PREPACKED.get("rung") != _rung():
            raise RuntimeError(
                f"operator rung={_PREPACKED.get('rung')} runtime={_rung()}"
            )
        entry = _PREPACKED["entries"][f"{kind}:{layer:02d}"]
        expected_basis = str(Path(os.environ["MG35_KVCODEC_BASIS_DIR"]).resolve())
        from nanokv.artifacts import portable_codec_verified
        portable = portable_codec_verified(prepacked_path)
        def same_artifact_dir(recorded: str, expected: str) -> bool:
            """Accept a compatibility symlink after the unified-root move."""
            if portable:
                return True
            recorded, expected = str(recorded or ""), str(expected or "")
            if not recorded or not expected:
                return recorded == expected
            return Path(recorded).resolve() == Path(expected).resolve()
        if kind == "k":
            expected_decoder = ""
            if _rung() == "full":
                expected_decoder = str(Path(os.environ["MG35_KVCODEC_DECODER_DIR_K"]).resolve())
            if not same_artifact_dir(entry.get("basis_dir"), expected_basis) or not same_artifact_dir(
                entry.get("decoder_dir"), expected_decoder
            ):
                raise RuntimeError(f"prepacked K operator provenance mismatch L{layer}")
        else:
            expected_qat = ""
            if _rung() in {"c64_vqat", "full"}:
                expected_qat = str(Path(os.environ["MG35_KVCODEC_DECODER_DIR_V"]).resolve())
            if not same_artifact_dir(entry.get("basis_dir"), expected_basis) or not same_artifact_dir(
                entry.get("qat_dir"), expected_qat
            ):
                raise RuntimeError(f"prepacked V operator provenance mismatch L{layer}")
        def move(packed):
            return {
                "storage": "int8",
                "qweight": packed["qweight"].to(device=device),
                "scale": packed["scale"].to(device=device),
                "group_size": int(packed["group_size"]),
            }
        value = {
            "mean": entry["mean"].to(device=device, dtype=torch.float32),
            "encoder": move(entry["encoder"]),
            "decoder": move(entry["decoder"]),
        }
        _OPERATORS[key] = value
        return value
    if kind == "k":
        basis_root = Path(os.environ["MG35_KVCODEC_BASIS_DIR"])
        basis = torch.load(
            basis_root / "k_c64" / f"layer_{layer:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        _check_center_provenance(basis, context=f"K basis L{layer}")
        encoder = basis["components"].float()
        decoder_root = os.environ.get("MG35_KVCODEC_DECODER_DIR_K", "").strip()
        if decoder_root:
            checkpoint = torch.load(
                Path(decoder_root) / f"layer_{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("target") != "k_c64":
                raise RuntimeError(f"bad WorldPlay K decoder L{layer}: {decoder_root}")
            _check_center_provenance(checkpoint, context=f"K decoder L{layer}")
            decoder = checkpoint["decoder"].float()
        else:
            decoder = encoder.t().contiguous()
        value = {"mean": basis["mean"].float(), "encoder": encoder, "decoder": decoder}
    else:
        root = Path(os.environ["MG35_KVCODEC_DECODER_DIR_V"])
        checkpoint = torch.load(root / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "mg35_worldplay_v_c64_int4_qat_v1":
            raise RuntimeError(f"bad WorldPlay V-QAT checkpoint L{layer}: {root}")
        _check_center_provenance(checkpoint, context=f"V-QAT L{layer}")
        value = {
            "mean": checkpoint["mean"].float(),
            "encoder": checkpoint["encoder"].float(),
            "decoder": checkpoint["decoder"].float(),
        }
    value = {
        "mean": value["mean"].to(device=device, dtype=torch.float32),
        "encoder": _pack_operator(value["encoder"], device),
        "decoder": _pack_operator(value["decoder"], device),
    }
    _OPERATORS[key] = value
    return value


@torch.no_grad()
def preload_operator_weights(device: torch.device) -> None:
    """Preload every consolidated operator as static model-resident state.

    Formal timing must not charge checkpoint I/O to the first cache write.
    WorldPlay's frozen deployment does the same preload before timed rollout.
    """
    path = os.environ.get("MG35_KVCODEC_PREPACKED_OPERATOR_PATH", "").strip()
    if not path:
        raise RuntimeError(
            "formal worldplay_method requires MG35_KVCODEC_PREPACKED_OPERATOR_PATH"
        )
    global _PREPACKED
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "mg35_worldplay_operator_int8_v1":
        raise RuntimeError(f"bad prepacked WorldPlay operator: {path}")
    _check_center_provenance(payload, context="prepacked operator")
    if payload.get("rung") != _rung():
        raise RuntimeError(f"operator rung={payload.get('rung')} runtime={_rung()}")
    _PREPACKED = payload
    for name in sorted(payload["entries"]):
        kind, layer = name.split(":", 1)
        _operator(kind, int(layer), device)


def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.int16)
    unsigned = flat + 7
    if torch.any(unsigned < 0) or torch.any(unsigned >= 16):
        raise RuntimeError("INT4 codes outside frozen symmetric range [-7,7]")
    if unsigned.numel() % 2:
        unsigned = torch.cat(
            (unsigned, torch.zeros(1, device=unsigned.device, dtype=unsigned.dtype))
        )
    grouped = unsigned.reshape(-1, 2)
    return (grouped[:, 0] | (grouped[:, 1] << 4)).to(torch.uint8).contiguous()


def _unpack_int4(packed: torch.Tensor, count: int, shape) -> torch.Tensor:
    packed = packed.reshape(-1).to(torch.uint8)
    unsigned = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=1).reshape(-1)
    return unsigned[:count].to(torch.int16).sub_(7).to(torch.int8).reshape(shape)


@torch.no_grad()
def semantic_residual(value: torch.Tensor, layer: int):
    """Exact deterministic joint-head C64 used by frozen HY."""
    batch, sequence, dim = value.shape
    rows = value.float()
    clusters = int(os.environ.get("MG35_KVCODEC_C64_CLUSTERS", "64"))
    iterations = int(os.environ.get("MG35_KVCODEC_C64_ITERS", "2"))
    if clusters <= 0 or clusters > sequence or clusters > 256:
        raise ValueError(f"C64 clusters must be in [1,min(S,256)], got {clusters}")
    generator = torch.Generator(device="cpu").manual_seed(20260804 + int(layer))
    initial = torch.randperm(sequence, generator=generator)[:clusters].to(rows.device)
    centers = rows.index_select(1, initial).clone()
    row_norm = rows.square().sum(-1, keepdim=True)
    labels = torch.zeros((batch, sequence), dtype=torch.long, device=rows.device)
    for _ in range(iterations):
        distances = (
            row_norm
            + centers.square().sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(rows, centers.transpose(1, 2))
        )
        labels = distances.argmin(-1)
        assignment = F.one_hot(labels, num_classes=clusters).transpose(1, 2).float()
        counts = assignment.sum(-1)
        updated = torch.bmm(assignment, rows) / counts.clamp_min(1.0).unsqueeze(-1)
        centers = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    center_payload, decoded_centers = store_centers(centers)
    assigned = torch.gather(
        decoded_centers, 1, labels[..., None].expand(-1, -1, dim)
    )
    return rows - assigned, center_payload, labels.to(torch.uint8)


@torch.no_grad()
def encode_tensor(kind: str, layer: int, value: torch.Tensor) -> dict:
    if kind not in {"k", "v"}:
        raise ValueError(kind)
    if not _target_enabled(kind):
        return {"codec": "worldplay_raw_v1", "tensor": value.detach()}
    semantic = _rung() != "pca"
    if semantic:
        residual, centers, ids = semantic_residual(value, layer)
    else:
        residual, centers, ids = value.float(), None, None
    operator = _operator(kind, layer, value.device)
    coeff = _operator_linear(
        residual.reshape(-1, residual.shape[-1]) - operator["mean"],
        operator["encoder"],
    )
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / 7.0
    codes = (coeff / scale).round().clamp(-7, 7).to(torch.int8)
    stored_scale = scale.to(torch.bfloat16)
    return {
        "codec": "worldplay_c64_pca_int4_v1",
        "kind": kind,
        "layer": int(layer),
        "shape": tuple(int(v) for v in value.shape),
        "latent_shape": tuple(int(v) for v in codes.shape),
        "latent_numel": int(codes.numel()),
        "packed_codes": _pack_int4(codes),
        "scale": stored_scale,
        "centers": centers,
        "ids": ids,
        "semantic": semantic,
        "source_dtype": str(value.dtype).replace("torch.", ""),
    }


@torch.no_grad()
def decode_tensor(payload: dict, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if payload.get("codec") == "worldplay_raw_v1":
        return payload["tensor"].to(device=device, dtype=dtype)
    if payload.get("codec") != "worldplay_c64_pca_int4_v1":
        raise RuntimeError(f"unsupported WorldPlay payload {payload.get('codec')}")
    kind, layer = str(payload["kind"]), int(payload["layer"])
    operator = _operator(kind, layer, device)
    codes = _unpack_int4(
        payload["packed_codes"].to(device),
        int(payload["latent_numel"]),
        tuple(payload["latent_shape"]),
    )
    latent = codes.float() * payload["scale"].to(device=device, dtype=torch.float32)
    residual = _operator_linear(latent, operator["decoder"], operator["mean"])
    batch, sequence, dim = (int(v) for v in payload["shape"])
    residual = residual.reshape(batch, sequence, dim)
    if payload.get("semantic", True):
        centers = restore_centers(payload["centers"], device=device)
        ids = payload["ids"].to(device=device, dtype=torch.long)
        assigned = torch.gather(centers, 1, ids[..., None].expand(-1, -1, dim))
        residual = residual + assigned
    return residual.to(dtype)


@torch.no_grad()
def _decode_method_segments(
    parts: list[dict], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode same-layer immutable method segments with one decoder GEMM.

    Segment-local scales and C64 centers remain unchanged.  Only independent
    latent rows are concatenated before the shared linear decoder, which is
    mathematically identical and bit-exact for the frozen BF16 implementation.
    """
    first = parts[0]
    kind, layer = str(first["kind"]), int(first["layer"])
    batch, _, dim = (int(value) for value in first["shape"])
    operator = _operator(kind, layer, device)
    latent_parts = []
    for part in parts:
        if (
            part.get("codec") != "worldplay_c64_pca_int4_v1"
            or str(part["kind"]) != kind
            or int(part["layer"]) != layer
            or int(part["shape"][0]) != batch
            or int(part["shape"][2]) != dim
        ):
            raise ValueError("incompatible WorldPlay method segments")
        packed = part["packed_codes"].to(device)
        scales = part["scale"].to(device=device)
        if packed.is_cuda:
            from .nanokv_codec_kernels import unpack_int4_scaled

            latent = unpack_int4_scaled(
                packed, scales, tuple(part["latent_shape"])
            )
        else:
            codes = _unpack_int4(
                packed, int(part["latent_numel"]), tuple(part["latent_shape"])
            )
            # Preserve the reference FP32 latent for the CPU/FP32-operator
            # path.  Frozen deployment operators are INT8 and cast their
            # input to BF16 inside `_operator_linear`, exactly like the HY
            # WorldPlay implementation; forcing BF16 here made the generic
            # FP32 parity test unnecessarily lossy.
            latent = codes.float() * scales.float()
        latent_parts.append(
            latent.reshape(batch, int(part["shape"][1]), -1)
        )
    latent = torch.cat(latent_parts, dim=1)
    residual = _operator_linear(
        latent.reshape(-1, latent.shape[-1]),
        operator["decoder"],
        operator["mean"],
    ).reshape(batch, latent.shape[1], dim)
    offset = 0
    for part in parts:
        sequence = int(part["shape"][1])
        if part.get("semantic", True):
            centers = restore_centers(part["centers"], device=device)
            ids = part["ids"].to(device=device, dtype=torch.long)
            assigned = torch.gather(
                centers, 1, ids[..., None].expand(-1, -1, dim)
            )
            residual[:, offset : offset + sequence].add_(assigned)
        offset += sequence
    return residual.to(dtype)


def encode_raw_tensor(value: torch.Tensor) -> dict:
    """Store an immutable BF16 segment in a compact, cache-owned storage.

    Attention K/V tensors are views into a larger QKV projection allocation.
    ``detach`` alone therefore keeps the complete QKV backing storage alive and
    can make a logical K/V cache consume almost twice its reported bytes.  A
    cache-owned clone makes physical resident bytes equal logical K/V bytes.
    """
    return {
        "codec": "worldplay_raw_v1",
        "tensor": value.detach().clone(memory_format=torch.contiguous_format),
    }


def is_payload(value: Any) -> bool:
    if isinstance(value, dict) and value.get("codec") in {
        "worldplay_kvtc_layer_v1",
        "worldplay_kvtc_stream_layer_v1",
    }:
        return True
    if isinstance(value, dict):
        from .training_free_kv_baselines import is_payload as is_baseline_payload
        if is_baseline_payload(value):
            return True
    return isinstance(value, dict) and value.get("codec") in {
        "worldplay_raw_v1",
        "worldplay_c64_pca_int4_v1",
        "worldplay_segments_v1",
    }


def value_shape(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return tuple(int(v) for v in value.shape)
    if value.get("codec") in {
        "worldplay_kvtc_layer_v1",
        "worldplay_kvtc_stream_layer_v1",
    }:
        from .kvtc_codec import layer_shape
        return layer_shape(value)
    from .training_free_kv_baselines import is_payload as is_baseline_payload
    if is_baseline_payload(value):
        from .training_free_kv_baselines import value_shape as baseline_shape
        return baseline_shape(value)
    if value.get("codec") == "worldplay_raw_v1":
        return tuple(int(v) for v in value["tensor"].shape)
    if value.get("codec") == "worldplay_c64_pca_int4_v1":
        return tuple(int(v) for v in value["shape"])
    if value.get("codec") == "worldplay_segments_v1":
        shapes = [value_shape(item) for item in value["segments"]]
        if not shapes:
            return None
        return (shapes[0][0], sum(shape[1] for shape in shapes), shapes[0][2])
    raise RuntimeError(f"unsupported cache value {value.get('codec')}")


def decode_value(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    if value.get("codec") in {
        "worldplay_kvtc_layer_v1",
        "worldplay_kvtc_stream_layer_v1",
    }:
        from .kvtc_codec import decode_value as decode_kvtc_value
        return decode_kvtc_value(value, dtype, device)
    if value.get("codec") == "worldplay_segments_v1":
        parts = value["segments"]
        if parts and all(
            part.get("codec") == "worldplay_c64_pca_int4_v1" for part in parts
        ):
            return _decode_method_segments(parts, dtype=dtype, device=device)
        parts = [decode_value(item, dtype=dtype, device=device) for item in value["segments"]]
        return torch.cat(parts, dim=1) if parts else None
    from .training_free_kv_baselines import is_payload as is_baseline_payload
    if is_baseline_payload(value):
        from .training_free_kv_baselines import decode_value as decode_baseline
        return decode_baseline(value, dtype=dtype, device=device)
    return decode_tensor(value, dtype=dtype, device=device)


def decode_pair_values(
    key: Any, value: Any, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Decode a K/V cache pair while preserving paired-baseline kernels."""
    if key is None or value is None:
        return (
            decode_value(key, dtype=dtype, device=device),
            decode_value(value, dtype=dtype, device=device),
        )
    from .training_free_kv_baselines import is_payload as is_baseline_payload

    use_batched = os.environ.get(
        "MG35_BASELINE_BATCHED_SEGMENT_DECODE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if is_baseline_payload(key) and is_baseline_payload(value):
        if use_batched:
            from .training_free_kv_baselines import decode_pair

            return decode_pair(key, value, dtype=dtype, device=device)
        return (
            decode_value(key, dtype=dtype, device=device),
            decode_value(value, dtype=dtype, device=device),
        )
    if (
        isinstance(key, dict)
        and isinstance(value, dict)
        and key.get("codec") == "worldplay_segments_v1"
        and value.get("codec") == "worldplay_segments_v1"
    ):
        key_parts, value_parts = key["segments"], value["segments"]
        if (
            key_parts
            and len(key_parts) == len(value_parts)
            and all(
                is_baseline_payload(k) and is_baseline_payload(v)
                for k, v in zip(key_parts, value_parts)
            )
        ):
            # Segment batching is tensor-exact, but changing the allocation
            # schedule can bifurcate a long autoregressive rollout.  Keep the
            # quality-frozen reference order by default; the opt-in path is
            # reserved for paired efficiency/quality revalidation.
            if not use_batched:
                return (
                    decode_value(key, dtype=dtype, device=device),
                    decode_value(value, dtype=dtype, device=device),
                )
            from .training_free_kv_baselines import decode_segment_pairs

            return decode_segment_pairs(
                key_parts, value_parts, dtype=dtype, device=device
            )
    return (
        decode_value(key, dtype=dtype, device=device),
        decode_value(value, dtype=dtype, device=device),
    )


def concat_values(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if not is_payload(left) or not is_payload(right):
        raise RuntimeError("WorldPlay packed-cache concat received a raw/payload mixture")
    left_parts = left["segments"] if left.get("codec") == "worldplay_segments_v1" else [left]
    right_parts = right["segments"] if right.get("codec") == "worldplay_segments_v1" else [right]
    return {"codec": "worldplay_segments_v1", "segments": list(left_parts) + list(right_parts)}


def select_value(
    value: Any,
    token_idx: torch.Tensor,
    *,
    kind: str,
    layer: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """Select tokens while preserving complete immutable payload segments."""
    if torch.is_tensor(value):
        return value.index_select(1, token_idx).contiguous()
    if value.get("codec") == "worldplay_kvtc_stream_layer_v1":
        from .kvtc_codec import select_stream_layer

        return select_stream_layer(value, token_idx)
    parts = value["segments"] if value.get("codec") == "worldplay_segments_v1" else [value]
    wanted = token_idx.detach().cpu().tolist()
    selected_parts = []
    selected_ranges = []
    offset = 0
    cursor = 0
    aligned = True
    for part in parts:
        length = int(value_shape(part)[1])
        local = []
        while cursor < len(wanted) and wanted[cursor] < offset + length:
            local.append(wanted[cursor])
            cursor += 1
        if local:
            if local != list(range(offset, offset + length)):
                aligned = False
            selected_ranges.append((part, offset, length, local))
            if local == list(range(offset, offset + length)):
                selected_parts.append(part)
        offset += length
    if aligned and cursor == len(wanted):
        if not selected_parts:
            return None
        if len(selected_parts) == 1:
            return selected_parts[0]
        return {"codec": "worldplay_segments_v1", "segments": selected_parts}
    if cursor != len(wanted) or wanted != sorted(wanted):
        raise RuntimeError(
            "segmented cache selection expects sorted in-range token indices"
        )

    # Partial-frame window compaction must not materialize the complete history.
    # Reuse fully retained immutable segments and decode/select only a boundary
    # segment that is actually cut.  The previous implementation decoded and
    # concatenated every segment before index_select, temporarily keeping old
    # history + full decoded history + the compacted result resident together.
    from .training_free_kv_baselines import is_payload as is_baseline_payload

    compacted_parts = []
    for part, part_offset, length, global_indices in selected_ranges:
        local_indices = [int(index - part_offset) for index in global_indices]
        if local_indices == list(range(length)):
            compacted_parts.append(part)
            continue
        part_tensor = decode_value(part, dtype=dtype, device=device)
        local_idx = torch.as_tensor(
            local_indices, device=part_tensor.device, dtype=torch.long
        )
        selected = part_tensor.index_select(1, local_idx).contiguous()
        if part.get("codec") == "worldplay_raw_v1":
            compacted = encode_raw_tensor(selected)
        elif is_baseline_payload(part):
            from .training_free_kv_baselines import encode_single

            codec = part["codec"]
            mode = "qvg" if "qvg" in codec else "turboquant"
            compacted = encode_single(
                mode, selected, target=kind, layer_idx=layer
            )
        else:
            compacted = encode_tensor(kind, layer, selected)
        compacted_parts.append(compacted)
    if not compacted_parts:
        return None
    if len(compacted_parts) == 1:
        return compacted_parts[0]
    return {"codec": "worldplay_segments_v1", "segments": compacted_parts}


def payload_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(payload_bytes(v) for k, v in value.items() if not str(k).startswith("_"))
    if isinstance(value, (list, tuple)):
        return sum(payload_bytes(v) for v in value)
    return 0
