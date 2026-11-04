"""Paper-faithful KVTC adapter for the HY pre-position vision KV cache.

This module intentionally does not share transforms or corrections with the
project codec.  A KVTC artifact contains a joint PCA over ``layers * heads *
head_dim`` and a static DP quantization plan.  Consecutive layers share one
payload; an attention layer reconstructs only its own decoder slice.

DEFLATE is deliberately not used on the online path.  KVTC itself describes
entropy coding as a storage/transfer mode and skips it for serving.
"""

from __future__ import annotations

import os
import atexit
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


_ARTIFACT_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_STATS = {
    "groups": 0,
    "original_bytes": 0,
    "payload_bytes": 0,
    "decode_calls": 0,
    "short_context_bypass_groups": 0,
    "current_original_bytes": 0,
    "current_payload_bytes": 0,
    "final_original_bytes": 0,
    "final_payload_bytes": 0,
}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("HY_KVTC_ENABLE")


def dump_calibration_sample(layer_idx: int, kind: str, tensor: torch.Tensor) -> None:
    """Persist a small, token-matched pre-KV calibration shard."""
    if not _flag("HY_KVTC_CALIB_DUMP_ENABLE"):
        return
    chunk = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    positions = {int(v) for v in os.environ.get("HY_KVTC_CALIB_DUMP_POSITIONS", "5").split(",") if v}
    target = os.environ.get("HY_KV_MULTIPOS_TARGET", "kv").strip().lower()
    if chunk not in positions or kind not in {"k", "v"} or target not in {kind, "kv"}:
        return
    sink = int(os.environ.get("HY_KVTC_SINK_TOKENS", "4"))
    window = int(os.environ.get("HY_KVTC_WINDOW_TOKENS", "128"))
    first, stop = min(sink, tensor.shape[2]), max(min(sink, tensor.shape[2]), tensor.shape[2] - window)
    available = max(stop - first, 0)
    if available == 0:
        return
    count = min(int(os.environ.get("HY_KVTC_CALIB_TOKENS_PER_SNAPSHOT", "32")), available)
    token_ids = torch.linspace(first, stop - 1, count, device=tensor.device).round().long()
    sample = tensor[:, :, token_ids, :].detach().float().cpu()
    scale = sample.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    codes = (sample / scale).round().clamp(-127, 127).to(torch.int8)
    root = Path(os.environ.get("HY_KVTC_CALIB_DUMP_DIR", "kv_ae_outputs/ae16x_v2/kvtc_calibration"))
    case = os.environ.get("HY_KV_MULTIPOS_CASE_ID", "unknown")
    path = root / case / f"position_{chunk:02d}" / f"layer{layer_idx:02d}_target-{kind}-sample.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _flag("HY_KV_MULTIPOS_FORCE"):
        return
    torch.save({
        "format": "hy_kvtc_sample_v1", "layer_idx": int(layer_idx), "kind": kind,
        "token_ids": token_ids.cpu(), "shape": tuple(tensor.shape),
        "tensor_int8": codes, "int8_scale": scale.to(torch.bfloat16),
    }, path)


def _artifact_root(kind: str | None = None) -> Path:
    # Formal calibration may fit K and V independently to avoid retaining both
    # raw dump trees.  Keep the original single-root interface as a fallback.
    value = ""
    if kind in {"k", "v"}:
        value = os.environ.get(f"HY_KVTC_ARTIFACT_DIR_{kind.upper()}", "").strip()
    if not value:
        value = os.environ.get("HY_KVTC_ARTIFACT_DIR", "").strip()
    if not value:
        suffix = f"_{kind.upper()}" if kind in {"k", "v"} else ""
        raise RuntimeError(
            f"HY_KVTC_ENABLE=1 requires HY_KVTC_ARTIFACT_DIR{suffix} "
            "or HY_KVTC_ARTIFACT_DIR"
        )
    return Path(value)


def _load_manifest(kind: str = "k") -> Dict[str, Any]:
    root = _artifact_root(kind)
    key = (str(root.resolve()), f"manifest:{kind}")
    if key not in _ARTIFACT_CACHE:
        manifest = torch.load(root / "manifest.pt", map_location="cpu", weights_only=False)
        if manifest.get("format") != "hy_kvtc_v1":
            raise RuntimeError(f"Unsupported KVTC artifact format: {manifest.get('format')}")
        if kind not in manifest.get("kinds", (kind,)):
            raise RuntimeError(f"KVTC {kind.upper()} manifest does not contain kind={kind}: {root}")
        _ARTIFACT_CACHE[key] = manifest
    return _ARTIFACT_CACHE[key]


def _load_group(group_start: int, kind: str, device: torch.device) -> Dict[str, Any]:
    root = str(_artifact_root(kind).resolve())
    cache_key = (root, f"{group_start}:{kind}:{device}")
    entry = _ARTIFACT_CACHE.get(cache_key)
    if entry is None:
        path = Path(root) / f"group_{group_start:02d}_{kind}.pt"
        entry = torch.load(path, map_location="cpu", weights_only=False)
        if entry.get("format") != "hy_kvtc_group_v1":
            raise RuntimeError(f"Bad KVTC group artifact: {path}")
        entry["mean"] = entry["mean"].to(device=device, dtype=torch.float32)
        entry["basis"] = entry["basis"].to(device=device, dtype=torch.bfloat16)
        _ARTIFACT_CACHE[cache_key] = entry
    return entry


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for k, v in value.items() if not k.startswith("_"))
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _pack_lowbit(values: torch.Tensor, bits: int) -> torch.Tensor:
    values = values.to(torch.uint8).flatten()
    if bits == 8:
        return values.contiguous()
    if bits not in (2, 4):
        raise RuntimeError(f"Online KVTC supports INT2/INT4/FP8, got bits={bits}")
    per_byte = 8 // bits
    pad = (-values.numel()) % per_byte
    if pad:
        values = torch.cat((values, torch.zeros(pad, dtype=torch.uint8, device=values.device)))
    values = values.reshape(-1, per_byte).to(torch.int16)
    packed = torch.zeros(values.shape[0], dtype=torch.int16, device=values.device)
    for offset in range(per_byte):
        packed |= values[:, offset] << (offset * bits)
    return packed.to(torch.uint8)


def _unpack_lowbit(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    if bits == 8:
        return packed[:count]
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int16) * bits
    values = ((packed.to(torch.int16)[:, None] >> shifts[None]) & ((1 << bits) - 1))
    return values.to(torch.uint8).flatten()[:count]


def _encode_coefficients(coeff: torch.Tensor, plan: Sequence[Dict[str, int]]) -> Dict[str, Any]:
    rows = int(coeff.shape[0])
    sections: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[int, str, int], List[int]] = {}
    for item in plan:
        bits = int(item["bits"])
        if bits:
            quant_type = str(item.get("type", f"int{bits}"))
            grouped.setdefault((bits, quant_type, int(item["size"])), []).append(int(item["start"]))
    for (bits, quant_type, width), starts in grouped.items():
        start_tensor = torch.tensor(starts, device=coeff.device, dtype=torch.long)
        indices = start_tensor[:, None] + torch.arange(width, device=coeff.device)[None]
        block = coeff[:, indices].float()  # [rows, groups, width]
        if quant_type == "fp8":
            shift = block.mean(dim=(0, 2), keepdim=True)
            scale = (block - shift).abs().amax(dim=(0, 2), keepdim=True).clamp_min(1e-8) / 448.0
            codes = ((block - shift) / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            shift = block.amin(dim=(0, 2), keepdim=True)
            high = block.amax(dim=(0, 2), keepdim=True)
            scale = ((high - shift) / float((1 << bits) - 1)).clamp_min(1e-8)
            codes = ((block - shift) / scale).round().clamp_(0, (1 << bits) - 1).to(torch.uint8)
        sections.append({
            "starts": starts,
            "size": width,
            "groups": len(starts),
            "bits": bits,
            "type": quant_type,
            "count": codes.numel(),
            "codes": _pack_lowbit(codes, bits),
            "scale": scale.to(torch.bfloat16),
            "shift": shift.to(torch.bfloat16),
        })
    return {"rows": rows, "rank": int(coeff.shape[1]), "sections": sections}


def _decode_coefficients(payload: Dict[str, Any], device: torch.device) -> torch.Tensor:
    cached = payload.get("_decoded")
    if torch.is_tensor(cached) and cached.device == device:
        return cached
    rows, rank = int(payload["rows"]), int(payload["rank"])
    output = torch.zeros((rows, rank), device=device, dtype=torch.bfloat16)
    for section in payload["sections"]:
        width, bits, groups = int(section["size"]), int(section["bits"]), int(section["groups"])
        codes = _unpack_lowbit(section["codes"].to(device), bits, int(section["count"]))
        if section.get("type") == "fp8":
            codes = codes.view(torch.float8_e4m3fn).reshape(rows, groups, width).to(torch.float32)
        else:
            codes = codes.reshape(rows, groups, width).to(torch.float32)
        scale = section["scale"].to(device=device, dtype=torch.float32)
        shift = section["shift"].to(device=device, dtype=torch.float32)
        starts = torch.tensor(section["starts"], device=device, dtype=torch.long)
        indices = starts[:, None] + torch.arange(width, device=device)[None]
        output[:, indices.reshape(-1)] = (codes * scale + shift).reshape(rows, -1).to(torch.bfloat16)
    payload["_decoded"] = output
    return output


def _encode_group(caches: Sequence[Dict[str, Any]], group_start: int, group_end: int, kind: str) -> Dict[str, Any]:
    name = "k_vision" if kind == "k" else "v_vision"
    tensors = [caches[layer][name] for layer in range(group_start, group_end)]
    if not all(torch.is_tensor(tensor) and tensor.ndim == 4 for tensor in tensors):
        raise RuntimeError(f"KVTC expected raw [B,H,T,D] {kind} tensors for layers {group_start}:{group_end}")
    shape = tuple(int(v) for v in tensors[0].shape)
    if any(tuple(t.shape) != shape for t in tensors):
        raise RuntimeError("KVTC layer group contains mismatched cache shapes")
    batch, heads, tokens, head_dim = shape
    manifest = _load_manifest(kind)
    sink_len = min(int(manifest.get("sink_tokens", 4)), tokens)
    window_len = min(int(manifest.get("window_tokens", 128)), max(tokens - sink_len, 0))
    middle_start, middle_end = sink_len, tokens - window_len
    middle_tokens = middle_end - middle_start
    rows = batch * middle_tokens
    if middle_tokens <= 0:
        raise RuntimeError("KVTC internal error: short context was not bypassed")
    # Same token is one PCA sample; layer/head/channel form the feature axis.
    middle = [tensor[:, :, middle_start:middle_end, :] for tensor in tensors]
    matrix = torch.stack(middle, dim=1).permute(0, 3, 1, 2, 4).reshape(rows, -1)
    entry = _load_group(group_start, kind, matrix.device)
    expected = (group_end - group_start) * heads * head_dim
    if int(entry["feature_dim"]) != expected:
        raise RuntimeError(f"KVTC artifact feature_dim={entry['feature_dim']} but runtime={expected}")
    centered = matrix.float() - entry["mean"]
    coeff = centered.to(torch.bfloat16) @ entry["basis"]
    quantized = _encode_coefficients(coeff, entry["plan"])
    shared = {
        "codec": "kvtc_shared_v1",
        "kind": kind,
        "group_start": group_start,
        "group_end": group_end,
        "shape": shape,
        "sink_len": sink_len,
        "window_len": window_len,
        "protected_sink": torch.stack([tensor[:, :, :sink_len, :] for tensor in tensors], dim=0),
        "protected_window": torch.stack([tensor[:, :, middle_end:, :] for tensor in tensors], dim=0),
        "quantized": quantized,
    }
    _STATS["original_bytes"] += sum(t.numel() * t.element_size() for t in tensors)
    _STATS["payload_bytes"] += _tensor_bytes(shared)
    _STATS["current_original_bytes"] += sum(t.numel() * t.element_size() for t in tensors)
    _STATS["current_payload_bytes"] += _tensor_bytes(shared)
    return shared


def compress_ready_group(caches: Sequence[Dict[str, Any]], layer_idx: int) -> None:
    """Compress a completed consecutive layer group in-place."""
    if not enabled():
        return
    manifest = _load_manifest("k")
    manifest_v = _load_manifest("v")
    for field in ("num_layers", "layer_group_size", "sink_tokens", "window_tokens"):
        if manifest.get(field) != manifest_v.get(field):
            raise RuntimeError(
                f"KVTC K/V manifest mismatch for {field}: "
                f"{manifest.get(field)} != {manifest_v.get(field)}"
            )
    group_size = int(manifest["layer_group_size"])
    total_layers = int(manifest["num_layers"])
    group_start = (int(layer_idx) // group_size) * group_size
    group_end = min(group_start + group_size, total_layers)
    if int(layer_idx) != group_end - 1:
        return
    if (
        _flag("HY_KV_CODEC_ENABLE")
        or _flag("HY_QVG_ENABLE")
        or _flag("HY_KV_AE3D_ENABLE")
        or _flag("HY_JENSEN_KV_ENABLE")
        or _flag("HY_TURBOQUANT_ENABLE")
    ):
        raise RuntimeError(
            "KVTC baseline must run without project codec, QVG, AE, Jensen, or TurboQuant"
        )
    sample = caches[group_start].get("k_vision")
    tokens = int(sample.shape[2]) if torch.is_tensor(sample) else 0
    if tokens <= int(manifest.get("sink_tokens", 4)) + int(manifest.get("window_tokens", 128)):
        _STATS["short_context_bypass_groups"] += 1
        return
    if group_start == 0:
        _STATS["current_original_bytes"] = 0
        _STATS["current_payload_bytes"] = 0
    shared_by_kind = {kind: _encode_group(caches, group_start, group_end, kind) for kind in ("k", "v")}
    for layer in range(group_start, group_end):
        for kind, name in (("k", "k_vision"), ("v", "v_vision")):
            caches[layer][name] = {
                "codec": "kvtc_layer_v1",
                "kind": kind,
                "layer_idx": layer,
                "shared": shared_by_kind[kind],
            }
    _STATS["groups"] += 1
    if group_end == total_layers:
        _STATS["final_original_bytes"] = _STATS["current_original_bytes"]
        _STATS["final_payload_bytes"] = _STATS["current_payload_bytes"]


def is_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") == "kvtc_layer_v1"


def decompress_pair(key_value: Any, value_value: Any, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not is_payload(key_value):
        return None
    if not is_payload(value_value):
        raise RuntimeError("KVTC K/V payload mismatch")
    restored = []
    for payload in (key_value, value_value):
        shared = payload["shared"]
        group_start = int(shared["group_start"])
        layer_idx = int(payload["layer_idx"])
        entry = _load_group(group_start, payload["kind"], device)
        coeff = _decode_coefficients(shared["quantized"], device)
        batch, heads, tokens, head_dim = (int(v) for v in shared["shape"])
        local = layer_idx - group_start
        start = local * heads * head_dim
        stop = start + heads * head_dim
        basis_slice = entry["basis"][start:stop]
        decoded = coeff @ basis_slice.transpose(0, 1)
        decoded = decoded.float() + entry["mean"][start:stop]
        sink_len, window_len = int(shared["sink_len"]), int(shared["window_len"])
        middle_tokens = tokens - sink_len - window_len
        decoded = decoded.reshape(batch, middle_tokens, heads, head_dim).permute(0, 2, 1, 3)
        output = torch.empty((batch, heads, tokens, head_dim), device=device, dtype=dtype)
        output[:, :, sink_len : tokens - window_len, :] = decoded.to(dtype=dtype)
        output[:, :, :sink_len, :] = shared["protected_sink"][local].to(device=device, dtype=dtype)
        if window_len:
            output[:, :, tokens - window_len :, :] = shared["protected_window"][local].to(device=device, dtype=dtype)
        restored.append(output)
        _STATS["decode_calls"] += 1
    # Last layer is the final consumer in a normal sequential transformer pass.
    shared = key_value["shared"]
    if int(key_value["layer_idx"]) == int(shared["group_end"]) - 1:
        key_value["shared"]["quantized"].pop("_decoded", None)
        value_value["shared"]["quantized"].pop("_decoded", None)
    return restored[0], restored[1]


def stats() -> Dict[str, Any]:
    result = dict(_STATS)
    result["actual_ratio"] = result["original_bytes"] / max(result["payload_bytes"], 1)
    result["final_payload_ratio"] = result["final_original_bytes"] / max(result["final_payload_bytes"], 1)
    return result


def _write_summary() -> None:
    path = os.environ.get("HY_KVTC_SUMMARY", "").strip()
    if not path or not enabled():
        return
    result = stats()
    roots = {_artifact_root(kind).resolve() for kind in ("k", "v")}
    result["artifact_bytes"] = sum(
        p.stat().st_size for root in roots for p in root.glob("*.pt")
    )
    result["artifact_roots"] = {kind: str(_artifact_root(kind)) for kind in ("k", "v")}
    result["final_system_ratio_with_artifact"] = result["final_original_bytes"] / max(
        result["final_payload_bytes"] + result["artifact_bytes"], 1
    )
    if torch.cuda.is_available():
        result["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


atexit.register(_write_summary)
