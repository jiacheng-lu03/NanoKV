"""HY-WorldPlay KVTC baseline port for Matrix-Game-3.5.

The quantizer, DP plan interpretation, raw sink/window protection, shared group
payload and layer-slice decoder follow HY's frozen `kvtc_adapter.py`.  Tensor
layout is adapted from [B,H,T,Dh] to MG3.5's equivalent [B,T,H*Dh].
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


_ARTIFACTS = {}
_PENDING = {}
_STATS = {"groups": 0, "original_bytes": 0, "payload_bytes": 0}
_STREAM_SLICE_CACHE = {}
_PROFILE_EVENTS = []
_PROFILE_COUNTS = {}


def _profile_begin():
    if not os.environ.get("MG35_KVTC_PROFILE_PATH", "").strip():
        return None
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, time.perf_counter_ns()


def _profile_end(label: str, marker, **metadata) -> None:
    _PROFILE_COUNTS[label] = _PROFILE_COUNTS.get(label, 0) + 1
    if marker is None:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    start, cpu_start_ns = marker
    _PROFILE_EVENTS.append((label, start, end, time.perf_counter_ns() - cpu_start_ns, metadata))


def write_profile() -> None:
    """Write opt-in diagnostic timings without changing the default KVTC path."""
    path_text = os.environ.get("MG35_KVTC_PROFILE_PATH", "").strip()
    if not path_text:
        return
    torch.cuda.synchronize()
    rows = []
    totals = {}
    for label, start, end, cpu_ns, metadata in _PROFILE_EVENTS:
        cuda_ms = float(start.elapsed_time(end))
        row = {"label": label, "cuda_ms": cuda_ms, "cpu_ms": cpu_ns / 1e6, **metadata}
        rows.append(row)
        total = totals.setdefault(label, {"calls": 0, "cuda_ms": 0.0, "cpu_ms": 0.0})
        total["calls"] += 1
        total["cuda_ms"] += cuda_ms
        total["cpu_ms"] += cpu_ns / 1e6
    for label, count in _PROFILE_COUNTS.items():
        totals.setdefault(label, {"calls": 0, "cuda_ms": 0.0, "cpu_ms": 0.0})
        totals[label]["observed_calls"] = count
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"totals": totals, "events": rows}, indent=2) + "\n")


def _root() -> Path:
    return Path(os.environ["MG35_KVTC_ARTIFACT_DIR"])


def _manifest() -> dict:
    key = (str(_root().resolve()), "manifest")
    if key not in _ARTIFACTS:
        value = torch.load(_root() / "manifest.pt", map_location="cpu", weights_only=False)
        if value.get("format") != "mg35_worldplay_kvtc_v1":
            raise RuntimeError(f"unsupported MG3.5 WorldPlay KVTC manifest: {_root()}")
        _ARTIFACTS[key] = value
    return _ARTIFACTS[key]


def _group(start: int, kind: str, device: torch.device) -> dict:
    path = _root() / f"group_{start:02d}_{kind}.pt"
    key = (str(path.resolve()), str(device))
    if key not in _ARTIFACTS:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("format") != "mg35_worldplay_kvtc_group_v1":
            raise RuntimeError(f"unsupported MG3.5 WorldPlay KVTC artifact: {path}")
        value["mean"] = value["mean"].to(device=device, dtype=torch.float32)
        value["basis"] = value["basis"].to(device=device, dtype=torch.bfloat16)
        _ARTIFACTS[key] = value
    return _ARTIFACTS[key]


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for k, v in value.items() if not str(k).startswith("_"))
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def _first_tensor(value: Any):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for child in value.values():
            found = _first_tensor(child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _first_tensor(child)
            if found is not None:
                return found
    return None


def _pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    values = values.to(torch.uint8).flatten()
    if bits == 8:
        return values.contiguous()
    if bits not in (2, 4):
        raise RuntimeError(f"KVTC supports INT2/INT4/FP8, got bits={bits}")
    per_byte = 8 // bits
    pad = (-values.numel()) % per_byte
    if pad:
        values = torch.cat((values, torch.zeros(pad, dtype=torch.uint8, device=values.device)))
    values = values.reshape(-1, per_byte).to(torch.int16)
    packed = torch.zeros(values.shape[0], dtype=torch.int16, device=values.device)
    for offset in range(per_byte):
        packed |= values[:, offset] << (offset * bits)
    return packed.to(torch.uint8)


def _unpack(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    if bits == 8:
        return packed[:count]
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int16) * bits
    values = (packed.to(torch.int16)[:, None] >> shifts[None]) & ((1 << bits) - 1)
    return values.to(torch.uint8).flatten()[:count]


def encode_coefficients(coeff: torch.Tensor, plan) -> dict:
    rows = int(coeff.shape[0])
    sections = []
    grouped = {}
    for item in plan:
        bits = int(item["bits"])
        if bits:
            quant_type = str(item.get("type", f"int{bits}"))
            grouped.setdefault((bits, quant_type, int(item["size"])), []).append(int(item["start"]))
    for (bits, quant_type, width), starts in grouped.items():
        start_tensor = torch.tensor(starts, device=coeff.device, dtype=torch.long)
        indices = start_tensor[:, None] + torch.arange(width, device=coeff.device)[None]
        block = coeff[:, indices].float()
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
            "codes": _pack(codes, bits),
            "scale": scale.to(torch.bfloat16),
            "shift": shift.to(torch.bfloat16),
        })
    return {"rows": rows, "rank": int(coeff.shape[1]), "sections": sections}


def decode_coefficients(payload: dict, device: torch.device) -> torch.Tensor:
    cached = payload.get("_decoded")
    if torch.is_tensor(cached) and cached.device == device:
        _PROFILE_COUNTS["decode_coefficients_hit"] = _PROFILE_COUNTS.get("decode_coefficients_hit", 0) + 1
        return cached
    marker = _profile_begin()
    rows, rank = int(payload["rows"]), int(payload["rank"])
    output = torch.zeros((rows, rank), device=device, dtype=torch.bfloat16)
    for section in payload["sections"]:
        width, bits, groups = int(section["size"]), int(section["bits"]), int(section["groups"])
        codes = _unpack(section["codes"].to(device), bits, int(section["count"]))
        if section.get("type") == "fp8":
            codes = codes.view(torch.float8_e4m3fn).reshape(rows, groups, width).float()
        else:
            codes = codes.reshape(rows, groups, width).float()
        scale = section["scale"].to(device=device, dtype=torch.float32)
        shift = section["shift"].to(device=device, dtype=torch.float32)
        starts = torch.tensor(section["starts"], device=device, dtype=torch.long)
        indices = starts[:, None] + torch.arange(width, device=device)[None]
        output[:, indices.reshape(-1)] = (codes * scale + shift).reshape(rows, -1).to(torch.bfloat16)
    payload["_decoded"] = output
    _profile_end(
        "decode_coefficients_miss", marker, rows=rows, rank=rank,
        sections=len(payload["sections"]),
    )
    return output


def encode_group(
    tensors: list[torch.Tensor], start: int, kind: str, *,
    sink_len: int | None = None, window_len: int | None = None,
) -> dict:
    shape = tuple(int(v) for v in tensors[0].shape)
    if any(tuple(item.shape) != shape for item in tensors):
        raise RuntimeError(f"KVTC group {start} contains mismatched tensors")
    manifest = _manifest()
    sink_len = min(
        int(manifest.get("sink_tokens", 4)) if sink_len is None else int(sink_len),
        shape[1],
    )
    window_len = min(
        int(manifest.get("window_tokens", 128)) if window_len is None else int(window_len),
        max(shape[1] - sink_len, 0),
    )
    middle_end = shape[1] - window_len
    if middle_end <= sink_len:
        # Match frozen HY: a short context is left as the original raw cache;
        # it is not copied into a synthetic shared wrapper.
        return None
    marker = _profile_begin()
    matrix = torch.cat([item[:, sink_len:middle_end] for item in tensors], dim=-1)
    matrix = matrix.reshape(-1, matrix.shape[-1])
    _profile_end(
        "encode_prepare", marker, kind=kind, group_start=start,
        rows=int(matrix.shape[0]), feature_dim=int(matrix.shape[1]),
    )
    marker = _profile_begin()
    artifact = _group(start, kind, matrix.device)
    _profile_end("artifact_load", marker, kind=kind, group_start=start)
    if int(artifact["feature_dim"]) != int(matrix.shape[-1]):
        raise RuntimeError(
            f"KVTC feature mismatch {start} {kind}: "
            f"{artifact['feature_dim']} != {matrix.shape[-1]}"
        )
    marker = _profile_begin()
    centered = (matrix.float() - artifact["mean"]).to(torch.bfloat16)
    _profile_end(
        "encode_center", marker, kind=kind, group_start=start,
        rows=int(matrix.shape[0]), feature_dim=int(matrix.shape[1]),
    )
    marker = _profile_begin()
    coeff = centered @ artifact["basis"]
    _profile_end(
        "encode_project", marker, kind=kind, group_start=start,
        rows=int(matrix.shape[0]), feature_dim=int(matrix.shape[1]),
        rank=int(coeff.shape[1]),
    )
    marker = _profile_begin()
    quantized = encode_coefficients(coeff, artifact["plan"])
    _profile_end(
        "encode_quantize", marker, kind=kind, group_start=start,
        rows=int(coeff.shape[0]), rank=int(coeff.shape[1]),
    )
    return {
        "codec": "worldplay_kvtc_shared_v1",
        "kind": kind,
        "group_start": start,
        "group_end": start + len(tensors),
        "shape": shape,
        "sink_len": sink_len,
        "window_len": window_len,
        "protected_sink": torch.stack([item[:, :sink_len] for item in tensors]),
        "protected_window": torch.stack([item[:, middle_end:] for item in tensors]),
        "quantized": quantized,
    }


def decode_layer(shared: dict, layer_idx: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    start = int(shared["group_start"])
    local = int(layer_idx) - start
    artifact = _group(start, str(shared["kind"]), device)
    coeff = decode_coefficients(shared["quantized"], device)
    batch, tokens, channels = (int(v) for v in shared["shape"])
    feature_start, feature_stop = local * channels, (local + 1) * channels
    marker = _profile_begin()
    decoded = coeff @ artifact["basis"][feature_start:feature_stop].transpose(0, 1)
    _profile_end(
        "decode_reconstruct_gemm", marker, kind=str(shared["kind"]),
        group_start=start, layer=layer_idx, rows=int(coeff.shape[0]),
        rank=int(coeff.shape[1]), channels=channels,
    )
    marker = _profile_begin()
    decoded = decoded.float() + artifact["mean"][feature_start:feature_stop]
    sink_len, window_len = int(shared["sink_len"]), int(shared["window_len"])
    middle_tokens = tokens - sink_len - window_len
    output = torch.empty((batch, tokens, channels), device=device, dtype=dtype)
    output[:, sink_len : tokens - window_len] = decoded.reshape(batch, middle_tokens, channels).to(dtype)
    output[:, :sink_len] = shared["protected_sink"][local].to(device=device, dtype=dtype)
    if window_len:
        output[:, tokens - window_len :] = shared["protected_window"][local].to(device=device, dtype=dtype)
    _profile_end(
        "decode_reconstruct_post", marker, kind=str(shared["kind"]),
        group_start=start, layer=layer_idx, tokens=tokens, channels=channels,
    )
    return output


def is_layer_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") == "worldplay_kvtc_layer_v1"


def is_stream_layer_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("codec") == "worldplay_kvtc_stream_layer_v1"


def layer_shape(value: dict) -> tuple[int, int, int]:
    shared = value["shared"]
    return tuple(int(v) for v in shared["shape"])


def decode_value(value: dict, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if is_stream_layer_payload(value):
        shared = value["shared"]
        layer_idx = int(value["layer_idx"])
        start = int(shared["group_start"])
        local = layer_idx - start
        artifact = _group(start, str(shared["kind"]), device)
        compressed = [piece for piece in shared["pieces"] if piece["type"] == "compressed"]
        coeff = shared.get("_decoded_coefficients")
        if not torch.is_tensor(coeff) or coeff.device != device:
            coeff_parts = []
            for piece in compressed:
                source = piece["shared"]
                batch, tokens, _ = (int(v) for v in source["shape"])
                current = decode_coefficients(source["quantized"], device)
                coeff_parts.append(current.reshape(batch, tokens, -1))
            if coeff_parts:
                coeff = torch.cat(coeff_parts, dim=1).contiguous()
            else:
                batch = int(shared["shape"][0])
                coeff = torch.empty(
                    (batch, 0, int(artifact["basis"].shape[1])),
                    device=device, dtype=torch.bfloat16,
                )
            shared["_decoded_coefficients"] = coeff
        channels = int(shared["shape"][2])
        feature_start, feature_stop = local * channels, (local + 1) * channels
        if int(coeff.shape[1]):
            marker = _profile_begin()
            decoded_middle = (
                coeff.reshape(-1, coeff.shape[-1])
                @ artifact["basis"][feature_start:feature_stop].transpose(0, 1)
            )
            _profile_end(
                "decode_reconstruct_gemm", marker, kind=str(shared["kind"]),
                group_start=start, layer=layer_idx,
                rows=int(coeff.shape[0] * coeff.shape[1]),
                rank=int(coeff.shape[-1]), channels=channels,
            )
            marker = _profile_begin()
            decoded_middle = (
                decoded_middle.float() + artifact["mean"][feature_start:feature_stop]
            ).reshape(coeff.shape[0], coeff.shape[1], channels).to(dtype)
            _profile_end(
                "decode_reconstruct_post", marker, kind=str(shared["kind"]),
                group_start=start, layer=layer_idx,
                tokens=int(coeff.shape[1]), channels=channels,
            )
        else:
            decoded_middle = torch.empty(
                (int(shared["shape"][0]), 0, channels), device=device, dtype=dtype
            )
        parts = []
        middle_cursor = 0
        for piece in shared["pieces"]:
            if piece["type"] == "raw":
                parts.append(piece["tensors"][local].to(device=device, dtype=dtype))
            elif piece["type"] == "compressed":
                tokens = int(piece["shared"]["shape"][1])
                parts.append(decoded_middle[:, middle_cursor : middle_cursor + tokens])
                middle_cursor += tokens
            else:
                raise RuntimeError(f"unsupported KVTC stream piece {piece.get('type')}")
        # As in the frozen HY decoder, coefficient matrices are shared only
        # across the consecutive consumers of one layer group.  They are a
        # transient decode workspace, never persistent cache state.
        if layer_idx == int(shared["group_end"]) - 1:
            shared.pop("_decoded_coefficients", None)
            for piece in shared["pieces"]:
                if piece["type"] == "compressed":
                    piece["shared"]["quantized"].pop("_decoded", None)
        marker = _profile_begin()
        result = torch.cat(parts, dim=1) if parts else torch.empty(
            (shared["shape"][0], 0, shared["shape"][2]), device=device, dtype=dtype
        )
        _profile_end(
            "decode_assemble", marker, kind=str(shared["kind"]),
            group_start=start, layer=layer_idx, pieces=len(parts),
        )
        return result
    if not is_layer_payload(value):
        raise RuntimeError("not a WorldPlay KVTC layer payload")
    shared = value["shared"]
    layer_idx = int(value["layer_idx"])
    restored = decode_layer(shared, layer_idx, dtype, device)
    # Frozen HY retains one decoded coefficient matrix across the consecutive
    # consumers in a group and releases it after the last layer.
    if layer_idx == int(shared["group_end"]) - 1:
        shared["quantized"].pop("_decoded", None)
    return restored


def _piece_tokens(piece: dict) -> int:
    if piece["type"] == "raw":
        return int(piece["tensors"].shape[2])
    return int(piece["shared"]["shape"][1])


def _raw_piece(tensors: list[torch.Tensor], role: str) -> dict | None:
    if not tensors or int(tensors[0].shape[1]) == 0:
        return None
    return {
        "type": "raw",
        "role": str(role),
        "tensors": torch.stack([item.contiguous() for item in tensors]),
    }


def _compressed_piece(tensors: list[torch.Tensor], start: int, kind: str) -> dict | None:
    if not tensors or int(tensors[0].shape[1]) == 0:
        return None
    shared = encode_group(tensors, start, kind, sink_len=0, window_len=0)
    if shared is None:
        raise RuntimeError("non-empty streaming KVTC segment unexpectedly bypassed")
    return {"type": "compressed", "shared": shared}


def _make_stream_shared(
    tensors: list[torch.Tensor], start: int, kind: str, *, keep_global_sink: bool,
) -> dict:
    shape = tuple(int(v) for v in tensors[0].shape)
    manifest = _manifest()
    sink = min(int(manifest.get("sink_tokens", 4)), shape[1]) if keep_global_sink else 0
    window = min(int(manifest.get("window_tokens", 128)), max(shape[1] - sink, 0))
    middle_end = shape[1] - window
    pieces = []
    raw_sink = _raw_piece([item[:, :sink] for item in tensors], "sink")
    middle = _compressed_piece(
        [item[:, sink:middle_end] for item in tensors], start, kind
    )
    raw_window = _raw_piece([item[:, middle_end:] for item in tensors], "window")
    pieces.extend(item for item in (raw_sink, middle, raw_window) if item is not None)
    return {
        "codec": "worldplay_kvtc_stream_shared_v1",
        "kind": kind,
        "group_start": int(start),
        "group_end": int(start + len(tensors)),
        "shape": shape,
        "pieces": pieces,
        "history_lifecycle": "clean_tokens_encoded_once_streaming",
    }


def _wrap_stream(shared: dict, layer: int) -> dict:
    return {
        "codec": "worldplay_kvtc_stream_layer_v1",
        "kind": str(shared["kind"]),
        "layer_idx": int(layer),
        "shared": shared,
    }


def initialize_stream_caches(caches) -> list[dict]:
    """Encode one raw cache list into a group-shared streaming resident state."""
    result = [dict(cache) for cache in caches]
    manifest = _manifest()
    group_size, total = int(manifest["layer_group_size"]), int(manifest["num_layers"])
    if len(result) != total:
        raise RuntimeError(f"KVTC expected {total} cache layers, got {len(result)}")
    for start in range(0, total, group_size):
        end = min(start + group_size, total)
        for kind in ("k", "v"):
            tensors = [result[layer][kind] for layer in range(start, end)]
            if not all(torch.is_tensor(item) for item in tensors):
                if all(is_stream_layer_payload(item) for item in tensors):
                    continue
                raise RuntimeError(
                    f"streaming KVTC initialization requires raw tensors for {kind} L{start}:{end}"
                )
            shared = _make_stream_shared(tensors, start, kind, keep_global_sink=True)
            for layer in range(start, end):
                result[layer][kind] = _wrap_stream(shared, layer)
    return result


def _normalize_concat_pieces(pieces: list[dict], start: int, kind: str) -> list[dict]:
    """Keep only the global first sink and final tail raw; encode other raw spans once."""
    output = []
    last = len(pieces) - 1
    for index, piece in enumerate(pieces):
        if piece["type"] != "raw":
            output.append(piece)
            continue
        keep_raw = (
            (index == 0 and piece.get("role") == "sink")
            or (index == last and piece.get("role") == "window")
        )
        if keep_raw:
            output.append(piece)
            continue
        tensors = [piece["tensors"][local] for local in range(piece["tensors"].shape[0])]
        encoded = _compressed_piece(tensors, start, kind)
        if encoded is not None:
            output.append(encoded)
    return output


def concat_stream_caches(left, right) -> list[dict]:
    """Concatenate cache lists without decoding or requantizing historical coefficients."""
    if not left:
        return initialize_stream_caches(right)
    if not right:
        return [dict(cache) for cache in left]
    left_stream = (
        [dict(cache) for cache in left]
        if is_stream_layer_payload(left[0].get("k"))
        else initialize_stream_caches(left)
    )
    right_stream = (
        [dict(cache) for cache in right]
        if is_stream_layer_payload(right[0].get("k"))
        else initialize_stream_caches(right)
    )
    result = [dict(cache) for cache in left_stream]
    manifest = _manifest()
    group_size, total = int(manifest["layer_group_size"]), int(manifest["num_layers"])
    for start in range(0, total, group_size):
        end = min(start + group_size, total)
        for kind in ("k", "v"):
            left_shared = left_stream[start][kind]["shared"]
            right_shared = right_stream[start][kind]["shared"]
            pieces = _normalize_concat_pieces(
                list(left_shared["pieces"]) + list(right_shared["pieces"]), start, kind
            )
            left_shape, right_shape = left_shared["shape"], right_shared["shape"]
            shared = {
                "codec": "worldplay_kvtc_stream_shared_v1",
                "kind": kind,
                "group_start": start,
                "group_end": end,
                "shape": (
                    int(left_shape[0]), int(left_shape[1]) + int(right_shape[1]),
                    int(left_shape[2]),
                ),
                "pieces": pieces,
                "history_lifecycle": "clean_tokens_encoded_once_streaming",
            }
            for layer in range(start, end):
                result[layer][kind] = _wrap_stream(shared, layer)
    return result


def _slice_coefficients(payload: dict, token_indices: list[int], batch: int, tokens: int) -> dict:
    row_indices = torch.as_tensor(
        [b * tokens + index for b in range(batch) for index in token_indices],
        device=payload["sections"][0]["codes"].device if payload["sections"] else "cpu",
        dtype=torch.long,
    )
    sections = []
    for section in payload["sections"]:
        bits, count = int(section["bits"]), int(section["count"])
        groups, width = int(section["groups"]), int(section["size"])
        codes = _unpack(section["codes"], bits, count).reshape(batch * tokens, groups, width)
        selected = codes.index_select(0, row_indices)
        copy = {k: v for k, v in section.items() if k not in {"codes", "count"}}
        copy["count"] = int(selected.numel())
        copy["codes"] = _pack(selected, bits)
        sections.append(copy)
    return {"rows": batch * len(token_indices), "rank": int(payload["rank"]), "sections": sections}


def _slice_stream_shared(shared: dict, token_indices: list[int]) -> dict:
    pieces, selected = [], list(token_indices)
    cursor = 0
    for piece in shared["pieces"]:
        length = _piece_tokens(piece)
        local = [index - cursor for index in selected if cursor <= index < cursor + length]
        cursor += length
        if not local:
            continue
        if local == list(range(length)):
            pieces.append(piece)
            continue
        if piece["type"] == "raw":
            index = torch.as_tensor(local, device=piece["tensors"].device, dtype=torch.long)
            pieces.append({
                "type": "raw", "role": piece.get("role", "boundary"),
                "tensors": piece["tensors"].index_select(2, index).contiguous(),
            })
            continue
        source = piece["shared"]
        if int(source["sink_len"]) or int(source["window_len"]):
            raise RuntimeError("streaming compressed piece unexpectedly contains raw protection")
        batch, tokens, channels = (int(v) for v in source["shape"])
        sliced = {**source}
        sliced["shape"] = (batch, len(local), channels)
        sliced["quantized"] = _slice_coefficients(
            source["quantized"], local, batch, tokens
        )
        sliced["protected_sink"] = source["protected_sink"][:, :, :0].contiguous()
        sliced["protected_window"] = source["protected_window"][:, :, :0].contiguous()
        pieces.append({"type": "compressed", "shared": sliced})
    shape = shared["shape"]
    return {
        **{k: v for k, v in shared.items() if not str(k).startswith("_")},
        "shape": (int(shape[0]), len(selected), int(shape[2])),
        "pieces": pieces,
    }


def select_stream_layer(value: dict, token_idx: torch.Tensor) -> dict:
    shared = value["shared"]
    indices = tuple(int(item) for item in token_idx.detach().cpu().tolist())
    key = (id(shared), indices)
    selected = _STREAM_SLICE_CACHE.get(key)
    if selected is None:
        selected = _slice_stream_shared(shared, list(indices))
        _STREAM_SLICE_CACHE[key] = selected
    layer = int(value["layer_idx"])
    result = _wrap_stream(selected, layer)
    if layer == int(shared["group_end"]) - 1:
        _STREAM_SLICE_CACHE.pop(key, None)
    return result


@torch.no_grad()
def compress_caches_in_place(caches) -> None:
    """Replace every raw layer tensor with HY-style group-shared payloads."""
    manifest = _manifest()
    group_size, total = int(manifest["layer_group_size"]), int(manifest["num_layers"])
    if len(caches) != total:
        raise RuntimeError(f"KVTC expected {total} cache layers, got {len(caches)}")
    for start in range(0, total, group_size):
        end = min(start + group_size, total)
        if any(caches[layer].get("k") is None for layer in range(start, end)):
            continue
        for kind in ("k", "v"):
            tensors = []
            for layer in range(start, end):
                value = caches[layer][kind]
                if is_layer_payload(value):
                    resident = _first_tensor(value)
                    if resident is None:
                        raise RuntimeError(f"KVTC packed payload L{layer} has no resident tensor")
                    tensor = decode_value(
                        value,
                        torch.bfloat16,
                        resident.device,
                    )
                elif torch.is_tensor(value):
                    tensor = value
                else:
                    raise RuntimeError(f"KVTC cache L{layer} {kind} is neither raw nor packed")
                tensors.append(tensor)
            shared = encode_group(tensors, start, kind)
            if shared is None:
                # Exact short-context bypass: keep every tensor untouched.
                continue
            for layer in range(start, end):
                caches[layer][kind] = {
                    "codec": "worldplay_kvtc_layer_v1",
                    "kind": kind,
                    "layer_idx": layer,
                    "shared": shared,
                }


@torch.no_grad()
def roundtrip(layer_idx: int, k: torch.Tensor, v: torch.Tensor):
    """Numerical parity gate; resident integration uses `encode_group` directly."""
    manifest = _manifest()
    group_size, total = int(manifest["layer_group_size"]), int(manifest["num_layers"])
    start = (int(layer_idx) // group_size) * group_size
    end = min(start + group_size, total)
    pending = _PENDING.setdefault(start, {})
    pending[int(layer_idx)] = (k, v)
    if int(layer_idx) != end - 1:
        return k, v
    if set(pending) != set(range(start, end)):
        raise RuntimeError(f"incomplete KVTC group {start}:{end}: {sorted(pending)}")
    for kind, offset in (("k", 0), ("v", 1)):
        tensors = [pending[layer][offset] for layer in range(start, end)]
        shared = encode_group(tensors, start, kind)
        if shared is None:
            continue
        shared["group_start"] = start
        for layer, tensor in zip(range(start, end), tensors):
            tensor.copy_(decode_layer(shared, layer, tensor.dtype, tensor.device))
        shared["quantized"].pop("_decoded", None)
        _STATS["original_bytes"] += sum(item.numel() * item.element_size() for item in tensors)
        _STATS["payload_bytes"] += tensor_bytes(shared)
    _STATS["groups"] += 1
    result = dict(_STATS)
    result["dynamic_ratio"] = result["original_bytes"] / max(result["payload_bytes"], 1)
    result["artifact_bytes"] = sum(path.stat().st_size for path in _root().glob("*.pt"))
    result["system_ratio_for_observed_writes"] = result["original_bytes"] / max(
        result["payload_bytes"] + result["artifact_bytes"], 1
    )
    out = Path(os.environ.get("MG35_KVCODEC_OUTPUT", "outputs/kvcodec_probe"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "worldplay_kvtc_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _PENDING.pop(start, None)
    return k, v
