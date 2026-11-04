"""Cross-layer KVTC32 runtime for LingBot-World 2.0 committed history."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch


_CACHE: dict[tuple[Any, ...], Any] = {}
_MEMORY_TRACE_ENABLED = os.environ.get("LINGBOT_V2_KVTC_MEMORY_TRACE", "0") == "1"
_MEMORY_EVENTS: list[dict[str, Any]] = []
_MEMORY_TRACE_PEAK = 0


def _mark_memory(label: str) -> None:
    """Record only newly observed allocator peaks on an opt-in debug run."""
    global _MEMORY_TRACE_PEAK
    if not _MEMORY_TRACE_ENABLED or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak <= _MEMORY_TRACE_PEAK:
        return
    _MEMORY_TRACE_PEAK = peak
    _MEMORY_EVENTS.append({
        "label": str(label),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": peak,
    })


def memory_events() -> list[dict[str, Any]]:
    return list(_MEMORY_EVENTS)


@torch.no_grad()
def preload(
    *, root_k: str | Path, root_v: str | Path, group_size: int,
    num_layers: int, sp_rank: int, device: torch.device,
) -> None:
    """Load all resident group PCA artifacts before timed generation."""
    for kind, root in (("k", root_k), ("v", root_v)):
        _load_manifest(root, kind, int(sp_rank))
        for start in range(0, int(num_layers), int(group_size)):
            _load_group(root, kind, int(sp_rank), start, device)


def _load_manifest(root: str | Path, kind: str, sp_rank: int):
    path = Path(root) / f"rank_{sp_rank:02d}" / f"manifest_{kind}.pt"
    key = (str(path.resolve()), "manifest")
    if key not in _CACHE:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != "lingbot_v2_kvtc_v1":
            raise RuntimeError(f"bad KVTC manifest: {path}")
        _CACHE[key] = item
    return _CACHE[key]


def _load_group(root, kind, sp_rank, start, device):
    path = Path(root) / f"rank_{sp_rank:02d}" / f"group_{start:02d}_{kind}.pt"
    key = (str(path.resolve()), str(device))
    if key not in _CACHE:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != "lingbot_v2_kvtc_group_v1":
            raise RuntimeError(f"bad KVTC group: {path}")
        item["mean"] = item["mean"].to(device=device, dtype=torch.float32)
        item["basis"] = item["basis"].to(device=device, dtype=torch.bfloat16)
        _CACHE[key] = item
    return _CACHE[key]


def _pack(values: torch.Tensor, bits: int):
    values = values.to(torch.uint8).flatten()
    if bits == 8:
        return values.contiguous()
    per_byte = 8 // bits
    pad = (-values.numel()) % per_byte
    if pad:
        values = torch.cat((values, values.new_zeros(pad)))
    values = values.reshape(-1, per_byte).to(torch.int16)
    packed = torch.zeros(values.shape[0], device=values.device, dtype=torch.int16)
    for offset in range(per_byte):
        packed |= values[:, offset] << (offset * bits)
    return packed.to(torch.uint8)


def _unpack(values: torch.Tensor, bits: int, count: int):
    if bits == 8:
        return values[:count]
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=values.device, dtype=torch.int16) * bits
    return (((values.to(torch.int16)[:, None] >> shifts) & ((1 << bits) - 1))
            .to(torch.uint8).flatten()[:count])


def _encode_coefficients(coeff: torch.Tensor, plan: Sequence[dict[str, Any]]):
    rows, sections = coeff.shape[0], []
    grouped: dict[tuple[int, str, int], list[int]] = {}
    for item in plan:
        if int(item["bits"]):
            grouped.setdefault(
                (int(item["bits"]), str(item["type"]), int(item["size"])), []
            ).append(int(item["start"]))
    for (bits, qtype, width), starts in grouped.items():
        indices = (torch.tensor(starts, device=coeff.device)[:, None]
                   + torch.arange(width, device=coeff.device)[None])
        block = coeff[:, indices].float()
        if qtype == "fp8":
            shift = block.mean(dim=(0, 2), keepdim=True)
            scale = (block - shift).abs().amax(dim=(0, 2), keepdim=True).clamp_min(1e-8) / 448
            codes = ((block - shift) / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            shift = block.amin(dim=(0, 2), keepdim=True)
            high = block.amax(dim=(0, 2), keepdim=True)
            scale = ((high - shift) / ((1 << bits) - 1)).clamp_min(1e-8)
            codes = ((block - shift) / scale).round().clamp(0, (1 << bits) - 1).to(torch.uint8)
        sections.append({
            "bits": bits, "type": qtype, "width": width, "starts": starts,
            "groups": len(starts), "count": codes.numel(), "codes": _pack(codes, bits),
            "scale": scale.to(torch.bfloat16), "shift": shift.to(torch.bfloat16),
        })
    return {"rows": rows, "rank": coeff.shape[1], "sections": sections}


def _decode_coefficients(payload, device):
    cached = payload.get("_decoded")
    if torch.is_tensor(cached) and cached.device == device:
        return cached
    rows, rank = int(payload["rows"]), int(payload["rank"])
    output = torch.zeros((rows, rank), device=device, dtype=torch.bfloat16)
    for section in payload["sections"]:
        bits, width, groups = map(int, (
            section["bits"], section["width"], section["groups"]
        ))
        codes = _unpack(section["codes"].to(device), bits, int(section["count"]))
        if section["type"] == "fp8":
            codes = codes.view(torch.float8_e4m3fn).reshape(rows, groups, width).float()
        else:
            codes = codes.reshape(rows, groups, width).float()
        restored = (codes * section["scale"].to(device).float()
                    + section["shift"].to(device).float())
        indices = (torch.tensor(section["starts"], device=device)[:, None]
                   + torch.arange(width, device=device)[None])
        output[:, indices.flatten()] = restored.reshape(rows, -1).to(torch.bfloat16)
    # One quantized segment is shared by every layer in its group.  Match the
    # WorldPlay baseline by unpacking once and reusing through the last layer;
    # attention_inputs removes this private transient immediately afterwards.
    payload["_decoded"] = output
    _mark_memory("decode_coefficients")
    return output


@torch.no_grad()
def _encode_group(caches, start, end, kind, tensors, global_start, global_end):
    shape = tuple(tensors[0].shape)
    if any(tuple(x.shape) != shape for x in tensors):
        raise RuntimeError(f"KVTC shape mismatch group {start}:{end}")
    batch, tokens, heads, head_dim = map(int, shape)
    root = caches[start][f"kvtc_root_{kind}"]
    sp_rank = int(caches[start].get("codec_sp_rank", 0))
    artifact = _load_group(root, kind, sp_rank, start, tensors[0].device)
    feature_dim = (end - start) * heads * head_dim
    # Preserve one full GEMM, including its exact shape and reduction order,
    # while bounding only the FP32 centering workspace.  Splitting the GEMM by
    # rows can select a different CUDA kernel and perturb the rollout even
    # though the algebra is row independent.  Instead, materialise the BF16
    # matrix once, centre it back into the same storage a few rows at a time,
    # then execute the original full-size GEMM.  This is bit-identical to the
    # former default path but avoids simultaneous full-size BF16 input, FP32
    # input, FP32 centred result and BF16 centred result allocations.
    matrix = torch.stack(tensors, dim=2).reshape(batch * tokens, feature_dim)
    _mark_memory(f"encode_group_{kind}_matrix_L{start:02d}")
    tile_rows = max(
        1, int(os.environ.get("LINGBOT_V2_KVTC_PROJECTION_TILE_ROWS", "256"))
    )
    for row_start in range(0, int(matrix.shape[0]), tile_rows):
        row_end = min(row_start + tile_rows, int(matrix.shape[0]))
        matrix[row_start:row_end].copy_(
            (matrix[row_start:row_end].float() - artifact["mean"])
            .to(torch.bfloat16)
        )
    _mark_memory(f"encode_group_{kind}_centered_L{start:02d}")
    coeff = matrix @ artifact["basis"]
    _mark_memory(f"encode_group_{kind}_projection_L{start:02d}")
    result = {
        "format": "lingbot_v2_kvtc_segment_v1", "kind": kind,
        "group_start": start, "group_end": end, "sp_rank": sp_rank,
        "global_start": int(global_start), "global_end": int(global_end),
        "shape": shape, "root": str(Path(root).resolve()),
        "quantized": _encode_coefficients(coeff, artifact["plan"]),
    }
    _mark_memory(f"encode_group_{kind}_quantized_L{start:02d}")
    return result


@torch.no_grad()
def _decode_layer_segment(segment, layer, dtype, device):
    start, end = int(segment["group_start"]), int(segment["group_end"])
    artifact = _load_group(
        segment["root"], segment["kind"], int(segment["sp_rank"]), start, device
    )
    batch, tokens, heads, head_dim = map(int, segment["shape"])
    coeff = _decode_coefficients(segment["quantized"], device)
    local = layer - start
    begin, stop = local * heads * head_dim, (local + 1) * heads * head_dim
    rows = coeff @ artifact["basis"][begin:stop].T
    rows = rows.float() + artifact["mean"][begin:stop]
    _mark_memory(f"decode_layer_{segment['kind']}_L{layer:02d}")
    return rows.reshape(batch, tokens, heads, head_dim).to(dtype)


def _active_intervals(committed_end, capacity, sink_tokens):
    sink_stop = min(sink_tokens, committed_end)
    recent_start = max(sink_stop, committed_end - max(capacity - sink_stop, 0))
    return [(0, sink_stop), (recent_start, committed_end)]


@torch.no_grad()
def _history(cache, kind, committed_end, capacity, sink_tokens, dtype, device):
    pieces = []
    intervals = _active_intervals(committed_end, capacity, sink_tokens)
    sink = cache.get(f"kvtc_sink_{kind}")
    if sink is not None:
        pieces.append((0, sink.shape[1], sink.to(device=device, dtype=dtype)))
    for segment in cache.get(f"kvtc_segments_{kind}", []):
        start, end = int(segment["global_start"]), int(segment["global_end"])
        intersections = [(max(start, a), min(end, b)) for a, b in intervals
                         if max(start, a) < min(end, b)]
        if not intersections:
            continue
        decoded = _decode_layer_segment(
            segment, int(cache["codec_layer"]), dtype, device
        )
        for begin, stop in intersections:
            pieces.append((begin, stop, decoded[:, begin - start:stop - start]))
    tail = cache.get(f"kvtc_tail_{kind}")
    if tail is not None:
        start = int(cache["kvtc_tail_start"])
        pieces.append((start, start + tail.shape[1], tail.to(device=device, dtype=dtype)))
    pieces.sort(key=lambda x: x[0])
    if not pieces:
        return None
    return (
        torch.cat([x[2] for x in pieces], dim=1),
        torch.cat([torch.arange(x[0], x[1], device=device) for x in pieces]),
    )


@torch.no_grad()
def _finalize_group(
    caches, start, end, current_start, current_end, *, retention_sink_tokens
):
    # KVTC protects only a tiny raw prefix (the paper baseline setting), while
    # LingBot's rolling attention logically retains a much larger frame sink.
    # The latter must remain addressable, but may remain compressed.  Keeping
    # these two notions separate avoids either unfairly storing the full model
    # sink in BF16 or incorrectly pruning compressed sink segments.
    raw_sink_tokens = int(caches[start]["kvtc_sink_tokens"])
    window_tokens = int(caches[start]["kvtc_window_tokens"])
    combined = {"k": [], "v": []}
    combined_start = None
    for layer in range(start, end):
        cache = caches[layer]
        pending = cache.pop("kvtc_pending")
        for kind in ("k", "v"):
            previous = cache.get(f"kvtc_tail_{kind}")
            tensor = pending[kind] if previous is None else torch.cat((previous, pending[kind]), dim=1)
            combined[kind].append(tensor)
        previous_start = int(cache.get("kvtc_tail_start", current_start))
        combined_start = previous_start if combined_start is None else combined_start
    eligible_start = max(combined_start, raw_sink_tokens)
    eligible_end = max(eligible_start, current_end - window_tokens)
    local_start, local_end = eligible_start - combined_start, eligible_end - combined_start
    for kind in ("k", "v"):
        tensors = combined[kind]
        if eligible_end > eligible_start:
            shared = _encode_group(
                caches, start, end, kind,
                [x[:, local_start:local_end] for x in tensors],
                eligible_start, eligible_end,
            )
            for layer in range(start, end):
                caches[layer].setdefault(f"kvtc_segments_{kind}", []).append(shared)
        _mark_memory(f"finalize_group_{kind}_L{start:02d}")
        for local, layer in enumerate(range(start, end)):
            tensor = tensors[local]
            if (combined_start < raw_sink_tokens
                    and caches[layer].get(f"kvtc_sink_{kind}") is None):
                count = min(raw_sink_tokens - combined_start, tensor.shape[1])
                # A detached slice is still a view and therefore retains the
                # complete backing storage of ``tensor``.  Since ``tensor``
                # can be the concatenation of the previous tail and a full
                # newly committed chunk, retaining that view made a logical
                # four-token sink pin the whole dense allocation on CUDA.
                # Clone the tiny resident slice so its physical storage agrees
                # with the cache payload accounted by the lifecycle audit.
                caches[layer][f"kvtc_sink_{kind}"] = (
                    tensor[:, :count].detach().clone()
                )
            # The rolling tail must likewise own a compact allocation.  A
            # plain ``detach()`` here previously kept every superseded
            # old-tail+new-chunk concatenation alive and caused roughly 10 GiB
            # of unreported dense history to accumulate over an 8-chunk run.
            caches[layer][f"kvtc_tail_{kind}"] = (
                tensor[:, -window_tokens:].detach().clone()
            )
            caches[layer]["kvtc_tail_start"] = current_end - min(window_tokens, tensor.shape[1])
            caches[layer]["kvtc_committed_end"] = current_end
            capacity = int(caches[layer]["codec_capacity_tokens"])
            sink_stop = min(int(retention_sink_tokens), current_end)
            recent_start = max(sink_stop, current_end - max(capacity - sink_stop, 0))
            caches[layer][f"kvtc_segments_{kind}"] = [
                segment for segment in caches[layer][f"kvtc_segments_{kind}"]
                if int(segment["global_start"]) < sink_stop
                or int(segment["global_end"]) > recent_start
            ]


@torch.no_grad()
def attention_inputs(cache, key, value, *, current_start, capacity, sink_tokens, max_attention_size):
    caches = cache["kvtc_all_caches"]
    committed = int(cache.get("kvtc_committed_end", 0))
    history_k = _history(cache, "k", committed, capacity, sink_tokens, value.dtype, value.device)
    _mark_memory(f"history_k_L{int(cache['codec_layer']):02d}")
    history_v = _history(cache, "v", committed, capacity, sink_tokens, value.dtype, value.device)
    _mark_memory(f"history_v_L{int(cache['codec_layer']):02d}")
    group = int(cache["kvtc_group_size"])
    layer, total = int(cache["codec_layer"]), len(caches)
    start, end = (layer // group) * group, min((layer // group + 1) * group, total)
    if layer == end - 1:
        for kind in ("k", "v"):
            for segment in cache.get(f"kvtc_segments_{kind}", []):
                segment["quantized"].pop("_decoded", None)
    current_positions = torch.arange(
        current_start, current_start + key.shape[1], device=key.device
    )
    key_cache = key if history_k is None else torch.cat((history_k[0], key), dim=1)
    value_cache = value if history_v is None else torch.cat((history_v[0], value), dim=1)
    _mark_memory(f"attention_inputs_L{layer:02d}")
    positions = (current_positions if history_k is None else
                 torch.cat((history_k[1], current_positions)))
    if key_cache.shape[1] > max_attention_size:
        protected = min(sink_tokens, committed, max_attention_size)
        tail = max_attention_size - protected
        if tail:
            key_cache = torch.cat((key_cache[:, :protected], key_cache[:, -tail:]), 1)
            value_cache = torch.cat((value_cache[:, :protected], value_cache[:, -tail:]), 1)
            positions = torch.cat((positions[:protected], positions[-tail:]))
        else:
            key_cache = key_cache[:, :protected]
            value_cache = value_cache[:, :protected]
            positions = positions[:protected]
    current_end = current_start + key.shape[1]
    if cache.get("codec_commit", False):
        cache["kvtc_pending"] = {"k": key.detach(), "v": value.detach()}
        if layer == end - 1:
            _finalize_group(
                caches, start, end, current_start, current_end,
                retention_sink_tokens=sink_tokens,
            )
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(min(current_end, capacity))
    return key_cache, value_cache, positions


def configure(caches, *, root_k, root_v, group_size=8, sink_tokens=4, window_tokens=128):
    for layer, cache in enumerate(caches):
        cache.update({
            "kvtc_enabled": True, "kvtc_all_caches": caches,
            "kvtc_root_k": str(root_k), "kvtc_root_v": str(root_v),
            "kvtc_group_size": int(group_size), "kvtc_sink_tokens": int(sink_tokens),
            "kvtc_window_tokens": int(window_tokens), "kvtc_committed_end": 0,
            "kvtc_segments_k": [], "kvtc_segments_v": [],
        })


def clear_transient_coefficients(caches) -> int:
    """Release any defensive leftover of the group-local decoded workspace."""
    seen, released = set(), 0
    for cache in caches:
        for kind in ("k", "v"):
            for segment in cache.get(f"kvtc_segments_{kind}", []):
                quantized = segment.get("quantized", {})
                decoded = quantized.get("_decoded")
                if torch.is_tensor(decoded) and id(decoded) not in seen:
                    seen.add(id(decoded))
                    released += decoded.numel() * decoded.element_size()
                quantized.pop("_decoded", None)
    return released
