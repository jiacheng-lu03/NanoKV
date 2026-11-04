"""Actual resident-tensor byte audit for WorldPlay-parity MG3.5 codecs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


_PEAK_SNAPSHOTS = {}


def record_cuda_phase(name: str, **metadata) -> None:
    """Append a lightweight allocator phase marker for peak attribution."""
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    payload = {
        "unix_ns": time.time_ns(),
        "phase": str(name),
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        **metadata,
    }
    root = Path(os.environ.get("MG35_KVCODEC_OUTPUT", "outputs/kvcodec_probe"))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "cuda_phase_trace.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _unique_tensor_bytes(value, seen: set[int], *, private_only=False, skip_private=False) -> int:
    if torch.is_tensor(value):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            private = str(key).startswith("_")
            if private_only and not private:
                continue
            if skip_private and private:
                continue
            total += _unique_tensor_bytes(
                child, seen, private_only=private_only, skip_private=skip_private
            )
        return total
    if isinstance(value, (list, tuple)):
        return sum(
            _unique_tensor_bytes(
                child, seen, private_only=private_only, skip_private=skip_private
            )
            for child in value
        )
    return 0


def _private_tensor_bytes(value, seen: set[int], *, inside_private=False) -> int:
    """Count tensors reachable below private bookkeeping keys at any depth."""
    if torch.is_tensor(value):
        if not inside_private or id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(
            _private_tensor_bytes(
                child,
                seen,
                inside_private=inside_private or str(key).startswith("_"),
            )
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _private_tensor_bytes(child, seen, inside_private=inside_private)
            for child in value
        )
    return 0


def _unique_storage_bytes(
    value, seen: set[tuple], *, skip_private: bool = False
) -> int:
    """Count physical tensor storages, including storage retained by a view."""
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        identity = (
            str(value.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
        if identity in seen:
            return 0
        seen.add(identity)
        return int(storage.nbytes())
    if isinstance(value, dict):
        return sum(
            _unique_storage_bytes(child, seen, skip_private=skip_private)
            for key, child in value.items()
            if not (skip_private and str(key).startswith("_"))
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _unique_storage_bytes(child, seen, skip_private=skip_private)
            for child in value
        )
    return 0


def _operator_bytes(mode: str) -> int:
    seen = set()
    if mode == "worldplay_method":
        from .nanokv_codec import _OPERATORS
        return _unique_tensor_bytes(_OPERATORS, seen)
    if mode == "worldplay_kvtc":
        from .kvtc_codec import _ARTIFACTS
        return _unique_tensor_bytes(_ARTIFACTS, seen)
    if mode in {"qvg", "turboquant"}:
        from .training_free_kv_baselines import resident_codec_bytes
        return resident_codec_bytes(mode, layers=30, head_dim=128)
    return 0


def record_live_roots(roots: dict, *, scope: str) -> dict:
    """Audit all live cache roots using storage-identity de-duplication.

    ``record`` intentionally reports the formal persistent rolling cache.  The
    rollout also creates temporary context/read/window cache views.  A raw BF16
    view can keep an evicted cache segment alive even when it is no longer
    reachable from the rolling cache, so allocator memory cannot be reconciled
    from the formal cache alone.  This audit reports both each root's reachable
    bytes and the physical union across every live root.
    """
    per_root = {}
    union_seen = set()
    union_bytes = 0
    for name, value in roots.items():
        root_seen = set()
        per_root[str(name)] = _unique_storage_bytes(value, root_seen)
        union_bytes += _unique_storage_bytes(value, union_seen)
    snapshot = {
        "scope": str(scope),
        "per_root_bytes": per_root,
        "union_unique_tensor_bytes": int(union_bytes),
        "union_unique_tensor_mib": float(union_bytes / (1024 ** 2)),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        snapshot["cuda_memory"] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    root = Path(os.environ.get("MG35_KVCODEC_OUTPUT", "outputs/kvcodec_probe"))
    root.mkdir(parents=True, exist_ok=True)
    safe_scope = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(scope)
    )
    (root / f"worldplay_live_roots_{safe_scope}.json").write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )
    return snapshot


def record(caches, *, scope: str) -> dict:
    from .nanokv_codec import value_shape
    mode = os.environ.get("MG35_KVCODEC_MODE", "").strip().lower()
    seen = set()
    # Report physical resident storage, not only tensor view numel.  This keeps
    # the formal cache ratio reconcilable with CUDA allocator measurements.
    payload = _unique_storage_bytes(caches, seen, skip_private=True)
    transient = _private_tensor_bytes(caches, set())
    raw = 0
    for cache in caches:
        for kind in ("k", "v"):
            shape = value_shape(cache.get(kind))
            if shape is not None:
                raw += int(shape[0] * shape[1] * shape[2] * 2)
    operators = _operator_bytes(mode)
    system = payload + operators
    snapshot = {
        "scope": str(scope),
        "mode": mode,
        "rung": (
            os.environ.get("MG35_WORLDPLAY_RUNG", "full").strip().lower()
            if mode == "worldplay_method" else None
        ),
        "raw_bytes": raw,
        "payload_bytes": payload,
        "operator_bytes": operators,
        "private_transient_bytes": transient,
        "actual_live_bytes": system + transient,
        "system_bytes": system,
        "payload_ratio": raw / max(payload, 1),
        "system_ratio": raw / max(system, 1),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        snapshot["cuda_memory"] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    if mode in {"qvg", "turboquant"}:
        from .training_free_kv_baselines import method_metadata
        snapshot["baseline_provenance"] = method_metadata(mode)
    # Ratios must use numerator and denominator from the same resident-state
    # snapshot.  Independently maximizing each field can manufacture a ratio
    # that never existed at runtime.
    peaks = _PEAK_SNAPSHOTS.setdefault(
        str(scope), {"payload": None, "system": None}
    )
    if peaks["payload"] is None or payload > peaks["payload"]["payload_bytes"]:
        peaks["payload"] = dict(snapshot)
    if peaks["system"] is None or system > peaks["system"]["system_bytes"]:
        peaks["system"] = dict(snapshot)
    payload_peak = peaks["payload"]
    system_peak = peaks["system"]
    snapshot["peak_payload_snapshot"] = payload_peak
    snapshot["peak_system_snapshot"] = system_peak
    snapshot["peak_payload_ratio"] = payload_peak["raw_bytes"] / max(
        payload_peak["payload_bytes"], 1
    )
    snapshot["peak_system_ratio"] = system_peak["raw_bytes"] / max(
        system_peak["system_bytes"], 1
    )
    root = Path(os.environ.get("MG35_KVCODEC_OUTPUT", "outputs/kvcodec_probe"))
    root.mkdir(parents=True, exist_ok=True)
    safe_scope = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(scope))
    (root / f"worldplay_resident_summary_{safe_scope}.json").write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )
    if scope == "persistent_rolling":
        (root / "worldplay_resident_summary.json").write_text(
            json.dumps(snapshot, indent=2) + "\n"
        )
    return snapshot
