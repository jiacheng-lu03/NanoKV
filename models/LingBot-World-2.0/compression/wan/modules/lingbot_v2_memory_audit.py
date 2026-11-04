"""Exact tensor-byte accounting for the dynamic self-KV inference arms."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _bytes(value, seen=None):
    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        # Runtime operator tables use both string metadata keys and tuple
        # tensor-entry keys such as ``(sp_rank, layer, kind)``.  Only private
        # string keys are excluded from accounting.
        return sum(
            _bytes(v, seen)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_"))
        )
    if isinstance(value, (list, tuple)):
        return sum(_bytes(v, seen) for v in value)
    return 0


def _snapshot(caches):
    if not caches:
        return None
    mode = ("kvtc32" if caches[0].get("kvtc_enabled") else
            str(caches[0].get("baseline_mode"))
            if caches[0].get("baseline_enabled") else
            "codec32" if caches[0].get("codec_enabled") else
            "clean_pre_rope" if caches[0].get("pre_rope_clean") else "clean")
    raw = 0
    for cache in caches:
        if mode == "codec32":
            committed = int(cache.get("codec_committed_end", 0))
        elif mode == "kvtc32":
            committed = int(cache.get("kvtc_committed_end", 0))
        elif mode in {"qvg", "turboquant"}:
            committed = int(cache.get("baseline_committed_end", 0))
        else:
            committed = int(cache["global_end_index"].item())
        capacity = int(cache.get("codec_capacity_tokens", cache["k"].shape[1]))
        tokens = min(committed, capacity)
        heads, head_dim = map(int, cache["k"].shape[-2:])
        raw += tokens * heads * head_dim * 2 * 2
    seen = set()
    if mode == "codec32":
        dynamic = sum(_bytes(cache.get("codec_segments", []), seen) for cache in caches)
        from .lingbot_v2_kv_codec import _OPERATORS
        static = _bytes(_OPERATORS, set())
    elif mode == "kvtc32":
        dynamic = 0
        for cache in caches:
            for kind in ("k", "v"):
                dynamic += _bytes(cache.get(f"kvtc_segments_{kind}", []), seen)
                dynamic += _bytes(cache.get(f"kvtc_sink_{kind}"), seen)
                dynamic += _bytes(cache.get(f"kvtc_tail_{kind}"), seen)
        from .lingbot_v2_kvtc import _CACHE
        static = _bytes(_CACHE, set())
    elif mode in {"qvg", "turboquant"}:
        dynamic = sum(
            _bytes(cache.get("baseline_segments", []), seen) for cache in caches
        )
        from .lingbot_v2_training_free_baselines import static_operator_bytes

        static = static_operator_bytes(mode)
    else:
        dynamic, static = raw, 0
    return {
        "mode": mode,
        "raw_self_kv_bytes": raw, "dynamic_payload_bytes": dynamic,
        "static_operator_bytes": static, "system_bytes": dynamic + static,
        "dynamic_ratio": raw / dynamic if dynamic else None,
        "system_ratio": raw / (dynamic + static) if dynamic + static else None,
    }


def record_snapshot(caches):
    """Record one post-commit persistent-cache lifecycle point."""
    snapshot = _snapshot(caches)
    if snapshot is not None:
        caches[0].setdefault("_kv_audit_snapshots", []).append(snapshot)


def write_summary(caches, path: str | Path):
    if not caches:
        return
    final = _snapshot(caches)
    snapshots = list(caches[0].get("_kv_audit_snapshots", []))
    if not snapshots or snapshots[-1] != final:
        snapshots.append(final)
    # A strict persistent-cache peak includes dynamic payload and the static
    # operators resident at the same lifecycle point.  It deliberately omits
    # transient decode workspaces, which are reported by CUDA peak metrics.
    worst = max(snapshots, key=lambda item: int(item["system_bytes"]))
    payload = {
        "format": "lingbot_v2_kv_memory_audit_v2", "mode": final["mode"],
        "raw_reference": "logical BF16 pre-RoPE K + raw projected V",
        "raw_self_kv_bytes": worst["raw_self_kv_bytes"],
        "dynamic_payload_bytes": worst["dynamic_payload_bytes"],
        "static_operator_bytes": worst["static_operator_bytes"],
        "system_bytes": worst["system_bytes"],
        "dynamic_ratio": worst["dynamic_ratio"],
        "system_ratio": worst["system_ratio"],
        "worst_persistent_snapshot": worst,
        "final_persistent_snapshot": final,
        "lifecycle_snapshots": snapshots,
        "layers": len(caches), "sp_rank": int(caches[0].get("codec_sp_rank", 0)),
        "scope": "single-GPU global-40-head dynamic self-KV; cross-attention excluded",
        "protocol": "post-x0-commit snapshots; strict maximum persistent system bytes",
        "method_metadata": (
            __import__(
                "wan.modules.lingbot_v2_training_free_baselines",
                fromlist=["metadata"],
            ).metadata(final["mode"])
            if final["mode"] in {"qvg", "turboquant"}
            else None
        ),
        "kvtc_transient_released_before_summary_bytes": int(
            caches[0].get("_kvtc_transient_released_before_summary", 0)
        ),
    }
    if final["mode"] == "kvtc32":
        from .lingbot_v2_kvtc import memory_events
        events = memory_events()
        if events:
            payload["kvtc_memory_events"] = events
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
