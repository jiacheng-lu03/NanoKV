"""Optional instrumentation and codec hook for Matrix-Game-3.5 causal KV.

The default mode is ``off`` and is intentionally a no-op.  Keeping the hook at
the cache-write boundary is important: K is stored before RoPE and V is stored
raw, exactly matching the tensors consumed by the rolling causal cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import torch


_COLLECTED = {}
_COLLECT_SEEN = {}
_BASIS_CACHE = {}
_DECODER_CACHE = {}
_KVTC_CACHE = {}
_KVTC_PENDING = {}
_KVTC_STATS = {"original_bytes": 0, "payload_bytes": 0, "groups": 0}
_STAT_WRITES = {}


def _mode() -> str:
    return os.environ.get("MG35_KVCODEC_MODE", "off").strip().lower()


def _output_dir() -> Path:
    return Path(os.environ.get("MG35_KVCODEC_OUTPUT", "outputs/kvcodec_probe"))


def _append_stat(layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_index = int(_STAT_WRITES.get(int(layer_idx), 0))
    _STAT_WRITES[int(layer_idx)] = write_index + 1
    item = {
        "layer": int(layer_idx),
        "write_index": write_index,
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "dtype": str(k.dtype),
        "device": str(k.device),
        "k_bytes": int(k.numel() * k.element_size()),
        "v_bytes": int(v.numel() * v.element_size()),
    }
    with (out / "cache_write_shapes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _collect(target: str, layer_idx: int, value: torch.Tensor) -> None:
    max_rows = int(os.environ.get("MG35_KVCODEC_MAX_ROWS", "16384"))
    rows_per_write = int(os.environ.get("MG35_KVCODEC_ROWS_PER_WRITE", "512"))
    key = (target, int(layer_idx))
    write_idx = int(_COLLECT_SEEN.get(key, 0))
    _COLLECT_SEEN[key] = write_idx + 1
    selected_writes = {
        int(item)
        for item in os.environ.get("MG35_KVCODEC_COLLECT_WRITE_IDS", "").split(",")
        if item.strip()
    }
    if selected_writes and write_idx not in selected_writes:
        return
    old = _COLLECTED.get(key)
    old_rows = 0 if old is None else int(old["codes"].shape[0])
    remaining = max_rows - old_rows
    if remaining <= 0:
        return

    flat = value.detach().reshape(-1, value.shape[-1])
    take = min(int(flat.shape[0]), rows_per_write, remaining)
    if take <= 0:
        return
    # Deterministic evenly-spaced sampling avoids perturbing the model RNG.
    indices = torch.linspace(0, flat.shape[0] - 1, take, device=flat.device).long()
    sample = flat.index_select(0, indices).to(device="cpu", dtype=torch.float32)
    # Matrix-Game-3.5's 3072-wide causal self-attention is 24 x 128.
    # This grouping is calibration transport metadata, so keep the model's
    # canonical head layout instead of merely choosing any divisor of 3072.
    heads = int(os.environ.get("MG35_KVCODEC_NUM_HEADS", "24"))
    if int(sample.shape[-1]) % heads:
        raise RuntimeError(
            f"calibration channels={sample.shape[-1]} not divisible by heads={heads}"
        )
    shaped = sample.reshape(sample.shape[0], heads, sample.shape[-1] // heads)
    scale = shaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    codes = (shaped / scale).round().clamp(-127, 127).to(torch.int8)
    stored = {
        "codes": codes if old is None else torch.cat([old["codes"], codes]),
        "scale": scale.to(torch.bfloat16) if old is None else torch.cat(
            [old["scale"], scale.to(torch.bfloat16)]
        ),
        "sample_ids": (
            torch.stack((
                torch.full((take,), write_idx, dtype=torch.int32),
                indices.cpu().to(torch.int32),
            ), dim=1)
            if old is None
            else torch.cat([
                old["sample_ids"],
                torch.stack((
                    torch.full((take,), write_idx, dtype=torch.int32),
                    indices.cpu().to(torch.int32),
                ), dim=1),
            ])
        ),
    }
    _COLLECTED[key] = stored

    out = _output_dir() / target
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"layer_{layer_idx:02d}.pt.tmp"
    dst = out / f"layer_{layer_idx:02d}.pt"
    torch.save({
        "format": "mg35_worldplay_pca_samples_v3",
        "target": target,
        "layer": int(layer_idx),
        "channels": int(sample.shape[-1]),
        "num_heads": heads,
        "tensor_int8": stored["codes"],
        "int8_scale": stored["scale"],
        "sample_ids": stored["sample_ids"],
        "center_storage": (
            os.environ.get("MG35_KVCODEC_CENTER_STORAGE", "bf16").strip().lower()
            if target.endswith("_c64") else "none"
        ),
        "center_block_size": (
            int(os.environ.get("MG35_KVCODEC_CENTER_BLOCK_SIZE", "64"))
            if target.endswith("_c64") else 0
        ),
        "c64_clusters": (
            int(os.environ.get("MG35_KVCODEC_C64_CLUSTERS", "64"))
            if target.endswith("_c64") else 0
        ),
        "c64_iters": (
            int(os.environ.get("MG35_KVCODEC_C64_ITERS", "2"))
            if target.endswith("_c64") else 0
        ),
        "c64_seed_rule": (
            "20260804+layer" if target.endswith("_c64") else "none"
        ),
    }, tmp)
    tmp.replace(dst)


def _target_enabled(target: str) -> bool:
    selected = os.environ.get("MG35_KVCODEC_TARGET", "both").strip().lower()
    if selected not in {"k", "v", "both"}:
        raise ValueError(f"Unsupported MG35_KVCODEC_TARGET={selected!r}")
    return selected == "both" or selected == target


def _load_basis(target: str, layer_idx: int, device: torch.device):
    root = Path(os.environ["MG35_KVCODEC_BASIS_DIR"])
    path = root / target / f"layer_{layer_idx:02d}.pt"
    kind = target.split("_", 1)[0].upper()
    requested_rank = int(os.environ.get(f"MG35_KVCODEC_RANK_{kind}", "0") or 0)
    key = (str(path.resolve()), str(device), int(requested_rank))
    cached = _BASIS_CACHE.get(key)
    if cached is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        components = payload["components"]
        if requested_rank:
            if requested_rank < 1 or requested_rank > int(components.shape[0]):
                raise RuntimeError(
                    f"invalid {kind} codec rank={requested_rank} for "
                    f"basis rank={components.shape[0]} at {path}"
                )
            # PCA artifacts store eigenvectors in ascending eigenvalue order;
            # every formal rank-truncation/export path therefore keeps the
            # suffix.  The causal A1 round-trip must use the identical slice.
            components = components[-requested_rank:]
        cached = (
            payload["mean"].to(device=device, dtype=torch.float32),
            components.to(device=device, dtype=torch.float32),
        )
        _BASIS_CACHE[key] = cached
    return cached


def _load_decoder(target: str, layer_idx: int, components: torch.Tensor):
    kind = target.split("_", 1)[0].upper()
    value = os.environ.get(f"MG35_KVCODEC_DECODER_DIR_{kind}", "").strip()
    if not value:
        return components.t()
    path = Path(value) / f"layer_{layer_idx:02d}.pt"
    # During strict causal group training, later groups have not produced A1
    # checkpoints yet and therefore retain the frozen PCA transpose decoder.
    if not path.is_file():
        return components.t()
    key = (str(path.resolve()), str(components.device))
    cached = _DECODER_CACHE.get(key)
    if cached is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = f"{kind.lower()}_c64"
        if payload.get("target") != expected or int(payload.get("layer", -1)) != layer_idx:
            raise RuntimeError(f"Decoder artifact does not match {expected} L{layer_idx}: {path}")
        cached = payload["decoder"].to(device=components.device, dtype=torch.float32)
        _DECODER_CACHE[key] = cached
    return cached


def _layer_enabled(target: str, layer_idx: int) -> bool:
    raw = os.environ.get(f"MG35_KVCODEC_LAYERS_{target.upper()}", "").strip()
    if not raw:
        return True
    selected = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    return int(layer_idx) in selected


def _load_kvtc_group(start: int, target: str, device: torch.device):
    root = Path(os.environ["MG35_KVTC_ARTIFACT_DIR"])
    path = root / f"group_{start:02d}_{target}.pt"
    key = (str(path.resolve()), str(device))
    cached = _KVTC_CACHE.get(key)
    if cached is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "mg35_kvtc_group_v1":
            raise RuntimeError(f"Bad MG3.5 KVTC artifact: {path}")
        payload["mean"] = payload["mean"].to(device=device, dtype=torch.float32)
        payload["components"] = payload["components"].to(
            device=device, dtype=torch.float32
        )
        cached = payload
        _KVTC_CACHE[key] = cached
    return cached


@torch.no_grad()
def _kvtc_roundtrip_group(layer_idx: int, k: torch.Tensor, v: torch.Tensor):
    group_size = int(os.environ.get("MG35_KVTC_GROUP_SIZE", "6"))
    num_layers = int(os.environ.get("MG35_KVTC_NUM_LAYERS", "30"))
    start = (layer_idx // group_size) * group_size
    end = min(start + group_size, num_layers)
    pending = _KVTC_PENDING.setdefault(start, {})
    pending[layer_idx] = (k, v)
    if layer_idx != end - 1:
        return k, v
    if set(pending) != set(range(start, end)):
        raise RuntimeError(f"Incomplete KVTC group {start}:{end}: {sorted(pending)}")

    for target, offset in (("k", 0), ("v", 1)):
        tensors = [pending[layer][offset] for layer in range(start, end)]
        shape = tensors[0].shape
        if any(item.shape != shape for item in tensors):
            raise RuntimeError(f"Mismatched KVTC cache-write group {start}:{end}")
        matrix = torch.cat(tensors, dim=-1).reshape(-1, sum(t.shape[-1] for t in tensors))
        artifact = _load_kvtc_group(start, target, matrix.device)
        if int(artifact["feature_dim"]) != int(matrix.shape[-1]):
            raise RuntimeError(
                f"KVTC feature mismatch group {start} {target}: "
                f"{artifact['feature_dim']} != {matrix.shape[-1]}"
            )
        coeff = torch.nn.functional.linear(
            matrix.float() - artifact["mean"], artifact["components"]
        )
        scale = coeff.abs().amax(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 7.0
        stored_scale = scale.to(torch.bfloat16)
        codes = (coeff / scale).round().clamp(-7, 7)
        restored = torch.nn.functional.linear(
            codes * stored_scale.float(), artifact["components"].t(), artifact["mean"]
        ).reshape(*shape[:-1], -1)
        chunks = restored.split(shape[-1], dim=-1)
        for tensor, decoded in zip(tensors, chunks):
            tensor.copy_(decoded.to(dtype=tensor.dtype))
        original_bytes = sum(t.numel() * t.element_size() for t in tensors)
        payload_bytes = coeff.numel() * 0.5 + coeff.shape[0] * 2
        _KVTC_STATS["original_bytes"] += int(original_bytes)
        _KVTC_STATS["payload_bytes"] += int(payload_bytes)
    _KVTC_STATS["groups"] += 1
    root = Path(os.environ["MG35_KVTC_ARTIFACT_DIR"])
    artifact_bytes = sum(path.stat().st_size for path in root.glob("*.pt"))
    summary = dict(_KVTC_STATS)
    summary["dynamic_ratio"] = summary["original_bytes"] / max(summary["payload_bytes"], 1)
    summary["artifact_bytes"] = artifact_bytes
    summary["system_ratio_for_observed_writes"] = summary["original_bytes"] / max(
        summary["payload_bytes"] + artifact_bytes, 1
    )
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "kvtc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _KVTC_PENDING.pop(start, None)
    return k, v


@torch.no_grad()
def _semantic_residual(value: torch.Tensor, layer_idx: int):
    """Return C64 residual plus the exact deployed centers and labels."""
    batch, sequence, dim = value.shape
    rows = value.float()
    clusters = int(os.environ.get("MG35_KVCODEC_C64_CLUSTERS", "64"))
    iters = int(os.environ.get("MG35_KVCODEC_C64_ITERS", "2"))
    if not 0 < clusters <= min(sequence, 256):
        raise ValueError(f"C64 clusters must be in [1,min(S,256)], got {clusters}")
    generator = torch.Generator(device="cpu").manual_seed(20260804 + layer_idx)
    initial = torch.randperm(sequence, generator=generator)[:clusters].to(rows.device)
    centers = rows.index_select(1, initial).clone()
    row_norm = rows.square().sum(-1, keepdim=True)
    labels = torch.zeros((batch, sequence), dtype=torch.long, device=rows.device)
    for _ in range(iters):
        distances = (
            row_norm
            + centers.square().sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(rows, centers.transpose(1, 2))
        )
        labels = distances.argmin(-1)
        assignment = torch.nn.functional.one_hot(
            labels, num_classes=clusters
        ).transpose(1, 2).float()
        counts = assignment.sum(-1)
        updated = torch.bmm(assignment, rows) / counts.clamp_min(1.0).unsqueeze(-1)
        centers = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    # The default is the exact frozen HY BF16 center path.  The optional
    # block-INT8 side-information experiment must affect calibration too, so
    # PCA/QAT rows and deployment never see different residual definitions.
    from .worldplay_kv_codec import store_centers
    stored, decoded_centers = store_centers(centers)
    assigned = torch.gather(
        decoded_centers, 1, labels[..., None].expand(-1, -1, dim)
    )
    return rows - assigned, stored, labels


@torch.no_grad()
def _pca_int4_roundtrip(
    target: str, layer_idx: int, value: torch.Tensor, semantic: bool
) -> torch.Tensor:
    if target == "k" and semantic:
        from .causal_k_a1 import enabled as k_a1_enabled, reconstruct as k_a1_reconstruct
        if k_a1_enabled():
            # Keep all subsequent cache fills synchronized with the decoder
            # updated by the preceding strict A1 atom, matching HY reapply.
            return k_a1_reconstruct(layer_idx, value, track_grad=False)
    if target == "v" and semantic:
        from .causal_v_a1 import (
            enabled as v_a1_enabled,
            reconstruct as v_a1_reconstruct,
            selected_layers as v_a1_selected_layers,
        )
        if v_a1_enabled() and int(layer_idx) in set(v_a1_selected_layers()):
            # Reapply the just-updated V decoder to later cache writes in the
            # same trajectory.  Inactive groups continue to use the frozen
            # V-QAT checkpoints through the ordinary decoder loader below.
            return v_a1_reconstruct(layer_idx, value, track_grad=False)
    source_dtype = value.dtype
    if semantic:
        rows, centers, labels = _semantic_residual(value, layer_idx)
    else:
        rows, centers, labels = value.float(), None, None
    basis_target = f"{target}_c64" if semantic else target
    mean, components = _load_basis(basis_target, layer_idx, value.device)
    decoder = _load_decoder(basis_target, layer_idx, components)
    coeff = torch.nn.functional.linear(rows.reshape(-1, rows.shape[-1]) - mean, components)
    scale = coeff.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / 7.0
    stored_scale = scale.to(torch.bfloat16).float()
    quantized = (coeff / scale).round().clamp(-7, 7)
    decoded = torch.nn.functional.linear(quantized * stored_scale, decoder, mean)
    decoded = decoded.reshape_as(rows)
    if semantic:
        from .worldplay_kv_codec import restore_centers
        assigned = torch.gather(
            restore_centers(centers, device=decoded.device),
            1,
            labels[..., None].expand_as(decoded),
        )
        decoded = decoded + assigned
    return decoded.to(dtype=source_dtype)


def process_cache_write(
    layer_idx: int, k_pre_rope: torch.Tensor, v_raw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process newly appended cache tensors without touching current attention.

    ``identity`` and ``stats`` return the original tensor objects. ``collect``
    additionally stores deterministic row samples but also remains numerically
    identical. Future learned codecs plug into this single function.
    """
    mode = _mode()
    if mode == "off":
        return k_pre_rope, v_raw
    if mode not in {
        "identity", "stats", "collect", "pca_int4", "c64_pca_int4", "kvtc",
        "worldplay_kvtc", "worldplay_method",
        "qvg", "turboquant",
    }:
        raise ValueError(f"Unsupported MG35_KVCODEC_MODE={mode!r}")

    _append_stat(layer_idx, k_pre_rope, v_raw)
    if mode == "kvtc":
        return _kvtc_roundtrip_group(layer_idx, k_pre_rope, v_raw)
    if mode == "worldplay_kvtc":
        # The source-faithful baseline packs a shared payload only after the
        # complete consecutive layer group has been written.  That happens at
        # the end of `model_fn_causal_kv`, not at this per-layer hook.
        return k_pre_rope, v_raw
    if mode == "collect":
        _collect("k", layer_idx, k_pre_rope)
        _collect("v", layer_idx, v_raw)
        if os.environ.get("MG35_KVCODEC_COLLECT_C64", "1") != "0":
            k_residual, _, _ = _semantic_residual(k_pre_rope, layer_idx)
            v_residual, _, _ = _semantic_residual(v_raw, layer_idx)
            _collect("k_c64", layer_idx, k_residual)
            _collect("v_c64", layer_idx, v_residual)
    elif mode in {"pca_int4", "c64_pca_int4"}:
        semantic = mode == "c64_pca_int4"
        if _target_enabled("k") and _layer_enabled("k", layer_idx):
            k_pre_rope = _pca_int4_roundtrip(
                "k", layer_idx, k_pre_rope, semantic
            )
        if _target_enabled("v") and _layer_enabled("v", layer_idx):
            v_raw = _pca_int4_roundtrip("v", layer_idx, v_raw, semantic)
    return k_pre_rope, v_raw
