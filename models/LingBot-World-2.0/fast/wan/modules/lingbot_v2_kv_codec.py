"""LingBot-World 2.0 semantic residual KV codec.

The formal paper path is the single-rank path: pre-RoPE K and raw V are
jointly flattened over all 40 heads (5120 channels), C64-centred, projected to
rank 640 and quantized with one symmetric INT4 scale per token.  The runtime
stores committed x0 chunks only; noisy denoising overwrites are never encoded.

Sequence-parallel payloads are deliberately tagged by ``sp_rank``.  They are
useful for engineering experiments, but are not paper-equivalent to global
C64 because each rank sees only its local head shard.
"""
from __future__ import annotations

import hashlib
import os
from functools import cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


FORMAT = "lingbot_v2_semantic_residual_int4_v1"
SAMPLE_FORMAT = "lingbot_v2_paired_raw_c64_v1"
OPERATOR_FORMAT = "lingbot_v2_operator_int8_v1"
_OPERATORS: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
_PREPACKED: dict[str, dict[str, Any]] = {}


@cache
def _resolved_path(value: str) -> str:
    """Canonicalize immutable artifact paths once, outside the KV hot path."""
    return str(Path(value).resolve())


def _artifact_identity(value: str | Path) -> str:
    """Return a repository-relative identity that survives checkout relocation.

    Frozen prepacked operators were created under the source server's absolute
    checkout path.  Comparing that path literally rejects a byte-identical
    deployment after moving the repository.  The formal runner separately
    verifies the frozen operator SHA256; here we retain the semantic guard by
    matching its path from ``outputs/`` onward.
    """
    parts = Path(value).parts
    try:
        start = parts.index("outputs")
    except ValueError:
        return _resolved_path(str(value))
    return Path(*parts[start:]).as_posix()


@torch.no_grad()
def preload_operators(
    *, basis_root_k: str | Path, basis_root_v: str | Path,
    linear_root_k: str | Path | None, linear_root_v: str | Path | None,
    rank: int, layers: int, joint_dim: int, sp_rank: int,
    device: torch.device,
) -> None:
    """Load resident codec weights before the timed generation interval.

    The packed INT8 tensors remain the deployed representation and are still
    included by the memory auditor.  This only moves one-time artifact I/O to
    the same model-ready setup phase used by the DiT weights.
    """
    for layer in range(int(layers)):
        _load_operator(
            basis_root_k, target="k", sp_rank=int(sp_rank), layer=layer,
            joint_dim=int(joint_dim), rank=int(rank), device=device,
            linear_root=linear_root_k,
        )
        _load_operator(
            basis_root_v, target="v", sp_rank=int(sp_rank), layer=layer,
            joint_dim=int(joint_dim), rank=int(rank), device=device,
            linear_root=linear_root_v,
        )


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for k, v in value.items() if not k.startswith("_"))
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def _artifact_path(root: str | Path, sp_rank: int, layer: int) -> Path:
    root = Path(root)
    ranked = root / f"rank_{sp_rank:02d}" / f"layer_{layer:02d}" / "global_head_pca.pt"
    legacy = root / f"layer_{layer:02d}" / "global_head_pca.pt"
    return ranked if ranked.exists() or not legacy.exists() else legacy


def _packed_to_device(item, device):
    return {"qweight": item["qweight"].to(device),
            "scale": item["scale"].to(device=device, dtype=torch.bfloat16)}


def _linear(rows, operator, bias=None):
    if torch.is_tensor(operator):
        return F.linear(rows.to(torch.bfloat16), operator, bias)
    weight = operator["qweight"].to(torch.bfloat16) * operator["scale"]
    return F.linear(rows.to(torch.bfloat16), weight, bias)


def _load_operator(
    root: str | Path,
    *,
    target: str,
    sp_rank: int,
    layer: int,
    joint_dim: int,
    rank: int,
    device: torch.device,
    linear_root: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    root_key = _resolved_path(str(root))
    linear_key = "" if linear_root is None else _resolved_path(str(linear_root))
    prepacked_path = os.environ.get("LINGBOT_V2_KV_CODEC_PREPACKED_OPERATOR_PATH", "").strip()
    prepacked_key = "" if not prepacked_path else _resolved_path(prepacked_path)
    key = (root_key, linear_key, prepacked_key,
           target, sp_rank, layer, joint_dim, rank, str(device))
    if key in _OPERATORS:
        return _OPERATORS[key]
    if prepacked_path:
        resolved = prepacked_key
        if resolved not in _PREPACKED:
            packed = torch.load(resolved, map_location="cpu", weights_only=False)
            if packed.get("format") != OPERATOR_FORMAT:
                raise RuntimeError(f"bad prepacked operator: {resolved}")
            _PREPACKED[resolved] = packed
        packed = _PREPACKED[resolved]
        entry_key = f"rank{sp_rank:02d}:{target}:{layer:02d}"
        entry = packed["entries"].get(entry_key)
        if entry is None:
            raise KeyError(f"prepacked operator missing {entry_key}")
        if int(packed.get("rank", -1)) != rank:
            raise RuntimeError(f"prepacked rank mismatch: {packed.get('rank')} != {rank}")
        from nanokv.artifacts import portable_codec_verified
        portable = portable_codec_verified(prepacked_path)
        if not portable and _artifact_identity(entry["basis_root"]) != _artifact_identity(root_key):
            raise RuntimeError(f"prepacked basis provenance mismatch: {entry_key}")
        recorded_linear = entry.get("linear_root", "")
        if recorded_linear:
            recorded_linear = _artifact_identity(recorded_linear)
        requested_linear = "" if not linear_key else _artifact_identity(linear_key)
        if not portable and recorded_linear != requested_linear:
            raise RuntimeError(f"prepacked linear provenance mismatch: {entry_key}")
        result = {
            "mean": entry["mean"].to(device=device, dtype=torch.float32),
            "encoder": _packed_to_device(entry["encoder"], device),
            "decoder": _packed_to_device(entry["decoder"], device),
        }
        _OPERATORS[key] = result
        return result

    path = _artifact_path(root, sp_rank, layer)
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("target", item.get("tensor_name")) != target:
        raise RuntimeError(f"operator target mismatch: {path}")
    if int(item.get("sp_rank", sp_rank)) != sp_rank:
        raise RuntimeError(f"operator SP-rank mismatch: {path}")
    mean, components = item["mean"].float(), item["components"].float()
    if tuple(mean.shape) != (joint_dim,) or components.ndim != 2:
        raise RuntimeError(f"operator shape mismatch: {path}")
    if components.shape[1] != joint_dim or components.shape[0] < rank:
        raise RuntimeError(
            f"operator cannot serve dim/rank={joint_dim}/{rank}: {tuple(components.shape)}"
        )
    # Exact covariance artifacts store retained eigenvectors in ascending
    # eigenvalue order.  The highest-variance rank-r subspace is the suffix.
    encoder = components[-rank:].contiguous()
    decoder = encoder.T.contiguous()
    if linear_root is not None:
        checkpoint = Path(linear_root)
        if checkpoint.is_dir():
            checkpoint = (
                checkpoint / f"rank_{sp_rank:02d}" / f"layer_{layer:02d}"
                / "checkpoint_best.pt"
            )
        trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if target == "v":
            if trained.get("format") != "lingbot_v2_c64_int4_qat_v1":
                raise RuntimeError(f"bad V-QAT checkpoint: {checkpoint}")
            state = trained["model"]
            encoder = state["basis"].float() + state["encoder_delta"].float()
            decoder = state["basis"].float().T + state["decoder_delta"].float()
            mean = state["mean"].float()
        else:
            if trained.get("format") != "lingbot_v2_k_a1_decoder_delta_v1":
                raise RuntimeError(f"bad K-A1 checkpoint: {checkpoint}")
            saved = trained["decoder_delta"].get(str(layer))
            # Intermediate causal stages intentionally contain only completed
            # layer groups; untouched layers retain the tied PCA decoder.
            if saved is not None:
                decoder = decoder + saved.float()
    result = {
        "mean": mean.to(device=device, dtype=torch.float32),
        "encoder": encoder.to(device=device, dtype=torch.bfloat16),
        "decoder": decoder.contiguous().to(device=device, dtype=torch.bfloat16),
    }
    _OPERATORS[key] = result
    return result


@torch.no_grad()
def semantic_residual(
    value: torch.Tensor, *, clusters: int = 64, iters: int = 2, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C64 over the joint head/channel axis of one immutable source chunk."""
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"expected [1,S,H,D], got {tuple(value.shape)}")
    batch, sequence, heads, head_dim = value.shape
    if not 0 < clusters <= min(sequence, 256):
        raise ValueError(f"invalid C={clusters} for S={sequence}")
    rows = value.reshape(batch, sequence, heads * head_dim).float()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(sequence, generator=generator)[:clusters].to(value.device)
    centers = rows.index_select(1, initial).clone()
    labels = torch.zeros((batch, sequence), device=value.device, dtype=torch.long)
    row_norm = rows.square().sum(-1, keepdim=True)
    for _ in range(int(iters)):
        distance = (
            row_norm + centers.square().sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(rows, centers.transpose(1, 2))
        )
        labels = distance.argmin(-1)
        assignment = F.one_hot(labels, num_classes=clusters).transpose(1, 2).float()
        counts = assignment.sum(-1)
        updated = torch.bmm(assignment, rows) / counts.clamp_min(1).unsqueeze(-1)
        centers = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    stored = centers.to(torch.bfloat16)
    assigned = torch.gather(
        stored.float(), 1, labels[..., None].expand(-1, -1, heads * head_dim)
    )
    return rows - assigned, stored, labels.to(torch.uint8)


def _pack_int4(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    values = codes.reshape(-1).to(torch.int16).add(7).to(torch.uint8)
    count = values.numel()
    if count % 2:
        values = F.pad(values, (0, 1), value=7)
    return (values[0::2] | (values[1::2] << 4)).contiguous(), count


def _unpack_int4(packed: torch.Tensor, count: int, shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.stack((packed & 15, (packed >> 4) & 15), dim=1).reshape(-1)[:count]
    return values.to(torch.int16).sub_(7).to(torch.int8).reshape(shape)


@torch.no_grad()
def encode_tensor(
    value: torch.Tensor,
    *,
    target: str,
    layer: int,
    sp_rank: int,
    basis_root: str | Path,
    rank: int,
    clusters: int = 64,
    iters: int = 2,
    linear_root: str | Path | None = None,
) -> dict[str, Any]:
    batch, sequence, heads, head_dim = map(int, value.shape)
    joint_dim = heads * head_dim
    residual, centers, ids = semantic_residual(
        value,
        clusters=clusters,
        iters=iters,
        seed=20260826 + layer * 17 + sp_rank * 1009 + (target == "v"),
    )
    operator = _load_operator(
        basis_root, target=target, sp_rank=sp_rank, layer=layer,
        joint_dim=joint_dim, rank=rank, device=value.device, linear_root=linear_root,
    )
    latent = _linear(residual - operator["mean"], operator["encoder"])
    if latent.is_cuda:
        from .lingbot_v2_kv_codec_triton import quantize_pack_int4_per_row
        packed, scale, count = quantize_pack_int4_per_row(latent)
    else:
        scale = latent.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
        codes = (latent.float() / scale).round().clamp(-7, 7).to(torch.int8)
        packed, count = _pack_int4(codes)
    return {
        "format": FORMAT, "target": target, "layer": layer, "sp_rank": sp_rank,
        "shape": (batch, sequence, heads, head_dim), "rank": rank,
        "clusters": clusters, "centers": centers, "ids": ids,
        "latent_packed": packed, "latent_count": count,
        "latent_shape": (batch, sequence, rank),
        "latent_scale": scale.to(torch.bfloat16),
        "basis_root": str(Path(basis_root).resolve()),
        "linear_root": "" if linear_root is None else str(Path(linear_root).resolve()),
    }


@torch.no_grad()
def decode_tensor(
    payload: dict[str, Any], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"bad codec payload: {payload.get('format')}")
    batch, sequence, heads, head_dim = map(int, payload["shape"])
    operator = _load_operator(
        payload["basis_root"], target=payload["target"],
        sp_rank=int(payload["sp_rank"]), layer=int(payload["layer"]),
        joint_dim=heads * head_dim, rank=int(payload["rank"]), device=device,
        linear_root=payload.get("linear_root") or None,
    )
    codes = _unpack_int4(
        payload["latent_packed"].to(device), int(payload["latent_count"]),
        tuple(map(int, payload["latent_shape"])),
    )
    latent = codes.to(torch.bfloat16) * payload["latent_scale"].to(
        device=device, dtype=torch.bfloat16
    )
    rows = _linear(latent, operator["decoder"], operator["mean"].to(torch.bfloat16))
    centers = payload["centers"].to(device=device, dtype=torch.bfloat16)
    ids = payload["ids"].to(device=device, dtype=torch.long)
    rows += torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, heads * head_dim)
    )
    return rows.reshape(batch, sequence, heads, head_dim).to(dtype)


@torch.no_grad()
def direct_latent_rows(
    payload: dict[str, Any], row_index: torch.Tensor, *, device: torch.device
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Read selected formal INT4/C64 rows without reconstructing native K/V."""
    if payload.get("format") != FORMAT:
        raise RuntimeError("codec-native attention accepts only the frozen codec payload")
    index = row_index.to(device=device, dtype=torch.long)
    shape = tuple(map(int, payload["latent_shape"]))
    if device.type == "cuda" and os.environ.get("LINGBOT_CODEC_NATIVE_FUSED_UNPACK", "0") == "1":
        from .lingbot_v2_kv_codec_triton import unpack_int4_scaled
        latent = unpack_int4_scaled(
            payload["latent_packed"].to(device),
            payload["latent_scale"].to(device=device, dtype=torch.bfloat16), shape,
        ).index_select(1, index)
    else:
        codes = _unpack_int4(
            payload["latent_packed"].to(device), int(payload["latent_count"]), shape
        )
        latent = (
            codes.to(torch.bfloat16)
            * payload["latent_scale"].to(device=device, dtype=torch.bfloat16)
        ).index_select(1, index)
    centers = payload["centers"].to(device=device, dtype=torch.bfloat16)
    ids = payload["ids"].to(device=device, dtype=torch.long).index_select(1, index)
    batch, _, heads, head_dim = map(int, payload["shape"])
    operator = _load_operator(
        payload["basis_root"], target=str(payload["target"]),
        sp_rank=int(payload["sp_rank"]), layer=int(payload["layer"]),
        joint_dim=heads * head_dim, rank=int(payload["rank"]), device=device,
        linear_root=payload.get("linear_root") or None,
    )
    return latent, (centers, ids), operator["mean"]


@torch.no_grad()
def direct_latent_slice(
    payload: dict[str, Any], start: int, stop: int, *, device: torch.device
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Read a contiguous interval without materializing an index-select copy."""
    if payload.get("format") != FORMAT:
        raise RuntimeError("codec-native attention accepts only the frozen codec payload")
    shape = tuple(map(int, payload["latent_shape"]));start=int(start);stop=int(stop)
    if not 0 <= start <= stop <= shape[1]:
        raise ValueError(f"invalid latent interval [{start}:{stop}] for {shape}")
    if device.type == "cuda" and os.environ.get("LINGBOT_CODEC_NATIVE_FUSED_UNPACK", "0") == "1":
        from .lingbot_v2_kv_codec_triton import unpack_int4_scaled_slice
        latent = unpack_int4_scaled_slice(
            payload["latent_packed"].to(device),
            payload["latent_scale"].to(device=device,dtype=torch.bfloat16),
            shape,start,stop,
        )
    else:
        codes=_unpack_int4(payload["latent_packed"].to(device),int(payload["latent_count"]),shape)
        latent=(codes.to(torch.bfloat16)*payload["latent_scale"].to(device=device,dtype=torch.bfloat16))[:,start:stop]
    centers=payload["centers"].to(device=device,dtype=torch.bfloat16)
    # The source labels are immutable.  Reuse their small device/long slice
    # across denoise calls; this is derived runtime state, not KV payload.
    cache=payload.setdefault("_codec_native_slice_ids",{})
    cache_key=(str(device),start,stop)
    ids=cache.get(cache_key)
    if ids is None:
        ids=payload["ids"].to(device=device,dtype=torch.long)[:,start:stop]
        if len(cache)>=4:
            cache.pop(next(iter(cache)))
        cache[cache_key]=ids
    batch,_,heads,head_dim=map(int,payload["shape"])
    operator=_load_operator(payload["basis_root"],target=str(payload["target"]),
        sp_rank=int(payload["sp_rank"]),layer=int(payload["layer"]),
        joint_dim=heads*head_dim,rank=int(payload["rank"]),device=device,
        linear_root=payload.get("linear_root") or None)
    return latent,(centers,ids),operator["mean"]


@torch.no_grad()
def decode_tensor_slice(
    payload: dict[str, Any], start: int, stop: int, *,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    latent,(centers,ids),mean=direct_latent_slice(payload,start,stop,device=device)
    batch,_,heads,head_dim=map(int,payload["shape"])
    operator=_load_operator(payload["basis_root"],target=str(payload["target"]),
        sp_rank=int(payload["sp_rank"]),layer=int(payload["layer"]),
        joint_dim=heads*head_dim,rank=int(payload["rank"]),device=device,
        linear_root=payload.get("linear_root") or None)
    rows=_linear(latent,operator["decoder"],mean.to(torch.bfloat16))
    rows+=torch.gather(centers,1,ids[...,None].expand(-1,-1,heads*head_dim))
    return rows.reshape(batch,stop-start,heads,head_dim).to(dtype)


@torch.no_grad()
def decode_tensor_rows(
    payload: dict[str, Any], row_index: torch.Tensor, *,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Decode only policy-protected rows to native head width."""
    latent, (centers, ids), mean = direct_latent_rows(payload, row_index, device=device)
    batch, _, heads, head_dim = map(int, payload["shape"])
    operator = _load_operator(
        payload["basis_root"], target=str(payload["target"]),
        sp_rank=int(payload["sp_rank"]), layer=int(payload["layer"]),
        joint_dim=heads * head_dim, rank=int(payload["rank"]), device=device,
        linear_root=payload.get("linear_root") or None,
    )
    rows = _linear(latent, operator["decoder"], mean.to(torch.bfloat16))
    rows.add_(torch.gather(
        centers, 1, ids[..., None].expand(-1, -1, heads * head_dim)
    ))
    return rows.reshape(batch, int(row_index.numel()), heads, head_dim).to(dtype)


@torch.no_grad()
def encode_pair(key: torch.Tensor, value: torch.Tensor, *, cache: dict[str, Any]) -> dict[str, Any]:
    common = {
        "layer": int(cache["codec_layer"]), "sp_rank": int(cache.get("codec_sp_rank", 0)),
        "rank": int(cache["codec_rank"]), "clusters": int(cache.get("codec_clusters", 64)),
        "iters": int(cache.get("codec_iters", 2)),
    }
    result = {
        "format": FORMAT,
        "global_start": int(cache["codec_commit_start"]),
        "global_end": int(cache["codec_commit_start"] + key.shape[1]),
        "k": encode_tensor(
            key, target="k", basis_root=cache["codec_basis_root_k"],
            linear_root=cache.get("codec_linear_root_k"), **common
        ),
        "v": encode_tensor(
            value, target="v", basis_root=cache["codec_basis_root_v"],
            linear_root=cache.get("codec_linear_root_v"), **common
        ),
    }
    result["resident_bytes"] = tensor_bytes(result)
    return result


@torch.no_grad()
def decode_segments(
    segments: list[dict[str, Any]], *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not segments:
        return None
    return tuple(
        _decode_method_segments(
            [segment[target] for segment in segments], dtype=dtype, device=device
        )
        for target in ("k", "v")
    )


@torch.no_grad()
def _decode_method_segments(
    parts: list[dict[str, Any]], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode immutable same-layer segments with one shared decoder GEMM.

    Segment-local INT4 scales and C64 assignments are unchanged.  This is the
    LingBot equivalent of the frozen WorldPlay/MG3.5 lifecycle optimization:
    concatenate independent latent rows before the shared linear operator.
    """
    first = parts[0]
    target, layer = str(first["target"]), int(first["layer"])
    batch, _, heads, head_dim = map(int, first["shape"])
    joint_dim, rank = heads * head_dim, int(first["rank"])
    operator = _load_operator(
        first["basis_root"], target=target, sp_rank=int(first["sp_rank"]),
        layer=layer, joint_dim=joint_dim, rank=rank, device=device,
        linear_root=first.get("linear_root") or None,
    )
    latent_parts = []
    for part in parts:
        if (
            part.get("format") != FORMAT
            or str(part["target"]) != target
            or int(part["layer"]) != layer
            or tuple(map(int, part["shape"][::3])) != (batch, head_dim)
            or int(part["shape"][2]) != heads
            or int(part["rank"]) != rank
        ):
            raise ValueError("incompatible LingBot codec segments")
        packed = part["latent_packed"].to(device)
        scales = part["latent_scale"].to(device=device)
        shape = tuple(map(int, part["latent_shape"]))
        if packed.is_cuda:
            from .lingbot_v2_kv_codec_triton import unpack_int4_scaled
            latent = unpack_int4_scaled(packed, scales, shape)
        else:
            codes = _unpack_int4(packed, int(part["latent_count"]), shape)
            latent = codes.to(torch.bfloat16) * scales.to(torch.bfloat16)
        latent_parts.append(latent.reshape(batch, int(part["shape"][1]), rank))
    latent = torch.cat(latent_parts, dim=1)
    rows = _linear(
        latent.reshape(-1, rank), operator["decoder"],
        operator["mean"].to(torch.bfloat16),
    ).reshape(batch, latent.shape[1], joint_dim)
    offset = 0
    for part in parts:
        sequence = int(part["shape"][1])
        centers = part["centers"].to(device=device, dtype=torch.bfloat16)
        ids = part["ids"].to(device=device, dtype=torch.long)
        target_rows = rows[:, offset : offset + sequence]
        if target_rows.is_cuda:
            from .lingbot_v2_kv_codec_triton import add_centers_inplace
            add_centers_inplace(target_rows, centers, ids)
        else:
            target_rows.add_(
                torch.gather(
                    centers, 1, ids[..., None].expand(-1, -1, joint_dim)
                )
            )
        offset += sequence
    return rows.reshape(batch, latent.shape[1], heads, head_dim).to(dtype)


def _stable_token_ids(sequence: int, count: int, identity: str) -> torch.Tensor:
    digest = hashlib.sha256(identity.encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    )
    return torch.randperm(sequence, generator=generator)[: min(sequence, count)].sort().values


@torch.no_grad()
def dump_paired_source(
    value: torch.Tensor,
    *,
    path: Path,
    target: str,
    layer: int,
    sp_rank: int,
    case_id: str,
    chunk_id: int,
    sample_tokens: int = 1024,
    clusters: int = 64,
    iters: int = 2,
) -> None:
    """Persist raw and C64-residual rows at exactly the same token IDs."""
    if path.exists():
        return
    sequence, heads, head_dim = map(int, value.shape[1:])
    ids = _stable_token_ids(
        sequence, sample_tokens, f"{case_id}|{chunk_id}|{sp_rank}|{sequence}|20260826"
    )
    selected = ids.to(value.device)
    raw = value[0].index_select(0, selected).float().reshape(ids.numel(), -1)
    residual, centers, labels = semantic_residual(
        value, clusters=clusters, iters=iters,
        seed=20260826 + layer * 17 + sp_rank * 1009 + (target == "v"),
    )
    residual = residual[0].index_select(0, selected)

    def quantize(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
        return (rows / scale).round().clamp(-127, 127).to(torch.int8).cpu(), scale.to(torch.bfloat16).cpu()

    raw_q, raw_s = quantize(raw)
    res_q, res_s = quantize(residual)
    payload = {
        "format": SAMPLE_FORMAT, "case_id": case_id, "chunk_id": int(chunk_id),
        "target": target, "layer": int(layer), "sp_rank": int(sp_rank),
        "token_ids": ids.to(torch.int32), "source_shape": tuple(value.shape),
        "joint_dim": heads * head_dim, "sample_tokens": ids.numel(),
        "raw_tensor_int8": raw_q, "raw_int8_scale": raw_s,
        "residual_tensor_int8": res_q, "residual_int8_scale": res_s,
        "clusters": clusters, "cluster_iters": iters,
        "center_l2": centers.float().cpu().norm(dim=-1),
        "cluster_histogram": torch.bincount(labels.flatten().cpu().long(), minlength=clusters),
        "cache_stage": "pre_rope_k/raw_v_committed_x0",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)
