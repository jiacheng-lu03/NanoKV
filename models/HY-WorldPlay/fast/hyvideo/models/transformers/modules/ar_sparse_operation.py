"""Block layout and selective-decode utilities for pre-position vision KV."""

import json
import os
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .kv_codecs import decode_global_pca_selected


@dataclass(frozen=True)
class SparseBlockLayout:
    tile_size: Tuple[int, int, int]
    original_thw: Tuple[int, int, int]
    padded_thw: Tuple[int, int, int]
    grid_thw: Tuple[int, int, int]
    num_blocks: int
    block_valid_mask: torch.Tensor
    block_to_token_ids: torch.Tensor

    @classmethod
    def build(
        cls,
        thw: Tuple[int, int, int],
        tile_size: Tuple[int, int, int] = (4, 8, 4),
        device=None,
    ) -> "SparseBlockLayout":
        t, h, w = (int(value) for value in thw)
        tt, th, tw = (int(value) for value in tile_size)
        if min(t, h, w, tt, th, tw) <= 0:
            raise ValueError(f"THW and tile size must be positive: {thw}, {tile_size}")
        nt, nh, nw = ceil(t / tt), ceil(h / th), ceil(w / tw)

        bt = torch.arange(nt, device=device)[:, None, None, None, None, None]
        bh = torch.arange(nh, device=device)[None, :, None, None, None, None]
        bw = torch.arange(nw, device=device)[None, None, :, None, None, None]
        dt = torch.arange(tt, device=device)[None, None, None, :, None, None]
        dh = torch.arange(th, device=device)[None, None, None, None, :, None]
        dw = torch.arange(tw, device=device)[None, None, None, None, None, :]
        coord_t = bt * tt + dt
        coord_h = bh * th + dh
        coord_w = bw * tw + dw
        coord_t, coord_h, coord_w = torch.broadcast_tensors(
            coord_t, coord_h, coord_w
        )
        valid = (coord_t < t) & (coord_h < h) & (coord_w < w)
        token_ids = (coord_t * h + coord_h) * w + coord_w
        token_ids = token_ids.masked_fill(~valid, -1)
        block_size = tt * th * tw
        num_blocks = nt * nh * nw
        return cls(
            tile_size=(tt, th, tw),
            original_thw=(t, h, w),
            padded_thw=(nt * tt, nh * th, nw * tw),
            grid_thw=(nt, nh, nw),
            num_blocks=num_blocks,
            block_valid_mask=valid.reshape(num_blocks, block_size),
            block_to_token_ids=token_ids.reshape(num_blocks, block_size).long(),
        )

    @property
    def block_size(self) -> int:
        tt, th, tw = self.tile_size
        return tt * th * tw

    @property
    def num_tokens(self) -> int:
        t, h, w = self.original_thw
        return t * h * w

    def selected_tokens(self, block_ids: torch.Tensor):
        block_ids = torch.as_tensor(
            block_ids, dtype=torch.long, device=self.block_to_token_ids.device
        ).reshape(-1)
        if block_ids.numel() and (
            int(block_ids.min()) < 0 or int(block_ids.max()) >= self.num_blocks
        ):
            raise IndexError("selected visual block id is out of bounds")
        token_ids = self.block_to_token_ids.index_select(0, block_ids)
        valid = self.block_valid_mask.index_select(0, block_ids)
        return token_ids, valid


@dataclass
class CompressedVisionKV:
    k_code_pre: Any
    v_code_pre: Any
    k_proxy_rope: torch.Tensor
    k_proxy_prope: torch.Tensor
    block_layout: SparseBlockLayout
    frame_indices: Optional[torch.Tensor] = None
    camera_metadata: Optional[dict] = None


@dataclass
class CompactSparseKV:
    k: torch.Tensor
    v: torch.Tensor
    indices: torch.Tensor
    counts: torch.Tensor
    kv_valid_mask: torch.Tensor
    original_to_compact: List[torch.Tensor]
    compact_visual_block_ids: List[torch.Tensor]


def tile_visual_tokens(x: torch.Tensor, layout: SparseBlockLayout) -> torch.Tensor:
    """Tile ``[B,H,T*Y*X,D]`` into block-major ``[B,H,N,block,D]``."""
    if x.ndim != 4 or x.shape[2] != layout.num_tokens:
        raise ValueError(
            f"expected [B,H,{layout.num_tokens},D], got {tuple(x.shape)}"
        )
    safe_ids = layout.block_to_token_ids.clamp_min(0).reshape(-1)
    tiled = x.index_select(2, safe_ids).reshape(
        x.shape[0], x.shape[1], layout.num_blocks, layout.block_size, x.shape[-1]
    )
    mask = layout.block_valid_mask[None, None, :, :, None]
    return tiled.masked_fill(~mask, 0)


def masked_block_mean(tiled: torch.Tensor, layout: SparseBlockLayout) -> torch.Tensor:
    if tiled.ndim != 5 or tiled.shape[2:4] != (
        layout.num_blocks,
        layout.block_size,
    ):
        raise ValueError("tiled tensor does not match sparse block layout")
    mask = layout.block_valid_mask.to(tiled.device)[None, None, :, :, None]
    count = mask.sum(dim=3).clamp_min(1)
    return (tiled.float() * mask).sum(dim=3).div(count).to(tiled.dtype)


def tile_and_pool_visual_tokens(
    x: torch.Tensor, layout: SparseBlockLayout
):
    tiled = tile_visual_tokens(x, layout)
    return tiled, masked_block_mean(tiled, layout)


def untile_visual_tokens(tiled: torch.Tensor, layout: SparseBlockLayout) -> torch.Tensor:
    """Restore valid tokens in their original time-major order."""
    if tiled.ndim != 5 or tiled.shape[2:4] != (
        layout.num_blocks,
        layout.block_size,
    ):
        raise ValueError("tiled tensor does not match sparse block layout")
    output = tiled.new_empty(
        tiled.shape[0], tiled.shape[1], layout.num_tokens, tiled.shape[-1]
    )
    valid = layout.block_valid_mask.reshape(-1)
    token_ids = layout.block_to_token_ids.reshape(-1)[valid]
    values = tiled.reshape(
        tiled.shape[0], tiled.shape[1], -1, tiled.shape[-1]
    )[:, :, valid]
    output.index_copy_(2, token_ids, values)
    return output


def decode_selected_visual_blocks(
    payload,
    layout: SparseBlockLayout,
    block_ids: torch.Tensor,
    source_batch: int,
    dtype: torch.dtype,
    device,
):
    """Decode valid tokens from selected blocks and preserve block-major order."""
    token_ids, valid = layout.selected_tokens(block_ids)
    flat_valid = valid.reshape(-1)
    valid_token_ids = token_ids.reshape(-1)[flat_valid]
    decoded = decode_global_pca_selected(
        payload,
        valid_token_ids,
        source_batch=source_batch,
        dtype=dtype,
        device=device,
    )
    output = decoded.new_zeros(
        decoded.shape[0],
        token_ids.shape[0],
        layout.block_size,
        decoded.shape[-1],
    )
    output.reshape(decoded.shape[0], -1, decoded.shape[-1])[:, flat_valid] = decoded
    return output, token_ids, valid


def apply_rope_selected(
    x: torch.Tensor,
    token_ids: torch.Tensor,
    freqs_cis,
) -> torch.Tensor:
    """Apply native RoPE to ``[B,H,S,D]`` using original dense token IDs."""
    from .posemb_layers import apply_rotary_emb_one

    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=x.device).reshape(-1)
    if isinstance(freqs_cis, tuple):
        selected_freqs = tuple(
            part.to(x.device).index_select(0, token_ids) for part in freqs_cis
        )
    else:
        selected_freqs = freqs_cis.to(x.device).index_select(0, token_ids)
    result = apply_rotary_emb_one(
        x.transpose(1, 2), selected_freqs, head_first=False
    )
    return result.transpose(1, 2).contiguous()


def apply_prope_selected(
    x: torch.Tensor,
    token_ids: torch.Tensor,
    full_sequence_length: int,
    viewmats: torch.Tensor,
    Ks,
) -> torch.Tensor:
    """Apply exact native PRoPE KV transforms using original token IDs."""
    from hyvideo.prope.camera_rope import apply_prope_kv_selected

    return apply_prope_kv_selected(
        x,
        token_ids,
        full_sequence_length,
        viewmats,
        Ks,
    )


def generate_visual_block_indices(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    topk_ratio: float,
    recent_protect: int = 0,
    mandatory_visual_block_ids: Optional[torch.Tensor] = None,
    sim_metric: str = "cosine",
):
    """Select original visual block IDs without mixing in text offsets."""
    if q_proxy.ndim != 4 or k_proxy.ndim != 4:
        raise ValueError("Q/K proxies must be [B,H,N_blocks,D]")
    if q_proxy.shape[:2] != k_proxy.shape[:2] or q_proxy.shape[-1] != k_proxy.shape[-1]:
        raise ValueError("Q/K proxy batch, heads and dimension must match")
    if not 0.0 <= topk_ratio <= 1.0:
        raise ValueError("topk_ratio must be in [0,1]")
    if sim_metric == "cosine":
        q_score = F.normalize(q_proxy.float(), dim=-1)
        k_score = F.normalize(k_proxy.float(), dim=-1)
        scores = torch.matmul(q_score, k_score.transpose(-2, -1))
    elif sim_metric == "scaled_dot_product":
        scores = torch.matmul(
            q_proxy.float(), k_proxy.float().transpose(-2, -1)
        ) / q_proxy.shape[-1] ** 0.5
    elif sim_metric == "dot":
        scores = torch.matmul(q_proxy.float(), k_proxy.float().transpose(-2, -1))
    else:
        raise ValueError(f"unsupported sparse similarity metric={sim_metric}")

    num_keys = k_proxy.shape[2]
    mandatory = torch.zeros(num_keys, dtype=torch.bool, device=k_proxy.device)
    if recent_protect:
        mandatory[max(0, num_keys - int(recent_protect)) :] = True
    if mandatory_visual_block_ids is not None:
        ids = torch.as_tensor(
            mandatory_visual_block_ids, dtype=torch.long, device=k_proxy.device
        ).reshape(-1)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= num_keys):
            raise IndexError("mandatory visual block id is out of bounds")
        mandatory[ids] = True
    candidates = (~mandatory).nonzero(as_tuple=False).flatten()
    keep_optional = ceil(int(candidates.numel()) * float(topk_ratio))
    mandatory_ids = mandatory.nonzero(as_tuple=False).flatten()
    selected_count = int(mandatory_ids.numel()) + keep_optional
    if selected_count == 0:
        raise ValueError("sparse selection produced no visual blocks")

    batch, heads, query_blocks, _ = q_proxy.shape
    output = torch.empty(
        batch,
        heads,
        query_blocks,
        selected_count,
        device=q_proxy.device,
        dtype=torch.int32,
    )
    cursor = 0
    if mandatory_ids.numel():
        output[..., : mandatory_ids.numel()] = mandatory_ids.to(torch.int32)
        cursor = int(mandatory_ids.numel())
    if keep_optional:
        candidate_scores = scores.index_select(-1, candidates)
        top_local = torch.topk(
            candidate_scores, keep_optional, dim=-1, sorted=True
        ).indices
        optional_ids = candidates[top_local]
        output[..., cursor:] = optional_ids.to(torch.int32)
    counts = torch.full(
        (batch, heads, query_blocks),
        selected_count,
        device=q_proxy.device,
        dtype=torch.int32,
    )
    return output, counts


def selected_block_union(
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> List[torch.Tensor]:
    """Union selected visual IDs over all heads and query blocks per batch."""
    unions = []
    for batch_idx in range(indices.shape[0]):
        pieces = []
        for head_idx in range(indices.shape[1]):
            for query_idx in range(indices.shape[2]):
                count = int(counts[batch_idx, head_idx, query_idx])
                pieces.append(indices[batch_idx, head_idx, query_idx, :count])
        unions.append(torch.unique(torch.cat(pieces).long(), sorted=True))
    return unions


def generate_oracle_visual_indices(
    q_tiled: torch.Tensor,
    text_key: torch.Tensor,
    visual_key_tiled: torch.Tensor,
    visual_valid_mask: torch.Tensor,
    topk_ratio: float,
    mandatory_visual_block_ids: torch.Tensor,
):
    """Generate debug-only block indices from exact dense attention mass."""
    batch, heads, query_blocks, block_size, dim = q_tiled.shape
    visual_blocks = visual_key_tiled.shape[2]
    text_len = text_key.shape[2]
    visual_flat = visual_key_tiled.flatten(2, 3)
    dense_key = torch.cat([text_key, visual_flat], dim=2)
    logits = torch.einsum(
        "bhnqd,bhkd->bhnqk", q_tiled.float(), dense_key.float()
    ) / dim**0.5
    dense_valid = torch.cat(
        [
            torch.ones(
                batch, text_len, dtype=torch.bool, device=q_tiled.device
            ),
            visual_valid_mask.reshape(batch, -1),
        ],
        dim=1,
    )
    logits.masked_fill_(~dense_valid[:, None, None, None, :], -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    visual_mass = probabilities[..., text_len:].reshape(
        batch, heads, query_blocks, block_size, visual_blocks, block_size
    ).sum(dim=(3, 5))

    mandatory = torch.zeros(
        visual_blocks, dtype=torch.bool, device=q_tiled.device
    )
    mandatory_ids = torch.as_tensor(
        mandatory_visual_block_ids, dtype=torch.long, device=q_tiled.device
    ).reshape(-1)
    mandatory[mandatory_ids] = True
    candidates = (~mandatory).nonzero(as_tuple=False).flatten()
    optional_count = ceil(candidates.numel() * float(topk_ratio))
    width = mandatory_ids.numel() + optional_count
    indices = torch.empty(
        batch,
        heads,
        query_blocks,
        width,
        dtype=torch.int32,
        device=q_tiled.device,
    )
    indices[..., : mandatory_ids.numel()] = mandatory_ids.to(torch.int32)
    if optional_count:
        top_local = torch.topk(
            visual_mass.index_select(-1, candidates),
            optional_count,
            dim=-1,
        ).indices
        indices[..., mandatory_ids.numel() :] = candidates[top_local].to(
            torch.int32
        )
    counts = torch.full(
        indices.shape[:-1],
        width,
        dtype=torch.int32,
        device=q_tiled.device,
    )
    return indices, counts, visual_mass


def compact_sparse_kv(
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    visual_k_blocks: List[torch.Tensor],
    visual_v_blocks: List[torch.Tensor],
    visual_valid_masks: List[torch.Tensor],
    compact_visual_block_ids: List[torch.Tensor],
    visual_indices: torch.Tensor,
    visual_counts: torch.Tensor,
    total_visual_blocks: int,
    block_size: int = 128,
) -> CompactSparseKV:
    """Pack text and selected visual blocks and remap sparse indices."""
    if text_k.shape != text_v.shape or text_k.ndim != 4:
        raise ValueError("text K/V must have matching [B,H,S,D] shapes")
    batch, heads, text_len, dim = text_k.shape
    if (
        len(visual_k_blocks) != batch
        or len(visual_v_blocks) != batch
        or len(visual_valid_masks) != batch
    ):
        raise ValueError("one visual block tensor is required per virtual batch")
    text_blocks = ceil(text_len / block_size)
    max_visual = max(int(ids.numel()) for ids in compact_visual_block_ids)
    total_compact_blocks = text_blocks + max_visual
    compact_k = text_k.new_zeros(
        batch, heads, total_compact_blocks, block_size, dim
    )
    compact_v = text_v.new_zeros(compact_k.shape)
    kv_valid = torch.zeros(
        batch,
        total_compact_blocks,
        block_size,
        dtype=torch.bool,
        device=text_k.device,
    )
    compact_k[:, :, :text_blocks].reshape(batch, heads, -1, dim)[
        :, :, :text_len
    ] = text_k
    compact_v[:, :, :text_blocks].reshape(batch, heads, -1, dim)[
        :, :, :text_len
    ] = text_v
    kv_valid[:, :text_blocks].reshape(batch, -1)[:, :text_len] = True

    mappings = []
    for batch_idx in range(batch):
        ids = compact_visual_block_ids[batch_idx].long()
        k_blocks = visual_k_blocks[batch_idx]
        v_blocks = visual_v_blocks[batch_idx]
        if k_blocks.shape != v_blocks.shape or k_blocks.shape != (
            heads,
            ids.numel(),
            block_size,
            dim,
        ):
            raise ValueError("decoded visual blocks do not match compact IDs")
        count = int(ids.numel())
        compact_k[batch_idx, :, text_blocks : text_blocks + count] = k_blocks
        compact_v[batch_idx, :, text_blocks : text_blocks + count] = v_blocks
        block_valid = visual_valid_masks[batch_idx].to(
            device=text_k.device, dtype=torch.bool
        )
        if block_valid.shape != (count, block_size):
            raise ValueError("visual valid mask does not match decoded blocks")
        kv_valid[batch_idx, text_blocks : text_blocks + count] = block_valid
        mapping = torch.full(
            (total_visual_blocks,),
            -1,
            dtype=torch.int32,
            device=text_k.device,
        )
        mapping[ids] = (
            torch.arange(count, device=text_k.device, dtype=torch.int32)
            + text_blocks
        )
        mappings.append(mapping)

    max_visual_per_query = visual_indices.shape[-1]
    final_width = text_blocks + max_visual_per_query
    remapped = torch.zeros(
        *visual_indices.shape[:-1],
        final_width,
        dtype=torch.int32,
        device=text_k.device,
    )
    text_ids = torch.arange(text_blocks, device=text_k.device, dtype=torch.int32)
    remapped[..., :text_blocks] = text_ids
    final_counts = visual_counts.to(torch.int32) + text_blocks
    for batch_idx in range(batch):
        mapped = mappings[batch_idx][visual_indices[batch_idx].long()]
        valid_slots = (
            torch.arange(max_visual_per_query, device=text_k.device)
            [None, None, :]
            < visual_counts[batch_idx, :, :, None]
        )
        if torch.any(mapped[valid_slots] < 0):
            raise ValueError("a sparse index references a visual block not decoded")
        remapped[batch_idx, ..., text_blocks:][valid_slots] = mapped[valid_slots]

    return CompactSparseKV(
        k=compact_k.flatten(2, 3).contiguous(),
        v=compact_v.flatten(2, 3).contiguous(),
        indices=remapped,
        counts=final_counts,
        kv_valid_mask=kv_valid.flatten(1, 2).contiguous(),
        original_to_compact=mappings,
        compact_visual_block_ids=compact_visual_block_ids,
    )


def _position_selected_blocks(
    blocks: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    branch: int,
    kind: str,
    full_sequence_length: int,
    freqs_cis,
    viewmats: torch.Tensor,
    Ks,
) -> torch.Tensor:
    heads, num_blocks, block_size, dim = blocks.shape
    flat_valid = valid.reshape(-1)
    selected = blocks.reshape(heads, -1, dim)[:, flat_valid][None]
    original_ids = token_ids.reshape(-1)[flat_valid]
    if branch == 0:
        transformed = (
            apply_rope_selected(selected, original_ids, freqs_cis)
            if kind == "k"
            else selected
        )
    else:
        transformed = apply_prope_selected(
            selected,
            original_ids,
            full_sequence_length,
            viewmats,
            Ks,
        )
    output = blocks.new_zeros(heads, num_blocks * block_size, dim)
    output[:, flat_valid] = transformed[0]
    return output.reshape(heads, num_blocks, block_size, dim)


def _sparse_debug_enabled(params) -> bool:
    return bool(params.get("sparse_debug_compare_dense", False)) or os.environ.get(
        "HY_SPARSE_DECODE_DEBUG", "0"
    ).lower() in {"1", "true", "yes", "on"}


def _write_sparse_stats(record: dict) -> None:
    path_value = os.environ.get("HY_SPARSE_DECODE_STATS", "").strip()
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def sparse_pre_position_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    text_key: torch.Tensor,
    text_value: torch.Tensor,
    layer_cache: dict,
    current_thw: Tuple[int, int, int],
    params: dict,
    layer_idx: int,
) -> torch.Tensor:
    """Selective-decode pre-position history and run compact LongCat attention."""
    from .longcat_kernel import longcat_flash_attn

    started = time.perf_counter()
    virtual_batch, heads, query_tokens, dim = query.shape
    source_batch = virtual_batch // 2
    if virtual_batch != source_batch * 2:
        raise ValueError("sparse dual-branch attention expects virtual batch 2B")
    tile_size = tuple(
        int(value) for value in params.get("sparse_tile_size", (4, 8, 4))
    )
    if tuple(layer_cache.get("sparse_proxy_tile_size", ())) != tile_size:
        raise RuntimeError("cached proxy tile size does not match sparse runtime")
    history_layout = SparseBlockLayout.build(
        tuple(int(value) for value in layer_cache["vision_thw"]),
        tile_size,
        device=query.device,
    )
    current_layout = SparseBlockLayout.build(
        tuple(int(value) for value in current_thw),
        tile_size,
        device=query.device,
    )
    if query_tokens != current_layout.num_tokens:
        raise ValueError("current query length does not match vision_thw")

    q_tiled = tile_visual_tokens(query, current_layout)
    q_proxy = masked_block_mean(q_tiled, current_layout)
    current_k_tiled = tile_visual_tokens(current_key, current_layout)
    current_v_tiled = tile_visual_tokens(current_value, current_layout)
    current_k_proxy = masked_block_mean(current_k_tiled, current_layout)
    history_proxy = torch.cat(
        [layer_cache["k_proxy_rope"], layer_cache["k_proxy_prope"]], dim=0
    ).to(device=query.device, dtype=q_proxy.dtype)
    k_proxy = torch.cat([history_proxy, current_k_proxy], dim=2)
    history_blocks = history_layout.num_blocks
    current_ids = torch.arange(
        history_blocks,
        history_blocks + current_layout.num_blocks,
        device=query.device,
    )
    recent_count = min(
        int(params.get("sparse_recent_protect", 0)), history_blocks
    )
    if recent_count:
        recent_ids = torch.arange(
            history_blocks - recent_count,
            history_blocks,
            device=query.device,
        )
        mandatory_ids = torch.cat([recent_ids, current_ids])
    else:
        mandatory_ids = current_ids
    if "_indices_override" in params:
        indices = params["_indices_override"].to(query.device)
        counts = params["_counts_override"].to(query.device)
    else:
        indices, counts = generate_visual_block_indices(
            q_proxy,
            k_proxy,
            topk_ratio=float(params.get("sparse_topk_ratio", 1.0)),
            recent_protect=0,
            mandatory_visual_block_ids=mandatory_ids,
            sim_metric=str(params.get("sparse_sim_metric", "cosine")),
        )
    if not bool(params.get("sparse_dual_branch_index", True)):
        indices[source_batch:] = indices[:source_batch]
        counts[source_batch:] = counts[:source_batch]
    unions = selected_block_union(indices, counts)
    index_ms = (time.perf_counter() - started) * 1000

    decode_started = time.perf_counter()
    visual_k_blocks = []
    visual_v_blocks = []
    visual_valid_masks = []
    compact_ids = []
    decoded_history_tokens = 0
    selective_decode_s = 0.0
    position_encoding_s = 0.0
    for virtual_idx, union_ids in enumerate(unions):
        source_idx = virtual_idx % source_batch
        branch = virtual_idx // source_batch
        history_ids = union_ids[union_ids < history_blocks]
        current_selected = union_ids[union_ids >= history_blocks] - history_blocks
        if history_ids.numel():
            selective_started = time.perf_counter()
            k_hist, token_ids, valid = decode_selected_visual_blocks(
                layer_cache["k_vision"],
                history_layout,
                history_ids,
                source_idx,
                query.dtype,
                query.device,
            )
            v_hist, _, _ = decode_selected_visual_blocks(
                layer_cache["v_vision"],
                history_layout,
                history_ids,
                source_idx,
                query.dtype,
                query.device,
            )
            selective_decode_s += time.perf_counter() - selective_started
            position_started = time.perf_counter()
            decoded_history_tokens += int(valid.sum())
            viewmats = layer_cache["vision_viewmats"][source_idx : source_idx + 1]
            Ks = layer_cache.get("vision_Ks")
            if Ks is not None:
                Ks = Ks[source_idx : source_idx + 1]
            k_hist = _position_selected_blocks(
                k_hist,
                token_ids,
                valid,
                branch,
                "k",
                history_layout.num_tokens,
                layer_cache["vision_freqs_cis"],
                viewmats,
                Ks,
            )
            position_encoding_s += time.perf_counter() - position_started
            v_hist = _position_selected_blocks(
                v_hist,
                token_ids,
                valid,
                branch,
                "v",
                history_layout.num_tokens,
                layer_cache["vision_freqs_cis"],
                viewmats,
                Ks,
            )
        else:
            k_hist = query.new_empty(heads, 0, current_layout.block_size, dim)
            v_hist = query.new_empty(heads, 0, current_layout.block_size, dim)
            valid = torch.empty(
                0,
                current_layout.block_size,
                dtype=torch.bool,
                device=query.device,
            )
        k_current = current_k_tiled[virtual_idx].index_select(1, current_selected)
        v_current = current_v_tiled[virtual_idx].index_select(1, current_selected)
        current_valid = current_layout.block_valid_mask.index_select(
            0, current_selected
        )
        visual_k_blocks.append(torch.cat([k_hist, k_current], dim=1))
        visual_v_blocks.append(torch.cat([v_hist, v_current], dim=1))
        visual_valid_masks.append(torch.cat([valid, current_valid], dim=0))
        compact_ids.append(union_ids)
    decode_ms = (time.perf_counter() - decode_started) * 1000

    compact_started = time.perf_counter()
    compact = compact_sparse_kv(
        text_key,
        text_value,
        visual_k_blocks,
        visual_v_blocks,
        visual_valid_masks,
        compact_ids,
        indices,
        counts,
        history_blocks + current_layout.num_blocks,
        current_layout.block_size,
    )
    compact_ms = (time.perf_counter() - compact_started) * 1000
    attention_started = time.perf_counter()
    output_tiled = longcat_flash_attn(
        q_tiled.flatten(2, 3),
        compact.k,
        compact.v,
        compact.indices,
        compact.counts,
        kv_valid_mask=compact.kv_valid_mask,
        sm_scale=1.0 / dim**0.5,
        logical_block_size=current_layout.block_size,
    )
    attention_ms = (time.perf_counter() - attention_started) * 1000
    untile_started = time.perf_counter()
    output = untile_visual_tokens(
        output_tiled.reshape(
            virtual_batch,
            heads,
            current_layout.num_blocks,
            current_layout.block_size,
            dim,
        ),
        current_layout,
    )
    untile_ms = (time.perf_counter() - untile_started) * 1000

    branch_overlaps = []
    for source_idx in range(source_batch):
        overlap_items = []
        for head_idx in range(heads):
            for query_idx in range(current_layout.num_blocks):
                main_count = int(counts[source_idx, head_idx, query_idx])
                prope_count = int(
                    counts[source_batch + source_idx, head_idx, query_idx]
                )
                main_ids = indices[
                    source_idx, head_idx, query_idx, :main_count
                ]
                prope_ids = indices[
                    source_batch + source_idx,
                    head_idx,
                    query_idx,
                    :prope_count,
                ]
                intersection = torch.isin(main_ids, prope_ids).sum()
                denominator = torch.unique(
                    torch.cat([main_ids, prope_ids])
                ).numel()
                overlap_items.append(intersection / max(denominator, 1))
        branch_overlaps.append(float(torch.stack(overlap_items).mean()))
    _write_sparse_stats(
        {
            "layer": int(layer_idx),
            "tile_size": list(tile_size),
            "q_blocks": current_layout.num_blocks,
            "history_blocks": history_blocks,
            "selected_blocks_rope": [
                int(unions[index].numel()) for index in range(source_batch)
            ],
            "selected_blocks_prope": [
                int(unions[source_batch + index].numel())
                for index in range(source_batch)
            ],
            "selected_union": [int(ids.numel()) for ids in unions],
            "selected_blocks_per_query_min": int(counts.min()),
            "selected_blocks_per_query_max": int(counts.max()),
            "selected_blocks_per_query_mean": float(counts.float().mean()),
            "branch_overlap_ratio": branch_overlaps,
            "decode_ratio": decoded_history_tokens
            / max(virtual_batch * history_layout.num_tokens, 1),
            "decoded_token_count": decoded_history_tokens,
            "text_blocks": ceil(text_key.shape[2] / current_layout.block_size),
            "mandatory_blocks": current_layout.num_blocks,
            "compact_blocks": compact.k.shape[2] // current_layout.block_size,
            "padding_tokens": int((~compact.kv_valid_mask).sum()),
            "index_ms": index_ms,
            "decode_ms": selective_decode_s * 1000,
            "position_encoding_ms": position_encoding_s * 1000,
            "decode_and_position_total_ms": decode_ms,
            "compact_ms": compact_ms,
            "attention_ms": attention_ms,
            "untile_ms": untile_ms,
        }
    )
    return output
