"""Taylor-corrected attention for the Jensen INT2 baseline.

The paper derives the correction for quantized keys consumed directly by
attention.  HY-WorldPlay instead uses one canonical pre-position cache and
reconstructs Main-RoPE and ProPE branches at read time.  Main-RoPE is
group-local orthogonal for group size 32.  ProPE is not orthogonal, so its
effective query energy is evaluated with the exact key-side projection matrix
for each historical camera.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from hyvideo.models.transformers.modules.jensen_kv_adapter import (
    key_steps,
    randomized_hadamard,
)
from hyvideo.prope.camera_rope import _prepare_projection_matrices


_compiled_flex_attention = torch.compile(_flex_attention, dynamic=False)


def _group_energy(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[-1] % group_size:
        raise ValueError("query head dimension must be divisible by group size")
    return x.float().reshape(*x.shape[:-1], -1, group_size).square().sum(-1)


def _prope_group_energy(
    query_prope: torch.Tensor,
    p_inv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Return ||P_inv(camera)^T q||^2 per query/camera/channel-group.

    Shape: query [B,H,Q,D], P_inv [B,C,4,4] -> [B,H,Q,C,D/g].
    The direct quadratic-form contraction avoids materializing transformed
    queries with an additional D axis for every historical camera.
    """
    if group_size % 4 or query_prope.shape[-1] % group_size:
        raise ValueError("ProPE correction requires groups aligned to 4D tiles")
    groups = query_prope.shape[-1] // group_size
    tiles_per_group = group_size // 4
    q = query_prope.float().reshape(
        *query_prope.shape[:-1], groups, tiles_per_group, 4
    )
    transform_gram = torch.matmul(
        p_inv.float(), p_inv.float().transpose(-1, -2)
    )
    # Force the memory-safe contraction order.  A three-operand einsum may
    # broadcast q over every camera before reducing the 8x4 tile axes, which
    # creates a >100 GiB intermediate at production sequence lengths.
    query_second_moment = torch.einsum(
        "bhqgri,bhqgrj->bhqgij", q, q
    )
    return torch.einsum(
        "bhqgij,bcij->bhqcg", query_second_moment, transform_gram
    )


def corrected_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_cache: dict[str, Any],
    text_tokens: int,
    history_tokens: int,
) -> torch.Tensor:
    """Run dense attention and correct only quantized historical visual scores."""
    if query.ndim != 4 or query.shape[0] % 2:
        raise ValueError("HY Jensen attention expects virtual Main/ProPE batch 2B")
    if history_tokens <= 0:
        raise ValueError("Taylor correction requires a non-empty historical cache")
    base_batch = query.shape[0] // 2
    head_dim = int(query.shape[-1])
    group_size = int(layer_cache["k_vision"]["group_size"])
    groups = head_dim // group_size
    if groups != 4:
        raise ValueError(
            f"paper-default d=128,g=32 expects four groups, got d={head_dim},g={group_size}"
        )

    steps = key_steps(layer_cache, query.device)
    step_factor = steps.float().square() / float(24 * head_dim)
    cache_mode = layer_cache.get("vision_kv_position_mode")
    if cache_mode == "post":
        expected = (query.shape[0], query.shape[1], history_tokens, groups)
        if tuple(steps.shape) != tuple(expected):
            raise ValueError(f"K step shape {tuple(steps.shape)} != expected {expected}")
        layer_idx = int(layer_cache["vision_block_idx"])
        # Cached post-position K was quantized in H-space.  Decompression
        # restores it to the ordinary post space so the surrounding cache path
        # stays conventional; rotating the complete Q/K pair here maps the
        # history back to its quantized H-space and preserves text/current
        # scores at the same time.
        query = randomized_hadamard(query, layer_idx)
        key = randomized_hadamard(key, layer_idx)
        cameras = 1
        patches_per_camera = history_tokens
        branch_energy = _group_energy(query, group_size).unsqueeze(3).contiguous()
        branch_step = step_factor.contiguous()
    elif cache_mode == "pre":
        expected = (base_batch, query.shape[1], history_tokens, groups)
        if tuple(steps.shape) != tuple(expected):
            raise ValueError(f"K step shape {tuple(steps.shape)} != expected {expected}")
        query_main, query_prope = torch.chunk(query, 2, dim=0)
        main_energy = _group_energy(query_main, group_size)
        viewmats = layer_cache.get("vision_viewmats")
        Ks = layer_cache.get("vision_Ks")
        if viewmats is None:
            raise RuntimeError("ProPE Taylor correction requires historical view matrices")
        vision_thw = layer_cache.get("vision_thw")
        if vision_thw is None:
            raise RuntimeError("ProPE Taylor correction requires historical T/H/W")
        time_tokens, height_tokens, width_tokens = (int(v) for v in vision_thw)
        if time_tokens * height_tokens * width_tokens != history_tokens:
            raise ValueError(
                f"history shape {vision_thw} does not match {history_tokens} tokens"
            )
        # HY expands each temporal camera matrix over all HxW spatial tokens before
        # entering the transformer.  Deduplicate those identical matrices before
        # forming query-camera energies; native ProPE still consumes the expanded
        # matrices when reconstructing K/V.
        if int(viewmats.shape[1]) == history_tokens:
            viewmats = viewmats.reshape(
                viewmats.shape[0], time_tokens, height_tokens, width_tokens, 4, 4
            )[:, :, 0, 0]
            if Ks is not None:
                Ks = Ks.reshape(
                    Ks.shape[0], time_tokens, height_tokens, width_tokens, 3, 3
                )[:, :, 0, 0]
        _, _, p_inv = _prepare_projection_matrices(
            viewmats.to(query.device), None if Ks is None else Ks.to(query.device)
        )
        cameras = int(p_inv.shape[1])
        if history_tokens % cameras:
            raise ValueError(
                f"history length {history_tokens} is not divisible by cameras {cameras}"
            )
        patches_per_camera = history_tokens // cameras
        prope_energy = _prope_group_energy(query_prope, p_inv, group_size)
        # Match HY's virtual batch exactly.  FlexAttention has an efficient batching
        # rule for direct captured-buffer indexing, whereas PyTorch 2.6's aten.take
        # fallback expands over QxK.  Main energy is camera-invariant and is only
        # expanded here to give both branches the same indexing contract.
        branch_energy = torch.cat(
            (
                main_energy.unsqueeze(3).expand(-1, -1, -1, cameras, -1),
                prope_energy,
            ),
            dim=0,
        ).contiguous()
        branch_step = torch.cat((step_factor, step_factor), dim=0).contiguous()
    else:
        raise ValueError(f"unsupported Jensen cache mode {cache_mode!r}")
    energy_groups = tuple(t.contiguous() for t in branch_energy.unbind(-1))
    step_groups = tuple(t.contiguous() for t in branch_step.unbind(-1))

    # Import the unwrapped API here.  It is compiled together with this
    # score_mod closure, allowing the four-term bias to be fused into online
    # softmax without materializing [B,H,Q,K] corrections.
    # PyTorch 2.6 represents the score-mod indices as scalar tensors.  Dynamic
    # indexing of captured per-query/per-key metadata therefore needs scalar
    # output capture enabled during the higher-order-op trace.
    torch._dynamo.config.capture_scalar_outputs = True

    history_start = int(text_tokens)
    history_end = history_start + int(history_tokens)

    def score_mod(score, b, h, q_idx, kv_idx):
        in_history = (kv_idx >= history_start) & (kv_idx < history_end)
        history_idx = torch.clamp(kv_idx - history_start, 0, history_tokens - 1)
        camera_idx = torch.clamp(
            history_idx // patches_per_camera, 0, cameras - 1
        )
        bias = score.new_zeros(())
        # Static unrolling is intentional: G=4 is the paper default and avoids
        # a graph-level loop or a dense correction tensor.
        for group_idx in range(4):
            # Explicit aten advanced indexing preserves the scalar indices as
            # tensors and has a vmap rule in torch 2.6.  Python __getitem__
            # attempts to call .item(), while aten.take lacks a batching rule.
            nu = torch.ops.aten.index.Tensor(
                energy_groups[group_idx], [b, h, q_idx, camera_idx]
            )
            factor = torch.ops.aten.index.Tensor(
                step_groups[group_idx], [b, h, history_idx]
            )
            bias = bias + factor * nu
        return score - torch.where(in_history, bias, score.new_zeros(()))

    return _compiled_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=1.0 / math.sqrt(head_dim),
        # The default autotuned 128x128 tile and a 64x64 tile both exceed A100
        # shared memory once the four correction metadata groups are captured.
        # Keep the production default conservative; the environment overrides
        # exist only for the standalone kernel-shape benchmark.
        kernel_options={
            "BLOCK_M": int(os.environ.get("HY_JENSEN_BLOCK_M", "64")),
            "BLOCK_N": int(os.environ.get("HY_JENSEN_BLOCK_N", "32")),
        },
    )
