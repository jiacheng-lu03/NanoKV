"""LongCat block-sparse attention with optional token validity masking."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_bsa_align(
    Q,
    K,
    V,
    KV_VALID,
    sm_scale,
    Out,
    block_indices,
    block_indices_lens,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_mz,
    stride_mn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_lz,
    stride_lh,
    stride_lm,
    H,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_KV_VALID_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    q_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q + q_offset + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qk,
        mask=q_offsets[:, None] < Q_LEN,
        other=0.0,
    )
    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    selected_count = tl.load(block_indices_lens)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for i in range(selected_count):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        key_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(
            K
            + k_offset
            + key_offsets[None, :] * stride_kn
            + d_offsets[:, None] * stride_kk
        )
        v = tl.load(
            V
            + v_offset
            + key_offsets[:, None] * stride_vn
            + d_offsets[None, :] * stride_vk
        )
        qk = tl.dot(q, k)
        if HAS_KV_VALID_MASK:
            valid = tl.load(
                KV_VALID + off_z * stride_mz + key_offsets * stride_mn
            ).to(tl.int1)
            qk = tl.where(valid[None, :], qk, -float("inf"))

        block_max = tl.max(qk, axis=1) * qk_scale
        m_ij = tl.maximum(m_i, block_max)
        alpha = tl.exp2(m_i - m_ij)
        shifted = qk * qk_scale - m_ij[:, None]
        p = tl.exp2(shifted)
        if HAS_KV_VALID_MASK:
            p = tl.where(valid[None, :], p, 0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    output = acc / l_i[:, None]
    output = tl.where((q_offsets < Q_LEN)[:, None], output, 0.0)
    tl.store(
        Out
        + o_offset
        + q_offsets[:, None] * stride_om
        + d_offsets[None, :] * stride_ok,
        output.to(Out.type.element_ty),
        mask=q_offsets[:, None] < Q_LEN,
    )


def longcat_flash_attn(
    q,
    k,
    v,
    block_indices,
    block_counts,
    kv_valid_mask=None,
    sm_scale=None,
    logical_block_size=128,
):
    """Run aligned block-sparse attention over compact physical KV blocks."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.to(torch.int32).contiguous()
    block_counts = block_counts.to(torch.int32).contiguous()
    batch, heads, q_len, head_dim = q.shape
    if k.shape != v.shape or k.shape[:2] != (batch, heads):
        raise ValueError("K/V shape must match Q batch and heads")
    k_len = k.shape[2]
    block_n = int(logical_block_size)
    if k_len % block_n:
        raise ValueError(
            f"K length ({k_len}) must be divisible by block size ({block_n})"
        )
    if block_indices.shape[:3] != (
        batch,
        heads,
        math.ceil(q_len / 128),
    ):
        raise ValueError("block_indices shape is incompatible with Q")
    if block_counts.shape != block_indices.shape[:3]:
        raise ValueError("block_counts shape is incompatible with indices")
    if block_indices.numel() and (
        int(block_indices.min()) < 0
        or int(block_indices.max()) >= k_len // block_n
    ):
        raise ValueError("sparse block index is out of bounds")
    if torch.any(block_counts < 1) or torch.any(
        block_counts > block_indices.shape[-1]
    ):
        raise ValueError("each query block must reference at least one valid KV block")

    has_mask = kv_valid_mask is not None
    if has_mask:
        if kv_valid_mask.shape != (batch, k_len):
            raise ValueError(
                f"kv_valid_mask must be {(batch, k_len)}, got {tuple(kv_valid_mask.shape)}"
            )
        kv_valid_mask = kv_valid_mask.to(device=q.device, dtype=torch.uint8).contiguous()
    else:
        kv_valid_mask = q

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(q)
    grid = (triton.cdiv(q_len, 128), batch * heads)
    _attn_fwd_bsa_align[grid](
        Q=q,
        K=k,
        V=v,
        KV_VALID=kv_valid_mask,
        sm_scale=sm_scale,
        Out=output,
        block_indices=block_indices,
        block_indices_lens=block_counts,
        stride_qz=q.stride(0),
        stride_qh=q.stride(1),
        stride_qm=q.stride(2),
        stride_qk=q.stride(3),
        stride_kz=k.stride(0),
        stride_kh=k.stride(1),
        stride_kn=k.stride(2),
        stride_kk=k.stride(3),
        stride_vz=v.stride(0),
        stride_vh=v.stride(1),
        stride_vn=v.stride(2),
        stride_vk=v.stride(3),
        stride_mz=kv_valid_mask.stride(0) if has_mask else 0,
        stride_mn=kv_valid_mask.stride(1) if has_mask else 0,
        stride_oz=output.stride(0),
        stride_oh=output.stride(1),
        stride_om=output.stride(2),
        stride_ok=output.stride(3),
        stride_bz=block_indices.stride(0),
        stride_bh=block_indices.stride(1),
        stride_bm=block_indices.stride(2),
        stride_bs=block_indices.stride(3),
        stride_lz=block_counts.stride(0),
        stride_lh=block_counts.stride(1),
        stride_lm=block_counts.stride(2),
        H=heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        HEAD_DIM=head_dim,
        BLOCK_M=128,
        BLOCK_N=block_n,
        HAS_KV_VALID_MASK=has_mask,
        num_warps=8,
        num_stages=3,
    )
    return output
