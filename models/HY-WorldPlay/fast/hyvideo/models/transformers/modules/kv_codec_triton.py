"""Triton hot-path kernels for the resident global-PCA KV codec.

The implementation is adapted to the joint-head layout used by this project
(``F = H * D = 2048``).  Quantized byte layout intentionally matches the
existing PyTorch codec: the even INT4 value occupies the low nibble.

The decomposition follows the public Quant-VideoGen Triton implementation,
but uses tiled feature loops so joint-head C64 does not materialize the full
``[tokens, clusters]`` distance matrix or require a 2048-wide register tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _int4_scale_per_row_kernel(
    x_ptr,
    scale_fp32_ptr,
    scale_storage_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    qmax: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, width)
    x = tl.load(x_ptr + row * width + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / qmax, 1.1920928955078125e-7)
    tl.store(scale_fp32_ptr + row, scale)
    tl.store(scale_storage_ptr + row, scale)


@triton.jit
def _int4_quant_pack_given_scale_kernel(
    x_ptr,
    scale_fp32_ptr,
    packed_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    qmax: tl.constexpr,
):
    row = tl.program_id(0)
    scale = tl.load(scale_fp32_ptr + row).to(tl.float32)
    packed_offsets = tl.arange(0, width // 2)
    even_x = tl.load(x_ptr + row * width + 2 * packed_offsets).to(tl.float32)
    odd_x = tl.load(x_ptr + row * width + 2 * packed_offsets + 1).to(tl.float32)
    # torch.round uses round-to-nearest-even; CUDA libdevice.round instead
    # rounds halfway cases away from zero, which changes real cache bytes.
    even = libdevice.rint(libdevice.div_rn(even_x, scale))
    odd = libdevice.rint(libdevice.div_rn(odd_x, scale))
    even = tl.maximum(tl.minimum(even, qmax), -qmax).to(tl.int32) + qmax
    odd = tl.maximum(tl.minimum(odd, qmax), -qmax).to(tl.int32) + qmax
    packed = even | (odd << 4)
    tl.store(packed_ptr + row * (width // 2) + packed_offsets, packed.to(tl.uint8))


def int4_quant_pack_per_row(x: torch.Tensor, scale_dtype: torch.dtype):
    """Quantize contiguous ``[..., width]`` rows and pack two codes per byte."""
    if not x.is_cuda:
        raise ValueError("Triton INT4 quantization requires a CUDA tensor")
    width = int(x.shape[-1])
    if width <= 0 or width % 2 or not triton.next_power_of_2(width) == width:
        raise ValueError(f"INT4 Triton width must be an even power of two, got {width}")
    rows = x.numel() // width
    source = x.reshape(rows, width).contiguous()
    packed = torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8)
    scale = torch.empty((rows, 1), device=x.device, dtype=scale_dtype)
    scale_fp32 = torch.empty((rows,), device=x.device, dtype=torch.float32)
    _int4_scale_per_row_kernel[(rows,)](
        source, scale_fp32, scale,
        rows=rows, width=width, qmax=7, num_warps=4,
    )
    _int4_quant_pack_given_scale_kernel[(rows,)](
        source, scale_fp32, packed,
        rows=rows, width=width, qmax=7, num_warps=4,
    )
    return packed.reshape(-1), scale.reshape(*x.shape[:-1], 1)


@triton.jit
def _int4_unpack_dequant_per_row_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    qmax: tl.constexpr,
):
    row = tl.program_id(0)
    packed_offsets = tl.arange(0, width // 2)
    packed = tl.load(
        packed_ptr + row * (width // 2) + packed_offsets
    ).to(tl.int32)
    even = packed & 0xF
    odd = (packed >> 4) & 0xF
    scale = tl.load(scale_ptr + row).to(tl.float32)
    even = even.to(tl.float32) - qmax
    odd = odd.to(tl.float32) - qmax
    tl.store(out_ptr + row * width + 2 * packed_offsets, even * scale)
    tl.store(out_ptr + row * width + 2 * packed_offsets + 1, odd * scale)


def int4_unpack_dequant_per_row(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Unpack the project's low-nibble-first INT4 format and dequantize."""
    if not packed.is_cuda:
        raise ValueError("Triton INT4 dequantization requires a CUDA tensor")
    width = int(shape[-1])
    rows = 1
    for value in shape[:-1]:
        rows *= int(value)
    if packed.numel() != rows * width // 2:
        raise ValueError("packed INT4 payload size does not match requested shape")
    output = torch.empty((rows, width), device=packed.device, dtype=output_dtype)
    _int4_unpack_dequant_per_row_kernel[(rows,)](
        packed.reshape(-1), scale.reshape(-1), output,
        rows=rows, width=width, qmax=7, num_warps=4,
    )
    return output.reshape(shape)


@triton.jit
def _subtract_joint_centers_kernel(
    rows_ptr,
    centers_ptr,
    labels_ptr,
    output_ptr,
    sequence: tl.constexpr,
    n_clusters: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    width: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    token = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_f = tl.program_id(2)
    f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    valid = f < width
    label = tl.load(labels_ptr + pid_b * sequence + token).to(tl.int32)
    row = tl.load(
        rows_ptr + (pid_b * sequence + token) * width + f,
        mask=valid, other=0.0,
    ).to(tl.float32)
    center = tl.load(
        centers_ptr + (pid_b * n_clusters + label) * width + f,
        mask=valid, other=0.0,
    ).to(tl.float32)
    h = f // head_dim
    d = f % head_dim
    out_offsets = ((pid_b * heads + h) * sequence + token) * head_dim + d
    tl.store(output_ptr + out_offsets, row - center, mask=valid)


def subtract_joint_centers_from_rows(
    rows: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
    heads: int,
    head_dim: int,
    output_dtype: torch.dtype,
):
    """Subtract BF16 stored centers and directly restore ``[B,H,S,D]`` layout."""
    batch, sequence, width = rows.shape
    n_clusters = int(centers.shape[1])
    output = torch.empty(
        (batch, heads, sequence, head_dim), device=rows.device, dtype=output_dtype
    )
    _subtract_joint_centers_kernel[(sequence, batch, triton.cdiv(width, 256))](
        rows, centers, labels, output,
        sequence=sequence, n_clusters=n_clusters, heads=heads,
        head_dim=head_dim, width=width, BLOCK_F=256, num_warps=4,
    )
    return output


@triton.jit
def _add_joint_centers_layout_kernel(
    rows_ptr,
    centers_ptr,
    labels_ptr,
    output_ptr,
    sequence: tl.constexpr,
    n_clusters: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    width: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    token = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_f = tl.program_id(2)
    f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    valid = f < width
    label = tl.load(labels_ptr + pid_b * sequence + token).to(tl.int32)
    row = tl.load(
        rows_ptr + (pid_b * sequence + token) * width + f,
        mask=valid, other=0.0,
    ).to(tl.float32)
    center = tl.load(
        centers_ptr + (pid_b * n_clusters + label) * width + f,
        mask=valid, other=0.0,
    ).to(tl.float32)
    h = f // head_dim
    d = f % head_dim
    out_offsets = ((pid_b * heads + h) * sequence + token) * head_dim + d
    tl.store(output_ptr + out_offsets, row + center, mask=valid)


def add_joint_centers_and_layout(
    rows: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
):
    """Fuse center gather/add and ``[B,S,H*D] -> [B,H,S,D]`` materialization."""
    batch, heads, sequence, head_dim = shape
    width = heads * head_dim
    n_clusters = int(centers.shape[1])
    output = torch.empty(shape, device=rows.device, dtype=output_dtype)
    _add_joint_centers_layout_kernel[(sequence, batch, triton.cdiv(width, 256))](
        rows, centers, labels, output,
        sequence=sequence, n_clusters=n_clusters, heads=heads,
        head_dim=head_dim, width=width, BLOCK_F=256, num_warps=4,
    )
    return output


@triton.jit
def _joint_euclid_assign_kernel(
    x_ptr,
    centers_ptr,
    labels_ptr,
    n_tokens: tl.constexpr,
    n_clusters: tl.constexpr,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    valid_n = n < n_tokens
    valid_k = k < n_clusters
    distance = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for start_d in range(0, width, BLOCK_D):
        d = start_d + tl.arange(0, BLOCK_D)
        x = tl.load(
            x_ptr + (pid_b * n_tokens + n[:, None]) * width + d[None, :],
            mask=valid_n[:, None], other=0.0,
        )
        c = tl.load(
            centers_ptr + (pid_b * n_clusters + k[:, None]) * width + d[None, :],
            mask=valid_k[:, None], other=0.0,
        )
        # ||x-c||^2 accumulated in feature tiles.  Keeping the distance in one
        # kernel avoids materializing the large FP32 [B,N,K] tensor.
        delta = (x[:, None, :] - c[None, :, :]).to(tl.float32)
        distance += tl.sum(delta * delta, axis=2)
    distance = tl.where(valid_k[None, :], distance, float("inf"))
    label = tl.argmin(distance, axis=1)
    tl.store(labels_ptr + pid_b * n_tokens + n, label, mask=valid_n)


@triton.jit
def _joint_centroid_sum_kernel(
    x_ptr,
    labels_ptr,
    sums_ptr,
    n_tokens: tl.constexpr,
    n_clusters: tl.constexpr,
    width: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = d < width
    label = tl.load(labels_ptr + pid_b * n_tokens + token).to(tl.int32)
    values = tl.load(
        x_ptr + (pid_b * n_tokens + token) * width + d,
        mask=valid_d, other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        sums_ptr + (pid_b * n_clusters + label) * width + d,
        values, mask=valid_d,
    )


@triton.jit
def _joint_centroid_count_kernel(
    labels_ptr,
    counts_ptr,
    n_tokens: tl.constexpr,
    n_clusters: tl.constexpr,
):
    token = tl.program_id(0)
    pid_b = tl.program_id(1)
    label = tl.load(labels_ptr + pid_b * n_tokens + token).to(tl.int32)
    tl.atomic_add(counts_ptr + pid_b * n_clusters + label, 1)


def joint_kmeans_iteration(rows: torch.Tensor, centers: torch.Tensor):
    """One exact Euclidean assignment/update iteration for joint-head rows."""
    if not rows.is_cuda or not centers.is_cuda:
        raise ValueError("Triton joint K-Means requires CUDA tensors")
    batch, n_tokens, width = rows.shape
    n_clusters = int(centers.shape[1])
    if centers.shape != (batch, n_clusters, width):
        raise ValueError("joint K-Means center shape mismatch")
    block_n = 4
    block_k = triton.next_power_of_2(n_clusters)
    block_d = 32
    labels = torch.empty((batch, n_tokens), device=rows.device, dtype=torch.int32)
    _joint_euclid_assign_kernel[(triton.cdiv(n_tokens, block_n), batch)](
        rows, centers, labels,
        n_tokens=n_tokens, n_clusters=n_clusters, width=width,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_D=block_d,
        num_warps=4,
    )
    sums = torch.zeros((batch, n_clusters, width), device=rows.device, dtype=torch.float32)
    counts = torch.zeros((batch, n_clusters), device=rows.device, dtype=torch.int32)
    _joint_centroid_sum_kernel[(n_tokens, batch, triton.cdiv(width, 128))](
        rows, labels, sums,
        n_tokens=n_tokens, n_clusters=n_clusters, width=width,
        BLOCK_D=128, num_warps=4,
    )
    _joint_centroid_count_kernel[(n_tokens, batch)](
        labels, counts, n_tokens=n_tokens, n_clusters=n_clusters, num_warps=1
    )
    updated = sums / counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
    updated = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    return updated, labels
