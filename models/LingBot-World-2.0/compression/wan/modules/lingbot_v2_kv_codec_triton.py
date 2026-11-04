"""Fused CUDA primitives for the LingBot 2.0 SAGE-KV codec."""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _int4_scale_per_row_kernel(
    source, scale_fp32, scale_storage, rows: tl.constexpr,
    width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, block)
    valid = column < width
    value = tl.load(source + row * width + column, mask=valid, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), axis=0), 1.0e-12) / 7.0
    tl.store(scale_fp32 + row, scale)
    tl.store(scale_storage + row, scale)


@triton.jit
def _int4_pack_given_scale_kernel(
    source, scale_fp32, packed, rows: tl.constexpr,
    width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.arange(0, block)
    valid = pair < width // 2
    scale = tl.load(scale_fp32 + row).to(tl.float32)
    even = tl.load(source + row * width + pair * 2, mask=valid, other=0.0).to(tl.float32)
    odd = tl.load(source + row * width + pair * 2 + 1, mask=valid, other=0.0).to(tl.float32)
    even = libdevice.rint(libdevice.div_rn(even, scale))
    odd = libdevice.rint(libdevice.div_rn(odd, scale))
    even = tl.maximum(tl.minimum(even, 7.0), -7.0).to(tl.int32) + 7
    odd = tl.maximum(tl.minimum(odd, 7.0), -7.0).to(tl.int32) + 7
    tl.store(packed + row * (width // 2) + pair, (even | (odd << 4)).to(tl.uint8), mask=valid)


def quantize_pack_int4_per_row(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize arbitrary even-width rows and pack low nibble first."""
    if not value.is_cuda:
        raise ValueError("fused INT4 quantization requires CUDA")
    width = int(value.shape[-1])
    if width <= 0 or width % 2:
        raise ValueError(f"INT4 width must be positive and even, got {width}")
    rows = value.numel() // width
    source = value.reshape(rows, width).contiguous()
    packed = torch.empty((rows, width // 2), device=value.device, dtype=torch.uint8)
    scale_fp32 = torch.empty(rows, device=value.device, dtype=torch.float32)
    scale = torch.empty((rows, 1), device=value.device, dtype=torch.bfloat16)
    block = triton.next_power_of_2(width)
    _int4_scale_per_row_kernel[(rows,)](
        source, scale_fp32, scale, rows=rows, width=width, block=block, num_warps=4
    )
    pair_block = triton.next_power_of_2(width // 2)
    _int4_pack_given_scale_kernel[(rows,)](
        source, scale_fp32, packed, rows=rows, width=width,
        block=pair_block, num_warps=4,
    )
    return packed.reshape(-1), scale.reshape(*value.shape[:-1], 1), value.numel()


@triton.jit
def _unpack_int4_scaled_kernel(
    packed, scales, output, count, dim: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < count
    byte = tl.load(packed + index // 2, mask=valid, other=0).to(tl.int32)
    shift = (index & 1) * 4
    code = ((byte >> shift) & 0x0F) - 7
    row = index // dim
    scale = tl.load(scales + row, mask=valid, other=0.0).to(tl.float32)
    tl.store(output + index, code.to(tl.float32) * scale, mask=valid)


def unpack_int4_scaled(
    packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """Unpack low-nibble-first symmetric INT4 directly to scaled BF16."""
    packed = packed.contiguous()
    scales = scales.contiguous().reshape(-1)
    count = math.prod(int(value) for value in shape)
    dim = int(shape[-1])
    rows = count // dim
    if not packed.is_cuda or not scales.is_cuda:
        raise ValueError("fused INT4 unpack requires CUDA tensors")
    if packed.numel() * 2 < count or scales.numel() != rows:
        raise ValueError("invalid packed INT4 payload")
    output = torch.empty(shape, device=packed.device, dtype=torch.bfloat16)
    block = 256
    _unpack_int4_scaled_kernel[(triton.cdiv(count, block),)](
        packed, scales, output, count, dim=dim, block=block, num_warps=4
    )
    return output


@triton.jit
def _add_centers_inplace_kernel(
    rows, centers, labels, sequence: tl.constexpr,
    clusters: tl.constexpr, width: tl.constexpr, block: tl.constexpr,
):
    token = tl.program_id(0)
    batch = tl.program_id(1)
    feature = tl.program_id(2) * block + tl.arange(0, block)
    valid = feature < width
    label = tl.load(labels + batch * sequence + token).to(tl.int32)
    row_offset = (batch * sequence + token) * width + feature
    value = tl.load(rows + row_offset, mask=valid, other=0.0).to(tl.float32)
    center = tl.load(
        centers + (batch * clusters + label) * width + feature,
        mask=valid, other=0.0,
    ).to(tl.float32)
    tl.store(rows + row_offset, value + center, mask=valid)


def add_centers_inplace(
    rows: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor
) -> None:
    """Fuse C64 center gather and add without a full gathered temporary."""
    batch, sequence, width = map(int, rows.shape)
    clusters = int(centers.shape[1])
    block = 256
    _add_centers_inplace_kernel[(sequence, batch, triton.cdiv(width, block))](
        rows, centers, labels, sequence=sequence, clusters=clusters,
        width=width, block=block, num_warps=4,
    )
