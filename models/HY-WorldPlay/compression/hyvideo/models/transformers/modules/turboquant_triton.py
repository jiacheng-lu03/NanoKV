"""Lossless hot-path kernels for the TurboQuant reference adapter.

These kernels only fuse scalar codebook lookup and exact bit packing.  The
random rotation, QJL projection, position encoding, and attention operators
remain unchanged.  Consequently this module changes neither the quantizer nor
the cache format; it is an implementation optimization of Algorithms 1/2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_ROWS = 8


@triton.jit
def _quantize_pack_lookup_kernel(
    values_ptr,
    centroids_ptr,
    boundaries_ptr,
    packed_ptr,
    reconstructed_ptr,
    rows,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BITS: tl.constexpr,
    N_BOUNDARIES: tl.constexpr,
    STORE_RECONSTRUCTED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = row < rows
    lane = tl.arange(0, 8)
    shifts = lane * BITS
    packed_width: tl.constexpr = GROUPS * BITS

    for group in range(GROUPS):
        offsets = row[:, None] * D + group * 8 + lane[None, :]
        values = tl.load(
            values_ptr + offsets,
            mask=valid_row[:, None],
            other=0.0,
        ).to(tl.float32)
        indices = tl.zeros((BLOCK_ROWS, 8), dtype=tl.int32)
        for boundary_index in range(N_BOUNDARIES):
            boundary = tl.load(boundaries_ptr + boundary_index).to(tl.float32)
            indices += values > boundary

        if STORE_RECONSTRUCTED:
            reconstructed = tl.load(centroids_ptr + indices).to(tl.float32)
            tl.store(
                reconstructed_ptr + offsets,
                reconstructed,
                mask=valid_row[:, None],
            )

        words = tl.sum(indices << shifts[None, :], axis=1)
        for byte_index in range(BITS):
            output_offset = row * packed_width + group * BITS + byte_index
            byte = (words >> (8 * byte_index)) & 0xFF
            tl.store(
                packed_ptr + output_offset,
                byte.to(tl.uint8),
                mask=valid_row,
            )


@triton.jit
def _unpack_lookup_kernel(
    packed_ptr,
    centroids_ptr,
    output_ptr,
    rows,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = row < rows
    lane = tl.arange(0, 8)
    shifts = lane * BITS
    packed_width: tl.constexpr = GROUPS * BITS
    value_mask: tl.constexpr = (1 << BITS) - 1

    for group in range(GROUPS):
        words = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
        for byte_index in range(BITS):
            input_offset = row * packed_width + group * BITS + byte_index
            byte = tl.load(
                packed_ptr + input_offset,
                mask=valid_row,
                other=0,
            ).to(tl.int32)
            words |= byte << (8 * byte_index)
        indices = (words[:, None] >> shifts[None, :]) & value_mask
        output_offset = row[:, None] * D + group * 8 + lane[None, :]
        values = tl.load(centroids_ptr + indices).to(tl.float32)
        tl.store(
            output_ptr + output_offset,
            values,
            mask=valid_row[:, None],
        )


@triton.jit
def _pack_signs_kernel(
    values_ptr,
    packed_ptr,
    rows,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = row < rows
    lane = tl.arange(0, 8)

    for group in range(GROUPS):
        offsets = row[:, None] * D + group * 8 + lane[None, :]
        values = tl.load(
            values_ptr + offsets,
            mask=valid_row[:, None],
            other=0.0,
        ).to(tl.float32)
        bits = (values >= 0.0).to(tl.int32)
        byte = tl.sum(bits << lane[None, :], axis=1)
        tl.store(
            packed_ptr + row * GROUPS + group,
            byte.to(tl.uint8),
            mask=valid_row,
        )


@triton.jit
def _unpack_signs_kernel(
    packed_ptr,
    output_ptr,
    rows,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = row < rows
    lane = tl.arange(0, 8)

    for group in range(GROUPS):
        byte = tl.load(
            packed_ptr + row * GROUPS + group,
            mask=valid_row,
            other=0,
        ).to(tl.int32)
        bits = (byte[:, None] >> lane[None, :]) & 1
        values = bits.to(tl.float32) * 2.0 - 1.0
        offsets = row[:, None] * D + group * 8 + lane[None, :]
        tl.store(output_ptr + offsets, values, mask=valid_row[:, None])


def _validate(tensor: torch.Tensor) -> tuple[int, int, int]:
    if not tensor.is_cuda:
        raise ValueError("TurboQuant Triton kernels require CUDA tensors")
    if not tensor.is_contiguous():
        raise ValueError("TurboQuant Triton kernels require contiguous tensors")
    dim = int(tensor.shape[-1])
    if dim <= 0 or dim % 8:
        raise ValueError(f"TurboQuant dimension must be divisible by 8, got {dim}")
    rows = int(tensor.numel() // dim)
    return rows, dim, dim // 8


def quantize_pack_lookup(
    values: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    bits: int,
    *,
    return_reconstructed: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Nearest-codebook quantize and exact-pack contiguous vectors."""
    source = values.contiguous()
    rows, dim, groups = _validate(source)
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"unsupported TurboQuant bit width {bits}")
    if int(centroids.numel()) != 1 << bits:
        raise ValueError("centroid count does not match bit width")
    if int(boundaries.numel()) != (1 << bits) - 1:
        raise ValueError("boundary count does not match bit width")
    packed = torch.empty(
        (*source.shape[:-1], groups * bits),
        device=source.device,
        dtype=torch.uint8,
    )
    reconstructed = (
        torch.empty_like(source, dtype=torch.float32)
        if return_reconstructed
        else None
    )
    dummy = reconstructed if reconstructed is not None else packed
    grid = (triton.cdiv(rows, _BLOCK_ROWS),)
    _quantize_pack_lookup_kernel[grid](
        source,
        centroids,
        boundaries,
        packed,
        dummy,
        rows,
        D=dim,
        GROUPS=groups,
        BITS=bits,
        N_BOUNDARIES=(1 << bits) - 1,
        STORE_RECONSTRUCTED=return_reconstructed,
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=4,
    )
    if reconstructed is not None:
        reconstructed = reconstructed.reshape_as(source)
    return packed, reconstructed


def unpack_lookup(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Exact-unpack indices and map them to FP32 codebook centroids."""
    if not packed.is_cuda or not packed.is_contiguous():
        packed = packed.contiguous()
    dim = int(shape[-1])
    groups = dim // 8
    rows = math_prod(shape[:-1])
    expected = rows * groups * bits
    if int(packed.numel()) != expected:
        raise ValueError(f"packed payload has {packed.numel()} bytes, expected {expected}")
    output = torch.empty(shape, device=packed.device, dtype=torch.float32)
    grid = (triton.cdiv(rows, _BLOCK_ROWS),)
    _unpack_lookup_kernel[grid](
        packed,
        centroids,
        output,
        rows,
        D=dim,
        GROUPS=groups,
        BITS=bits,
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=4,
    )
    return output


def pack_signs(values: torch.Tensor) -> torch.Tensor:
    """Pack ``values >= 0`` using exactly one bit per coordinate."""
    source = values.contiguous()
    rows, dim, groups = _validate(source)
    packed = torch.empty(
        (*source.shape[:-1], groups), device=source.device, dtype=torch.uint8
    )
    grid = (triton.cdiv(rows, _BLOCK_ROWS),)
    _pack_signs_kernel[grid](
        source,
        packed,
        rows,
        D=dim,
        GROUPS=groups,
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=4,
    )
    return packed


def unpack_signs(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Unpack one-bit signs directly to FP32 ``{-1,+1}`` values."""
    source = packed.contiguous()
    dim = int(shape[-1])
    groups = dim // 8
    rows = math_prod(shape[:-1])
    if int(source.numel()) != rows * groups:
        raise ValueError("packed sign payload size does not match requested shape")
    output = torch.empty(shape, device=source.device, dtype=torch.float32)
    grid = (triton.cdiv(rows, _BLOCK_ROWS),)
    _unpack_signs_kernel[grid](
        source,
        output,
        rows,
        D=dim,
        GROUPS=groups,
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=4,
    )
    return output


def math_prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result
