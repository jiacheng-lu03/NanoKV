"""Fused CUDA primitives for the MG3.5 WorldPlay-parity codec."""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_int4_scaled_kernel(
    packed,
    scales,
    output,
    count,
    dim: tl.constexpr,
    block: tl.constexpr,
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
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Unpack symmetric INT4 and apply one FP32 scale per latent row.

    The result is BF16 because the following decoder GEMM consumes BF16.  This
    replaces the old uint8->int16->int8->float32 chain and its temporary codes
    tensor without changing the represented values.
    """
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
        packed,
        scales,
        output,
        count,
        dim=dim,
        block=block,
        num_warps=4,
    )
    return output
