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


@triton.jit
def _unpack_int8_weight_kernel(Q, S, Y, COUNT:tl.constexpr,
                              GROUP:tl.constexpr, BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=ix<COUNT
    # All signed INT8 values are exactly representable in BF16. The reference
    # materializes that cast, then multiplies by its BF16 group scale.
    q=tl.load(Q+ix,valid,0).to(tl.bfloat16).to(tl.float32)
    scale=tl.load(S+ix//GROUP,valid,0).to(tl.float32)
    tl.store(Y+ix,q*scale,valid)


def unpack_int8_weight(qweight, scales, group_size):
    if not (qweight.is_contiguous() and scales.is_contiguous()):
        raise ValueError('INT8 weight unpack requires contiguous tensors')
    out=torch.empty_like(qweight,dtype=torch.bfloat16)
    _unpack_int8_weight_kernel[(triton.cdiv(out.numel(),512),)](
        qweight,scales,out,out.numel(),group_size,512)
    return out
