"""Bit-exact TurboQuant unpack kernels for MG3.5."""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _lookup_kernel(packed, centers, output, rows, D: tl.constexpr, GROUPS: tl.constexpr,
                   BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    lane = tl.arange(0, 8)
    for group in range(GROUPS):
        word = tl.zeros((BLOCK,), tl.int32)
        for byte in range(BITS):
            x = tl.load(packed + row * (GROUPS * BITS) + group * BITS + byte,
                        mask=valid, other=0).to(tl.int32)
            word |= x << (8 * byte)
        idx = (word[:, None] >> (lane[None, :] * BITS)) & ((1 << BITS) - 1)
        offsets = row[:, None] * D + group * 8 + lane[None, :]
        tl.store(output + offsets, tl.load(centers + idx).to(tl.float32), mask=valid[:, None])

@triton.jit
def _sign_kernel(packed, output, rows, D: tl.constexpr, GROUPS: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    lane = tl.arange(0, 8)
    for group in range(GROUPS):
        byte = tl.load(packed + row * GROUPS + group, mask=valid, other=0).to(tl.int32)
        values = (((byte[:, None] >> lane[None, :]) & 1).to(tl.float32) * 2.0) - 1.0
        offsets = row[:, None] * D + group * 8 + lane[None, :]
        tl.store(output + offsets, values, mask=valid[:, None])

def unpack_lookup(packed, centers, bits, shape):
    packed = packed.contiguous(); dim = int(shape[-1]); groups = dim // 8
    rows = math.prod(int(v) for v in shape[:-1])
    if not packed.is_cuda or dim % 8 or packed.numel() != rows * groups * int(bits):
        raise ValueError("invalid TurboQuant packed lookup input")
    output = torch.empty(shape, device=packed.device, dtype=torch.float32)
    _lookup_kernel[(triton.cdiv(rows, 8),)](
        packed, centers, output, rows, D=dim, GROUPS=groups, BITS=int(bits), BLOCK=8,
        num_warps=4,
    )
    return output

def unpack_signs(packed, shape):
    packed = packed.contiguous(); dim = int(shape[-1]); groups = dim // 8
    rows = math.prod(int(v) for v in shape[:-1])
    if not packed.is_cuda or dim % 8 or packed.numel() != rows * groups:
        raise ValueError("invalid TurboQuant packed sign input")
    output = torch.empty(shape, device=packed.device, dtype=torch.float32)
    _sign_kernel[(triton.cdiv(rows, 8),)](
        packed, output, rows, D=dim, GROUPS=groups, BLOCK=8, num_warps=4,
    )
    return output
