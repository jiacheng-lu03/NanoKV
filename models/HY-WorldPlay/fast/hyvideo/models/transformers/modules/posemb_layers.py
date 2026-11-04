# Modified for NanoKV: KV-cache codec integration or evaluation configuration.
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
import atexit
import torch
from typing import Union, Tuple, List

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

_FUSED_ROPE_ONE_REAL_HITS = 0
_FUSED_ROPE_ONE_COMPLEX_HITS = 0
_FUSED_ROPE_ONE_REAL_MISS_PRINTED = False


def _print_fused_rope_one_stats():
    if os.environ.get("HY_FUSED_ROPE_ONE_DEBUG", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    print(
        "[Fused RoPE One] "
        f"real_hits={_FUSED_ROPE_ONE_REAL_HITS} "
        f"complex_hits={_FUSED_ROPE_ONE_COMPLEX_HITS}"
    )


atexit.register(_print_fused_rope_one_stats)


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if (
        os.environ.get("HY_LOSSLESS_FUSED_ROTARY_PAIR", "0").lower()
        in {"1", "true", "yes", "on"}
        and isinstance(freqs_cis, tuple)
        and triton is not None and not head_first
        and xq.is_cuda and xk.is_cuda and xq.ndim == xk.ndim == 4
        and xq.shape[1:] == xk.shape[1:] and xq.shape[-1] % 2 == 0
        and freqs_cis[0].dtype == freqs_cis[1].dtype == torch.float32
        and freqs_cis[0].shape == freqs_cis[1].shape == (xq.shape[1], xq.shape[-1])
    ):
        # Reuse the strict staged kernel: two independently rounded FP32
        # products, then FP32 addition and BF16 cast; no reassociation/FMA.
        cos, sin = (part.to(xq.device) for part in freqs_cis)
        return (
            _apply_rotary_emb_one_triton_real_exact(xq, cos, sin),
            _apply_rotary_emb_one_triton_real_exact(xk, cos, sin),
        )
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def apply_rotary_emb_one(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> torch.Tensor:
    """Apply rotary embedding to one tensor.

    This is equivalent to the single-output half of apply_rotary_emb(), and is
    used when only cached K needs to be restored.
    """
    if isinstance(freqs_cis, tuple):
        if _use_exact_rotary_emb_one_real(x, freqs_cis, head_first):
            return _apply_rotary_emb_one_triton_real_exact(
                x, freqs_cis[0], freqs_cis[1]
            )
        if _use_fused_rotary_emb_one_real(x, freqs_cis, head_first):
            return _apply_rotary_emb_one_triton_real(x, freqs_cis[0], freqs_cis[1])
        cos, sin = reshape_for_broadcast(freqs_cis, x, head_first)
        cos, sin = cos.to(x.device), sin.to(x.device)
        if os.environ.get("HY_PRE_POSITION_ROPE_WORK_REUSE", "1").lower() in {
            "1", "true", "yes", "on"
        }:
            work = x.float()
            return (work * cos + rotate_half(work) * sin).type_as(x)
        return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)

    if _use_fused_rotary_emb_one(x, freqs_cis, head_first):
        return _apply_rotary_emb_one_triton(x, freqs_cis)

    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_, head_first).to(x.device)
    return torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)


def _use_fused_rotary_emb_one(x: torch.Tensor, freqs_cis: torch.Tensor, head_first: bool) -> bool:
    if os.environ.get("HY_FUSED_ROPE_ONE", "0").lower() not in {"1", "true", "yes", "on"}:
        return False
    if triton is None:
        return False
    if head_first:
        return False
    if not x.is_cuda or x.ndim != 4 or x.shape[-1] % 2 != 0:
        return False
    if not torch.is_complex(freqs_cis):
        return False
    return (
        freqs_cis.is_cuda
        and freqs_cis.ndim == 2
        and freqs_cis.shape[0] == x.shape[1]
        and freqs_cis.shape[1] == x.shape[-1] // 2
    )


def _use_fused_rotary_emb_one_real(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    head_first: bool,
) -> bool:
    global _FUSED_ROPE_ONE_REAL_MISS_PRINTED
    if os.environ.get("HY_FUSED_ROPE_ONE", "0").lower() not in {"1", "true", "yes", "on"}:
        return False
    debug = os.environ.get("HY_FUSED_ROPE_ONE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    miss_reason = None
    if triton is None or _rotary_one_real_kernel is None:
        miss_reason = "triton_unavailable"
    elif head_first:
        miss_reason = "head_first"
    elif not x.is_cuda or x.ndim != 4 or x.shape[-1] % 2 != 0:
        miss_reason = f"x_invalid device={x.device} ndim={x.ndim} shape={tuple(x.shape)}"
    else:
        cos, sin = freqs_cis
        if not (
            cos.is_cuda
            and sin.is_cuda
            and cos.ndim == 2
            and sin.ndim == 2
            and cos.shape == sin.shape
            and cos.shape[0] == x.shape[1]
            and cos.shape[1] == x.shape[-1]
        ):
            miss_reason = (
                f"freq_invalid cos_device={cos.device} sin_device={sin.device} "
                f"cos_shape={tuple(cos.shape)} sin_shape={tuple(sin.shape)} "
                f"x_shape={tuple(x.shape)}"
            )
    if miss_reason is not None:
        if debug and not _FUSED_ROPE_ONE_REAL_MISS_PRINTED:
            print(f"[Fused RoPE One] real_miss={miss_reason}")
            _FUSED_ROPE_ONE_REAL_MISS_PRINTED = True
        return False
    cos, sin = freqs_cis
    return True


def _use_exact_rotary_emb_one_real(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    head_first: bool,
) -> bool:
    if os.environ.get("HY_FUSED_ROPE_ONE_EXACT", "0").lower() not in {
        "1", "true", "yes", "on"
    }:
        return False
    if triton is None or _rotary_one_real_mul_kernel is None:
        return False
    if head_first or not x.is_cuda or x.ndim != 4 or x.shape[-1] % 2:
        return False
    cos, sin = freqs_cis
    return (
        cos.is_cuda
        and sin.is_cuda
        and cos.ndim == 2
        and sin.ndim == 2
        and cos.shape == sin.shape == (x.shape[1], x.shape[-1])
    )


def _apply_rotary_emb_one_triton_real_exact(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Bitwise-preserving staged real RoPE.

    The tempting one-pass kernel contracts multiply/add operations and changes
    long-rollout video bits.  This path deliberately materializes the two FP32
    products, then performs the FP32 add and output cast in a third kernel,
    matching eager PyTorch's rounding boundaries while avoiding its separate
    rotate/stack and duplicate BF16->FP32 materializations.
    """
    if os.environ.get("HY_LOSSLESS_ROTARY_SINGLEPASS", "0").lower() in {
        "1", "true", "yes", "on"
    }:
        return _apply_rotary_emb_one_triton_real(x, cos, sin, disable_fma=True)
    bsz, seq_len, num_heads, head_dim = x.shape
    total = x.numel()
    term_a = torch.empty(total, device=x.device, dtype=torch.float32)
    term_b = torch.empty_like(term_a)
    # ``x`` is normally a non-contiguous [B,S,H,D] transpose view.  The
    # kernel writes logical BSHD-linear offsets, so its destination must be an
    # explicitly contiguous tensor rather than ``empty_like(x)`` (which may
    # preserve the transposed strides).
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block = 256
    grid = (triton.cdiv(total, block),)
    common = (
        total, seq_len, num_heads, head_dim,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
    )
    _rotary_one_real_mul_kernel[grid](
        x, cos, sin, term_a, term_b, *common, BLOCK=block
    )
    _rotary_one_real_add_cast_kernel[grid](
        term_a, term_b, out, total, BLOCK=block
    )
    return out


def _apply_rotary_emb_one_triton_real(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    disable_fma: bool = False,
) -> torch.Tensor:
    global _FUSED_ROPE_ONE_REAL_HITS
    _FUSED_ROPE_ONE_REAL_HITS += 1
    bsz, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    out = torch.empty((bsz, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    total_pairs = bsz * seq_len * num_heads * half_dim
    block = 256
    _rotary_one_real_kernel[(triton.cdiv(total_pairs, block),)](
        x,
        cos,
        sin,
        out,
        total_pairs,
        seq_len,
        num_heads,
        head_dim,
        half_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        BLOCK=block, enable_fp_fusion=not disable_fma,
    )
    return out


def _apply_rotary_emb_one_triton(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    global _FUSED_ROPE_ONE_COMPLEX_HITS
    _FUSED_ROPE_ONE_COMPLEX_HITS += 1
    bsz, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    out = torch.empty((bsz, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    total_pairs = bsz * seq_len * num_heads * half_dim
    block = 256
    freq_real = freqs_cis.real
    freq_imag = freqs_cis.imag
    _rotary_one_kernel[(triton.cdiv(total_pairs, block),)](
        x,
        freq_real,
        freq_imag,
        out,
        total_pairs,
        seq_len,
        num_heads,
        head_dim,
        half_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        freq_real.stride(0),
        freq_real.stride(1),
        BLOCK=block,
    )
    return out


if triton is not None:
    @triton.jit
    def _rotary_one_kernel(
        x,
        freq_real,
        freq_imag,
        out,
        total_pairs: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        freq_stride_s: tl.constexpr,
        freq_stride_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_pairs

        d2 = offsets % half_dim
        tmp = offsets // half_dim
        h = tmp % num_heads
        tmp = tmp // num_heads
        s = tmp % seq_len
        b = tmp // seq_len

        in_base = b * x_stride_b + s * x_stride_s + h * x_stride_h + (2 * d2) * x_stride_d
        x0 = tl.load(x + in_base, mask=mask).to(tl.float32)
        x1 = tl.load(x + in_base + x_stride_d, mask=mask).to(tl.float32)

        freq_base = s * freq_stride_s + d2 * freq_stride_d
        c = tl.load(freq_real + freq_base, mask=mask).to(tl.float32)
        sn = tl.load(freq_imag + freq_base, mask=mask).to(tl.float32)

        y0 = x0 * c - x1 * sn
        y1 = x1 * c + x0 * sn

        out_base = ((b * seq_len + s) * num_heads + h) * head_dim + 2 * d2
        tl.store(out + out_base, y0, mask=mask)
        tl.store(out + out_base + 1, y1, mask=mask)

else:
    _rotary_one_kernel = None


if triton is not None:
    @triton.jit
    def _rotary_one_real_kernel(
        x,
        cos,
        sin,
        out,
        total_pairs: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        cos_stride_s: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_s: tl.constexpr,
        sin_stride_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_pairs

        d2 = offsets % half_dim
        tmp = offsets // half_dim
        h = tmp % num_heads
        tmp = tmp // num_heads
        s = tmp % seq_len
        b = tmp // seq_len

        d0 = 2 * d2
        d1 = d0 + 1
        in_base = b * x_stride_b + s * x_stride_s + h * x_stride_h + d0 * x_stride_d
        x0 = tl.load(x + in_base, mask=mask).to(tl.float32)
        x1 = tl.load(x + in_base + x_stride_d, mask=mask).to(tl.float32)

        cos0 = tl.load(cos + s * cos_stride_s + d0 * cos_stride_d, mask=mask).to(tl.float32)
        cos1 = tl.load(cos + s * cos_stride_s + d1 * cos_stride_d, mask=mask).to(tl.float32)
        sin0 = tl.load(sin + s * sin_stride_s + d0 * sin_stride_d, mask=mask).to(tl.float32)
        sin1 = tl.load(sin + s * sin_stride_s + d1 * sin_stride_d, mask=mask).to(tl.float32)

        y0 = x0 * cos0 - x1 * sin0
        y1 = x1 * cos1 + x0 * sin1

        out_base = ((b * seq_len + s) * num_heads + h) * head_dim + d0
        tl.store(out + out_base, y0, mask=mask)
        tl.store(out + out_base + 1, y1, mask=mask)
else:
    _rotary_one_real_kernel = None


if triton is not None:
    @triton.jit
    def _rotary_one_real_mul_kernel(
        x, cos, sin, term_a, term_b,
        total: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        cos_stride_s: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_s: tl.constexpr,
        sin_stride_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        d = offsets % head_dim
        tmp = offsets // head_dim
        h = tmp % num_heads
        tmp = tmp // num_heads
        s = tmp % seq_len
        b = tmp // seq_len
        x_offset = (
            b * x_stride_b + s * x_stride_s + h * x_stride_h
            + d * x_stride_d
        )
        partner_d = tl.where((d & 1) == 0, d + 1, d - 1)
        partner_offset = (
            b * x_stride_b + s * x_stride_s + h * x_stride_h
            + partner_d * x_stride_d
        )
        value = tl.load(x + x_offset, mask=mask).to(tl.float32)
        partner = tl.load(x + partner_offset, mask=mask).to(tl.float32)
        rotated = tl.where((d & 1) == 0, -partner, partner)
        c = tl.load(
            cos + s * cos_stride_s + d * cos_stride_d, mask=mask
        ).to(tl.float32)
        sn = tl.load(
            sin + s * sin_stride_s + d * sin_stride_d, mask=mask
        ).to(tl.float32)
        tl.store(term_a + offsets, value * c, mask=mask)
        tl.store(term_b + offsets, rotated * sn, mask=mask)

    @triton.jit
    def _rotary_one_real_add_cast_kernel(
        term_a, term_b, out, total: tl.constexpr, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        a = tl.load(term_a + offsets, mask=mask).to(tl.float32)
        b = tl.load(term_b + offsets, mask=mask).to(tl.float32)
        tl.store(out + offsets, a + b, mask=mask)
else:
    _rotary_one_real_mul_kernel = None
    _rotary_one_real_add_cast_kernel = None


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis
