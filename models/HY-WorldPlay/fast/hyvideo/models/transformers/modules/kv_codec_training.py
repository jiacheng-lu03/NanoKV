import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_V_FINAL_PRED_TENSOR_AUDIT: Dict[int, Dict[str, torch.Tensor]] = {}
_V_BIN_REFERENCE_MODELS: Dict[int, nn.Module] = {}
_K_PCA_TAIL_SCORE_TENSORS: Dict[int, Dict[str, torch.Tensor]] = {}
_FUNCTIONAL_PRODUCTION_BASELINES: Dict[Tuple[str, int], torch.Tensor] = {}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _apply_functional_decoder_delta_to_production_base(
    out: torch.Tensor,
    latent: torch.Tensor,
    model: nn.Module,
    target: str,
    layer_idx: int,
) -> torch.Tensor:
    """Apply only the decoder-weight change on top of the deployed cache value."""
    baseline = _FUNCTIONAL_PRODUCTION_BASELINES.get((target, int(layer_idx)))
    if baseline is None:
        return out
    if not isinstance(model, GlobalPCALinearFunctionalCodec):
        raise RuntimeError(
            "Production-base joint FT currently requires global_pca_linear_d512"
        )
    if not hasattr(model, "_hy_joint_decoder_weight0"):
        model._hy_joint_decoder_weight0 = (
            model.decoder_delta.weight.detach().float().cpu().clone()
        )
    old_weight = model._hy_joint_decoder_weight0.to(
        device=model.decoder_delta.weight.device,
        dtype=model.decoder_delta.weight.dtype,
    )
    latent_rows = latent.transpose(1, 2).reshape(-1, model.latent_dim).to(
        dtype=model.decoder_delta.weight.dtype
    )
    correction_rows = F.linear(
        latent_rows, model.decoder_delta.weight - old_weight
    )
    batch, heads, sequence, dim = out.shape
    correction = (
        correction_rows.reshape(batch, sequence, heads, dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return (baseline.float() + correction.float()).to(dtype=out.dtype)


def _timing_profile_enabled() -> bool:
    return _env_flag("HY_KV_AE3D_PROFILE", "0")


def _parse_layers(value: str):
    value = value.strip().lower()
    if value == "all":
        return "all"
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _layer_enabled(layer_idx: int, layers) -> bool:
    return layers == "all" or int(layer_idx) in layers


def _tensor_bytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(_tensor_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(v) for v in x)
    return 0


class PreKV3DAE(nn.Module):
    """A small shared 3D AE for pre-position KV cache.

    Input layout is [B, H, S, D], where S = T * h * w. The model reshapes it to
    [B*H, D, T, h, w], downsamples T/H/W by 2 and channel D by 2. For even
    T/H/W this is a 16x element compression.
    """

    def __init__(self, dim: int = 128, hidden: int = 128, latent_dim: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(self.dim, self.hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                self.hidden,
                self.latent_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_dim,
                self.hidden,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv3d(self.hidden, self.dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, h, s, d = x.shape
        t, hh, ww = [int(v) for v in thw]
        if s != t * hh * ww or d != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        x5 = x.reshape(b, h, t, hh, ww, d).permute(0, 1, 5, 2, 3, 4)
        x5 = x5.reshape(b * h, d, t, hh, ww).contiguous()
        return self.encoder(x5)

    def decode(
        self,
        z: torch.Tensor,
        shape: Tuple[int, int, int, int],
        thw: Tuple[int, int, int],
    ) -> torch.Tensor:
        b, h, s, d = [int(v) for v in shape]
        t, hh, ww = [int(v) for v in thw]
        x5 = self.decoder(z)
        x5 = x5[:, :, :t, :hh, :ww].contiguous()
        x = x5.reshape(b, h, d, t, hh, ww).permute(0, 1, 3, 4, 5, 2)
        return x.reshape(b, h, s, d).contiguous()


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(torch.nn.functional.silu(self.conv1(x)))


def _validate_dcae_stride(stride: Tuple[int, int, int]) -> Tuple[int, int, int]:
    stride = tuple(int(value) for value in stride)
    if len(stride) != 3 or any(value not in {1, 2, 3} for value in stride):
        raise ValueError(f"DCAE stride must contain only 1, 2, or 3, got {stride}")
    return stride


def space_time_to_channel(
    x: torch.Tensor, stride: Tuple[int, int, int]
) -> torch.Tensor:
    """Rearrange local stride neighborhoods into channels."""
    stride_t, stride_h, stride_w = _validate_dcae_stride(stride)
    n, c, t, h, w = x.shape
    if t % stride_t or h % stride_h or w % stride_w:
        raise ValueError(
            f"T/H/W={(t, h, w)} must be divisible by stride={stride}"
        )
    return (
        x.reshape(
            n,
            c,
            t // stride_t,
            stride_t,
            h // stride_h,
            stride_h,
            w // stride_w,
            stride_w,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(
            n,
            c * stride_t * stride_h * stride_w,
            t // stride_t,
            h // stride_h,
            w // stride_w,
        )
        .contiguous()
    )


def channel_to_space_time(
    x: torch.Tensor,
    output_channels: int,
    stride: Tuple[int, int, int],
) -> torch.Tensor:
    """Inverse channel-to-space-time rearrangement for a given stride."""
    stride_t, stride_h, stride_w = _validate_dcae_stride(stride)
    n, channels, t, h, w = x.shape
    expected = int(output_channels) * stride_t * stride_h * stride_w
    if channels != expected:
        raise ValueError(f"expected {expected} channels, got {channels}")
    return (
        x.reshape(
            n,
            output_channels,
            stride_t,
            stride_h,
            stride_w,
            t,
            h,
            w,
        )
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(
            n,
            output_channels,
            t * stride_t,
            h * stride_h,
            w * stride_w,
        )
        .contiguous()
    )


def space_time_to_channel_2x(x: torch.Tensor) -> torch.Tensor:
    return space_time_to_channel(x, (2, 2, 2))


def channel_to_space_time_2x(x: torch.Tensor, output_channels: int) -> torch.Tensor:
    return channel_to_space_time(x, output_channels, (2, 2, 2))


def make_hadamard8(device=None, dtype=None) -> torch.Tensor:
    """Return an orthonormal 8x8 Sylvester Hadamard matrix."""
    dtype = dtype or torch.float32
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    h4 = torch.kron(h2, h2)
    h8 = torch.kron(h2, h4)
    return h8 / math.sqrt(8.0)


def _pad_2x2x2(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
    t, h, w = [int(v) for v in x.shape[-3:]]
    pad_t = (-t) % 2
    pad_h = (-h) % 2
    pad_w = (-w) % 2
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
    return x, (pad_t, pad_h, pad_w), (t, h, w)


def block_unshuffle_2x2x2(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Convert [B,C,T,H,W] to [B,C,8,T/2,H/2,W/2] with fixed local order."""
    if x.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
    padded, pad, original_thw = _pad_2x2x2(x)
    n, c, t, h, w = padded.shape
    blocks = padded.reshape(n, c, t // 2, 2, h // 2, 2, w // 2, 2)
    blocks = blocks.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(
        n, c, 8, t // 2, h // 2, w // 2
    )
    return blocks.contiguous(), {
        "pad": pad,
        "original_thw": original_thw,
        "padded_thw": (t, h, w),
    }


def block_shuffle_2x2x2(block: torch.Tensor, pad_info: Dict[str, Any]) -> torch.Tensor:
    """Inverse of block_unshuffle_2x2x2."""
    if block.ndim != 6 or block.shape[2] != 8:
        raise ValueError(f"expected [B,C,8,T2,H2,W2], got {tuple(block.shape)}")
    n, c, _, t2, h2, w2 = block.shape
    x = (
        block.reshape(n, c, 2, 2, 2, t2, h2, w2)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(n, c, t2 * 2, h2 * 2, w2 * 2)
    )
    t, h, w = [int(v) for v in pad_info["original_thw"]]
    return x[:, :, :t, :h, :w].contiguous()


def hadamard_decompose_2x2x2(x: torch.Tensor):
    """Return block DC mean and flattened AC coefficients for [B,C,T,H,W]."""
    blocks, pad_info = block_unshuffle_2x2x2(x)
    work = blocks.float()
    h8 = make_hadamard8(work.device, work.dtype)
    coeff = torch.einsum("ab,ncbthw->ncathw", h8, work)
    dc_mean = coeff[:, :, 0] / math.sqrt(8.0)
    ac_coeff = coeff[:, :, 1:].reshape(
        coeff.shape[0], coeff.shape[1] * 7, coeff.shape[3], coeff.shape[4], coeff.shape[5]
    )
    return dc_mean.to(x.dtype), ac_coeff.to(x.dtype), pad_info


def hadamard_recompose_2x2x2(
    dc_mean: torch.Tensor, ac_coeff: torch.Tensor, pad_info: Dict[str, Any]
) -> torch.Tensor:
    """Recompose [B,C,T,H,W] from DC mean and AC coefficients."""
    n, c, t2, h2, w2 = dc_mean.shape
    if ac_coeff.ndim == 5:
        ac_coeff = ac_coeff.reshape(n, c, 7, t2, h2, w2)
    if ac_coeff.shape != (n, c, 7, t2, h2, w2):
        raise ValueError(
            f"bad AC shape {tuple(ac_coeff.shape)} for dc={tuple(dc_mean.shape)}"
        )
    coeff = torch.empty(n, c, 8, t2, h2, w2, device=dc_mean.device, dtype=torch.float32)
    coeff[:, :, 0] = dc_mean.float() * math.sqrt(8.0)
    coeff[:, :, 1:] = ac_coeff.float()
    h8 = make_hadamard8(coeff.device, coeff.dtype)
    blocks = torch.einsum("ba,ncathw->ncbthw", h8, coeff)
    return block_shuffle_2x2x2(blocks.to(dc_mean.dtype), pad_info)


def block_mean_expand_2x2x2(x: torch.Tensor) -> torch.Tensor:
    """Return the per-2x2x2 block mean expanded back to x's original shape."""
    blocks, pad_info = block_unshuffle_2x2x2(x)
    mean_blocks = blocks.mean(dim=2, keepdim=True).expand_as(blocks)
    return block_shuffle_2x2x2(mean_blocks.contiguous(), pad_info)


def _adaptive_channel_average(x: torch.Tensor, output_channels: int) -> torch.Tensor:
    """Parameter-free channel reduction, exact grouped mean when divisible."""
    n, channels, t, h, w = x.shape
    output_channels = int(output_channels)
    if channels % output_channels == 0:
        return x.reshape(
            n, output_channels, channels // output_channels, t, h, w
        ).mean(dim=2)
    positions = x.permute(0, 2, 3, 4, 1).reshape(-1, 1, channels)
    reduced = F.adaptive_avg_pool1d(positions, output_channels)
    return (
        reduced.reshape(n, t, h, w, output_channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def _adaptive_channel_duplicate(x: torch.Tensor, output_channels: int) -> torch.Tensor:
    """Parameter-free nearest channel expansion, exact repeat when divisible."""
    channels = x.shape[1]
    output_channels = int(output_channels)
    if output_channels % channels == 0:
        return x.repeat_interleave(output_channels // channels, dim=1)
    indices = torch.div(
        torch.arange(output_channels, device=x.device) * channels,
        output_channels,
        rounding_mode="floor",
    ).long()
    return x.index_select(1, indices)



class PreKV3DCAED1288x(nn.Module):
    """DCAE-style residual pre-KV AE with a parameter-free 2x2x2 shortcut."""

    def __init__(
        self,
        dim=128,
        hidden=128,
        latent_dim=128,
        stride=(2, 2, 2),
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.stride = _validate_dcae_stride(stride)
        if self.hidden != self.dim:
            raise ValueError("dcae_d128_8x currently requires hidden == dim")
        self.stem_conv = nn.Conv3d(self.dim, self.dim, 3, padding=1)
        self.stem_res = ResBlock3D(self.dim)
        self.learned_down = nn.Conv3d(
            self.dim, self.latent_dim, 3, stride=self.stride, padding=1
        )
        self.latent_encoder_res = ResBlock3D(self.latent_dim)
        self.latent_decoder_res = ResBlock3D(self.latent_dim)
        up_kernel = tuple(4 if value == 2 else 1 for value in self.stride)
        up_padding = tuple(1 if value == 2 else 0 for value in self.stride)
        self.learned_up = nn.ConvTranspose3d(
            self.latent_dim,
            self.dim,
            kernel_size=up_kernel,
            stride=self.stride,
            padding=up_padding,
        )
        self.output_res = ResBlock3D(self.dim)
        self.output_conv = nn.Conv3d(self.dim, self.dim, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _downsample_shortcut(self, x: torch.Tensor) -> torch.Tensor:
        packed = space_time_to_channel(x, self.stride)
        return _adaptive_channel_average(packed, self.latent_dim)

    def _upsample_shortcut(self, z: torch.Tensor) -> torch.Tensor:
        neighborhood = math.prod(self.stride)
        expanded = _adaptive_channel_duplicate(z, self.dim * neighborhood)
        return channel_to_space_time(expanded, self.dim, self.stride)

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, heads, sequence, dim = x.shape
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        x5 = (
            x.reshape(b, heads, t, h, w, dim)
            .permute(0, 1, 5, 2, 3, 4)
            .reshape(b * heads, dim, t, h, w)
            .contiguous()
        )
        x5 = self.stem_res(F.silu(self.stem_conv(x5)))
        pad_t = (-t) % self.stride[0]
        pad_h = (-h) % self.stride[1]
        pad_w = (-w) % self.stride[2]
        if pad_t or pad_h or pad_w:
            x5 = F.pad(x5, (0, pad_w, 0, pad_h, 0, pad_t))
        latent = self._downsample_shortcut(x5) + self.learned_down(x5)
        return self._mix_latent(self.latent_encoder_res(latent))

    def _mix_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def decode(self, z, shape, thw):
        return self.decode_prepared(self.prepare_decode_latent(z), shape, thw)

    def prepare_decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Run the deterministic, shape-preserving decoder prefix."""
        return self.latent_decoder_res(z)

    def decode_prepared(self, z, shape, thw):
        b, heads, sequence, dim = [int(value) for value in shape]
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad output shape {shape} for thw={thw}, dim={self.dim}")
        upsampled = self._upsample_shortcut(z) + self.learned_up(z)
        upsampled = upsampled[:, :, :t, :h, :w].contiguous()
        x5 = self._decode_output(upsampled)
        return (
            x5.reshape(b, heads, dim, t, h, w)
            .permute(0, 1, 3, 4, 5, 2)
            .reshape(b, heads, sequence, dim)
            .contiguous()
        )

    def _decode_output(self, upsampled: torch.Tensor) -> torch.Tensor:
        residual = self.output_conv(self.output_res(upsampled))
        return upsampled + residual


class PreKV3DCAED2568x(PreKV3DCAED1288x):
    """DCAE-style upper-bound variant with a 256-channel latent.

    The spatial-temporal downsample ratio stays 2x2x2, but the latent channel
    count is doubled from 128 to 256. This gives an effective resident element
    compression of about 4x for raw V cache, while keeping the input/output
    contract identical to dcae_d128_8x.
    """

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.shortcut_proj = nn.Conv3d(self.dim, self.latent_dim, kernel_size=1)
        self.shortcut_unproj = nn.Conv3d(self.latent_dim, self.dim, kernel_size=1)
        self._init_shortcut_projections()

    def _init_shortcut_projections(self) -> None:
        nn.init.zeros_(self.shortcut_proj.weight)
        nn.init.zeros_(self.shortcut_proj.bias)
        nn.init.zeros_(self.shortcut_unproj.weight)
        nn.init.zeros_(self.shortcut_unproj.bias)
        with torch.no_grad():
            shared = min(self.dim, self.latent_dim)
            for idx in range(shared):
                self.shortcut_proj.weight[idx, idx, 0, 0, 0] = 1.0
            for idx in range(self.dim):
                self.shortcut_unproj.weight[idx, idx, 0, 0, 0] = 1.0

    def _downsample_shortcut(self, x: torch.Tensor) -> torch.Tensor:
        packed = space_time_to_channel(x, self.stride)
        base = _adaptive_channel_average(packed, self.dim)
        return self.shortcut_proj(base)

    def _upsample_shortcut(self, z: torch.Tensor) -> torch.Tensor:
        base = self.shortcut_unproj(z)
        neighborhood = math.prod(self.stride)
        expanded = _adaptive_channel_duplicate(base, self.dim * neighborhood)
        return channel_to_space_time(expanded, self.dim, self.stride)


class PreKV3DCAED5128x(PreKV3DCAED2568x):
    """Raw K/V DCAE with a 512-channel latent.

    This keeps the 2x2x2 token downsample pattern and doubles the latent
    capacity relative to dcae_d256_8x. With BF16 latent storage it is a 2x
    element codec; with INT8 latent storage it targets roughly 4x resident
    compression.
    """

    model_variant = "dcae_d512_8x"

    def __init__(self, dim=128, hidden=128, latent_dim=512, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)


class ACOnlyRefineHead3D(nn.Module):
    """Full-resolution residual head whose output is projected to block-AC only."""

    def __init__(self, dim: int = 128, hidden: int = 128, depth: int = 2):
        super().__init__()
        blocks = [nn.Conv3d(dim, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(max(0, int(depth))):
            blocks.append(ResBlock3D(hidden))
        self.body = nn.Sequential(*blocks)
        self.out = nn.Conv3d(hidden, dim, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.out(self.body(x))
        return residual - block_mean_expand_2x2x2(residual)


class ScaledACOnlyRefineHead3D(ACOnlyRefineHead3D):
    """AC-only residual head with a learnable residual scale."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 128,
        depth: int = 2,
        init_scale: float = 1.0,
    ):
        super().__init__(dim=dim, hidden=hidden, depth=depth)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * super().forward(x)


class PreKV3DCAED256ACRefine8x(PreKV3DCAED2568x):
    """d256 4x DCAE with a zero-init output AC-only refinement head.

    The encoder, latent shape, and main decoder are identical to dcae_d256_8x.
    The added head can only change local 2x2x2 AC details because its per-block
    mean is removed before adding it to the base reconstruction.
    """

    model_variant = "dcae_d256_8x_ac_refine"

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.ac_refine = ACOnlyRefineHead3D(dim=self.dim, hidden=self.dim, depth=2)
        self._last_ac_refine_base = None
        self._last_ac_refine_residual = None

    def _decode_output(self, upsampled: torch.Tensor) -> torch.Tensor:
        base = super()._decode_output(upsampled)
        residual = self.ac_refine(base)
        self._last_ac_refine_base = base
        self._last_ac_refine_residual = residual
        return base + residual


class PreKV3DCAED256ACRefineH184D3_8x(PreKV3DCAED256ACRefine8x):
    """Stronger AC-only refinement head, kept below 20M total parameters."""

    model_variant = "dcae_d256_8x_ac_refine_h184_d3"

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.ac_refine = ACOnlyRefineHead3D(dim=self.dim, hidden=184, depth=3)


class PreKV3DCAED256ACRefineH184D3Scaled8x(PreKV3DCAED256ACRefine8x):
    """Stronger AC-only refinement head with a learnable residual scale."""

    model_variant = "dcae_d256_8x_ac_refine_h184_d3_scaled"

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.ac_refine = ScaledACOnlyRefineHead3D(dim=self.dim, hidden=184, depth=3, init_scale=1.0)


class LatentFullBandResidualDecoder3D(nn.Module):
    """Decode an unrestricted full-resolution residual directly from latent z.

    The final projection is zero initialized, making the containing D1 codec
    exactly equivalent to its D0 backbone before training.  Unlike the older
    AC refine head, this branch sees z directly and may correct both DC and AC.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        output_dim: int = 128,
        width: int = 160,
        depth: int = 3,
        stride: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.stride = _validate_dcae_stride(stride)
        self.latent_in = nn.Conv3d(latent_dim, width, kernel_size=3, padding=1)
        self.latent_res = ResBlock3D(width)
        up_kernel = tuple(4 if value == 2 else 1 for value in self.stride)
        up_padding = tuple(1 if value == 2 else 0 for value in self.stride)
        self.up = nn.ConvTranspose3d(
            width,
            width,
            kernel_size=up_kernel,
            stride=self.stride,
            padding=up_padding,
        )
        self.full_res = nn.Sequential(*[ResBlock3D(width) for _ in range(int(depth))])
        self.out = nn.Conv3d(width, self.output_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, shape, thw):
        b, heads, sequence, dim = [int(value) for value in shape]
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.output_dim:
            raise ValueError(f"bad output shape {shape} for thw={thw}, dim={self.output_dim}")
        x5 = self.latent_res(F.silu(self.latent_in(z)))
        x5 = self.full_res(F.silu(self.up(x5)))
        x5 = self.out(x5)[:, :, :t, :h, :w].contiguous()
        return (
            x5.reshape(b, heads, dim, t, h, w)
            .permute(0, 1, 3, 4, 5, 2)
            .reshape(b, heads, sequence, dim)
            .contiguous()
        )


class PreKV3DCAED256FullRefineH160D3_8x(PreKV3DCAED2568x):
    """D1: frozen d256 D0 plus a zero-init full-band decoder from the same z."""

    model_variant = "dcae_d256_8x_full_refine_h160_d3"

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.full_refine = LatentFullBandResidualDecoder3D(
            latent_dim=self.latent_dim,
            output_dim=self.dim,
            width=160,
            depth=3,
            stride=self.stride,
        )
        self._last_full_refine_base = None
        self._last_full_refine_residual = None

    def decode(self, z, shape, thw):
        base = super().decode(z, shape, thw)
        residual = self.full_refine(z, shape, thw)
        self._last_full_refine_base = base
        self._last_full_refine_residual = residual
        return base + residual


class VInt8CodeD5122x(PreKV3DCAED2568x):
    """2x V INT8-code AE with 2x2x2 token downsampling and 512 latent channels.

    This variant is intended to model normalized INT8 code, not raw BF16 V.  A
    2x2x2 local block contains 8 * 128 = 1024 code values; the latent stores 512
    values at the downsampled grid, giving a 2x element compression before any
    packed latent storage.
    """

    model_variant = "v_int8_code_d512_2x"
    is_int8_code_ae = True

    def __init__(self, dim=128, hidden=128, latent_dim=512, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)


class VInt8CodeD5122xFullRefineH160D3(VInt8CodeD5122x):
    """V D1: frozen INT8-code D0 plus zero-init full-band decoder from d512 z."""

    model_variant = "v_int8_code_d512_2x_full_refine_h160_d3"

    def __init__(self, dim=128, hidden=128, latent_dim=512, stride=(2, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)
        self.full_refine = LatentFullBandResidualDecoder3D(
            latent_dim=self.latent_dim,
            output_dim=self.dim,
            width=160,
            depth=3,
            stride=self.stride,
        )
        self._last_full_refine_base = None
        self._last_full_refine_residual = None

    def decode(self, z, shape, thw):
        base = super().decode(z, shape, thw)
        residual = self.full_refine(z, shape, thw)
        self._last_full_refine_base = base
        self._last_full_refine_residual = residual
        return base + residual


class VInt8CodeD256Spatial2x(PreKV3DCAED2568x):
    """2x V INT8-code AE that preserves time and downsamples only H/W.

    A 1x2x2 local block contains 4 * 128 = 512 code values; the latent stores
    256 values, also giving 2x element compression while avoiding temporal
    downsampling.
    """

    model_variant = "v_int8_code_d256_spatial2x"
    is_int8_code_ae = True

    def __init__(self, dim=128, hidden=128, latent_dim=256, stride=(1, 2, 2)):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=stride)


class PreKV3DCAED128Spatial4x(PreKV3DCAED1288x):
    def __init__(self, dim=128, hidden=128, latent_dim=128):
        super().__init__(dim, hidden, latent_dim, stride=(1, 2, 2))


class PreKV3DCAED256Spatial4x(PreKV3DCAED2568x):
    """D256 DCAE that preserves time and downsamples only H/W.

    This is a controlled diagnostic for temporal downsampling damage: it keeps
    the d256 latent width but changes stride from 2x2x2 to 1x2x2.  The byte
    ratio is therefore about 2x instead of 4x.
    """

    def __init__(self, dim=128, hidden=128, latent_dim=256):
        super().__init__(dim=dim, hidden=hidden, latent_dim=latent_dim, stride=(1, 2, 2))


class PerTokenChannelMLPAE(nn.Module):
    """Per-token channel codec for K diagnostics.

    This preserves token identity exactly in the latent layout: each
    [B,H,S,128] token is compressed independently to [B,H,S,latent_dim].
    """

    def __init__(self, dim=128, hidden=256, latent_dim=32):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.dim),
        )

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, heads, sequence, dim = x.shape
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        return self.encoder(x)

    def decode(self, z: torch.Tensor, shape, thw) -> torch.Tensor:
        b, heads, sequence, dim = [int(value) for value in shape]
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad output shape {shape} for thw={thw}, dim={self.dim}")
        if z.shape[:3] != (b, heads, sequence) or z.shape[-1] != self.latent_dim:
            raise ValueError(f"bad latent shape {tuple(z.shape)} for output shape={shape}")
        return self.decoder(z).contiguous()


class PerTokenChannelMLPD32AE(PerTokenChannelMLPAE):
    def __init__(self, dim=128, hidden=256, latent_dim=32):
        super().__init__(dim=dim, hidden=hidden, latent_dim=32)


class PerTokenChannelMLPD48AE(PerTokenChannelMLPAE):
    def __init__(self, dim=128, hidden=256, latent_dim=48):
        super().__init__(dim=dim, hidden=hidden, latent_dim=48)


class PerTokenChannelMLPD64AE(PerTokenChannelMLPAE):
    def __init__(self, dim=128, hidden=256, latent_dim=64):
        super().__init__(dim=dim, hidden=hidden, latent_dim=64)


class PreKV3DCAED128Temporal2x(PreKV3DCAED1288x):
    def __init__(self, dim=128, hidden=128, latent_dim=128):
        super().__init__(dim, hidden, latent_dim, stride=(2, 1, 1))


class ResidualPreKV3DCAED1288x(PreKV3DCAED1288x):
    """D128-8x network used to encode the residual over an INT2 V base."""

    is_residual_codec = True
    residual_base_bits = 2
    residual_base_granularity = "per_token"

    def _decode_output(self, upsampled: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.output_res(upsampled))


def match_spatial_temporal_shape(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Crop or interpolate a [N,C,T,H,W] tensor to the requested T/H/W shape."""
    target_t, target_h, target_w = [int(v) for v in target_shape]
    cur_t, cur_h, cur_w = [int(v) for v in x.shape[-3:]]
    if cur_t >= target_t and cur_h >= target_h and cur_w >= target_w:
        return x[:, :, :target_t, :target_h, :target_w].contiguous()
    return F.interpolate(
        x,
        size=(target_t, target_h, target_w),
        mode="nearest",
    ).contiguous()


def pad_to_even_hw(x: torch.Tensor):
    """Pad H/W of a [N,C,T,H,W] tensor to even sizes."""
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))
    return x, (pad_h, pad_w)


def crop_to_shape(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    target_t, target_h, target_w = [int(v) for v in target_shape]
    return x[:, :, :target_t, :target_h, :target_w].contiguous()


class SpaceToChannel2x2(nn.Module):
    """Rearrange [N,C,T,H,W] into [N,4C,T,H/2,W/2] without averaging."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = pad_to_even_hw(x)
        n, c, t, h, w = x.shape
        x = x.reshape(n, c, t, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        return x.reshape(n, c * 4, t, h // 2, w // 2)


class ChannelToSpace2x2(nn.Module):
    """Rearrange [N,4C,T,H,W] into [N,C,T,2H,2W]."""

    def __init__(self, output_channels: int):
        super().__init__()
        self.output_channels = int(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, channels, t, h, w = x.shape
        expected = self.output_channels * 4
        if channels != expected:
            raise ValueError(f"expected {expected} channels, got {channels}")
        x = x.reshape(n, self.output_channels, 2, 2, t, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return x.reshape(n, self.output_channels, t, h * 2, w * 2)


class ResDown3D(nn.Module):
    """Residual 3D downsample block with an average-pool shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 2, 2)):
        super().__init__()
        self.stride = tuple(int(v) for v in stride)
        self.shortcut_pool = nn.AvgPool3d(
            kernel_size=self.stride,
            stride=self.stride,
            ceil_mode=True,
        )
        self.shortcut_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.main_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.main_conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        self.main_conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.shortcut_pool(x.float()).to(dtype=x.dtype)
        shortcut = self.shortcut_proj(pooled)
        main = F.silu(self.main_conv1(x))
        main = F.silu(self.main_conv2(main))
        main = self.main_conv3(main)
        main = match_spatial_temporal_shape(main, shortcut.shape[-3:])
        return shortcut + main


class ResUp3D(nn.Module):
    """Residual 3D upsample block with nearest-neighbor shortcut."""

    def __init__(self, in_channels: int, out_channels: int, scale=(1, 2, 2)):
        super().__init__()
        self.scale = tuple(int(v) for v in scale)
        self.shortcut_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.main_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.main_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.main_conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self._upsample(x)
        shortcut = self.shortcut_proj(up)
        main = F.silu(self.main_conv1(up))
        main = F.silu(self.main_conv2(main))
        main = self.main_conv3(main)
        return shortcut + main


class RefineBlock3D(nn.Module):
    """Light decoder refinement block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.silu(self.conv1(x)))


class ResPatchPreKV3DAE(nn.Module):
    """Wan-inspired deterministic KV AE with staged spatial/temporal sampling."""

    def __init__(self, dim=128, hidden=128, latent_dim=128, dec_width=192):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.dec_width = int(dec_width)
        self.stem = nn.Sequential(
            nn.Conv3d(self.dim, self.hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.down_space = ResDown3D(self.hidden, self.hidden, stride=(1, 2, 2))
        self.encoder_res = ResBlock3D(self.hidden)
        self.down_time = ResDown3D(self.hidden, self.latent_dim, stride=(2, 1, 1))
        self.latent_proj = nn.Conv3d(self.latent_dim, self.latent_dim, kernel_size=1)
        self.latent_expand = nn.Sequential(
            nn.Conv3d(self.latent_dim, self.dec_width, kernel_size=1),
            nn.SiLU(),
        )
        self.up_time = ResUp3D(self.dec_width, self.dec_width, scale=(2, 1, 1))
        self.decoder_res = ResBlock3D(self.dec_width)
        self.up_space = ResUp3D(self.dec_width, self.dec_width, scale=(1, 2, 2))
        self.refine = RefineBlock3D(self.dec_width)
        self.output_proj = nn.Conv3d(self.dec_width, self.dim, kernel_size=3, padding=1)

    def _to_5d(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, heads, sequence, dim = x.shape
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        return (
            x.reshape(b, heads, t, h, w, dim)
            .permute(0, 1, 5, 2, 3, 4)
            .reshape(b * heads, dim, t, h, w)
            .contiguous()
        )

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        x5 = self._to_5d(x, thw)
        x5 = self.stem(x5)
        x5 = self.down_space(x5)
        x5 = self.encoder_res(x5)
        x5 = self.down_time(x5)
        return self.latent_proj(x5)

    def decode(self, z, shape, thw):
        b, heads, sequence, dim = [int(value) for value in shape]
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad output shape {shape} for thw={thw}, dim={self.dim}")
        x5 = self.latent_expand(z)
        x5 = self.up_time(x5)
        x5 = self.decoder_res(x5)
        x5 = self.up_space(x5)
        x5 = match_spatial_temporal_shape(x5, (t, h, w))
        x5 = self.refine(x5)
        x5 = self.output_proj(x5)
        if tuple(x5.shape[-3:]) != (t, h, w):
            raise RuntimeError(f"ResPatch output spatial shape {tuple(x5.shape[-3:])} != {(t, h, w)}")
        return (
            x5.reshape(b, heads, dim, t, h, w)
            .permute(0, 1, 3, 4, 5, 2)
            .reshape(b, heads, sequence, dim)
            .contiguous()
        )


class PatchD1288xAE(nn.Module):
    """Patchify-first deterministic raw V AE."""

    model_variant = "patch_d128_8x"

    def __init__(self, dim=128, hidden=128, latent_dim=128):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        patch_channels = self.dim * 4
        mix_channels = 192
        time_channels = 192

        self.space_to_channel = SpaceToChannel2x2()
        self.patch_stem = nn.Sequential(
            nn.Conv3d(patch_channels, mix_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(mix_channels, mix_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.patch_mix = nn.Sequential(
            ResBlock3D(mix_channels),
        )
        self.time_down = nn.Sequential(
            nn.Conv3d(
                mix_channels,
                time_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.SiLU(),
        )
        self.latent_proj = nn.Conv3d(time_channels, self.latent_dim, kernel_size=1)

        self.latent_expand = nn.Sequential(
            nn.Conv3d(self.latent_dim, time_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.time_up_conv = nn.Sequential(
            nn.Conv3d(time_channels, mix_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.decoder_mix = ResBlock3D(mix_channels)
        self.to_patch_channels = nn.Sequential(
            nn.Conv3d(mix_channels, patch_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.channel_to_space = ChannelToSpace2x2(self.dim)
        self.output_refine = nn.Sequential(
            nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(self.dim, self.dim, kernel_size=1),
        )
        nn.init.zeros_(self.output_refine[-1].weight)
        nn.init.zeros_(self.output_refine[-1].bias)

    def _to_5d(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, heads, sequence, dim = x.shape
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        return (
            x.reshape(b, heads, t, h, w, dim)
            .permute(0, 1, 5, 2, 3, 4)
            .reshape(b * heads, dim, t, h, w)
            .contiguous()
        )

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        x5 = self._to_5d(x, thw)
        x5 = self.space_to_channel(x5)
        x5 = self.patch_stem(x5)
        x5 = self.patch_mix(x5)
        x5 = self.time_down(x5)
        return self.latent_proj(x5)

    def decode(self, z, shape, thw):
        b, heads, sequence, dim = [int(value) for value in shape]
        t, h, w = [int(value) for value in thw]
        if sequence != t * h * w or dim != self.dim:
            raise ValueError(f"bad output shape {shape} for thw={thw}, dim={self.dim}")
        x5 = self.latent_expand(z)
        x5 = F.interpolate(x5, scale_factor=(2, 1, 1), mode="nearest")
        x5 = self.time_up_conv(x5)
        x5 = self.decoder_mix(x5)
        x5 = self.to_patch_channels(x5)
        x5 = self.channel_to_space(x5)
        x5 = crop_to_shape(x5, (t, h, w))
        x5 = x5 + self.output_refine(x5)
        if tuple(x5.shape[-3:]) != (t, h, w):
            raise RuntimeError(f"PatchAE output spatial shape {tuple(x5.shape[-3:])} != {(t, h, w)}")
        return (
            x5.reshape(b, heads, dim, t, h, w)
            .permute(0, 1, 3, 4, 5, 2)
            .reshape(b, heads, sequence, dim)
            .contiguous()
        )


class SeparableResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.depthwise2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise2 = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = torch.nn.functional.silu(self.pointwise1(self.depthwise1(x)))
        return residual + self.pointwise2(self.depthwise2(x))


class _PreKVAEBase(nn.Module):
    def __init__(self, dim=128, hidden=128, latent_dim=64):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)

    def encode(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> torch.Tensor:
        b, h, s, d = x.shape
        t, hh, ww = [int(v) for v in thw]
        if s != t * hh * ww or d != self.dim:
            raise ValueError(f"bad shape {tuple(x.shape)} for thw={thw}, dim={self.dim}")
        x = x.reshape(b, h, t, hh, ww, d).permute(0, 1, 5, 2, 3, 4)
        return self.encoder(x.reshape(b * h, d, t, hh, ww).contiguous())

    def decode(self, z, shape, thw):
        b, h, s, d = [int(v) for v in shape]
        t, hh, ww = [int(v) for v in thw]
        x = self.decoder(z)[:, :, :t, :hh, :ww].contiguous()
        x = x.reshape(b, h, d, t, hh, ww).permute(0, 1, 3, 4, 5, 2)
        return x.reshape(b, h, s, d).contiguous()


class BottleneckResPreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=128, latent_dim=64):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, latent_dim, 3, stride=2, padding=1),
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(latent_dim, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class Hidden192PreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=192, latent_dim=64):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            ResBlock3D(hidden),
            nn.Conv3d(hidden, latent_dim, 3, stride=2, padding=1),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(latent_dim, hidden, 4, stride=2, padding=1),
            ResBlock3D(hidden),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class EfficientSeparablePreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=128, latent_dim=64):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, latent_dim, 3, stride=2, padding=1),
            SeparableResBlock3D(latent_dim),
            SeparableResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            SeparableResBlock3D(latent_dim),
            SeparableResBlock3D(latent_dim),
            nn.ConvTranspose3d(latent_dim, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class Hidden192SeparablePreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=192, latent_dim=64):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            SeparableResBlock3D(hidden),
            SeparableResBlock3D(hidden),
            nn.Conv3d(hidden, latent_dim, 3, stride=2, padding=1),
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(latent_dim, hidden, 4, stride=2, padding=1),
            SeparableResBlock3D(hidden),
            SeparableResBlock3D(hidden),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class HybridSeparablePreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=128, latent_dim=64):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            SeparableResBlock3D(hidden),
            SeparableResBlock3D(hidden),
            nn.Conv3d(hidden, latent_dim, 3, stride=2, padding=1),
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(latent_dim, hidden, 4, stride=2, padding=1),
            SeparableResBlock3D(hidden),
            SeparableResBlock3D(hidden),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class TemporalPreservePreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=128, latent_dim=32):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                hidden, latent_dim, 3, stride=(1, 2, 2), padding=1
            ),
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(
                latent_dim,
                hidden,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class Channel96Spatial3PreKV3DAE(_PreKVAEBase):
    def __init__(self, dim=128, hidden=128, latent_dim=96):
        super().__init__(dim, hidden, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                hidden, latent_dim, 3, stride=(2, 2, 3), padding=1
            ),
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock3D(latent_dim),
            ResBlock3D(latent_dim),
            nn.ConvTranspose3d(
                latent_dim,
                hidden,
                kernel_size=(4, 4, 5),
                stride=(2, 2, 3),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(hidden, dim, 3, padding=1),
        )


class GlobalPCALinearFunctionalCodec(nn.Module):
    """Linear functional codec initialized from a global-head PCA codec.

    The PCA basis and mean remain immutable buffers. ``encoder_delta`` and
    ``decoder_delta`` are zero-initialized, so the initial model is exactly the
    original PCA reconstruction. Decoder-only A1 freezes ``encoder_delta``;
    full-codec A1 can optimize both untied deltas without changing the PCA
    decoder when it changes the encoder. The latent layout is [B,R,S] so the existing
    ``*_LATENT_INT4_TOKEN`` path quantizes across R independently per token,
    matching the resident global-PCA codec's [B,S,R] per-token quantizer.
    """

    def __init__(self, dim=128, hidden=128, latent_dim=512):
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        if self.dim != 128 or self.latent_dim not in {128, 256, 512}:
            raise ValueError(
                "GlobalPCALinearFunctionalCodec requires dim=128 and "
                "latent_dim in {128,256,512}"
            )
        self.register_buffer("mean", torch.zeros(2048, dtype=torch.float32))
        self.register_buffer(
            "encoder_basis",
            torch.zeros(self.latent_dim, 2048, dtype=torch.float32),
        )
        self.encoder_delta = nn.Parameter(
            torch.zeros(self.latent_dim, 2048, dtype=torch.float32)
        )
        self.decoder_delta = nn.Linear(self.latent_dim, 2048, bias=False)
        nn.init.zeros_(self.decoder_delta.weight)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        # Checkpoints produced before the encoder+decoder A1 gate have no
        # encoder_delta. Treat them as the exact zero-delta initialization so
        # all existing decoder-only artifacts remain loadable with strict=True.
        key = prefix + "encoder_delta"
        inserted = key not in state_dict
        if inserted:
            state_dict[key] = torch.zeros_like(self.encoder_delta)
        try:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys,
                unexpected_keys, error_msgs,
            )
        finally:
            if inserted:
                state_dict.pop(key, None)

    def encode(self, x: torch.Tensor, thw) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"global PCA expects [B,H,S,D], got {tuple(x.shape)}")
        batch, heads, sequence, dim = x.shape
        if heads * dim != 2048:
            raise ValueError(f"global PCA expects H*D=2048, got {heads}*{dim}")
        # The inference path keeps the codec in the incoming cache dtype
        # (normally BF16), while functional training can use an FP32 master
        # copy.  Do not unconditionally promote the activation to FP32 here:
        # doing so makes F.linear fail against BF16 PCA buffers at deployment.
        rows = (
            x.permute(0, 2, 1, 3)
            .reshape(batch, sequence, 2048)
            .to(dtype=self.encoder_basis.dtype)
        )
        coeff = F.linear(rows - self.mean, self.encoder_basis + self.encoder_delta)
        return coeff.transpose(1, 2).contiguous()

    def decode(self, z: torch.Tensor, shape, thw) -> torch.Tensor:
        batch, heads, sequence, dim = [int(value) for value in shape]
        if z.shape != (batch, self.latent_dim, sequence):
            raise ValueError(
                f"global PCA latent must be {(batch, self.latent_dim, sequence)}, "
                f"got {tuple(z.shape)}"
            )
        latent_rows = (
            z.transpose(1, 2)
            .reshape(-1, self.latent_dim)
            .to(dtype=self.encoder_basis.dtype)
        )
        rows = (
            F.linear(latent_rows, self.encoder_basis.T, self.mean)
            + self.decoder_delta(latent_rows)
        )
        return (
            rows.reshape(batch, sequence, heads, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )


def build_pre_kv_ae(config: Dict[str, Any]) -> nn.Module:
    architecture = config.get("architecture", "small")
    classes = {
        "small": PreKV3DAE,
        "bottleneck_res": BottleneckResPreKV3DAE,
        "hidden192": Hidden192PreKV3DAE,
        "efficient_separable": EfficientSeparablePreKV3DAE,
        "hidden192_separable": Hidden192SeparablePreKV3DAE,
        "hybrid_separable": HybridSeparablePreKV3DAE,
        "temporal_preserve": TemporalPreservePreKV3DAE,
        "channel96_spatial3": Channel96Spatial3PreKV3DAE,
        "dcae_d128_8x": PreKV3DCAED1288x,
        "dcae_d256_8x": PreKV3DCAED2568x,
        "dcae_d512_8x": PreKV3DCAED5128x,
        "dcae_d256_8x_ac_refine": PreKV3DCAED256ACRefine8x,
        "dcae_d256_8x_ac_refine_h184_d3": PreKV3DCAED256ACRefineH184D3_8x,
        "dcae_d256_8x_ac_refine_h184_d3_scaled": PreKV3DCAED256ACRefineH184D3Scaled8x,
        "dcae_d256_8x_full_refine_h160_d3": PreKV3DCAED256FullRefineH160D3_8x,
        "v_int8_code_d512_2x": VInt8CodeD5122x,
        "v_int8_code_d512_2x_full_refine_h160_d3": VInt8CodeD5122xFullRefineH160D3,
        "v_int8_code_d256_spatial2x": VInt8CodeD256Spatial2x,
        "dcae_d128_spatial4x": PreKV3DCAED128Spatial4x,
        "dcae_d256_spatial4x": PreKV3DCAED256Spatial4x,
        "token_mlp_d32": PerTokenChannelMLPD32AE,
        "token_mlp_d48": PerTokenChannelMLPD48AE,
        "token_mlp_d64": PerTokenChannelMLPD64AE,
        "dcae_d128_temporal2x": PreKV3DCAED128Temporal2x,
        "respatch_d128_8x": ResPatchPreKV3DAE,
        "patch_d128_8x": PatchD1288xAE,
        "residual_int2_dcae_d128_8x": ResidualPreKV3DCAED1288x,
        "global_pca_linear_d512": GlobalPCALinearFunctionalCodec,
    }
    if architecture not in classes:
        raise ValueError(f"Unknown pre-KV AE architecture: {architecture}")
    defaults = {
        "dim": int(config.get("dim", 128)),
        "hidden": int(config.get("hidden", 128)),
        "latent_dim": int(config.get("latent_dim", 64)),
    }
    if architecture in {"hidden192", "hidden192_separable"}:
        defaults["hidden"] = int(config.get("hidden", 192))
    if architecture == "temporal_preserve":
        defaults["latent_dim"] = int(config.get("latent_dim", 32))
    if architecture == "channel96_spatial3":
        defaults["latent_dim"] = int(config.get("latent_dim", 96))
    if architecture in {
        "dcae_d128_8x",
        "dcae_d256_8x",
        "dcae_d512_8x",
        "dcae_d256_8x_ac_refine",
        "dcae_d256_8x_ac_refine_h184_d3",
        "dcae_d256_8x_ac_refine_h184_d3_scaled",
        "dcae_d256_8x_full_refine_h160_d3",
        "dcae_d128_spatial4x",
        "dcae_d256_spatial4x",
        "dcae_d128_temporal2x",
        "respatch_d128_8x",
        "patch_d128_8x",
        "residual_int2_dcae_d128_8x",
    }:
        defaults["latent_dim"] = int(config.get("latent_dim", 128))
    if architecture in {"token_mlp_d32", "token_mlp_d48", "token_mlp_d64"}:
        defaults["hidden"] = int(config.get("hidden", 256))
        defaults["latent_dim"] = int(config.get("latent_dim", architecture.rsplit("d", 1)[1]))
    if architecture in {
        "dcae_d256_8x",
        "dcae_d256_8x_ac_refine",
        "dcae_d256_8x_ac_refine_h184_d3",
        "dcae_d256_8x_ac_refine_h184_d3_scaled",
        "dcae_d256_8x_full_refine_h160_d3",
        "dcae_d256_spatial4x",
        "v_int8_code_d256_spatial2x",
    }:
        defaults["latent_dim"] = int(config.get("latent_dim", 256))
    if architecture in {
        "v_int8_code_d512_2x",
        "v_int8_code_d512_2x_full_refine_h160_d3",
        "global_pca_linear_d512",
    }:
        defaults["latent_dim"] = int(config.get("latent_dim", 512))
    if architecture in {"dcae_d512_8x"}:
        defaults["latent_dim"] = int(config.get("latent_dim", 512))
    return classes[architecture](**defaults)


_MODELS: Dict[str, PreKV3DAE] = {}
_NORM_STATS: Optional[Dict[Tuple[int, str], Tuple[float, float]]] = None
_STATS: Dict[str, Any] = {
    "enabled": False,
    "layers": "",
    "latent_dtype": "",
    "compress_calls": 0,
    "decompress_calls": 0,
    "encoder_time_s": 0.0,
    "decoder_time_s": 0.0,
    "resident_bf16_bytes": 0,
    "resident_ae_bytes": 0,
    "resident_original_total_bytes": 0,
    "resident_effective_bytes": 0,
    "resident_estimated_effective_bytes": 0.0,
    "resident_estimated_codec_bytes": 0.0,
    "residual_codec_calls": 0,
    "layerwise_enabled": False,
    "layerwise_layers": "",
    "layerwise_compress_calls": 0,
    "layerwise_encoder_time_s": 0.0,
    "layerwise_decoder_time_s": 0.0,
    "layerwise_k_encoder_time_s": 0.0,
    "layerwise_v_encoder_time_s": 0.0,
    "layerwise_k_decoder_time_s": 0.0,
    "layerwise_v_decoder_time_s": 0.0,
    "layerwise_k_decoder_calls": 0,
    "layerwise_v_decoder_calls": 0,
    "layerwise_original_bytes": 0,
    "layerwise_payload_bytes": 0,
    "layerwise_k_original_bytes": 0,
    "layerwise_k_payload_bytes": 0,
    "layerwise_v_original_bytes": 0,
    "layerwise_v_payload_bytes": 0,
    "layerwise_latent_int8_calls": 0,
    "layerwise_k_calls": 0,
    "layerwise_v_calls": 0,
    "k_corruption_calls": 0,
    "k_corruption_encoder_time_s": 0.0,
    "k_corruption_decoder_time_s": 0.0,
    "residual_base_bits": 0,
    "residual_base_granularity": "",
    "fallback_count": 0,
    "fallback_reasons": {},
}
_CORRUPTION_STATS: Dict[str, Any] = {
    "enabled": False,
    "mode": "off",
    "variant": "",
    "layers": "",
    "calls": 0,
    "skipped": 0,
    "warnings": {},
    "sum_actual_rel_l2": 0.0,
    "sum_cosine": 0.0,
    "sum_l1": 0.0,
    "sum_dc_rel_l2": 0.0,
    "sum_ac_rel_l2": 0.0,
    "sum_ac_error_ratio": 0.0,
    "max_layer_rel_l2": 0.0,
    "per_layer": {},
}
_DUMP_COUNT = 0
_ATTN_DUMP_COUNT = 0
_ATTN_DUMP_KEYS = set()


def enabled() -> bool:
    return _env_flag("HY_KV_AE3D_ENABLE", "0")


def _record_fallback(reason: str) -> None:
    _STATS["fallback_count"] += 1
    _STATS["fallback_reasons"][reason] = _STATS["fallback_reasons"].get(reason, 0) + 1


def _group_for_layer(layer_idx: int) -> int:
    if layer_idx <= 12:
        return 0
    if layer_idx <= 26:
        return 1
    if layer_idx <= 40:
        return 2
    return 3


def _checkpoint_path(layer_idx: int, tensor_name: str) -> Path:
    root_env = (
        "HY_KV_AE3D_K_CHECKPOINT_ROOT"
        if tensor_name == "k_vision"
        else "HY_KV_AE3D_V_CHECKPOINT_ROOT"
    )
    root = os.environ.get(root_env, "").strip()
    if not root:
        root = os.environ.get("HY_KV_AE3D_CHECKPOINT_ROOT", "").strip()
    if root:
        group = _group_for_layer(layer_idx)
        suffix = "k" if tensor_name == "k_vision" else "v"
        checkpoint_kind = os.environ.get("HY_KV_AE3D_CHECKPOINT_KIND", "stage_b")
        if checkpoint_kind == "layerwise":
            return Path(root) / f"layer_{int(layer_idx):02d}" / "checkpoints" / "checkpoint_best.pt"
        if checkpoint_kind == "stage_a":
            return Path(root) / f"group{group}_{suffix}" / "prekv_ae3d.pt"
        return Path(root) / f"group{group}" / f"stage_b_{suffix}.pt"
    return Path(os.environ.get("HY_KV_AE3D_CKPT", "kv_ae_outputs/ae3d/prekv_ae3d.pt"))


def _model(device, dtype, checkpoint_path: Path) -> PreKV3DAE:
    key = str(checkpoint_path.resolve())
    if key not in _MODELS:
        ckpt_path = checkpoint_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("model_config", {})
        if not cfg:
            cfg = {"architecture": "dcae_d128_8x", "dim": 128, "hidden": 128, "latent_dim": 128}
        model = build_pre_kv_ae(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODELS[key] = model
        print(
            f"[KV-AE3D] loaded model_variant={cfg.get('architecture', 'dcae_d128_8x')} "
            f"from {ckpt_path}",
            flush=True,
        )
    return _MODELS[key].to(device=device, dtype=dtype)


def _normalization(layer_idx: int, tensor_name: str):
    global _NORM_STATS
    path = os.environ.get("HY_KV_AE3D_NORM_STATS", "").strip()
    if not path:
        return None
    if _NORM_STATS is None:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        _NORM_STATS = {
            (int(row["layer"]), row["tensor"]): (
                float(row["mean"]),
                max(float(row["std"]), 1e-6),
            )
            for row in rows
        }
    return _NORM_STATS.get((int(layer_idx), tensor_name))


def _tensor_enabled(tensor_name: str) -> bool:
    value = os.environ.get("HY_KV_AE3D_TENSORS", "K,V").upper()
    enabled_tensors = {item.strip() for item in value.split(",") if item.strip()}
    short_name = "K" if tensor_name == "k_vision" else "V"
    return short_name in enabled_tensors


def _corruption_enabled() -> bool:
    return _env_flag("HY_KV_CORRUPTION_ENABLE", "0")


def _record_corruption_warning(reason: str) -> None:
    _CORRUPTION_STATS["skipped"] += 1
    warnings = _CORRUPTION_STATS["warnings"]
    warnings[reason] = warnings.get(reason, 0) + 1


def _layer_accum(layer_idx: int) -> Dict[str, Any]:
    key = str(int(layer_idx))
    table = _CORRUPTION_STATS["per_layer"]
    if key not in table:
        table[key] = {
            "calls": 0,
            "sum_actual_rel_l2": 0.0,
            "sum_cosine": 0.0,
            "sum_l1": 0.0,
            "sum_dc_rel_l2": 0.0,
            "sum_ac_rel_l2": 0.0,
            "sum_ac_error_ratio": 0.0,
            "max_actual_rel_l2": 0.0,
        }
    return table[key]


def _pad_thw_even(x5: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, _, t, h, w = x5.shape
    pad_t = t % 2
    pad_h = h % 2
    pad_w = w % 2
    if pad_t or pad_h or pad_w:
        x5 = F.pad(x5, (0, pad_w, 0, pad_h, 0, pad_t))
    return x5, (pad_t, pad_h, pad_w)


def _unpad_thw(x5: torch.Tensor, pad: Tuple[int, int, int]) -> torch.Tensor:
    pad_t, pad_h, pad_w = pad
    t_end = x5.shape[2] - pad_t if pad_t else x5.shape[2]
    h_end = x5.shape[3] - pad_h if pad_h else x5.shape[3]
    w_end = x5.shape[4] - pad_w if pad_w else x5.shape[4]
    return x5[:, :, :t_end, :h_end, :w_end]


def _v_to_5d(v: torch.Tensor, thw) -> torch.Tensor:
    b, heads, seq, dim = v.shape
    t, h, w = [int(x) for x in thw]
    if seq != t * h * w:
        raise ValueError(f"bad V seq={seq} for thw={(t, h, w)}")
    return v.reshape(b, heads, t, h, w, dim).permute(0, 1, 5, 2, 3, 4).reshape(
        b * heads, dim, t, h, w
    )


def _v_from_5d(x5: torch.Tensor, shape: Tuple[int, int, int, int], thw) -> torch.Tensor:
    b, heads, seq, dim = [int(x) for x in shape]
    t, h, w = [int(x) for x in thw]
    return (
        x5.reshape(b, heads, dim, t, h, w)
        .permute(0, 1, 3, 4, 5, 2)
        .reshape(b, heads, seq, dim)
        .contiguous()
    )


def _block_mean_dc_5d(x5: torch.Tensor) -> torch.Tensor:
    padded, pad = _pad_thw_even(x5.float())
    n, c, t, h, w = padded.shape
    blocks = padded.reshape(n, c, t // 2, 2, h // 2, 2, w // 2, 2)
    dc = blocks.mean(dim=(3, 5, 7), keepdim=True).expand_as(blocks)
    dc = dc.reshape(n, c, t, h, w)
    return _unpad_thw(dc, pad)


def _ac_scaled(v: torch.Tensor, thw, beta: float) -> torch.Tensor:
    shape = tuple(int(x) for x in v.shape)
    x5 = _v_to_5d(v, thw)
    dc = _block_mean_dc_5d(x5)
    out5 = dc + float(beta) * (x5.float() - dc)
    return _v_from_5d(out5.to(v.dtype), shape, thw)


def _dcae_error_ac_scaled(clean: torch.Tensor, recon: torch.Tensor, thw, gamma: float) -> torch.Tensor:
    """Keep DCAE DC error unchanged and scale only 2x2x2 block AC error."""
    shape = tuple(int(x) for x in clean.shape)
    err5 = _v_to_5d(recon.float() - clean.float(), thw)
    err_dc = _block_mean_dc_5d(err5)
    err_ac = err5 - err_dc
    out5 = _v_to_5d(clean.float(), thw) + err_dc + float(gamma) * err_ac
    return _v_from_5d(out5.to(clean.dtype), shape, thw)


def _dc_ac_metrics(clean: torch.Tensor, corrupt: torch.Tensor, thw) -> Tuple[float, float, float]:
    x5 = _v_to_5d(clean, thw).float()
    y5 = _v_to_5d(corrupt, thw).float()
    dc_x = _block_mean_dc_5d(x5)
    dc_y = _block_mean_dc_5d(y5)
    ac_x = x5 - dc_x
    ac_y = y5 - dc_y
    diff = y5 - x5
    eps = 1e-8
    dc_rel = (dc_y - dc_x).norm() / (dc_x.norm() + eps)
    ac_rel = (ac_y - ac_x).norm() / (ac_x.norm() + eps)
    ac_err = ((ac_y - ac_x).pow(2).sum()) / (diff.pow(2).sum() + eps)
    return float(dc_rel.item()), float(ac_rel.item()), float(ac_err.item())


def _corruption_metrics(clean: torch.Tensor, corrupt: torch.Tensor, thw) -> Dict[str, float]:
    x = clean.float()
    y = corrupt.float()
    diff = y - x
    eps = 1e-8
    rel = diff.norm() / (x.norm() + eps)
    cosine = F.cosine_similarity(x.flatten(), y.flatten(), dim=0)
    l1 = diff.abs().mean()
    metrics_mode = os.environ.get("HY_KV_CORRUPTION_METRICS", "full").strip().lower()
    if metrics_mode in {"basic", "light"}:
        dc_rel, ac_rel, ac_err = 0.0, 0.0, 0.0
    else:
        dc_rel, ac_rel, ac_err = _dc_ac_metrics(clean, corrupt, thw)
    return {
        "actual_rel_l2": float(rel.item()),
        "cosine": float(cosine.item()),
        "l1": float(l1.item()),
        "dc_rel_l2": dc_rel,
        "ac_rel_l2": ac_rel,
        "ac_error_ratio": ac_err,
    }


@torch.no_grad()
def _layerwise_residual_ae_correct(
    clean: torch.Tensor, dequant: torch.Tensor, layer_idx: int, thw,
    ckpt_dir: str, variant: str
) -> Optional[torch.Tensor]:
    """Apply INT2 residual AE: V_hat = dequant + AE(V_clean - dequant)."""
    # build_pre_kv_ae is defined in this same module
    ckpt_path = (
        Path(ckpt_dir)
        / f"layer_{int(layer_idx):02d}"
        / "checkpoints"
        / "checkpoint_best.pt"
    )
    if not ckpt_path.exists():
        _record_corruption_warning(f"missing_int2res_checkpoint_{layer_idx}")
        return None
    key = f"int2res::corr::{ckpt_path.resolve()}"
    if key not in _LAYERWISE_AE_MODELS:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt_data.get("model_config", {})
        model = build_pre_kv_ae(cfg)
        model.load_state_dict(ckpt_data["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LAYERWISE_AE_MODELS[key] = model
    model = _LAYERWISE_AE_MODELS[key].to(device=clean.device, dtype=clean.dtype)
    residual = (clean.float() - dequant.float()).to(clean.dtype)
    z = model.encode(residual.contiguous(), thw)
    r_hat = model.decode(z, tuple(residual.shape), tuple(int(v) for v in thw)).to(clean.dtype)
    return (dequant.float() + r_hat.float()).to(clean.dtype)


def _layerwise_reconstruct_v(tensor: torch.Tensor, layer_idx: int, thw) -> Optional[torch.Tensor]:
    ckpt_dir = os.environ.get("HY_KV_CORRUPTION_DCAE_CKPT_DIR", "").strip()
    if not ckpt_dir:
        ckpt_dir = os.environ.get(
            "HY_LAYERWISE_RAW_V_AE_CKPT_DIR",
            "kv_ae_outputs/ae16x_v2/layerwise_raw_v_12000",
        )
    variant = os.environ.get("HY_KV_CORRUPTION_DCAE_MODEL_VARIANT", "dcae_d128_8x")
    ckpt_base = Path(ckpt_dir) / f"layer_{int(layer_idx):02d}" / "checkpoints"
    ckpt_path = ckpt_base / f"layer{int(layer_idx):02d}_v_{variant}_step12000.pt"
    if not ckpt_path.exists():
        for name in ("checkpoint_best.pt", "best.pt", "checkpoint_last.pt"):
            candidate = ckpt_base / name
            if candidate.exists():
                ckpt_path = candidate
                break
    if not ckpt_path.exists():
        _record_corruption_warning("missing_dcae_checkpoint")
        return None
    key = f"corruption::{ckpt_path.resolve()}"
    if key not in _LAYERWISE_AE_MODELS:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt_data.get("model_config", {})
        model = build_pre_kv_ae(cfg)
        model.load_state_dict(ckpt_data["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LAYERWISE_AE_MODELS[key] = model
    model = _LAYERWISE_AE_MODELS[key].to(device=tensor.device, dtype=tensor.dtype)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    start = time.perf_counter()
    z = model.encode(tensor.contiguous(), thw)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    _STATS["k_corruption_encoder_time_s"] += time.perf_counter() - start

    start = time.perf_counter()
    out = model.decode(z, tuple(tensor.shape), tuple(int(v) for v in thw)).to(tensor.dtype)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    _STATS["k_corruption_decoder_time_s"] += time.perf_counter() - start
    _STATS["k_corruption_calls"] += 1
    return out


@torch.no_grad()
def _apply_v_corruption(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    if not _corruption_enabled():
        return layer_cache
    mode = os.environ.get("HY_KV_CORRUPTION_MODE", "clean").strip().lower()
    layers = _parse_layers(os.environ.get("HY_KV_CORRUPTION_LAYERS", ""))
    if not _layer_enabled(layer_idx, layers):
        return layer_cache

    # Optional: skip corruption for early chunks (for ablation experiments)
    start_chunk = int(os.environ.get("HY_KV_CORRUPTION_START_CHUNK", "0"))
    current_chunk = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    if start_chunk > 0 and (current_chunk < 0 or current_chunk < start_chunk):
        return layer_cache
    name = "v_vision"
    tensor = layer_cache.get(name)
    thw = layer_cache.get("vision_thw")
    if not isinstance(tensor, torch.Tensor):
        _record_corruption_warning("missing_v_tensor")
        return layer_cache
    if thw is None:
        _record_corruption_warning("missing_thw")
        return layer_cache
    if tuple(tensor.shape)[-1] != 128 or tensor.ndim != 4:
        _record_corruption_warning(f"bad_shape_{tuple(tensor.shape)}")
        return layer_cache

    _CORRUPTION_STATS["enabled"] = True
    _CORRUPTION_STATS["mode"] = mode
    _CORRUPTION_STATS["variant"] = os.environ.get("HY_KV_CORRUPTION_VARIANT", "")
    _CORRUPTION_STATS["layers"] = os.environ.get("HY_KV_CORRUPTION_LAYERS", "")

    clean = tensor
    corrupt = clean
    if mode == "clean":
        corrupt = clean
    elif mode == "gaussian":
        target_rel = float(os.environ.get("HY_KV_CORRUPTION_TARGET_REL_L2", "0.1"))
        seed = int(os.environ.get("HY_KV_CORRUPTION_SEED", "1"))
        gen = torch.Generator(device=clean.device)
        gen.manual_seed(seed + int(layer_idx) * 1000003 + int(_CORRUPTION_STATS["calls"]))
        noise = torch.randn(clean.shape, device=clean.device, dtype=torch.float32, generator=gen)
        scale = target_rel * clean.float().norm() / (noise.norm() + 1e-8)
        corrupt = (clean.float() + noise * scale).to(clean.dtype)
    elif mode == "ac_scale":
        beta = float(os.environ.get("HY_KV_CORRUPTION_BETA", "0.8"))
        corrupt = _ac_scaled(clean, thw, beta)
    elif mode in {"dcae", "dcae_error_scaled", "dcae_error_ac_scaled"}:
        recon = _layerwise_reconstruct_v(clean, layer_idx, thw)
        if recon is None:
            return layer_cache
        if mode == "dcae":
            corrupt = recon
        elif mode == "dcae_error_scaled":
            gamma = float(os.environ.get("HY_KV_CORRUPTION_GAMMA", "1.0"))
            corrupt = (clean.float() + gamma * (recon.float() - clean.float())).to(clean.dtype)
        else:
            gamma = float(os.environ.get("HY_KV_CORRUPTION_GAMMA", "1.0"))
            corrupt = _dcae_error_ac_scaled(clean, recon, thw, gamma)
    elif mode in {"int2", "int4", "int2_residual"}:
        from hyvideo.models.transformers.modules.kv_codecs import quantize_kv_tensor, dequantize_kv_tensor
        bits = 2 if "int2" in mode else 4
        raw_gran = os.environ.get("HY_KV_CORRUPTION_SCALE_GRANULARITY", "per_token")
        granularity = {"token": "per_token", "tensor": "per_tensor"}.get(raw_gran, raw_gran)
        codes, scale = quantize_kv_tensor(clean, bits, granularity)
        dequant = dequantize_kv_tensor(codes, scale, clean.dtype)
        if mode == "int2_residual":
            ckpt_dir = os.environ.get("HY_KV_CORRUPTION_DCAE_CKPT_DIR", "").strip()
            variant = os.environ.get("HY_KV_CORRUPTION_DCAE_MODEL_VARIANT", "dcae_d128_8x")
            if ckpt_dir:
                residual_ae = _layerwise_residual_ae_correct(clean, dequant, layer_idx, thw, ckpt_dir, variant)
                if residual_ae is not None:
                    corrupt = residual_ae
                else:
                    corrupt = dequant
            else:
                corrupt = dequant
        else:
            corrupt = dequant
    else:
        _record_corruption_warning(f"unsupported_mode_{mode}")
        return layer_cache

    metrics = _corruption_metrics(clean, corrupt, thw)
    _CORRUPTION_STATS["calls"] += 1
    for key in (
        "actual_rel_l2",
        "cosine",
        "l1",
        "dc_rel_l2",
        "ac_rel_l2",
        "ac_error_ratio",
    ):
        stat_key = "sum_" + key
        _CORRUPTION_STATS[stat_key] += metrics[key]
    _CORRUPTION_STATS["max_layer_rel_l2"] = max(
        float(_CORRUPTION_STATS["max_layer_rel_l2"]), metrics["actual_rel_l2"]
    )
    layer_row = _layer_accum(layer_idx)
    layer_row["calls"] += 1
    for key in (
        "actual_rel_l2",
        "cosine",
        "l1",
        "dc_rel_l2",
        "ac_rel_l2",
        "ac_error_ratio",
    ):
        layer_row["sum_" + key] += metrics[key]
    layer_row["max_actual_rel_l2"] = max(
        float(layer_row["max_actual_rel_l2"]), metrics["actual_rel_l2"]
    )
    layer_cache[name] = corrupt
    return layer_cache


_K_CORRUPTION_STATS: Dict[str, Any] = {
    "calls": 0,
    "warnings": {},
}


def _k_corruption_enabled() -> bool:
    return _env_flag("HY_KV_CORRUPTION_K_ENABLE", "0")


def _apply_k_corruption(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """Corrupt K-cache with a per-layer DCAE (separate from V corruption).

    Controlled by env vars prefixed with HY_KV_CORRUPTION_K_.
    V corruption is unaffected.
    """
    if not _k_corruption_enabled():
        return layer_cache
    mode = os.environ.get("HY_KV_CORRUPTION_K_MODE", "dcae").strip().lower()
    layers = _parse_layers(os.environ.get("HY_KV_CORRUPTION_K_LAYERS", ""))

    # Optional: keep early chunks clean. This is useful when the earliest
    # resident K cache is a small fraction of the full cache but strongly
    # affects later rollout stability.
    start_chunk = int(os.environ.get("HY_KV_CORRUPTION_K_START_CHUNK", "0"))
    current_chunk = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    if start_chunk > 0 and (current_chunk < 0 or current_chunk < start_chunk):
        return layer_cache
    bypass_chunks = _parse_layers(
        os.environ.get("HY_KV_CORRUPTION_K_BYPASS_CHUNKS", "")
    )
    if current_chunk in bypass_chunks:
        return layer_cache

    name = "k_vision"
    tensor = layer_cache.get(name)
    thw = layer_cache.get("vision_thw")
    if not isinstance(tensor, torch.Tensor):
        return layer_cache
    if thw is None:
        return layer_cache
    if tuple(tensor.shape)[-1] != 128 or tensor.ndim != 4:
        return layer_cache

    # Reset stats if this is the first call
    if _K_CORRUPTION_STATS["calls"] == 0:
        _K_CORRUPTION_STATS["mode"] = mode
        _K_CORRUPTION_STATS["layers"] = os.environ.get("HY_KV_CORRUPTION_K_LAYERS", "")
        _K_CORRUPTION_STATS["enabled"] = True

    clean = tensor
    # Clean-suffix training only instantiates codecs in the trainable window,
    # but Teacher/Student still need a detached clean historical K view at all
    # suffix layers. Retain that view before filtering the corrupted layers.
    if (
        _k_functional_train_enabled()
        or _env_flag("HY_K_REPLAY_DUMP_ENABLE", "0")
        or _env_flag("HY_FUNCTIONAL_BLOCK_FT_RETAIN_CLEAN_K_ALL_LAYERS", "0")
    ):
        layer_cache["_k_vision_clean_for_ft"] = clean.detach()
    if not _layer_enabled(layer_idx, layers):
        return layer_cache
    if mode == "dcae":
        recon = _layerwise_reconstruct_k(clean, layer_idx, thw)
        if recon is None:
            return layer_cache
        if _k_functional_train_enabled():
            layer_cache["_k_vision_base_for_ft"] = recon
        corrupt = recon
    else:
        return layer_cache

    metrics = _corruption_metrics(clean, corrupt, thw)
    _K_CORRUPTION_STATS["calls"] += 1
    for key in ("actual_rel_l2", "cosine", "l1", "dc_rel_l2", "ac_rel_l2", "ac_error_ratio"):
        stat_key = "sum_" + key
        _K_CORRUPTION_STATS.setdefault(stat_key, 0.0)
        _K_CORRUPTION_STATS[stat_key] += metrics[key]
    _K_CORRUPTION_STATS["max_layer_rel_l2"] = max(
        float(_K_CORRUPTION_STATS.get("max_layer_rel_l2", 0.0)), metrics["actual_rel_l2"]
    )

    layer_cache[name] = corrupt
    return layer_cache


def _layerwise_reconstruct_k(
    tensor: torch.Tensor,
    layer_idx: int,
    thw,
    *,
    track_grad: bool = False,
) -> Optional[torch.Tensor]:
    """Per-layer K-cache DCAE reconstruction (reads K-specific env vars)."""
    ckpt_dir = os.environ.get("HY_KV_CORRUPTION_K_DCAE_CKPT_DIR", "").strip()
    if not ckpt_dir:
        return None
    variant = os.environ.get("HY_KV_CORRUPTION_K_DCAE_MODEL_VARIANT", "dcae_d256_8x")
    layer_dir = Path(ckpt_dir) / f"layer_{int(layer_idx):02d}"

    # Optional immutable source snapshot. This is separate from the output
    # directory so A1-P1 can start from final A1 while saving candidates into a
    # fresh tree without accidentally falling back to the pre-A1 Base codec.
    source_ft_ckpt_dir = os.environ.get(
        "HY_K_FUNCTIONAL_FT_SOURCE_CKPT_DIR", ""
    ).strip()
    ft_ckpt_dir = os.environ.get("HY_K_FUNCTIONAL_FT_CKPT_DIR", "").strip()
    if source_ft_ckpt_dir:
        source_candidate = (
            Path(source_ft_ckpt_dir)
            / f"layer_{int(layer_idx):02d}"
            / "checkpoints"
            / "checkpoint_functional_ft.pt"
        )
        candidates = [source_candidate] if source_candidate.exists() else []
        if not candidates:
            raise FileNotFoundError(
                f"Missing immutable K functional source checkpoint: {source_candidate}"
            )
    elif ft_ckpt_dir:
        ft_candidate = Path(ft_ckpt_dir) / f"layer_{int(layer_idx):02d}" / "checkpoints" / "checkpoint_functional_ft.pt"
        if ft_candidate.exists():
            candidates = [ft_candidate]
        else:
            candidates = [
                layer_dir / "checkpoints" / "checkpoint_functional_ft.pt",
                layer_dir / "checkpoints" / "checkpoint_best.pt",
                layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_k_{variant}_step12000.pt",
                layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_v_{variant}_step12000.pt",
                layer_dir / "checkpoints" / "checkpoint_final.pt",
                layer_dir / "checkpoints" / "checkpoint_last.pt",
                layer_dir / "checkpoint_best.pt",
            ]
    else:
        candidates = [
            layer_dir / "checkpoints" / "checkpoint_functional_ft.pt",
            layer_dir / "checkpoints" / "checkpoint_best.pt",
            layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_k_{variant}_step12000.pt",
            layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_v_{variant}_step12000.pt",
            layer_dir / "checkpoints" / "checkpoint_final.pt",
            layer_dir / "checkpoints" / "checkpoint_last.pt",
            layer_dir / "checkpoint_best.pt",
        ]
    ckpt_path = next((path for path in candidates if path.exists()), None)
    if ckpt_path is None:
        return None
    key = f"k_corruption::{ckpt_path.resolve()}"
    if key not in _LAYERWISE_AE_MODELS:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt_data.get("model_config", {})
        model = build_pre_kv_ae(cfg)
        model.load_state_dict(ckpt_data["model"])
        model._hy_functional_layer_id = int(layer_idx)
        model._hy_functional_source_ckpt = str(ckpt_path)
        model._hy_functional_model_config = cfg
        model._hy_codec_metadata = dict(ckpt_data.get("codec_metadata", {}))
        trainable = _k_functional_layer_trainable(layer_idx)
        if trainable:
            model.train()
            trainable_parameters = _configure_k_functional_parameters(model, True)
            lr = float(os.environ.get("HY_KV_CORRUPTION_K_LR", "1e-5"))
            wd = float(os.environ.get("HY_KV_CORRUPTION_K_WEIGHT_DECAY", "0.0"))
            opt = _build_k_functional_optimizer(
                trainable_parameters, lr=lr, weight_decay=wd
            )
            optimizer_state = ckpt_data.get("optimizer")
            optimizer_type = ckpt_data.get("optimizer_type", "adamw")
            requested_optimizer_type = os.environ.get(
                "HY_KV_CORRUPTION_K_OPTIMIZER", "adamw"
            ).strip().lower()
            saved_optimizer_params = (
                sum(len(group.get("params", ())) for group in optimizer_state.get("param_groups", ()))
                if isinstance(optimizer_state, dict)
                else 0
            )
            if (
                optimizer_state
                and not _env_flag("HY_FUNCTIONAL_RESET_OPTIMIZER", "0")
                and optimizer_type == requested_optimizer_type
                and saved_optimizer_params == len(trainable_parameters)
            ):
                opt.load_state_dict(optimizer_state)
            _K_TRAIN_OPTIMS[key] = opt
        else:
            model.eval()
            _configure_k_functional_parameters(model, False)
        _LAYERWISE_AE_MODELS[key] = model
    fp32_master = _env_flag("HY_KV_CORRUPTION_K_FP32_MASTER", "0")
    model_dtype = torch.float32 if fp32_master else tensor.dtype
    model = _LAYERWISE_AE_MODELS[key].to(device=tensor.device, dtype=model_dtype)
    trainable = _k_functional_layer_trainable(layer_idx)
    if trainable:
        model.train()
        trainable_parameters = _configure_k_functional_parameters(model, True)
        if key not in _K_TRAIN_OPTIMS:
            lr = float(os.environ.get("HY_KV_CORRUPTION_K_LR", "1e-5"))
            wd = float(os.environ.get("HY_KV_CORRUPTION_K_WEIGHT_DECAY", "0.0"))
            _K_TRAIN_OPTIMS[key] = _build_k_functional_optimizer(
                trainable_parameters, lr=lr, weight_decay=wd
            )
        _optimizer_state_to_parameter_devices(_K_TRAIN_OPTIMS[key])
    else:
        model.eval()
        _configure_k_functional_parameters(model, False)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    start = time.perf_counter()
    build_graph = trainable and track_grad
    amp_enabled = bool(
        fp32_master
        and tensor.is_cuda
        and tensor.dtype in {torch.bfloat16, torch.float16}
        and not _env_flag("HY_KV_CORRUPTION_K_FORCE_FP32_LINEAR", "0")
    )
    amp_ctx = torch.autocast(
        device_type="cuda",
        dtype=tensor.dtype,
        enabled=amp_enabled,
    )
    semantic_centers = None
    semantic_ids = None
    codec_input = tensor
    if _env_flag("HY_KV_CORRUPTION_K_SEMANTIC_SMOOTH", "0"):
        # Reuse the resident codec's exact C64 implementation.  Cluster
        # assignment and BF16 centers are cache side information, not A1
        # parameters, and therefore remain detached during decoder-only FT.
        from .kv_codecs import _joint_semantic_smooth

        codec_input, semantic_centers, semantic_ids = _joint_semantic_smooth(
            tensor.detach(), int(layer_idx)
        )
    with torch.set_grad_enabled(build_graph), amp_ctx:
        z = model.encode(codec_input.contiguous(), thw)
        encoder_output = z
        if _env_flag("HY_KV_CORRUPTION_K_LATENT_INT4_TOKEN", "0"):
            z_codes, z_scale = _quantize_latent_int4_token(z)
            # The deployed PCA codec stores its per-token scale in BF16.
            z_scale = z_scale.to(torch.bfloat16)
            z_deq = _dequantize_latent_int4_token(z_codes, z_scale, z.dtype)
            z = z + (z_deq - z).detach()
        elif _env_flag("HY_KV_CORRUPTION_K_LATENT_INT4", "0"):
            z_codes, z_scale = _quantize_latent_int4(z)
            z = _dequantize_latent_int4(z_codes, z_scale, z.dtype)
        elif _env_flag("HY_KV_CORRUPTION_K_LATENT_INT8", "0"):
            z_codes, z_scale = _quantize_latent_int8(z)
            z = _dequantize_latent_int8(z_codes, z_scale, z.dtype)
        functional_latent = z
    if trainable:
        _offload_k_frozen_encoder(model)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    _STATS["k_corruption_encoder_time_s"] += time.perf_counter() - start

    start = time.perf_counter()
    amp_ctx = torch.autocast(
        device_type="cuda",
        dtype=tensor.dtype,
        enabled=amp_enabled,
    )
    output_shape = tuple(tensor.shape)
    output_thw = tuple(int(v) for v in thw)
    with torch.set_grad_enabled(build_graph), amp_ctx:
        if build_graph and _env_flag("HY_K_T0_CODEC_CHECKPOINT", "0"):
            decoded = activation_checkpoint(
                lambda latent: model.decode(latent, output_shape, output_thw),
                z,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            decoded = model.decode(z, output_shape, output_thw)
        out = decoded
        if semantic_centers is not None:
            batch, heads, sequence, dim = tensor.shape
            assigned = torch.gather(
                semantic_centers.to(device=out.device, dtype=torch.float32),
                1,
                semantic_ids.to(device=out.device, dtype=torch.long)[..., None].expand(
                    -1, -1, heads * dim
                ),
            )
            assigned = (
                assigned.reshape(batch, sequence, heads, dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            # Resident decode adds the BF16 center in FP32 and casts only the
            # final reconstructed K. Keep the same order for a true zero-delta
            # initialization rather than introducing an extra BF16 rounding.
            out = out.float() + assigned.float()
        out = out.to(tensor.dtype)
        if build_graph:
            out = _apply_functional_decoder_delta_to_production_base(
                out, functional_latent, model, "k", int(layer_idx)
            )
        if track_grad and _env_flag("HY_FUNCTIONAL_FINAL_PRED_GRAD_AUDIT", "0"):
            _V_FINAL_PRED_TENSOR_AUDIT[int(layer_idx)] = {
                "encoder_output": encoder_output,
                "quantized_latent": functional_latent,
                "decoder_output": out,
            }
    if fp32_master and trainable:
        bad_dtypes = {
            str(param.dtype) for param in model.parameters() if param.requires_grad
        } - {"torch.float32"}
        if bad_dtypes:
            raise RuntimeError(
                f"K FP32-master mode has non-fp32 trainable parameters: {bad_dtypes}"
            )
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    _STATS["k_corruption_decoder_time_s"] += time.perf_counter() - start
    _STATS["k_corruption_calls"] += 1
    return out


def dump_pre_kv(layer_idx: int, tensor_name: str, tensor: torch.Tensor, thw) -> None:
    global _DUMP_COUNT
    if not _env_flag("HY_KV_AE3D_DUMP_ENABLE", "0"):
        return
    layers = _parse_layers(os.environ.get("HY_KV_AE3D_DUMP_LAYERS", "0,13"))
    if not _layer_enabled(layer_idx, layers):
        return
    max_cases = int(os.environ.get("HY_KV_AE3D_DUMP_MAX_CASES", "32"))
    if _DUMP_COUNT >= max_cases:
        return
    out_dir = Path(os.environ.get("HY_KV_AE3D_DUMP_DIR", "kv_ae_outputs/ae3d/dump"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{_DUMP_COUNT:04d}_layer{layer_idx}_{tensor_name}.pt"
    torch.save(
        {
            "layer_idx": int(layer_idx),
            "tensor_name": tensor_name,
            "shape": tuple(tensor.shape),
            "thw": tuple(int(v) for v in thw),
            "tensor": tensor.detach().to("cpu", dtype=torch.float16),
        },
        path,
    )
    _DUMP_COUNT += 1


def _multipos_quantize_int8_payload(metadata: Dict[str, Any], tensor: torch.Tensor, save_dtype):
    x = tensor.detach().to("cpu", dtype=save_dtype)
    work = x.float()
    eps = torch.finfo(torch.float32).eps
    scale = work.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    codes = (work / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return {
        **metadata,
        "storage_format": "int8_symmetric",
        "quantization": {
            "bits": 8,
            "scheme": "symmetric_signed",
            "granularity": "per_token",
            "qmin": -127,
            "qmax": 127,
            "scale_dtype": "bf16",
        },
        "original_dtype": str(save_dtype).replace("torch.", ""),
        "tensor_int8": codes,
        "int8_scale": scale.to(torch.bfloat16).contiguous(),
    }


def dump_multipos_pre_v(layer_idx: int, tensor_name: str, tensor: torch.Tensor, thw) -> None:
    if not _env_flag("HY_KV_MULTIPOS_DUMP_ENABLE", "0"):
        return
    if os.environ.get("HY_KV_MULTIPOS_DUMP_PASS", "pos") != "pos":
        return

    layers = _parse_layers(os.environ.get("HY_KV_MULTIPOS_DUMP_LAYERS", "13,27,45"))
    if not _layer_enabled(layer_idx, layers):
        return

    chunk_id = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    positions = {
        int(item.strip())
        for item in os.environ.get("HY_KV_MULTIPOS_DUMP_POSITIONS", "4").split(",")
        if item.strip()
    }
    if chunk_id not in positions:
        return

    target = os.environ.get("HY_KV_MULTIPOS_TARGET", "v").lower()
    if target == "kv":
        if tensor_name not in {"k_vision", "v_vision"}:
            return
        one_target = "k" if tensor_name == "k_vision" else "v"
        expected_tensor_name = tensor_name
    else:
        expected_tensor_name = "k_vision" if target == "k" else "v_vision"
        one_target = target
    if tensor_name != expected_tensor_name:
        return
    branch = os.environ.get("HY_KV_MULTIPOS_BRANCH", "main_rope")
    kv_stage = os.environ.get("HY_KV_MULTIPOS_KV_STAGE", "pre")
    if target not in {"k", "v", "kv"} or branch != "main_rope" or kv_stage != "pre":
        raise RuntimeError(
            "HY_KV_MULTIPOS only supports target=k/v/kv, branch=main_rope, kv_stage=pre"
        )

    dtype_name = os.environ.get("HY_KV_MULTIPOS_DTYPE", "bf16").lower()
    if dtype_name == "bf16":
        save_dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        save_dtype = torch.float16
    else:
        raise RuntimeError(f"unsupported HY_KV_MULTIPOS_DTYPE={dtype_name}")

    storage_format = os.environ.get("HY_KV_MULTIPOS_STORAGE_FORMAT", "bf16").lower()
    if storage_format not in {"bf16", "int8"}:
        raise RuntimeError(f"unsupported HY_KV_MULTIPOS_STORAGE_FORMAT={storage_format}")

    out_dir = Path(os.environ.get("HY_KV_MULTIPOS_DUMP_DIR", "kv_ae_outputs/ae16x_v2/dataset/demo_multipos_5chunk_27layer_pre_v_bf16"))
    split = os.environ.get("HY_KV_MULTIPOS_SPLIT", "unknown")
    case_id = os.environ.get("HY_KV_MULTIPOS_CASE_ID", "case_unknown")
    position_dir = out_dir / split / case_id / f"position_{chunk_id:02d}" / "kv"
    position_dir.mkdir(parents=True, exist_ok=True)

    path = (
        position_dir
        / f"case-{case_id}_position{chunk_id:02d}_layer{int(layer_idx):02d}_"
        f"target-{one_target}_branch-main_rope_kvstage-pre_{expected_tensor_name}.pt"
    )
    if path.exists() and not _env_flag("HY_KV_MULTIPOS_FORCE", "0"):
        return

    selected_frame_indices = json.loads(
        os.environ.get("HY_KV_MULTIPOS_SELECTED_FRAME_INDICES", "[]")
    )
    metadata = {
        "format_version": 1,
        "case_id": case_id,
        "split": split,
        "video_id": os.environ.get("HY_KV_MULTIPOS_VIDEO_ID", case_id),
        "seed": int(os.environ.get("HY_KV_MULTIPOS_SEED", "-1")),
        "num_frames": int(os.environ.get("HY_KV_MULTIPOS_NUM_FRAMES", "-1")),
        "chunk_position": chunk_id,
        "requested_position": chunk_id,
        "selected_frame_indices": selected_frame_indices,
        "actual_context_start": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_START", "-1")),
        "actual_context_end": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_END", "-1")),
        "actual_context_len": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_LEN", "-1")),
        "actual_context_latent_count": len(selected_frame_indices),
        "expected_context_len": int(os.environ.get("HY_KV_MULTIPOS_EXPECTED_CONTEXT_LEN", "5")),
        "layer_idx": int(layer_idx),
        "tensor_name": expected_tensor_name,
        "target": one_target,
        "requested_target": target,
        "branch": "main_rope",
        "kv_stage": "pre",
        "dtype": dtype_name,
        "storage_format": storage_format,
        "source_device": str(tensor.device),
        "shape": tuple(int(v) for v in tensor.shape),
        "thw": tuple(int(v) for v in thw),
        "dump_hook_name": "dump_multipos_pre_kv",
        "dump_module_path": "hyvideo.models.transformers.modules.kv_codec_training",
        "tensor_source": f"_kv_cache_new[layer]['{expected_tensor_name}'] after HY_PRE_POSITION_VISION_KV_CACHE remaps it from vision_kv['{expected_tensor_name}_pre_position']",
        "notes": [
            f"{one_target.upper()} dump from requested target={target}.",
            "main_rope branch only: saved before RoPE/ProPE positional transforms.",
            "No post-KV, attention output, layer output, decoded latent, or ProPE branch is saved.",
        ],
    }
    if storage_format == "int8":
        payload = _multipos_quantize_int8_payload(metadata, tensor, save_dtype)
    else:
        payload = {**metadata, "tensor": tensor.detach().to("cpu", dtype=save_dtype)}
    torch.save(payload, path)


def dump_qk_subset_pre_position(
    layer_idx: int,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    thw,
) -> None:
    if not _env_flag("HY_KV_QK_SUBSET_DUMP_ENABLE", "0"):
        return

    layers = _parse_layers(os.environ.get("HY_KV_QK_SUBSET_LAYERS", "13,45"))
    if not _layer_enabled(layer_idx, layers):
        return

    chunk_id = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    positions = {
        int(item.strip())
        for item in os.environ.get("HY_KV_QK_SUBSET_POSITIONS", "5,8,11,15").split(",")
        if item.strip()
    }
    if chunk_id not in positions:
        return

    dtype_name = os.environ.get("HY_KV_QK_SUBSET_DTYPE", "bf16").lower()
    if dtype_name == "bf16":
        save_dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        save_dtype = torch.float16
    elif dtype_name == "int8":
        save_dtype = torch.bfloat16
    else:
        raise RuntimeError(f"unsupported HY_KV_QK_SUBSET_DTYPE={dtype_name}")
    dump_targets = {
        item.strip().lower()
        for item in os.environ.get("HY_KV_QK_SUBSET_TARGETS", "q,k").split(",")
        if item.strip()
    }

    out_dir = Path(
        os.environ.get(
            "HY_KV_QK_SUBSET_DUMP_DIR",
            "kv_ae_outputs/ae16x_v2/dataset_qk_subset_attn_l13_l45_bf16",
        )
    )
    split = os.environ.get("HY_KV_QK_SUBSET_SPLIT", "unknown")
    case_id = os.environ.get("HY_KV_QK_SUBSET_CASE_ID", "case_unknown")
    position_dir = out_dir / split / case_id / f"position_{chunk_id:02d}" / "kv"
    position_dir.mkdir(parents=True, exist_ok=True)

    base = (
        f"case-{case_id}_position{chunk_id:02d}_layer{int(layer_idx):02d}_"
        "target-qk_branch-main_rope_kvstage-pre"
    )
    q_path = position_dir / f"{base}_q_vision.pt"
    k_path = position_dir / f"{base}_k_vision.pt"
    if (
        q_path.exists()
        and k_path.exists()
        and not _env_flag("HY_KV_QK_SUBSET_FORCE", "0")
    ):
        return

    selected_frame_indices = json.loads(
        os.environ.get("HY_KV_MULTIPOS_SELECTED_FRAME_INDICES", "[]")
    )
    source_v_map = json.loads(os.environ.get("HY_KV_QK_SUBSET_SOURCE_V_MAP", "{}"))
    source_v_path = source_v_map.get(str(int(layer_idx)), source_v_map.get(int(layer_idx), ""))
    common = {
        "format_version": 1,
        "case_id": case_id,
        "split": split,
        "video_id": os.environ.get("HY_KV_QK_SUBSET_VIDEO_ID", case_id),
        "seed": int(os.environ.get("HY_KV_QK_SUBSET_SEED", "-1")),
        "num_frames": int(os.environ.get("HY_KV_QK_SUBSET_NUM_FRAMES", "-1")),
        "chunk_position": chunk_id,
        "requested_position": chunk_id,
        "selected_frame_indices": selected_frame_indices,
        "actual_context_start": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_START", "-1")),
        "actual_context_end": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_END", "-1")),
        "actual_context_len": int(os.environ.get("HY_KV_MULTIPOS_CONTEXT_LEN", "-1")),
        "actual_context_latent_count": len(selected_frame_indices),
        "expected_context_len": int(os.environ.get("HY_KV_MULTIPOS_EXPECTED_CONTEXT_LEN", "5")),
        "layer_idx": int(layer_idx),
        "branch": "main_rope",
        "kv_stage": "pre",
        "dtype": dtype_name,
        "source_device": str(q_tensor.device),
        "thw": tuple(int(v) for v in thw),
        "dump_hook_name": "dump_qk_subset_pre_position",
        "dump_module_path": "hyvideo.models.transformers.modules.kv_codec_training",
        "tensor_source": "img_q/img_k before RoPE/ProPE positional transforms in forward_vision",
        "source_v_path": source_v_path,
        "notes": [
            "Q/K-only subset dump for sampled attention-output loss.",
            "main_rope branch only: saved from img_q/img_k before RoPE/ProPE positional transforms.",
            "No V tensor is saved by this hook.",
        ],
    }
    q_save = q_tensor.transpose(1, 2).contiguous()
    k_save = k_tensor.transpose(1, 2).contiguous()
    def _build_payload(tensor: torch.Tensor, tensor_name: str, target: str):
        base_payload = {
            **common,
            "tensor_name": tensor_name,
            "target": target,
            "shape": tuple(int(v) for v in tensor.shape),
        }
        tensor_cpu = tensor.detach().to("cpu", dtype=save_dtype)
        if dtype_name == "int8":
            return _multipos_quantize_int8_payload(base_payload, tensor_cpu, save_dtype)
        return {**base_payload, "tensor": tensor_cpu}

    if "q" in dump_targets:
        torch.save(_build_payload(q_save, "q_vision", "q"), q_path)
    if "k" in dump_targets:
        torch.save(_build_payload(k_save, "k_vision", "k"), k_path)


def _sample_indices(length: int, count: int, device) -> torch.Tensor:
    count = min(int(count), int(length))
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == length:
        return torch.arange(length, dtype=torch.long, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().long()


def _select_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Input is [B, S, H, D]; output is [B, H, sampled_S, D].
    return x.index_select(1, indices).permute(0, 2, 1, 3).contiguous()


def _select_sequence_metadata(x: Optional[torch.Tensor], indices: torch.Tensor):
    if x is None:
        return None
    sequence_dim = 1 if x.ndim >= 3 and x.shape[0] == 1 else 0
    return x.index_select(sequence_dim, indices.to(x.device)).detach().to(
        "cpu", dtype=torch.float16
    )


def _select_hidden(x: Optional[torch.Tensor], indices: torch.Tensor):
    if x is None:
        return None
    if x.ndim == 2:
        return x.index_select(0, indices.to(x.device)).detach().to(
            "cpu", dtype=torch.float16
        )
    return x.index_select(1, indices.to(x.device)).detach().to(
        "cpu", dtype=torch.float16
    )


def dump_attention_sample(
    layer_idx: int,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    v_main: torch.Tensor,
    q_prope: torch.Tensor,
    k_prope: torch.Tensor,
    v_prope: torch.Tensor,
    attn_main: torch.Tensor,
    attn_prope: torch.Tensor,
    thw,
    freqs_cis=None,
    viewmats: Optional[torch.Tensor] = None,
    Ks: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
    layer_output: Optional[torch.Tensor] = None,
    vec: Optional[torch.Tensor] = None,
    text_k: Optional[torch.Tensor] = None,
    text_v: Optional[torch.Tensor] = None,
    attn_prope_restored: Optional[torch.Tensor] = None,
) -> None:
    global _ATTN_DUMP_COUNT, _ATTN_DUMP_KEYS
    if not _env_flag("HY_KV_AE3D_ATTN_DUMP_ENABLE", "0"):
        return
    layers = _parse_layers(os.environ.get("HY_KV_AE3D_ATTN_DUMP_LAYERS", "6,20,33,47"))
    if not _layer_enabled(layer_idx, layers):
        return
    max_cases = int(os.environ.get("HY_KV_AE3D_ATTN_DUMP_MAX_CASES", "64"))
    if _ATTN_DUMP_COUNT >= max_cases:
        return

    out_dir = Path(os.environ.get("HY_KV_AE3D_ATTN_DUMP_DIR", "kv_ae_outputs/ae16x_v2/attention_samples"))
    out_dir.mkdir(parents=True, exist_ok=True)
    q_count = int(os.environ.get("HY_KV_AE3D_ATTN_DUMP_Q_TOKENS", "256"))
    k_count = int(os.environ.get("HY_KV_AE3D_ATTN_DUMP_K_TOKENS", "1024"))
    q_idx = _sample_indices(q_rope.shape[1], q_count, q_rope.device)
    k_idx = _sample_indices(k_rope.shape[1], k_count, k_rope.device)

    freqs_cos, freqs_sin = (freqs_cis if freqs_cis is not None else (None, None))
    selected_frame_indices = json.loads(
        os.environ.get("HY_KV_AE3D_SELECTED_FRAME_INDICES", "[]")
    )
    case_id = os.environ.get("HY_KV_AE3D_CASE_ID", "unknown")
    split = os.environ.get("HY_KV_AE3D_SPLIT", "unknown")
    ar_chunk_id = int(os.environ.get("HY_KV_AE3D_AR_CHUNK_ID", "-1"))
    requested_chunks = os.environ.get(
        "HY_KV_AE3D_ATTN_DUMP_AR_CHUNKS", ""
    ).strip()
    if requested_chunks:
        allowed_chunks = {
            int(item) for item in requested_chunks.split(",") if item.strip()
        }
        if ar_chunk_id not in allowed_chunks:
            return
    if _env_flag("HY_KV_AE3D_ATTN_DUMP_V_REFIT_ONLY", "0"):
        dump_key = (case_id, ar_chunk_id, int(layer_idx))
        if dump_key in _ATTN_DUMP_KEYS:
            return
        _ATTN_DUMP_KEYS.add(dump_key)
    pre_kv_case_dir = os.environ.get("HY_KV_AE3D_PRE_KV_CASE_DIR", "")

    path = out_dir / f"case{_ATTN_DUMP_COUNT:04d}_layer{layer_idx}_attn_sample.pt"
    if _env_flag("HY_KV_AE3D_ATTN_DUMP_V_REFIT_ONLY", "0"):
        # Lightweight calibration payload for the all-layer V residual-tail
        # refit.  The refit only needs deployment-distribution pre-position V
        # rows; retaining Q/K, full-context V, and block replay state would
        # multiply storage without changing the fitted residual subspace.
        torch.save({
            "format_version": 3,
            "dump_kind": "v_refit_only",
            "dump_index": int(_ATTN_DUMP_COUNT),
            "case_id": case_id,
            "split": split,
            "ar_chunk_id": ar_chunk_id,
            "selected_frame_indices": selected_frame_indices,
            "layer_idx": int(layer_idx),
            "k_indices": k_idx.detach().cpu(),
            "v_main": _select_tokens(v_main, k_idx).detach().to(
                "cpu", dtype=torch.float16
            ),
        }, path)
        _ATTN_DUMP_COUNT += 1
        return
    payload = {
            "format_version": 2,
            "dump_index": int(_ATTN_DUMP_COUNT),
            "case_id": case_id,
            "split": split,
            "ar_chunk_id": ar_chunk_id,
            "selected_frame_indices": selected_frame_indices,
            "pre_kv_case_dir": pre_kv_case_dir,
            "alignment_scope": "first_recache_context",
            "layer_idx": int(layer_idx),
            "shape": {
                "q_rope": tuple(q_rope.shape),
                "k_rope": tuple(k_rope.shape),
                "v_main": tuple(v_main.shape),
                "q_prope": tuple(q_prope.shape),
                "k_prope": tuple(k_prope.shape),
                "v_prope": tuple(v_prope.shape),
                "attn_main": tuple(attn_main.shape),
                "attn_prope": tuple(attn_prope.shape),
            },
            "thw": tuple(int(v) for v in thw),
            "q_indices": q_idx.detach().cpu(),
            "k_indices": k_idx.detach().cpu(),
            "freqs_cos_q": _select_sequence_metadata(freqs_cos, q_idx),
            "freqs_sin_q": _select_sequence_metadata(freqs_sin, q_idx),
            "freqs_cos_k": _select_sequence_metadata(freqs_cos, k_idx),
            "freqs_sin_k": _select_sequence_metadata(freqs_sin, k_idx),
            "viewmats_q": _select_sequence_metadata(viewmats, q_idx),
            "viewmats_k": _select_sequence_metadata(viewmats, k_idx),
            "Ks_q": _select_sequence_metadata(Ks, q_idx),
            "Ks_k": _select_sequence_metadata(Ks, k_idx),
            "layer_input": _select_hidden(layer_input, q_idx),
            "layer_output": _select_hidden(layer_output, q_idx),
            "vec": _select_hidden(vec, q_idx),
            "text_k": text_k.detach().to("cpu", dtype=torch.float16)
            if text_k is not None
            else None,
            "text_v": text_v.detach().to("cpu", dtype=torch.float16)
            if text_v is not None
            else None,
            "q_rope": _select_tokens(q_rope, q_idx).detach().to("cpu", dtype=torch.float16),
            "k_rope": _select_tokens(k_rope, k_idx).detach().to("cpu", dtype=torch.float16),
            "v_main": _select_tokens(v_main, k_idx).detach().to("cpu", dtype=torch.float16),
            "q_prope": _select_tokens(q_prope, q_idx).detach().to("cpu", dtype=torch.float16),
            "k_prope": _select_tokens(k_prope, k_idx).detach().to("cpu", dtype=torch.float16),
            "v_prope": _select_tokens(v_prope, k_idx).detach().to("cpu", dtype=torch.float16),
            "attn_main": _select_tokens(
                attn_main.reshape(attn_main.shape[0], attn_main.shape[1], -1, q_rope.shape[-1]),
                q_idx,
            ).detach().to("cpu", dtype=torch.float16),
            "attn_prope": _select_tokens(
                attn_prope.reshape(attn_prope.shape[0], attn_prope.shape[1], -1, q_rope.shape[-1]),
                q_idx,
            ).detach().to("cpu", dtype=torch.float16),
            "attn_prope_restored": _select_hidden(attn_prope_restored, q_idx),
        }
    if _env_flag("HY_KV_AE3D_ATTN_DUMP_FULL_KV", "0"):
        # Diagnostic-only payload used by axis-placement functional oracles.
        # Keep the normal sampled dump as the default because full context K/V
        # can be tens of MB per layer.
        payload["k_rope_full"] = k_rope.detach().to("cpu", dtype=torch.float16)
        payload["k_prope_full"] = k_prope.detach().to("cpu", dtype=torch.float16)
        payload["v_main_full"] = v_main.detach().to("cpu", dtype=torch.float16)
        payload["v_prope_full"] = v_prope.detach().to("cpu", dtype=torch.float16)
        # Preserve the compact camera-level metadata so a reconstructed
        # pre-position V can be passed through the exact PRoPE KV transform.
        payload["viewmats_full"] = (
            viewmats.detach().to("cpu") if viewmats is not None else None
        )
        payload["Ks_full"] = Ks.detach().to("cpu") if Ks is not None else None
    torch.save(payload, path)
    _ATTN_DUMP_COUNT += 1


@torch.no_grad()
def _compress_tensor(tensor: torch.Tensor, layer_idx: int, tensor_name: str, thw):
    checkpoint_path = _checkpoint_path(layer_idx, tensor_name)
    model = _model(tensor.device, tensor.dtype, checkpoint_path)
    normalization = _normalization(layer_idx, tensor_name)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    start = time.perf_counter()
    residual_codec = bool(getattr(model, "is_residual_codec", False))
    residual_payload = {}
    if residual_codec:
        if tensor_name != "v_vision":
            raise ValueError("residual_int2_dcae_d128_8x supports V only")
        from hyvideo.models.transformers.modules.kv_codecs import (
            dequantize_kv_tensor,
            pack_quantized_codes,
            quantize_kv_tensor,
        )

        bits = int(
            os.environ.get(
                "HY_PREKV_AE_RESIDUAL_BASE_BITS",
                os.environ.get("HY_KV_AE3D_RESIDUAL_BASE_BITS", "2"),
            )
        )
        granularity = os.environ.get(
            "HY_PREKV_AE_RESIDUAL_BASE_GRANULARITY",
            os.environ.get("HY_KV_AE3D_RESIDUAL_BASE_GRANULARITY", "per_token"),
        )
        codes, scale = quantize_kv_tensor(tensor, bits, granularity)
        scale_cache = scale.to(dtype=tensor.dtype)
        base = dequantize_kv_tensor(codes, scale_cache, tensor.dtype)
        packed_codes = pack_quantized_codes(codes, bits)
        residual = tensor - base
        residual_std = normalization[1] if normalization is not None else None
        model_input = residual / residual_std if residual_std is not None else residual
        residual_payload = {
            "residual_codec": True,
            "base_packed_codes": packed_codes,
            "base_numel": int(tensor.numel()),
            "base_shape": tuple(int(value) for value in tensor.shape),
            "base_scale": scale_cache,
            "base_bits": bits,
            "base_granularity": granularity,
            "residual_std": residual_std,
        }
    else:
        model_input = tensor
        if normalization is not None:
            mean, std = normalization
            model_input = (tensor - mean) / std
    z = model.encode(model_input.contiguous(), thw)
    if tensor.is_cuda and _timing_profile_enabled():
        torch.cuda.synchronize(tensor.device)
    _STATS["encoder_time_s"] += time.perf_counter() - start
    _STATS["compress_calls"] += 1
    latent_dtype = os.environ.get("HY_KV_AE3D_LATENT_DTYPE", "bf16").lower()
    if latent_dtype in {"bf16", "bfloat16"}:
        latent = z.to(torch.bfloat16)
    elif latent_dtype in {"fp16", "float16"}:
        latent = z.to(torch.float16)
    else:
        latent = z
    payload = {
        "kv_codec_training": True,
        "layer_idx": int(layer_idx),
        "tensor_name": tensor_name,
        "shape": tuple(int(v) for v in tensor.shape),
        "thw": tuple(int(v) for v in thw),
        "latent": latent,
        "latent_dtype": str(latent.dtype).replace("torch.", ""),
        "checkpoint_path": str(checkpoint_path),
        "normalization": None if residual_codec else normalization,
    }
    payload.update(residual_payload)
    if residual_codec:
        scale_bits = payload["base_scale"].numel() * 16
        latent_bits = latent.numel() * latent.element_size() * 8
        actual_bytes = _tensor_bytes(payload["base_packed_codes"]) + _tensor_bytes(
            payload["base_scale"]
        ) + _tensor_bytes(latent)
        payload["estimated_codec_bytes"] = actual_bytes
        payload["estimated_bits_per_value"] = actual_bytes * 8 / tensor.numel()
        _STATS["residual_codec_calls"] += 1
        _STATS["residual_base_bits"] = payload["base_bits"]
        _STATS["residual_base_granularity"] = payload["base_granularity"]
    return payload


def _quantize_latent_int8(z: torch.Tensor):
    scale = z.float().abs().amax(dim=tuple(range(1, z.ndim)), keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    ) / 127.0
    codes = (z.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return codes, scale.to(dtype=z.dtype)


def _dequantize_latent_int8(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.float()).to(dtype=dtype)


def _quantize_latent_int8_token(z: torch.Tensor):
    if z.ndim >= 2:
        scale = z.float().abs().amax(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 127.0
    else:
        scale = z.float().abs().amax().clamp_min(torch.finfo(torch.float32).eps) / 127.0
    codes = (z.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return codes, scale.to(dtype=z.dtype)


def _dequantize_latent_int8_token(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.float()).to(dtype=dtype)


def _quantize_latent_int4(z: torch.Tensor):
    scale = z.float().abs().amax(dim=tuple(range(1, z.ndim)), keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    ) / 7.0
    codes = (z.float() / scale).round().clamp(-7, 7).to(torch.int8)
    return codes, scale.to(dtype=z.dtype)


def _dequantize_latent_int4(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.float()).to(dtype=dtype)


def _quantize_latent_int4_token(z: torch.Tensor):
    if z.ndim >= 2:
        scale = z.float().abs().amax(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / 7.0
    else:
        scale = z.float().abs().amax().clamp_min(torch.finfo(torch.float32).eps) / 7.0
    codes = (z.float() / scale).round().clamp(-7, 7).to(torch.int8)
    return codes, scale.to(dtype=z.dtype)


def _dequantize_latent_int4_token(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.float()).to(dtype=dtype)


def _layerwise_target_settings(target: str, default_ckpt_dir: str, default_variant: str) -> Tuple[str, str]:
    prefix = "HY_LAYERWISE_RAW_K_AE" if target == "k" else "HY_LAYERWISE_RAW_V_AE"
    ckpt_dir = os.environ.get(f"{prefix}_CKPT_DIR", "").strip()
    if not ckpt_dir:
        ckpt_dir = default_ckpt_dir
    variant = os.environ.get(f"{prefix}_MODEL_VARIANT", "").strip()
    if not variant:
        variant = default_variant
    return ckpt_dir, variant


def _layerwise_target_latent_int8(target: str, int8_code_ae: bool) -> bool:
    prefix = "HY_LAYERWISE_RAW_K_AE" if target == "k" else "HY_LAYERWISE_RAW_V_AE"
    raw = os.environ.get(f"{prefix}_LATENT_INT8", "").strip()
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    return int8_code_ae and _env_flag("HY_LAYERWISE_RAW_V_AE_LATENT_INT8", "0")


def _layerwise_target_latent_int8_token(target: str) -> bool:
    prefix = "HY_LAYERWISE_RAW_K_AE" if target == "k" else "HY_LAYERWISE_RAW_V_AE"
    raw = os.environ.get(f"{prefix}_LATENT_INT8_TOKEN", "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def _layerwise_target_latent_int4(target: str) -> bool:
    prefix = "HY_LAYERWISE_RAW_K_AE" if target == "k" else "HY_LAYERWISE_RAW_V_AE"
    raw = os.environ.get(f"{prefix}_LATENT_INT4", "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def _layerwise_target_latent_int4_token(target: str) -> bool:
    prefix = "HY_LAYERWISE_RAW_K_AE" if target == "k" else "HY_LAYERWISE_RAW_V_AE"
    raw = os.environ.get(f"{prefix}_LATENT_INT4_TOKEN", "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def _compress_one_layerwise_target(
    layer_cache: Dict[str, Any],
    layer_idx: int,
    target: str,
    default_ckpt_dir: str,
    default_variant: str,
) -> Dict[str, Any]:
    name = "k_vision" if target == "k" else "v_vision"
    tensor = layer_cache.get(name)
    if not isinstance(tensor, torch.Tensor):
        return layer_cache

    ckpt_dir, default_variant = _layerwise_target_settings(
        target, default_ckpt_dir, default_variant
    )
    if not ckpt_dir:
        return layer_cache

    variants = [default_variant]
    layer_var_env = f"HY_LAYERWISE_RAW_{target.upper()}_AE_MODEL_VARIANT_LAYER_{layer_idx:02d}"
    layer_var = os.environ.get(layer_var_env, "").strip()
    if not layer_var and target == "v":
        layer_var = os.environ.get(
            f"HY_LAYERWISE_RAW_V_AE_MODEL_VARIANT_LAYER_{layer_idx:02d}", ""
        ).strip()
    if layer_var:
        variants = [layer_var]
    elif target == "v":
        variants.append("dcae_d128_spatial4x")
        variants.append("dcae_d128_temporal2x")

    ckpt_path = None
    used_variant = variants[0]
    if target == "v":
        ft_ckpt_dir = os.environ.get("HY_V_FUNCTIONAL_FT_CKPT_DIR", "").strip()
        if ft_ckpt_dir:
            ft_candidate = (
                Path(ft_ckpt_dir)
                / f"layer_{int(layer_idx):02d}"
                / "checkpoints"
                / "checkpoint_functional_ft.pt"
            )
            if ft_candidate.exists():
                ckpt_path = ft_candidate
    elif target == "k":
        ft_ckpt_dir = os.environ.get("HY_K_FUNCTIONAL_FT_CKPT_DIR", "").strip()
        if ft_ckpt_dir:
            ft_candidate = (
                Path(ft_ckpt_dir)
                / f"layer_{int(layer_idx):02d}"
                / "checkpoints"
                / "checkpoint_functional_ft.pt"
            )
            if ft_candidate.exists():
                ckpt_path = ft_candidate
    for v in variants:
        if ckpt_path is not None:
            break
        p = _layerwise_ckpt_path(ckpt_dir, layer_idx, v)
        if p is not None:
            ckpt_path = p
            used_variant = v
            break
    if ckpt_path is None:
        return layer_cache

    try:
        if (
            target == "v"
            and _env_flag("HY_FUNCTIONAL_BLOCK_FT_RETAIN_CLEAN_V_ALL_LAYERS", "0")
        ):
            # Clean-suffix final-prediction training consumes the clean historical
            # V directly outside the trainable window. It is an immutable input;
            # never retain the cache-build graph.
            layer_cache["_v_vision_clean_for_ft"] = tensor.detach()
        if (
            target == "v"
            and _v_functional_train_enabled()
            and _v_functional_layer_trainable(layer_idx)
        ):
            thw = layer_cache.get("vision_thw")
            if thw is None:
                return layer_cache
            layer_cache["_v_vision_clean_for_ft"] = tensor.detach()
            recon = _layerwise_reconstruct_v_for_ft(tensor, layer_idx, thw)
            if recon is not None:
                layer_cache["_v_vision_base_for_ft"] = recon
                layer_cache[name] = recon
            return layer_cache

        model_key = (
            f"layerwise::{Path(str(ckpt_dir)).resolve()}::"
            f"{used_variant}_layer_{layer_idx:02d}"
        )
        cache_models = _env_flag("HY_KV_EFF_OPT_MODEL_CACHE", "0")
        if cache_models and model_key in _LAYERWISE_AE_MODELS:
            model = _LAYERWISE_AE_MODELS[model_key]
            cfg = getattr(model, "_hy_kv_eff_model_config", {})
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt.get("model_config", ckpt.get("config", {}))
            model = build_pre_kv_ae(cfg)
            model.load_state_dict(ckpt["model"])
            model.to(device=tensor.device, dtype=tensor.dtype)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            model._hy_kv_eff_model_config = cfg
            if cache_models:
                _LAYERWISE_AE_MODELS[model_key] = model

        thw = layer_cache.get("vision_thw")
        if thw is None:
            return layer_cache

        start = time.perf_counter()
        int8_code_ae = bool(getattr(model, "is_int8_code_ae", False)) or str(
            cfg.get("architecture", "")
        ).startswith("v_int8_code")
        if int8_code_ae:
            codes, scale = _quantize_per_token_int8_for_code_ae(tensor)
            model_input = _normalized_code_from_int8(codes, tensor.dtype)
            z = model.encode(model_input, thw)
        else:
            scale = None
            z = model.encode(tensor, thw)
        decode_prefix_hoisted = (
            target == "k"
            and _env_flag("HY_KV_EFF_OPT_K_DECODE_PREFIX_HOIST", "0")
            and hasattr(model, "prepare_decode_latent")
            and hasattr(model, "decode_prepared")
        )
        if decode_prefix_hoisted:
            z = model.prepare_decode_latent(z)
        if tensor.is_cuda and _timing_profile_enabled():
            torch.cuda.synchronize(tensor.device)
        encode_elapsed = time.perf_counter() - start
        _STATS["layerwise_encoder_time_s"] += encode_elapsed
        _STATS[f"layerwise_{target}_encoder_time_s"] += encode_elapsed

        payload = {
            "layerwise_ae": True,
            "target": target,
            "shape": tuple(int(v) for v in tensor.shape),
            "thw": tuple(int(v) for v in thw),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "model_config": cfg,
            "layer_idx": int(layer_idx),
            "ckpt_dir": ckpt_dir,
            "model_variant": used_variant,
            "int8_code_ae": int8_code_ae,
            "decode_prefix_hoisted": decode_prefix_hoisted,
        }
        v_error_scale = os.environ.get("HY_LAYERWISE_RAW_V_AE_ERROR_SCALE", "").strip()
        v_error_scale_chunks = os.environ.get(
            "HY_LAYERWISE_RAW_V_AE_ERROR_SCALE_CHUNKS", ""
        ).strip()
        oracle_layers_spec = os.environ.get(
            "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_LAYERS", ""
        ).strip()
        oracle_steps_spec = os.environ.get(
            "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_STEPS", ""
        ).strip()
        oracle_layers = _parse_layers(oracle_layers_spec) if oracle_layers_spec else set()
        store_clean_for_step_oracle = (
            target == "v"
            and bool(oracle_steps_spec)
            and layer_idx in oracle_layers
        )
        if target == "v" and (
            v_error_scale or v_error_scale_chunks or store_clean_for_step_oracle
        ):
            # Capacity oracle only: retain clean V so decode can scale the codec error.
            payload["oracle_clean"] = tensor
            payload["oracle_error_scale"] = float(v_error_scale or "1.0")
        if _layerwise_target_latent_int8_token(target):
            z_codes, z_scale = _quantize_latent_int8_token(z)
            payload["latent_int8_token"] = z_codes
            payload["latent_scale"] = z_scale
            payload["latent_int8_token_storage"] = True
            _STATS["layerwise_latent_int8_calls"] += 1
        elif _layerwise_target_latent_int4_token(target):
            z_codes, z_scale = _quantize_latent_int4_token(z)
            payload["latent_int4_token"] = z_codes
            payload["latent_scale"] = z_scale
            payload["latent_int4_token_storage"] = True
        elif _layerwise_target_latent_int4(target):
            z_codes, z_scale = _quantize_latent_int4(z)
            payload["latent_int4"] = z_codes
            payload["latent_scale"] = z_scale
            payload["latent_int4_storage"] = True
        elif _layerwise_target_latent_int8(target, int8_code_ae):
            z_codes, z_scale = _quantize_latent_int8(z)
            payload["latent_int8"] = z_codes
            payload["latent_scale"] = z_scale
            payload["latent_int8_storage"] = True
            _STATS["layerwise_latent_int8_calls"] += 1
        else:
            payload["latent"] = z
        if int8_code_ae and scale is not None:
            payload["int8_scale"] = scale

        tensor_bytes = _tensor_bytes(tensor)
        payload_bytes = _tensor_bytes(payload)
        _STATS["layerwise_original_bytes"] += tensor_bytes
        _STATS["layerwise_payload_bytes"] += payload_bytes
        _STATS[f"layerwise_{target}_original_bytes"] += tensor_bytes
        _STATS[f"layerwise_{target}_payload_bytes"] += payload_bytes
        _STATS["layerwise_compress_calls"] += 1
        if target == "k":
            _STATS["layerwise_k_calls"] += 1
        else:
            _STATS["layerwise_v_calls"] += 1
        layer_cache[name] = payload

        defer_empty_cache = _env_flag("HY_KV_EFF_OPT_EMPTY_CACHE_PER_CHUNK", "0")
        segmented_empty_cache = int(
            os.environ.get("HY_KV_EFF_OPT_EMPTY_CACHE_INTERVAL_LAYERS", "0") or 0
        ) > 0
        if tensor.is_cuda and not (defer_empty_cache or segmented_empty_cache):
            torch.cuda.empty_cache()
    except Exception:
        pass

    return layer_cache


def _layerwise_raw_v_ae_compress(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """Apply a per-layer AE payload to selected pre-position vision K/V tensors.

    Independent of HY_KV_AE3D_ENABLE — controlled by its own env vars.
    Defaults to V for backward compatibility; set
    HY_LAYERWISE_RAW_V_AE_TARGET=k or kv to include pre-position K.
    """
    if not _env_flag("HY_LAYERWISE_RAW_V_AE", "0"):
        return layer_cache
    layers = _parse_layers(os.environ.get("HY_LAYERWISE_RAW_V_AE_LAYERS", ""))
    bypass_chunks = _parse_layers(
        os.environ.get("HY_LAYERWISE_RAW_KV_AE_BYPASS_CHUNKS", "")
    )
    current_chunk = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    if current_chunk in bypass_chunks:
        _STATS["layerwise_bypass_calls"] = (
            int(_STATS.get("layerwise_bypass_calls", 0)) + 1
        )
        _STATS["layerwise_bypass_chunks"] = ",".join(
            str(chunk) for chunk in bypass_chunks
        )
        return layer_cache
    target = os.environ.get("HY_LAYERWISE_RAW_V_AE_TARGET", "v").strip().lower()
    if target not in {"k", "v", "kv", "vk", "k,v", "v,k"}:
        return layer_cache

    # Clean-suffix final-prediction training needs a clean historical V view at
    # every suffix layer, but must not instantiate or run suffix codecs. Keep a
    # detached alias before filtering the actually compressed layer set.
    if (
        target in {"v", "kv", "vk", "k,v", "v,k"}
        and _env_flag("HY_FUNCTIONAL_BLOCK_FT_RETAIN_CLEAN_V_ALL_LAYERS", "0")
    ):
        clean_v = layer_cache.get("v_vision")
        if torch.is_tensor(clean_v):
            layer_cache["_v_vision_clean_for_ft"] = clean_v.detach()
    if not layers or layer_idx not in layers:
        return layer_cache

    _STATS["layerwise_enabled"] = True
    _STATS["layerwise_layers"] = ",".join(str(l) for l in layers)

    default_ckpt_dir = os.environ.get("HY_LAYERWISE_RAW_V_AE_CKPT_DIR", "").strip()
    default_variant = os.environ.get("HY_LAYERWISE_RAW_V_AE_MODEL_VARIANT", "dcae_d128_8x")
    targets = ("k", "v") if target in {"kv", "vk", "k,v", "v,k"} else (target,)
    for one_target in targets:
        layer_cache = _compress_one_layerwise_target(
            layer_cache, layer_idx, one_target, default_ckpt_dir, default_variant
        )

    return layer_cache


def compress_layer_cache(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    thw = layer_cache.get("vision_thw")
    if thw is None:
        _record_fallback("missing_thw")
        return layer_cache
    for name in ("k_vision", "v_vision"):
        tensor = layer_cache.get(name)
        if isinstance(tensor, torch.Tensor):
            dump_pre_kv(layer_idx, name, tensor, thw)
            dump_multipos_pre_v(layer_idx, name, tensor, thw)

    # Phase 4-KV-CALIB hook: corrupt pre-position K (separate control from V)
    layer_cache = _apply_k_corruption(layer_cache, layer_idx)

    # Phase 4-KV-CALIB hook: corrupt clean pre-position V at the same recache
    # replacement point used by layerwise online AE. K is intentionally untouched.
    layer_cache = _apply_v_corruption(layer_cache, layer_idx)
    if _corruption_enabled():
        return layer_cache

    # Layerwise raw V AE hook (independent of HY_KV_AE3D_ENABLE)
    layer_cache = _layerwise_raw_v_ae_compress(layer_cache, layer_idx)

    if not enabled():
        return layer_cache
    layers = _parse_layers(os.environ.get("HY_KV_AE3D_LAYERS", "all"))
    _STATS["enabled"] = True
    _STATS["layers"] = os.environ.get("HY_KV_AE3D_LAYERS", "all")
    _STATS["tensors"] = os.environ.get("HY_KV_AE3D_TENSORS", "K,V")
    _STATS["latent_dtype"] = os.environ.get("HY_KV_AE3D_LATENT_DTYPE", "bf16")
    if not _layer_enabled(layer_idx, layers):
        return layer_cache
    for name in ("k_vision", "v_vision"):
        tensor = layer_cache.get(name)
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_bytes = _tensor_bytes(tensor)
        _STATS["resident_original_total_bytes"] += tensor_bytes
        if not _tensor_enabled(name):
            _STATS["resident_effective_bytes"] += tensor_bytes
            _STATS["resident_estimated_effective_bytes"] += tensor_bytes
            continue
        try:
            _STATS["resident_bf16_bytes"] += tensor_bytes
            payload = _compress_tensor(tensor, layer_idx, name, thw)
            payload_bytes = _tensor_bytes(payload)
            _STATS["resident_ae_bytes"] += payload_bytes
            _STATS["resident_effective_bytes"] += payload_bytes
            estimated_payload_bytes = float(payload.get("estimated_codec_bytes", payload_bytes))
            _STATS["resident_estimated_codec_bytes"] += estimated_payload_bytes
            _STATS["resident_estimated_effective_bytes"] += estimated_payload_bytes
            layer_cache[name] = payload
        except Exception as exc:
            _STATS["resident_effective_bytes"] += tensor_bytes
            _STATS["resident_estimated_effective_bytes"] += tensor_bytes
            _record_fallback(type(exc).__name__)
            if _env_flag("HY_KV_AE3D_STRICT", "0"):
                raise
    return layer_cache


_LAYERWISE_AE_MODELS: Dict[str, nn.Module] = {}
_LAYERWISE_CKPT_PATH_CACHE: Dict[Tuple[str, int, str], Optional[Path]] = {}
_K_TRAIN_OPTIMS: Dict[str, torch.optim.Optimizer] = {}


def _latent_int8_ste(z: torch.Tensor) -> torch.Tensor:
    codes, scale = _quantize_latent_int8(z)
    z_deq = _dequantize_latent_int8(codes, scale, z.dtype)
    return z + (z_deq - z).detach()


def _latent_int8_token_ste(z: torch.Tensor) -> torch.Tensor:
    codes, scale = _quantize_latent_int8_token(z)
    z_deq = _dequantize_latent_int8_token(codes, scale, z.dtype)
    return z + (z_deq - z).detach()


def _latent_int4_ste(z: torch.Tensor) -> torch.Tensor:
    codes, scale = _quantize_latent_int4(z)
    z_deq = _dequantize_latent_int4(codes, scale, z.dtype)
    return z + (z_deq - z).detach()


def _latent_int4_token_ste(z: torch.Tensor) -> torch.Tensor:
    codes, scale = _quantize_latent_int4_token(z)
    z_deq = _dequantize_latent_int4_token(codes, scale, z.dtype)
    return z + (z_deq - z).detach()


def _k_functional_train_enabled() -> bool:
    return _env_flag("HY_KV_CORRUPTION_K_TRAINABLE", "0")


def _v_functional_train_enabled() -> bool:
    return _env_flag("HY_LAYERWISE_RAW_V_AE_TRAINABLE", "0")


def _k_functional_layer_trainable(layer_idx: int) -> bool:
    if not _k_functional_train_enabled():
        return False
    layers = _parse_layers(os.environ.get("HY_KV_CORRUPTION_K_TRAINABLE_LAYERS", ""))
    if not layers:
        return True
    return int(layer_idx) in layers


def _configure_k_functional_parameters(model: nn.Module, trainable: bool):
    """Select the K functional-training parameter surface.

    T0 freezes the encoder/code basis and trains only decoder parameters.  The
    legacy behavior remains full-codec training unless the flag is enabled.
    """
    decoder_only = _env_flag("HY_KV_CORRUPTION_K_DECODER_ONLY", "0")
    selected = []
    for name, parameter in model.named_parameters():
        enabled = bool(
            trainable
            and (not decoder_only or _functional_parameter_group(name) == "decoder")
        )
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if trainable and not selected:
        mode = "decoder-only" if decoder_only else "full-codec"
        raise RuntimeError(f"K functional {mode} mode selected no parameters")
    return selected


def _build_k_functional_optimizer(
    parameters,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_type = os.environ.get(
        "HY_KV_CORRUPTION_K_OPTIMIZER", "adamw"
    ).strip().lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_type == "sgd":
        momentum = float(
            os.environ.get("HY_KV_CORRUPTION_K_SGD_MOMENTUM", "0.0")
        )
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(
        f"Unsupported HY_KV_CORRUPTION_K_OPTIMIZER={optimizer_type}"
    )


def _offload_k_frozen_encoder(model: nn.Module) -> None:
    """Move only the frozen DCAE encoder half to CPU after z is materialized."""
    if not _env_flag("HY_K_T0_CPU_OFFLOAD_ENCODER", "0"):
        return
    decoder_modules = {
        "decoder_delta",
        "latent_decoder_res",
        "learned_up",
        "output_res",
        "output_conv",
        "shortcut_unproj",
    }
    for name, child in model.named_children():
        if name not in decoder_modules:
            child.to(device="cpu")


def _v_functional_layer_trainable(layer_idx: int) -> bool:
    if not _v_functional_train_enabled():
        return False
    layers = _parse_layers(os.environ.get("HY_LAYERWISE_RAW_V_AE_TRAINABLE_LAYERS", ""))
    if not layers:
        return True
    return int(layer_idx) in layers


def _configure_v_functional_parameters(model: nn.Module, trainable: bool):
    """Select the V functional-training parameter surface.

    PCA readout calibration keeps the QAT encoder/code immutable and adjusts
    only the decoder interpretation. Legacy V functional runs retain their
    full-codec behavior unless the decoder-only flag is explicitly enabled.
    """
    decoder_only = _env_flag("HY_LAYERWISE_RAW_V_AE_DECODER_ONLY", "0")
    selected = []
    for name, parameter in model.named_parameters():
        enabled = bool(
            trainable
            and (not decoder_only or _functional_parameter_group(name) == "decoder")
        )
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if trainable and not selected:
        mode = "decoder-only" if decoder_only else "full-codec"
        raise RuntimeError(f"V functional {mode} mode selected no parameters")
    return selected


def zero_k_functional_train_grads() -> None:
    for opt in _K_TRAIN_OPTIMS.values():
        opt.zero_grad(set_to_none=True)


def backward_k_functional_audit(loss: torch.Tensor) -> Dict[str, Any]:
    """Backward once and report per-layer encoder/decoder gradient norms.

    This intentionally performs no clipping and no optimizer step.  It is the
    hard Gate-1 audit for T0 before any checkpoint is allowed to change.
    """
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError("K functional audit loss has no autograd graph")
    loss.backward()
    rows = []
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("k_corruption::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is None:
            continue
        group_sq = {"encoder": 0.0, "decoder": 0.0, "other": 0.0}
        group_nonzero = {"encoder": 0, "decoder": 0, "other": 0}
        group_numel = {"encoder": 0, "decoder": 0, "other": 0}
        finite = True
        trainable_numel = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable_numel += int(parameter.numel())
            group = _functional_parameter_group(name)
            group_numel[group] += int(parameter.numel())
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(grad).all().cpu())
            group_sq[group] += float(grad.pow(2).sum().cpu())
            group_nonzero[group] += int(torch.count_nonzero(grad).cpu())
        row = {
            "layer": int(layer_id),
            "trainable_numel": int(trainable_numel),
            "finite": bool(finite),
        }
        for group in ("encoder", "decoder", "other"):
            row[f"{group}_grad_norm"] = math.sqrt(max(group_sq[group], 0.0))
            row[f"{group}_grad_nonzero"] = int(group_nonzero[group])
            row[f"{group}_numel"] = int(group_numel[group])
        rows.append(row)
    rows.sort(key=lambda item: item["layer"])
    return {
        "loss": float(loss.detach().float().cpu()),
        "num_layers": len(rows),
        "rows": rows,
    }


def save_k_functional_gradient_snapshot(
    path: str, *, atom_id: str, loss: float
) -> Dict[str, Any]:
    """Persist the exact post-backward decoder gradient for T0 line search."""
    layers: Dict[str, Dict[str, torch.Tensor]] = {}
    source_checkpoints: Dict[str, str] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("k_corruption::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is None:
            continue
        grads = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grads[name] = parameter.grad.detach().float().cpu().clone()
        layers[str(int(layer_id))] = grads
        source_checkpoints[str(int(layer_id))] = str(
            getattr(model, "_hy_functional_source_ckpt", "")
        )
    payload = {
        "version": 2,
        "kind": os.environ.get(
            "HY_FUNCTIONAL_GRADIENT_KIND", "k_t0_single_atom_gradient"
        ),
        "target": 1,
        "atom_ids": [str(atom_id)],
        "loss_before": float(loss),
        "layers": layers,
        "source_checkpoints": source_checkpoints,
        "complete": False,
        "collected": True,
        "created_at": time.time(),
    }
    output = Path(path)
    _atomic_torch_save(payload, output)
    return {
        "path": str(output),
        "layers": len(layers),
        "parameters": sum(len(value) for value in layers.values()),
    }


def _optimizer_state_to_parameter_devices(opt: torch.optim.Optimizer) -> None:
    for param, state in opt.state.items():
        for name, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            # Adam moments must follow the parameter dtype after the codec is
            # moved from its CPU construction dtype to bf16 on CUDA. Keep the
            # scalar step counter in fp32 as required by fused/foreach AdamW.
            target_dtype = (
                param.dtype
                if name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
                else value.dtype
            )
            if value.device != param.device or value.dtype != target_dtype:
                state[name] = value.to(device=param.device, dtype=target_dtype)


def reset_v_final_prediction_tensor_audit() -> None:
    _V_FINAL_PRED_TENSOR_AUDIT.clear()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _accumulate_functional_grads(atom_id: str) -> Tuple[bool, Dict[str, float]]:
    """Accumulate codec gradients across deterministic rollout processes.

    Each launcher update uses a fresh directory. All atoms therefore evaluate
    the same checkpoint, while only the final atom materializes the mean
    gradient and permits AdamW to step.
    """
    accum_dir_raw = os.environ.get("HY_FUNCTIONAL_ACCUM_DIR", "").strip()
    if not accum_dir_raw:
        return True, {}
    target = int(os.environ.get("HY_FUNCTIONAL_ACCUM_TARGET", "1"))
    if target <= 0:
        raise ValueError(f"HY_FUNCTIONAL_ACCUM_TARGET must be positive, got {target}")
    if not atom_id:
        raise RuntimeError("Functional accumulation requires a non-empty atom_id")
    bucket_by_step = _env_flag("HY_FUNCTIONAL_ACCUM_BUCKET_BY_STEP", "0")
    bucket_by_chunk_step = _env_flag(
        "HY_FUNCTIONAL_ACCUM_BUCKET_BY_CHUNK_STEP", "0"
    )
    bucket = ""
    if bucket_by_chunk_step:
        chunk_marker = ":chunk"
        step_marker = ":step"
        if chunk_marker not in atom_id or step_marker not in atom_id:
            raise RuntimeError(
                f"Cannot infer chunk/step bucket from atom_id={atom_id}"
            )
        chunk_bucket = atom_id.split(chunk_marker, 1)[1].split(":", 1)[0]
        step_bucket = atom_id.split(step_marker, 1)[1].split(":", 1)[0]
        bucket = f"chunk{chunk_bucket}_step{step_bucket}"
        state_path = Path(accum_dir_raw) / f"gradient_accumulator_{bucket}.pt"
    elif bucket_by_step:
        marker = ":step"
        if marker not in atom_id:
            raise RuntimeError(f"Cannot infer step bucket from atom_id={atom_id}")
        bucket = atom_id.split(marker, 1)[1].split(":", 1)[0]
        state_path = Path(accum_dir_raw) / f"gradient_accumulator_step{bucket}.pt"
    else:
        state_path = Path(accum_dir_raw) / "gradient_accumulator.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
    else:
        state = {
            "version": 1,
            "target": target,
            "atom_ids": [],
            "layers": {},
            "complete": False,
        }
    if int(state.get("target", target)) != target:
        raise RuntimeError(
            f"Accumulator target mismatch: file={state.get('target')} env={target}"
        )
    if state.get("complete", False) or state.get("collected", False):
        raise RuntimeError(f"Accumulator is already complete: {state_path}")
    atom_ids = list(state.get("atom_ids", []))
    if atom_id in atom_ids:
        raise RuntimeError(f"Duplicate functional-training atom: {atom_id}")
    if len(atom_ids) >= target:
        raise RuntimeError(
            f"Accumulator already has {len(atom_ids)} atoms for target={target}"
        )

    trainable_models = []
    for key, opt in _K_TRAIN_OPTIMS.items():
        model = _LAYERWISE_AE_MODELS.get(key)
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if model is None or layer_id is None:
            continue
        target_name = "v" if key.startswith("v_functional::") else "k"
        trainable_models.append((target_name, int(layer_id), model, opt))
    if not trainable_models:
        raise RuntimeError("No trainable codec models found for accumulation")

    for target_name, layer_id, model, _ in trainable_models:
        state_key = f"{target_name}:L{int(layer_id)}"
        layer_state = state["layers"].setdefault(state_key, {})
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad_cpu = param.grad.detach().float().cpu()
            if name in layer_state:
                if tuple(layer_state[name].shape) != tuple(grad_cpu.shape):
                    raise RuntimeError(
                        f"Gradient shape mismatch L{layer_id} {name}: "
                        f"{tuple(layer_state[name].shape)} vs {tuple(grad_cpu.shape)}"
                    )
                layer_state[name].add_(grad_cpu)
            else:
                layer_state[name] = grad_cpu.clone()

    atom_ids.append(atom_id)
    state["atom_ids"] = atom_ids
    count = len(atom_ids)
    collected_only = _env_flag("HY_FUNCTIONAL_ACCUM_NO_STEP", "0")
    ready = count == target and not collected_only
    if count == target and collected_only:
        state["collected"] = True
        state["completed_at"] = time.time()
    if ready:
        for target_name, layer_id, model, _ in trainable_models:
            state_key = f"{target_name}:L{int(layer_id)}"
            layer_state = state["layers"].get(state_key, {})
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                grad_sum = layer_state.get(name)
                param.grad = (
                    None
                    if grad_sum is None
                    else grad_sum.to(device=param.device, dtype=param.dtype).div_(target)
                )
    _atomic_torch_save(state, state_path)
    return ready, {
        "accum_count": float(count),
        "accum_target": float(target),
        "optimizer_step": float(ready),
        "accum_collected": float(count == target),
    }


def step_k_functional_train_optimizers(
    loss: torch.Tensor, *, atom_id: str = ""
) -> Dict[str, float]:
    if not _K_TRAIN_OPTIMS:
        return {"num_optimizers": 0.0, "max_grad_norm": 0.0}
    loss.backward()
    snapshot_path = os.environ.get(
        "HY_FUNCTIONAL_GRADIENT_SNAPSHOT", ""
    ).strip()
    if snapshot_path:
        snapshot = save_k_functional_gradient_snapshot(
            snapshot_path,
            atom_id=atom_id,
            loss=float(loss.detach().float().cpu()),
        )
        return {
            "num_optimizers": float(len(_K_TRAIN_OPTIMS)),
            "max_grad_norm": 0.0,
            "gradient_snapshot": 1.0,
            "snapshot_layers": float(snapshot["layers"]),
        }
    ready_to_step, accum_info = _accumulate_functional_grads(atom_id)
    if not ready_to_step:
        return {
            "num_optimizers": float(len(_K_TRAIN_OPTIMS)),
            "max_grad_norm": 0.0,
            **accum_info,
        }
    grad_clip = float(os.environ.get("HY_KV_CORRUPTION_K_GRAD_CLIP", "1.0"))
    max_grad = 0.0
    layer_grad_norms: Dict[int, float] = {}
    layer_update_norms: Dict[int, float] = {}
    layer_param_norms: Dict[int, float] = {}
    layer_group_update_rms: Dict[Tuple[int, str], float] = {}
    target_grad_norms: Dict[Tuple[str, int], float] = {}
    target_grad_rel_norms: Dict[Tuple[str, int], float] = {}
    target_update_norms: Dict[Tuple[str, int], float] = {}
    diagnose_updates = _env_flag("HY_FUNCTIONAL_BLOCK_FT_DIAG_UPDATE_NORMS", "0")
    for key, opt in _K_TRAIN_OPTIMS.items():
        params = [p for group in opt.param_groups for p in group["params"] if p.grad is not None]
        grad_sq = 0.0
        for param in params:
            grad_sq += float(param.grad.detach().float().pow(2).sum().cpu())
        grad_norm_preclip = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
        model = _LAYERWISE_AE_MODELS.get(key)
        layer_id = getattr(model, "_hy_functional_layer_id", None) if model is not None else None
        if (
            _env_flag("HY_FUNCTIONAL_BLOCK_FT_JOINT_GRAD_STRICT", "0")
            and model is not None
        ):
            leaked_encoder = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and _functional_parameter_group(name) == "encoder"
            ]
            if leaked_encoder:
                raise RuntimeError(
                    f"Joint functional FT has trainable encoder parameters: {leaked_encoder[:4]}"
                )
        if layer_id is not None:
            layer_grad_norms[int(layer_id)] = grad_norm_preclip
            target_name = "v" if key.startswith("v_functional::") else "k"
            target_grad_norms[(target_name, int(layer_id))] = grad_norm_preclip
            parameter_norm = math.sqrt(
                sum(
                    float(parameter.detach().float().pow(2).sum().cpu())
                    for group in opt.param_groups
                    for parameter in group["params"]
                )
            )
            target_grad_rel_norms[(target_name, int(layer_id))] = (
                grad_norm_preclip / max(parameter_norm, 1e-12)
            )
        if params and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            max_grad = max(max_grad, float(grad_norm.detach().cpu()))
        before_step = None
        if diagnose_updates and params:
            before_step = [param.detach().clone() for param in params]
        opt.step()
        if diagnose_updates and params and before_step is not None and layer_id is not None:
            update_sq = 0.0
            param_sq = 0.0
            for param, before in zip(params, before_step):
                update_sq += float(
                    (param.detach().float() - before.float()).pow(2).sum().cpu()
                )
                param_sq += float(param.detach().float().pow(2).sum().cpu())
            layer_update_norms[int(layer_id)] = math.sqrt(max(update_sq, 0.0))
            target_update_norms[(target_name, int(layer_id))] = math.sqrt(
                max(update_sq, 0.0)
            )
            layer_param_norms[int(layer_id)] = math.sqrt(max(param_sq, 0.0))
            names_by_id = {
                id(param): name for name, param in model.named_parameters()
            }
            group_sq = {"encoder": 0.0, "decoder": 0.0}
            group_numel = {"encoder": 0, "decoder": 0}
            for param, before in zip(params, before_step):
                group = _functional_parameter_group(names_by_id.get(id(param), ""))
                group_sq[group] += float(
                    (param.detach().float() - before.float()).pow(2).sum().cpu()
                )
                group_numel[group] += int(param.numel())
            for group in ("encoder", "decoder"):
                layer_group_update_rms[(int(layer_id), group)] = math.sqrt(
                    max(group_sq[group], 0.0) / max(group_numel[group], 1)
                )
            del before_step
    if _env_flag("HY_FUNCTIONAL_BLOCK_FT_JOINT_GRAD_STRICT", "0"):
        requested_layers = _parse_layers(
            os.environ.get("HY_FUNCTIONAL_BLOCK_FT_LAYERS", "")
        )
        missing = [
            f"{target_name.upper()}-L{layer_id}"
            for target_name in ("k", "v")
            for layer_id in sorted(requested_layers)
            if target_grad_norms.get((target_name, int(layer_id)), 0.0) <= 0.0
        ]
        if missing:
            raise RuntimeError(
                "Joint functional FT missing nonzero decoder gradients: "
                + ", ".join(missing)
            )
    info = {
        "num_optimizers": float(len(_K_TRAIN_OPTIMS)),
        "max_grad_norm": max_grad,
        **accum_info,
    }
    accum_dir_raw = os.environ.get("HY_FUNCTIONAL_ACCUM_DIR", "").strip()
    if accum_dir_raw:
        state_path = Path(accum_dir_raw) / "gradient_accumulator.pt"
        state = torch.load(state_path, map_location="cpu")
        state["complete"] = True
        state["completed_at"] = time.time()
        _atomic_torch_save(state, state_path)
    for layer_id, grad_norm in layer_grad_norms.items():
        info[f"grad_L{layer_id}"] = float(grad_norm)
    for (target_name, layer_id), grad_norm in target_grad_norms.items():
        info[f"grad_{target_name.upper()}_L{layer_id}"] = float(grad_norm)
    for (target_name, layer_id), grad_rel in target_grad_rel_norms.items():
        info[f"grad_rel_{target_name.upper()}_L{layer_id}"] = float(grad_rel)
    for (target_name, layer_id), update_norm in target_update_norms.items():
        info[f"update_{target_name.upper()}_L{layer_id}"] = float(update_norm)
    for layer_id, update_norm in layer_update_norms.items():
        info[f"update_L{layer_id}"] = float(update_norm)
        param_norm = layer_param_norms.get(layer_id, 0.0)
        info[f"update_rel_L{layer_id}"] = float(
            update_norm / max(param_norm, 1e-12)
        )
    for (layer_id, group), update_rms in layer_group_update_rms.items():
        info[f"update_rms_{group}_L{layer_id}"] = float(update_rms)
    return info


def v_functional_loss_grad_norm(loss: torch.Tensor) -> float:
    """Return the global trainable V-codec gradient norm without consuming the graph."""
    params = []
    for key, model in _LAYERWISE_AE_MODELS.items():
        if key.startswith("v_functional::"):
            params.extend(param for param in model.parameters() if param.requires_grad)
    if not params or loss is None or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    grad_sq = 0.0
    for grad in grads:
        if grad is not None:
            grad_sq += float(grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(max(grad_sq, 0.0))


def k_functional_anchor_loss_for_cache(
    kv_cache,
    layers,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Differentiable K reconstruction anchor for functional fine-tuning.

    This is intentionally lightweight: full DC/AC metrics allocate large 5D
    temporaries once the resident KV cache grows. For online functional FT we
    instead sample flattened K entries and keep the codec close to the clean K
    while block-output loss nudges function.
    """
    if kv_cache is None:
        return None, {}
    sample_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ANCHOR_SAMPLES", "65536"))
    recon_weight = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_RECON_WEIGHT", "0.0"))
    cosine_weight = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_COS_WEIGHT", "0.0"))
    diagnostics_only = _env_flag("HY_FUNCTIONAL_BLOCK_FT_DIAG_METRICS", "0")
    if recon_weight <= 0.0 and cosine_weight <= 0.0 and not diagnostics_only:
        return None, {}
    losses = []
    rel_values = []
    cos_values = []
    for layer_idx in layers:
        if int(layer_idx) >= len(kv_cache):
            continue
        layer_cache = kv_cache[int(layer_idx)]
        clean = layer_cache.get("_k_vision_clean_for_ft")
        corrupt = layer_cache.get("k_vision")
        if not (torch.is_tensor(clean) and torch.is_tensor(corrupt)):
            continue
        flat_clean = clean.detach().float().reshape(-1)
        flat_corrupt = corrupt.float().reshape(-1)
        n = int(flat_clean.numel())
        if n == 0:
            continue
        if sample_count > 0 and sample_count < n:
            step = max(1, n // sample_count)
            idx = torch.arange(0, n, step, device=flat_corrupt.device)[:sample_count]
            flat_clean = flat_clean.index_select(0, idx)
            flat_corrupt = flat_corrupt.index_select(0, idx)
        diff = flat_corrupt - flat_clean.to(device=flat_corrupt.device)
        rel = diff.norm() / flat_clean.to(device=flat_corrupt.device).norm().clamp_min(1e-8)
        cos = F.cosine_similarity(flat_corrupt, flat_clean.to(device=flat_corrupt.device), dim=0)
        layer_loss = recon_weight * rel + cosine_weight * (1.0 - cos)
        losses.append(layer_loss)
        rel_values.append(float(rel.detach().cpu()))
        cos_values.append(float(cos.detach().cpu()))
    if not losses:
        return None, {}
    loss = torch.stack(losses).mean()
    metrics = {
        "anchor_rel_l2": sum(rel_values) / max(1, len(rel_values)),
        "anchor_cosine": sum(cos_values) / max(1, len(cos_values)),
    }
    return loss, metrics


def k_functional_reconstruction_losses_for_cache(
    kv_cache,
    layers,
    *,
    sample_count: Optional[int] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Return unweighted differentiable K reconstruction components."""
    if kv_cache is None:
        return {}
    if sample_count is None:
        sample_count = int(
            os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ANCHOR_SAMPLES", "65536")
        )
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        if int(layer_idx) >= len(kv_cache):
            continue
        layer_cache = kv_cache[int(layer_idx)]
        clean = layer_cache.get("_k_vision_clean_for_ft")
        corrupt = layer_cache.get("k_vision")
        if not (torch.is_tensor(clean) and torch.is_tensor(corrupt)):
            continue
        flat_clean = clean.detach().float().reshape(-1)
        flat_corrupt = corrupt.float().reshape(-1)
        n = int(flat_clean.numel())
        if n == 0:
            continue
        if sample_count > 0 and sample_count < n:
            step = max(1, n // sample_count)
            idx = torch.arange(0, n, step, device=flat_corrupt.device)[
                :sample_count
            ]
            flat_clean = flat_clean.index_select(0, idx)
            flat_corrupt = flat_corrupt.index_select(0, idx)
        clean_dev = flat_clean.to(device=flat_corrupt.device)
        diff = flat_corrupt - clean_dev
        out[int(layer_idx)] = {
            "rel": diff.norm() / clean_dev.norm().clamp_min(1e-8),
            "cosine": F.cosine_similarity(flat_corrupt, clean_dev, dim=0),
        }
    return out


def run_k_p1_gradient_audit(
    prop1_losses: Dict[int, torch.Tensor],
    local_losses: Dict[int, torch.Tensor],
    reconstruction_losses: Dict[int, Dict[str, torch.Tensor]],
    *,
    chunk: int,
    step: int,
) -> Dict[str, Any]:
    """Audit independent P1/local/reconstruction decoder gradients."""
    models: Dict[int, nn.Module] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("k_corruption::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is not None:
            models[int(layer_id)] = model
    layers = sorted(
        set(prop1_losses)
        & set(local_losses)
        & set(reconstruction_losses)
        & set(models)
    )
    rows = []
    snapshot_layers: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    snapshot_losses: Dict[str, Dict[str, float]] = {}
    snapshot_sources: Dict[str, str] = {}
    for layer in layers:
        named = [
            (name, parameter)
            for name, parameter in models[layer].named_parameters()
            if parameter.requires_grad
        ]
        params = [parameter for _, parameter in named]
        if not params:
            continue
        loss_map = {
            "prop1": prop1_losses[layer],
            "local": local_losses[layer],
            "reconstruction": reconstruction_losses[layer]["rel"],
            "cosine": 1.0 - reconstruction_losses[layer]["cosine"],
        }
        grads = {
            name: torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            for name, loss in loss_map.items()
        }
        row = {"layer": int(layer), "group": "decoder"}
        for name in ("prop1", "local", "reconstruction", "cosine"):
            row[name] = _gradient_group_stats(named, grads[name], "decoder")
        row["cos_prop1_local"] = _gradient_group_cosine(
            named, grads["prop1"], grads["local"], "decoder"
        )
        row["cos_prop1_reconstruction"] = _gradient_group_cosine(
            named, grads["prop1"], grads["reconstruction"], "decoder"
        )
        row["cos_local_reconstruction"] = _gradient_group_cosine(
            named, grads["local"], grads["reconstruction"], "decoder"
        )
        rows.append(row)
        print("[KP1GradientAudit] " + json.dumps(row, sort_keys=True), flush=True)
        snapshot_layers[str(int(layer))] = {
            component: {
                parameter_name: (
                    torch.zeros_like(parameter.detach(), dtype=torch.float32, device="cpu")
                    if grad is None
                    else grad.detach().float().cpu().clone()
                )
                for (parameter_name, parameter), grad in zip(named, component_grads)
            }
            for component, component_grads in grads.items()
        }
        snapshot_losses[str(int(layer))] = {
            component: float(loss.detach().float().cpu())
            for component, loss in loss_map.items()
        }
        snapshot_sources[str(int(layer))] = str(
            getattr(models[layer], "_hy_functional_source_ckpt", "")
        )
    payload = {
        "chunk": int(chunk),
        "step": int(step),
        "rows": rows,
    }
    snapshot_path_raw = os.environ.get(
        "HY_K_P1_COMPONENT_GRADIENT_SNAPSHOT", ""
    ).strip()
    if snapshot_path_raw:
        atom_id = os.environ.get(
            "HY_FUNCTIONAL_ACCUM_ATOM_PREFIX", ""
        ).strip()
        if not atom_id:
            raise RuntimeError("K P1 component snapshot requires atom id")
        snapshot_payload = {
            "version": 1,
            "kind": "k_a1_p1_gate1_component_gradient",
            "atom_ids": [atom_id],
            "chunk": int(chunk),
            "step": int(step),
            "layers": snapshot_layers,
            "losses": snapshot_losses,
            "source_checkpoints": snapshot_sources,
            "complete": True,
            "created_at": time.time(),
        }
        snapshot_path = Path(snapshot_path_raw)
        _atomic_torch_save(snapshot_payload, snapshot_path)
        payload["component_snapshot"] = str(snapshot_path)
    out_root = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT", "").strip()
    if out_root:
        out_path = (
            Path(out_root)
            / f"k_p1_gradient_audit_chunk{int(chunk):02d}_step{int(step):02d}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["path"] = str(out_path)
    return payload


def v_functional_anchor_loss_for_cache(
    kv_cache,
    layers,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    if kv_cache is None:
        return None, {}
    sample_count = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ANCHOR_SAMPLES", "65536"))
    recon_weight = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_RECON_WEIGHT", "0.0"))
    cosine_weight = float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_COS_WEIGHT", "0.0"))
    diagnostics_only = _env_flag("HY_FUNCTIONAL_BLOCK_FT_DIAG_METRICS", "0")
    if recon_weight <= 0.0 and cosine_weight <= 0.0 and not diagnostics_only:
        return None, {}
    components = v_functional_reconstruction_losses_for_cache(
        kv_cache, layers, sample_count=sample_count
    )
    losses = []
    rel_values = []
    cos_values = []
    layer_values = []
    for layer_idx in layers:
        item = components.get(int(layer_idx))
        if item is None:
            continue
        rel = item["rel"]
        cos = item["cosine"]
        if recon_weight > 0.0 or cosine_weight > 0.0:
            losses.append(recon_weight * rel + cosine_weight * (1.0 - cos))
        rel_values.append(float(rel.detach().cpu()))
        cos_values.append(float(cos.detach().cpu()))
        layer_values.append(int(layer_idx))
    if not rel_values:
        return None, {}
    metrics = {
        "anchor_rel_l2": sum(rel_values) / max(1, len(rel_values)),
        "anchor_cosine": sum(cos_values) / max(1, len(cos_values)),
    }
    for layer_idx, rel, cos in zip(layer_values, rel_values, cos_values):
        metrics[f"anchor_rel_L{layer_idx}"] = float(rel)
        metrics[f"anchor_cos_L{layer_idx}"] = float(cos)
    loss = torch.stack(losses).mean() if losses else None
    return loss, metrics


def v_functional_reconstruction_losses_for_cache(
    kv_cache,
    layers,
    *,
    sample_count: Optional[int] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Return unweighted, differentiable per-layer V reconstruction metrics."""
    if kv_cache is None:
        return {}
    if sample_count is None:
        sample_count = int(
            os.environ.get("HY_FUNCTIONAL_BLOCK_FT_ANCHOR_SAMPLES", "65536")
        )
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        if int(layer_idx) >= len(kv_cache):
            continue
        layer_cache = kv_cache[int(layer_idx)]
        clean = layer_cache.get("_v_vision_clean_for_ft")
        corrupt = layer_cache.get("_v_vision_base_for_ft")
        if not torch.is_tensor(corrupt):
            corrupt = layer_cache.get("v_vision")
        if not (torch.is_tensor(clean) and torch.is_tensor(corrupt)):
            continue
        flat_clean = clean.detach().float().reshape(-1)
        flat_corrupt = corrupt.float().reshape(-1)
        n = int(flat_clean.numel())
        if n == 0:
            continue
        if sample_count > 0 and sample_count < n:
            step = max(1, n // sample_count)
            idx = torch.arange(0, n, step, device=flat_corrupt.device)[:sample_count]
            flat_clean = flat_clean.index_select(0, idx)
            flat_corrupt = flat_corrupt.index_select(0, idx)
        clean_dev = flat_clean.to(device=flat_corrupt.device)
        diff = flat_corrupt - clean_dev
        rel = diff.norm() / clean_dev.norm().clamp_min(1e-8)
        cos = F.cosine_similarity(flat_corrupt, clean_dev, dim=0)
        out[int(layer_idx)] = {"rel": rel, "cosine": cos}
    return out


def _functional_parameter_group(name: str) -> str:
    decoder_prefixes = (
        "decoder_delta",
        "latent_decoder_res",
        "learned_up",
        "output_res",
        "output_conv",
        "shortcut_unproj",
    )
    return "decoder" if name.startswith(decoder_prefixes) else "encoder"


def _gradient_group_stats(named_params, grads, group: str) -> Dict[str, float]:
    grad_sq = 0.0
    param_sq = 0.0
    nonzero = 0
    total = 0
    for (name, param), grad in zip(named_params, grads):
        if _functional_parameter_group(name) != group:
            continue
        total += int(param.numel())
        param_sq += float(param.detach().float().pow(2).sum().cpu())
        if grad is not None:
            grad_float = grad.detach().float()
            grad_sq += float(grad_float.pow(2).sum().cpu())
            nonzero += int((grad_float != 0).sum().cpu())
    grad_norm = math.sqrt(max(grad_sq, 0.0))
    param_norm = math.sqrt(max(param_sq, 0.0))
    return {
        "grad_norm": grad_norm,
        "relative_grad_norm": grad_norm / max(param_norm, 1e-12),
        "nonzero_fraction": float(nonzero) / max(total, 1),
        "parameter_norm": param_norm,
    }


def _gradient_group_cosine(named_params, left, right, group: str) -> float:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for (name, _), grad_left, grad_right in zip(named_params, left, right):
        if _functional_parameter_group(name) != group:
            continue
        if grad_left is None or grad_right is None:
            continue
        left_float = grad_left.detach().float()
        right_float = grad_right.detach().float()
        dot += float((left_float * right_float).sum().cpu())
        left_sq += float(left_float.pow(2).sum().cpu())
        right_sq += float(right_float.pow(2).sum().cpu())
    denom = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    return dot / max(denom, 1e-20)


def run_v_functional_gradient_audit(
    endpoint_loss: torch.Tensor,
    local_losses: Dict[int, torch.Tensor],
    reconstruction_losses: Dict[int, Dict[str, torch.Tensor]],
    *,
    chunk: int,
    step: int,
) -> Dict[str, Any]:
    """Measure endpoint/local/reconstruction gradients without optimizer steps."""
    if not endpoint_loss.requires_grad or endpoint_loss.grad_fn is None:
        raise RuntimeError("Gradient-audit endpoint loss has no autograd graph")
    for layer, loss in local_losses.items():
        if not loss.requires_grad or loss.grad_fn is None:
            raise RuntimeError(
                f"Gradient-audit local loss L{int(layer)} has no autograd graph"
            )
    for layer, components in reconstruction_losses.items():
        loss = components["rel"]
        if not loss.requires_grad or loss.grad_fn is None:
            raise RuntimeError(
                f"Gradient-audit reconstruction loss L{int(layer)} has no autograd graph"
            )
    models: Dict[int, nn.Module] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("v_functional::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is not None:
            models[int(layer_id)] = model
    layers = sorted(set(local_losses) & set(reconstruction_losses) & set(models))
    all_named = []
    slices = {}
    for layer in layers:
        start = len(all_named)
        named = [(name, param) for name, param in models[layer].named_parameters() if param.requires_grad]
        all_named.extend((layer, name, param) for name, param in named)
        slices[layer] = (start, len(all_named))
    all_params = [param for _, _, param in all_named]
    endpoint_all = torch.autograd.grad(
        endpoint_loss,
        all_params,
        retain_graph=True,
        allow_unused=True,
    )
    endpoint_grad_sq = 0.0
    for grad in endpoint_all:
        if grad is not None:
            endpoint_grad_sq += float(grad.detach().float().pow(2).sum().cpu())
    rows = []
    for layer in layers:
        start, end = slices[layer]
        named_params = [(name, param) for _, name, param in all_named[start:end]]
        params = [param for _, param in named_params]
        endpoint_grads = endpoint_all[start:end]
        local_grads = torch.autograd.grad(
            local_losses[layer], params, retain_graph=True, allow_unused=True
        )
        recon_grads = torch.autograd.grad(
            reconstruction_losses[layer]["rel"],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        for group in ("encoder", "decoder"):
            row = {
                "layer": int(layer),
                "group": group,
                "endpoint": _gradient_group_stats(named_params, endpoint_grads, group),
                "local": _gradient_group_stats(named_params, local_grads, group),
                "reconstruction": _gradient_group_stats(named_params, recon_grads, group),
                "cos_endpoint_local": _gradient_group_cosine(
                    named_params, endpoint_grads, local_grads, group
                ),
                "cos_endpoint_reconstruction": _gradient_group_cosine(
                    named_params, endpoint_grads, recon_grads, group
                ),
                "cos_local_reconstruction": _gradient_group_cosine(
                    named_params, local_grads, recon_grads, group
                ),
            }
            rows.append(row)
            print("[FunctionalGradientAudit] " + json.dumps(row, sort_keys=True), flush=True)
        del local_grads, recon_grads
    payload = {
        "chunk": int(chunk),
        "step": int(step),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "endpoint_grad_norm": math.sqrt(max(endpoint_grad_sq, 0.0)),
        "rows": rows,
    }
    out_root = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT", "").strip()
    if out_root:
        out_path = Path(out_root) / f"gradient_audit_chunk{int(chunk):02d}_step{int(step):02d}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        payload["path"] = str(out_path)
    del endpoint_all
    return payload


def run_v_final_prediction_gradient_audit(
    prediction_loss: torch.Tensor,
    layers,
    *,
    chunk: int,
    step: int,
) -> Dict[str, Any]:
    """Audit K/V final-prediction gradients through the frozen DiT suffix."""
    if not prediction_loss.requires_grad or prediction_loss.grad_fn is None:
        raise RuntimeError("Final-prediction audit loss has no autograd graph")
    requested = sorted(int(layer) for layer in layers)
    target = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TARGET", "k").strip().lower()
    target = "v" if target == "v" else "k"
    model_prefix = "v_functional::" if target == "v" else "k_corruption::"
    models: Dict[int, nn.Module] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith(model_prefix):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is not None and int(layer_id) in requested:
            models[int(layer_id)] = model
    missing = [layer for layer in requested if layer not in models]
    if missing:
        raise RuntimeError(
            f"Final-prediction audit missing trainable {target.upper()} models: {missing}"
        )

    all_named = []
    slices = {}
    for layer in requested:
        start = len(all_named)
        named = [
            (name, param)
            for name, param in models[layer].named_parameters()
            if param.requires_grad
        ]
        all_named.extend((layer, name, param) for name, param in named)
        slices[layer] = (start, len(all_named))
    tensor_entries = []
    attention_name = "attention_v" if target == "v" else "attention_k"
    for layer in requested:
        tensors = _V_FINAL_PRED_TENSOR_AUDIT.get(layer, {})
        for name in (
            "encoder_output",
            "quantized_latent",
            "decoder_output",
            attention_name,
        ):
            value = tensors.get(name)
            if torch.is_tensor(value) and value.requires_grad:
                tensor_entries.append((layer, name, value))
    inputs = [param for _, _, param in all_named] + [value for _, _, value in tensor_entries]
    grads = torch.autograd.grad(
        prediction_loss,
        inputs,
        retain_graph=True,
        allow_unused=True,
    )
    param_grads = grads[: len(all_named)]
    tensor_grads = grads[len(all_named) :]
    rows = []
    for layer in requested:
        start, end = slices[layer]
        named_params = [(name, param) for _, name, param in all_named[start:end]]
        layer_grads = param_grads[start:end]
        row = {
            "layer": layer,
            "target": target,
            "parameter_dtypes": sorted(
                {str(param.dtype) for _, param in named_params}
            ),
            "encoder": _gradient_group_stats(named_params, layer_grads, "encoder"),
            "decoder": _gradient_group_stats(named_params, layer_grads, "decoder"),
            "tensors": {},
        }
        for (tensor_layer, name, value), grad in zip(tensor_entries, tensor_grads):
            if tensor_layer != layer:
                continue
            row["tensors"][name] = {
                "requires_grad": bool(value.requires_grad),
                "has_grad_fn": value.grad_fn is not None,
                "shape": [int(dim) for dim in value.shape],
                "rms": float(value.detach().float().pow(2).mean().sqrt().cpu()),
                "grad_norm": 0.0
                if grad is None
                else float(grad.detach().float().norm().cpu()),
            }
        rows.append(row)
        print("[FinalPredictionGradientAudit] " + json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "chunk": int(chunk),
        "step": int(step),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "rows": rows,
    }
    out_root = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT", "").strip()
    if out_root:
        out_path = (
            Path(out_root)
            / f"final_prediction_gradient_audit_chunk{int(chunk):02d}_step{int(step):02d}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["path"] = str(out_path)
    return payload


def run_joint_functional_path_audit(
    loss: torch.Tensor,
    kv_cache,
    layers,
    *,
    chunk: int,
    step: int,
) -> Dict[str, Any]:
    """Trace final-output gradients to deployed cache tensors and K/V decoders."""
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError("Joint path audit loss has no autograd graph")

    requested = sorted(int(layer) for layer in layers)
    entries = []
    for layer in requested:
        if layer >= len(kv_cache):
            continue
        layer_cache = kv_cache[layer]
        for target, cache_name, model_prefix in (
            ("k", "k_vision", "k_corruption::"),
            ("v", "v_vision", "v_functional::"),
        ):
            cache_value = layer_cache.get(cache_name)
            model = next(
                (
                    candidate
                    for key, candidate in _LAYERWISE_AE_MODELS.items()
                    if key.startswith(model_prefix)
                    and int(getattr(candidate, "_hy_functional_layer_id", -1)) == layer
                ),
                None,
            )
            if model is None:
                raise RuntimeError(
                    f"Joint path audit missing {target.upper()} model at L{layer}"
                )
            decoder_params = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and _functional_parameter_group(name) == "decoder"
            ]
            entries.append((layer, target, cache_value, model, decoder_params))

    inputs = []
    input_meta = []
    for layer, target, cache_value, _model, decoder_params in entries:
        if torch.is_tensor(cache_value) and cache_value.requires_grad:
            inputs.append(cache_value)
            input_meta.append((layer, target, "cache", cache_value))
        layer_cache = kv_cache[layer]
        intermediate_names = (
            (
                "_joint_audit_key_pre",
                "_joint_audit_key_rope",
                "_joint_audit_key_prope",
                "_joint_audit_restored_key",
                "_joint_audit_sdpa_key",
            )
            if target == "k"
            else (
                "_joint_audit_value_pre",
                "_joint_audit_restored_value",
                "_joint_audit_sdpa_value",
            )
        )
        for name in intermediate_names:
            value = layer_cache.get(name)
            if torch.is_tensor(value) and value.requires_grad:
                inputs.append(value)
                input_meta.append((layer, target, f"intermediate:{name}", value))
        for name, parameter in decoder_params:
            inputs.append(parameter)
            input_meta.append((layer, target, f"parameter:{name}", parameter))
    if not inputs:
        raise RuntimeError("Joint path audit found no differentiable inputs")

    grads = torch.autograd.grad(
        loss,
        inputs,
        retain_graph=True,
        allow_unused=True,
    )
    rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for (layer, target, kind, value), grad in zip(input_meta, grads):
        row = rows.setdefault(
            (layer, target),
            {
                "layer": layer,
                "target": target,
                "cache_requires_grad": False,
                "cache_has_grad_fn": False,
                "cache_grad_norm": 0.0,
                "decoder_grad_norm": 0.0,
                "decoder_parameter_count": 0,
                "decoder_nonzero_parameter_count": 0,
                "intermediates": {},
            },
        )
        grad_norm = (
            0.0 if grad is None else float(grad.detach().float().norm().cpu())
        )
        if kind == "cache":
            row["cache_requires_grad"] = bool(value.requires_grad)
            row["cache_has_grad_fn"] = value.grad_fn is not None
            row["cache_grad_norm"] = grad_norm
        elif kind.startswith("intermediate:"):
            row["intermediates"][kind.split(":", 1)[1]] = {
                "requires_grad": bool(value.requires_grad),
                "has_grad_fn": value.grad_fn is not None,
                "grad_norm": grad_norm,
            }
        else:
            row["decoder_parameter_count"] += 1
            row["decoder_grad_norm"] = math.sqrt(
                row["decoder_grad_norm"] ** 2 + grad_norm ** 2
            )
            if grad_norm > 0.0:
                row["decoder_nonzero_parameter_count"] += 1

    payload = {
        "chunk": int(chunk),
        "step": int(step),
        "loss": float(loss.detach().float().cpu()),
        "rows": [rows[key] for key in sorted(rows)],
    }
    out_root = os.environ.get("HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT", "").strip()
    if out_root:
        out_path = Path(out_root) / (
            f"joint_path_audit_chunk{int(chunk):02d}_step{int(step):02d}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["path"] = str(out_path)
    print("[JointPathAudit] " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run_k_pca_tail_score_audit(
    block_loss: torch.Tensor,
    *,
    layer: int,
    chunk: int,
    step: int,
) -> Dict[str, Any]:
    """Score all PCA directions from one pre-RoPE K block-loss gradient."""
    if not block_loss.requires_grad or block_loss.grad_fn is None:
        raise RuntimeError("K PCA tail score block loss has no graph")
    tensors = _K_PCA_TAIL_SCORE_TENSORS.get(int(layer), {})
    clean = tensors.get("clean_pre_rope")
    recon = tensors.get("recon_pre_rope")
    if not (torch.is_tensor(clean) and torch.is_tensor(recon)):
        raise RuntimeError(f"K PCA tail score missing pre-RoPE tensors for L{layer}")
    grad = torch.autograd.grad(
        block_loss, recon, retain_graph=False, allow_unused=False
    )[0]
    if grad is None or not torch.isfinite(grad).all():
        raise RuntimeError(f"K PCA tail score invalid gradient for L{layer}")
    basis_path = Path(os.environ["HY_K_PCA_TAIL_FULL_BASIS"])
    basis_payload = torch.load(basis_path, map_location="cpu")
    components = basis_payload["components"].float()
    mean = basis_payload["mean"].float()
    eigenvalues = basis_payload.get("eigenvalues")
    if tuple(components.shape) != (2048, 2048):
        raise RuntimeError(
            f"K PCA tail score requires full 2048 basis, got {tuple(components.shape)}"
        )
    # [B,H,S,D] -> one 2048-D global-head observation per token.
    clean_rows = clean.detach().permute(0, 2, 1, 3).reshape(-1, 2048).float()
    grad_rows = grad.detach().permute(0, 2, 1, 3).reshape(-1, 2048).float()
    sample_count = int(os.environ.get("HY_K_PCA_TAIL_TOKEN_SAMPLES", "4096"))
    if 0 < sample_count < clean_rows.shape[0]:
        seed = (
            int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_SEED", "2027"))
            + int(layer) * 1009 + int(chunk) * 97 + int(step)
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(
            int(clean_rows.shape[0]), generator=generator
        )[:sample_count].sort().values.to(clean_rows.device)
        clean_rows = clean_rows.index_select(0, indices)
        grad_rows = grad_rows.index_select(0, indices)
    components_dev = components.to(device=clean_rows.device)
    mean_dev = mean.to(device=clean_rows.device)
    signed_sum = torch.zeros(2048, device=clean_rows.device, dtype=torch.float64)
    square_sum = torch.zeros_like(signed_sum)
    batch = int(os.environ.get("HY_K_PCA_TAIL_PROJECT_BATCH", "512"))
    count = 0
    for start in range(0, clean_rows.shape[0], batch):
        x = clean_rows[start : start + batch] - mean_dev
        g = grad_rows[start : start + batch]
        coefficients = F.linear(x, components_dev)
        projected_grad = F.linear(g, components_dev)
        product = coefficients.double() * projected_grad.double()
        signed_sum += product.sum(dim=0)
        square_sum += product.square().sum(dim=0)
        count += int(product.shape[0])
    benefit = (-signed_sum / max(count, 1)).float().cpu()
    sensitivity = (square_sum / max(count, 1)).float().cpu()
    payload = {
        "kind": "k_pca_tail_function_score_atom",
        "layer": int(layer),
        "chunk": int(chunk),
        "step": int(step),
        "block_loss": float(block_loss.detach().cpu()),
        "rows": int(count),
        "benefit": benefit,
        "sensitivity": sensitivity,
        "grad_norm": float(grad.float().norm().cpu()),
        "full_basis": str(basis_path.resolve()),
        "eigenvalues": (
            None if eigenvalues is None else eigenvalues.float().cpu()
        ),
        "complete": True,
    }
    output = Path(os.environ["HY_K_PCA_TAIL_SCORE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    _K_PCA_TAIL_SCORE_TENSORS.clear()
    print(
        "[KPCATailScore] "
        f"L{layer} chunk={chunk} step={step} rows={count} "
        f"loss={payload['block_loss']:.8e} grad={payload['grad_norm']:.8e} "
        f"output={output}",
        flush=True,
    )
    return {"path": str(output), "rows": count, "layer": int(layer)}


def _capture_functional_decoder_anchor(
    layer_cache: Dict[str, Any], cache_name: str, prefix: str
) -> None:
    """Keep a deterministic sample of the immutable pre-FT decoder output."""
    anchor_name = f"_{prefix}_old_decoder_anchor_for_ft"
    if anchor_name in layer_cache:
        return
    value = layer_cache.get(cache_name)
    if not torch.is_tensor(value) or value.numel() == 0:
        return
    sample_count = int(
        os.environ.get("HY_FUNCTIONAL_BLOCK_FT_TRUST_SAMPLES", "65536")
    )
    flat = value.detach().reshape(-1)
    if 0 < sample_count < flat.numel():
        step = max(1, flat.numel() // sample_count)
        indices = torch.arange(0, flat.numel(), step, device=flat.device)[:sample_count]
        samples = flat.index_select(0, indices)
    else:
        indices = torch.arange(flat.numel(), device=flat.device)
        samples = flat
    layer_cache[anchor_name] = {
        "indices": indices.detach().cpu(),
        "values": samples.detach().to(device="cpu", dtype=torch.float32),
    }


def joint_functional_trust_anchor_loss_for_cache(
    kv_cache, layers
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Anchor joint FT to the deployed K/V decoder functions, not clean KV."""
    if kv_cache is None:
        return None, {}
    weights = {
        "k": float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_K_TRUST_WEIGHT", "0.0")),
        "v": float(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_V_TRUST_WEIGHT", "0.0")),
    }
    losses = []
    metrics: Dict[str, float] = {}
    for prefix in ("k", "v"):
        rels = []
        for layer_idx in layers:
            if int(layer_idx) >= len(kv_cache):
                continue
            layer_cache = kv_cache[int(layer_idx)]
            current = layer_cache.get(f"{prefix}_vision")
            anchor = layer_cache.get(f"_{prefix}_old_decoder_anchor_for_ft")
            if not torch.is_tensor(current) or not isinstance(anchor, dict):
                continue
            indices = anchor["indices"].to(device=current.device)
            target = anchor["values"].to(device=current.device)
            sampled = current.float().reshape(-1).index_select(0, indices)
            rel_mse = (sampled - target).pow(2).sum() / target.pow(2).sum().clamp_min(1e-8)
            rels.append(rel_mse)
            metrics[f"trust_rel_{prefix}_L{int(layer_idx)}"] = float(
                rel_mse.detach().sqrt().cpu()
            )
        if rels:
            mean_rel_mse = torch.stack(rels).mean()
            metrics[f"trust_rel_{prefix}"] = float(mean_rel_mse.detach().sqrt().cpu())
            if weights[prefix] > 0.0:
                losses.append(weights[prefix] * mean_rel_mse)
    return (torch.stack(losses).sum() if losses else None), metrics


def reapply_k_corruption_for_ft(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """Re-run K corruption on the clean K in the cache using updated codec weights.
    Used by functional ft inner loop to refresh corrupted K after optimizer step."""
    start_chunk = int(os.environ.get("HY_KV_CORRUPTION_K_START_CHUNK", "0"))
    current_chunk = int(os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1"))
    if start_chunk > 0 and (current_chunk < 0 or current_chunk < start_chunk):
        return layer_cache
    clean_k = layer_cache.get("_k_vision_clean_for_ft")
    thw = layer_cache.get("vision_thw")
    if clean_k is None or thw is None:
        return layer_cache
    _capture_functional_decoder_anchor(layer_cache, "k_vision", "k")
    production_key = "_k_production_base_for_ft"
    if production_key not in layer_cache:
        layer_cache[production_key] = layer_cache["k_vision"].detach().to("cpu")
    _FUNCTIONAL_PRODUCTION_BASELINES[("k", int(layer_idx))] = layer_cache[
        production_key
    ].to(device=clean_k.device, dtype=clean_k.dtype)
    mode = os.environ.get("HY_KV_CORRUPTION_K_MODE", "dcae").strip().lower()
    track_grad = not _env_flag("HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY", "0")
    # Ensure explicit Student reapply can override the rollout no_grad context,
    # while evaluation and resident-cache rebuilds never retain codec graphs.
    try:
        with torch.set_grad_enabled(track_grad):
            recon = _layerwise_reconstruct_k(
                clean_k,
                int(layer_idx),
                thw,
                track_grad=track_grad,
            )
    finally:
        _FUNCTIONAL_PRODUCTION_BASELINES.pop(("k", int(layer_idx)), None)
    if recon is not None:
        layer_cache["_k_vision_base_for_ft"] = recon
        layer_cache["k_vision"] = recon
        if (
            track_grad
            and _env_flag("HY_K_PCA_TAIL_SCORE_ENABLE", "0")
            and int(layer_idx)
            == int(os.environ.get("HY_K_PCA_TAIL_SCORE_LAYER", "-1"))
        ):
            _K_PCA_TAIL_SCORE_TENSORS[int(layer_idx)] = {
                "clean_pre_rope": clean_k,
                "recon_pre_rope": recon,
            }
        if _env_flag("HY_FUNCTIONAL_FINAL_PRED_GRAD_AUDIT", "0"):
            _V_FINAL_PRED_TENSOR_AUDIT.setdefault(int(layer_idx), {})[
                "attention_k"
            ] = layer_cache["k_vision"]
        if track_grad and _env_flag("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT", "0"):
            if not recon.requires_grad or recon.grad_fn is None:
                raise RuntimeError(
                    f"K functional attention input lost autograd graph at layer {int(layer_idx)}"
                )
    return layer_cache


def _layerwise_reconstruct_v_for_ft(
    tensor: torch.Tensor,
    layer_idx: int,
    thw,
    *,
    track_grad: bool = False,
) -> Optional[torch.Tensor]:
    ckpt_dir = os.environ.get("HY_LAYERWISE_RAW_V_AE_CKPT_DIR", "").strip()
    if not ckpt_dir:
        return None
    variant = os.environ.get("HY_LAYERWISE_RAW_V_AE_MODEL_VARIANT", "dcae_d512_8x")
    layer_dir = Path(ckpt_dir) / f"layer_{int(layer_idx):02d}"
    ft_ckpt_dir = os.environ.get("HY_V_FUNCTIONAL_FT_CKPT_DIR", "").strip()
    if ft_ckpt_dir:
        ft_candidate = Path(ft_ckpt_dir) / f"layer_{int(layer_idx):02d}" / "checkpoints" / "checkpoint_functional_ft.pt"
        candidates = [ft_candidate] if ft_candidate.exists() else []
    else:
        candidates = []
    candidates.extend(
        [
            layer_dir / "checkpoints" / "checkpoint_functional_ft.pt",
            layer_dir / "checkpoints" / "checkpoint_best.pt",
            layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_v_{variant}_step12000.pt",
            layer_dir / "checkpoints" / "checkpoint_final.pt",
            layer_dir / "checkpoints" / "checkpoint_last.pt",
            layer_dir / "checkpoint_best.pt",
        ]
    )
    ckpt_path = next((path for path in candidates if path.exists()), None)
    if ckpt_path is None:
        return None
    key = f"v_functional::{ckpt_path.resolve()}"
    if key not in _LAYERWISE_AE_MODELS:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt_data.get("model_config", ckpt_data.get("config", {}))
        model = build_pre_kv_ae(cfg)
        model.load_state_dict(ckpt_data["model"])
        model._hy_functional_layer_id = int(layer_idx)
        model._hy_functional_source_ckpt = str(ckpt_path)
        model._hy_functional_model_config = cfg
        model._hy_codec_metadata = dict(ckpt_data.get("codec_metadata", {}))
        trainable = _v_functional_layer_trainable(layer_idx)
        if trainable:
            model.train()
            trainable_parameters = _configure_v_functional_parameters(model, True)
            lr = float(os.environ.get("HY_LAYERWISE_RAW_V_AE_LR", "1e-5"))
            wd = float(os.environ.get("HY_LAYERWISE_RAW_V_AE_WEIGHT_DECAY", "0.0"))
            opt = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=wd)
            optimizer_state = ckpt_data.get("optimizer")
            optimizer_type = ckpt_data.get("optimizer_type", "adamw")
            if (
                optimizer_state
                and not _env_flag("HY_FUNCTIONAL_RESET_OPTIMIZER", "0")
                and optimizer_type == "adamw"
            ):
                opt.load_state_dict(optimizer_state)
            _K_TRAIN_OPTIMS[key] = opt
        else:
            model.eval()
            _configure_v_functional_parameters(model, False)
        _LAYERWISE_AE_MODELS[key] = model
    model = _LAYERWISE_AE_MODELS[key].to(device=tensor.device, dtype=tensor.dtype)
    trainable = _v_functional_layer_trainable(layer_idx)
    if trainable:
        model.train()
        trainable_parameters = _configure_v_functional_parameters(model, True)
        if key not in _K_TRAIN_OPTIMS:
            lr = float(os.environ.get("HY_LAYERWISE_RAW_V_AE_LR", "1e-5"))
            wd = float(os.environ.get("HY_LAYERWISE_RAW_V_AE_WEIGHT_DECAY", "0.0"))
            _K_TRAIN_OPTIMS[key] = torch.optim.AdamW(
                trainable_parameters, lr=lr, weight_decay=wd
            )
        _optimizer_state_to_parameter_devices(_K_TRAIN_OPTIMS[key])
    else:
        model.eval()
        _configure_v_functional_parameters(model, False)
    # The semantic V codec operates on the residual after subtracting a
    # frozen, per-token C64 center.  Reproduce that deployed coordinate here;
    # otherwise functional fine-tuning would optimize the C64 weights on raw V.
    semantic_centers = None
    semantic_ids = None
    codec_input = tensor
    if _env_flag("HY_LAYERWISE_RAW_V_AE_SEMANTIC_SMOOTH", "0"):
        from .kv_codecs import _joint_semantic_smooth

        codec_input, semantic_centers, semantic_ids = _joint_semantic_smooth(
            tensor.detach(), int(layer_idx)
        )

    # Resident-cache rebuilds must never retain codec graphs. Gradients are
    # needed only for the explicit inner-loop reapply immediately before the
    # Student functional forward.
    with torch.set_grad_enabled(trainable and track_grad):
        z = model.encode(codec_input.contiguous(), thw)
        encoder_output = z
        bin_ref_root = os.environ.get(
            "HY_V_FUNCTIONAL_BIN_REFERENCE_CKPT_DIR", ""
        ).strip()
        bin_audit_step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
        if (
            bin_ref_root
            and _env_flag("HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY", "0")
            and bin_audit_step == 0
            and _layerwise_target_latent_int8(
                "v", bool(getattr(model, "is_int8_code_ae", False))
            )
        ):
            with torch.no_grad():
                ref_model = _V_BIN_REFERENCE_MODELS.get(int(layer_idx))
                if ref_model is None:
                    ref_dir = (
                        Path(bin_ref_root)
                        / f"layer_{int(layer_idx):02d}"
                        / "checkpoints"
                    )
                    ref_path = ref_dir / "checkpoint_functional_ft.pt"
                    if not ref_path.exists():
                        ref_path = ref_dir / "checkpoint_best.pt"
                    ref_payload = torch.load(ref_path, map_location="cpu")
                    ref_model = build_pre_kv_ae(
                        ref_payload.get("model_config", ref_payload.get("config", {}))
                    )
                    ref_model.load_state_dict(ref_payload["model"])
                    ref_model.eval().requires_grad_(False)
                    _V_BIN_REFERENCE_MODELS[int(layer_idx)] = ref_model
                ref_model = ref_model.to(device=tensor.device, dtype=tensor.dtype)
                ref_z = ref_model.encode(tensor.detach().contiguous(), thw)
                candidate_codes, candidate_scale = _quantize_latent_int8(
                    encoder_output.detach()
                )
                reference_codes, reference_scale = _quantize_latent_int8(ref_z)
                changed = int((candidate_codes != reference_codes).sum().cpu())
                total = int(candidate_codes.numel())
                layer_key = f"functional_int8_bin_changed_L{int(layer_idx)}"
                total_key = f"functional_int8_bin_total_L{int(layer_idx)}"
                _STATS[layer_key] = int(_STATS.get(layer_key, 0)) + changed
                _STATS[total_key] = int(_STATS.get(total_key, 0)) + total
                scale_rel = (
                    (candidate_scale.float() - reference_scale.float()).norm()
                    / reference_scale.float().norm().clamp_min(1e-8)
                )
                scale_key = f"functional_int8_scale_rel_sum_L{int(layer_idx)}"
                count_key = f"functional_int8_audit_calls_L{int(layer_idx)}"
                _STATS[scale_key] = float(_STATS.get(scale_key, 0.0)) + float(
                    scale_rel.cpu()
                )
                _STATS[count_key] = int(_STATS.get(count_key, 0)) + 1
        if _layerwise_target_latent_int8_token("v"):
            z = _latent_int8_token_ste(z)
        elif _layerwise_target_latent_int4_token("v"):
            z = _latent_int4_token_ste(z)
        elif _layerwise_target_latent_int4("v"):
            z = _latent_int4_ste(z)
        elif _layerwise_target_latent_int8("v", bool(getattr(model, "is_int8_code_ae", False))):
            z = _latent_int8_ste(z)
        quantized_latent = z
        out = model.decode(z, tuple(tensor.shape), tuple(int(v) for v in thw)).to(tensor.dtype)
        if semantic_centers is not None:
            batch, heads, sequence, dim = tensor.shape
            assigned = torch.gather(
                semantic_centers.to(device=out.device, dtype=torch.float32),
                1,
                semantic_ids.to(device=out.device, dtype=torch.long)[..., None].expand(
                    -1, -1, heads * dim
                ),
            )
            assigned = (
                assigned.reshape(batch, sequence, heads, dim)
                .permute(0, 2, 1, 3)
                .to(dtype=out.dtype)
            )
            out = out + assigned
        if trainable and track_grad:
            out = _apply_functional_decoder_delta_to_production_base(
                out, quantized_latent, model, "v", int(layer_idx)
            )
        if track_grad and _env_flag("HY_FUNCTIONAL_FINAL_PRED_GRAD_AUDIT", "0"):
            _V_FINAL_PRED_TENSOR_AUDIT[int(layer_idx)] = {
                "encoder_output": encoder_output,
                "quantized_latent": quantized_latent,
                "decoder_output": out,
            }
    return out


def reapply_v_corruption_for_ft(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    clean_v = layer_cache.get("_v_vision_clean_for_ft")
    thw = layer_cache.get("vision_thw")
    if clean_v is None or thw is None:
        return layer_cache
    _capture_functional_decoder_anchor(layer_cache, "v_vision", "v")
    production_key = "_v_production_base_for_ft"
    if production_key not in layer_cache:
        layer_cache[production_key] = layer_cache["v_vision"].detach().to("cpu")
    _FUNCTIONAL_PRODUCTION_BASELINES[("v", int(layer_idx))] = layer_cache[
        production_key
    ].to(device=clean_v.device, dtype=clean_v.dtype)
    track_grad = not _env_flag("HY_FUNCTIONAL_BLOCK_FT_EVAL_ONLY", "0")
    try:
        with torch.set_grad_enabled(track_grad):
            recon = _layerwise_reconstruct_v_for_ft(
                clean_v, int(layer_idx), thw, track_grad=track_grad
            )
    finally:
        _FUNCTIONAL_PRODUCTION_BASELINES.pop(("v", int(layer_idx)), None)
    if recon is not None:
        layer_cache["_v_vision_base_for_ft"] = recon
        scale_spec = os.environ.get("HY_V_FUNCTIONAL_ERROR_SCALES", "1.0").strip()
        scales = [float(item.strip()) for item in scale_spec.split(",") if item.strip()]
        if not scales:
            scales = [1.0]
        inner_idx = int(os.environ.get("HY_FUNCTIONAL_BLOCK_FT_INNER_INDEX", "0"))
        error_scale = scales[inner_idx % len(scales)]
        # The rollout caller normally runs under no_grad. Keep the explicit
        # functional reapply arithmetic inside enable_grad as well; otherwise
        # ``recon`` retains a graph in _v_vision_base_for_ft while the actual
        # v_vision consumed by attention is silently detached here.
        with torch.set_grad_enabled(track_grad):
            layer_cache["v_vision"] = clean_v + error_scale * (recon - clean_v)
        if _env_flag("HY_FUNCTIONAL_FINAL_PRED_GRAD_AUDIT", "0"):
            _V_FINAL_PRED_TENSOR_AUDIT.setdefault(int(layer_idx), {})[
                "attention_v"
            ] = layer_cache["v_vision"]
        if track_grad and _env_flag("HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT", "0"):
            if not recon.requires_grad or recon.grad_fn is None:
                raise RuntimeError(
                    f"V functional recon lost autograd graph at layer {int(layer_idx)}"
                )
            actual_v = layer_cache["v_vision"]
            if not actual_v.requires_grad or actual_v.grad_fn is None:
                raise RuntimeError(
                    f"V functional attention input lost autograd graph at layer {int(layer_idx)}"
                )
        layer_cache["_v_functional_error_scale"] = float(error_scale)
    return layer_cache


def detach_k_functional_cache_for_rollout(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """Refresh K with current codec weights and detach resident cache for continued rollout."""
    clean_k = layer_cache.get("_k_vision_clean_for_ft")
    thw = layer_cache.get("vision_thw")
    if clean_k is None or thw is None:
        k_vision = layer_cache.get("k_vision")
        if torch.is_tensor(k_vision):
            layer_cache["k_vision"] = k_vision.detach()
        return layer_cache
    with torch.no_grad():
        recon = _layerwise_reconstruct_k(
            clean_k, int(layer_idx), thw, track_grad=False
        )
    if recon is not None:
        detached = recon.detach()
        layer_cache["_k_vision_base_for_ft"] = detached
        layer_cache["k_vision"] = detached
    return layer_cache


def detach_v_functional_cache_for_rollout(layer_cache: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    clean_v = layer_cache.get("_v_vision_clean_for_ft")
    thw = layer_cache.get("vision_thw")
    if clean_v is None or thw is None:
        v_vision = layer_cache.get("v_vision")
        if torch.is_tensor(v_vision):
            layer_cache["v_vision"] = v_vision.detach()
        return layer_cache
    with torch.no_grad():
        recon = _layerwise_reconstruct_v_for_ft(
            clean_v, int(layer_idx), thw, track_grad=False
        )
    if recon is not None:
        detached = recon.detach()
        layer_cache["_v_vision_base_for_ft"] = detached
        layer_cache["v_vision"] = detached
    return layer_cache


def save_k_functional_train_checkpoints(output_root: str) -> Dict[str, str]:
    # Also save to HY_K_FUNCTIONAL_FT_CKPT_DIR if separately set
    ft_ckpt_dir = os.environ.get("HY_K_FUNCTIONAL_FT_CKPT_DIR", "").strip()
    out_root = Path(output_root)
    saved: Dict[str, str] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("k_corruption::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is None:
            continue
        layer_dir = out_root / f"layer_{int(layer_id):02d}" / "checkpoints"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / "checkpoint_functional_ft.pt"
        opt = _K_TRAIN_OPTIMS.get(key)
        payload = {
            "model": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "model_config": getattr(model, "_hy_functional_model_config", {}),
            "layer_id": int(layer_id),
            "target": "k",
            "source_checkpoint": getattr(model, "_hy_functional_source_ckpt", ""),
            "optimizer": None if opt is None else opt.state_dict(),
            "optimizer_type": os.environ.get(
                "HY_KV_CORRUPTION_K_OPTIMIZER", "adamw"
            ).strip().lower(),
            "parameter_dtypes": sorted(
                {str(param.dtype) for param in model.parameters()}
            ),
            "train_args": {
                "mode": "rollout_block_functional_ft",
                "lr": float(os.environ.get("HY_KV_CORRUPTION_K_LR", "1e-5")),
                "fp32_master": _env_flag("HY_KV_CORRUPTION_K_FP32_MASTER", "0"),
            },
        }
        codec_metadata = getattr(model, "_hy_codec_metadata", None)
        if isinstance(codec_metadata, dict):
            payload["model_kind"] = "global_pca_int4_linear_a1"
            payload["codec_metadata"] = dict(codec_metadata)
        torch.save(payload, path)
        saved[str(int(layer_id))] = str(path)
        # Also save to HY_K_FUNCTIONAL_FT_CKPT_DIR if set and different
        if ft_ckpt_dir and str(ft_ckpt_dir) != str(out_root):
            ft_layer_dir = Path(ft_ckpt_dir) / f"layer_{int(layer_id):02d}" / "checkpoints"
            ft_layer_dir.mkdir(parents=True, exist_ok=True)
            ft_path = ft_layer_dir / "checkpoint_functional_ft.pt"
            torch.save(payload, ft_path)
            saved[f"{int(layer_id)}_ft"] = str(ft_path)
    return saved


def save_v_functional_train_checkpoints(output_root: str) -> Dict[str, str]:
    ft_ckpt_dir = os.environ.get("HY_V_FUNCTIONAL_FT_CKPT_DIR", "").strip()
    out_root = Path(output_root)
    saved: Dict[str, str] = {}
    for key, model in _LAYERWISE_AE_MODELS.items():
        if not key.startswith("v_functional::"):
            continue
        layer_id = getattr(model, "_hy_functional_layer_id", None)
        if layer_id is None:
            continue
        payload = {
            "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "model_config": getattr(model, "_hy_functional_model_config", {}),
            "layer_id": int(layer_id),
            "target": "v",
            "source_checkpoint": getattr(model, "_hy_functional_source_ckpt", ""),
            "train_args": {
                "mode": "rollout_block_functional_ft",
                "lr": float(os.environ.get("HY_LAYERWISE_RAW_V_AE_LR", "1e-5")),
                "latent_int8": _env_flag("HY_LAYERWISE_RAW_V_AE_LATENT_INT8", "0"),
                "latent_int8_token": _env_flag("HY_LAYERWISE_RAW_V_AE_LATENT_INT8_TOKEN", "0"),
                "latent_int4": _env_flag("HY_LAYERWISE_RAW_V_AE_LATENT_INT4", "0"),
                "latent_int4_token": _env_flag("HY_LAYERWISE_RAW_V_AE_LATENT_INT4_TOKEN", "0"),
            },
        }
        opt = _K_TRAIN_OPTIMS.get(key)
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        codec_metadata = getattr(model, "_hy_codec_metadata", None)
        if isinstance(codec_metadata, dict):
            payload["model_kind"] = "global_pca_int4_linear_a1"
            payload["codec_metadata"] = dict(codec_metadata)
        layer_dir = out_root / f"layer_{int(layer_id):02d}" / "checkpoints"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / "checkpoint_functional_ft.pt"
        torch.save(payload, path)
        saved[str(int(layer_id))] = str(path)
        if ft_ckpt_dir and str(ft_ckpt_dir) != str(out_root):
            ft_layer_dir = Path(ft_ckpt_dir) / f"layer_{int(layer_id):02d}" / "checkpoints"
            ft_layer_dir.mkdir(parents=True, exist_ok=True)
            ft_path = ft_layer_dir / "checkpoint_functional_ft.pt"
            torch.save(payload, ft_path)
            saved[f"{int(layer_id)}_ft"] = str(ft_path)
    return saved


def _quantize_per_token_int8_for_code_ae(tensor: torch.Tensor):
    x = tensor.float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / 127.0
    codes = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return codes, scale.to(dtype=tensor.dtype)


def _normalized_code_from_int8(codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() / 127.0).to(dtype=dtype)


def _round_normalized_code_ste(pred_norm: torch.Tensor) -> torch.Tensor:
    clipped = pred_norm.float().clamp(-1.0, 1.0)
    rounded = (clipped * 127.0).round().clamp(-127.0, 127.0) / 127.0
    return clipped + (rounded - clipped).detach()


def _layerwise_ckpt_path(root: str, layer_idx: int, variant: str) -> Optional[Path]:
    cache_key = (str(root), int(layer_idx), str(variant))
    use_metadata_cache = _env_flag("HY_KV_EFF_OPT_METADATA_CACHE", "0")
    if use_metadata_cache and cache_key in _LAYERWISE_CKPT_PATH_CACHE:
        return _LAYERWISE_CKPT_PATH_CACHE[cache_key]
    layer_dir = Path(root) / f"layer_{int(layer_idx):02d}"
    candidates = [
        layer_dir / "checkpoints" / "checkpoint_functional_ft.pt",
        layer_dir / "checkpoints" / "checkpoint_best.pt",
        layer_dir / "checkpoints" / f"layer{int(layer_idx):02d}_v_{variant}_step12000.pt",
        layer_dir / "checkpoints" / "checkpoint_final.pt",
        layer_dir / "checkpoints" / "checkpoint_last.pt",
        layer_dir / "checkpoint_best.pt",
    ]
    for path in candidates:
        if path.exists():
            if use_metadata_cache:
                _LAYERWISE_CKPT_PATH_CACHE[cache_key] = path
            return path
    pattern = f"layer{int(layer_idx):02d}_*_{variant}_step*.pt"
    matches = sorted((layer_dir / "checkpoints").glob(pattern))
    result = matches[-1] if matches else None
    if use_metadata_cache:
        _LAYERWISE_CKPT_PATH_CACHE[cache_key] = result
    return result

@torch.no_grad()
def maybe_decompress(obj: Any, dtype: torch.dtype, device) -> Any:
    if not isinstance(obj, dict) or not obj.get("kv_codec_training"):
        if isinstance(obj, dict) and obj.get("kv_codec"):
            from hyvideo.models.transformers.modules.kv_codecs import (
                decompress_kv_tensor,
            )

            return decompress_kv_tensor(obj, dtype=dtype, device=device)
        if isinstance(obj, dict) and obj.get("layerwise_ae"):
            if obj.get("latent_int8_token_storage"):
                z = _dequantize_latent_int8_token(
                    obj["latent_int8_token"].to(device=device),
                    obj["latent_scale"].to(device=device, dtype=dtype),
                    dtype,
                )
            elif obj.get("latent_int4_token_storage"):
                z = _dequantize_latent_int4_token(
                    obj["latent_int4_token"].to(device=device),
                    obj["latent_scale"].to(device=device, dtype=dtype),
                    dtype,
                )
            elif obj.get("latent_int4_storage"):
                z = _dequantize_latent_int4(
                    obj["latent_int4"].to(device=device),
                    obj["latent_scale"].to(device=device, dtype=dtype),
                    dtype,
                )
            elif obj.get("latent_int8_storage"):
                z = _dequantize_latent_int8(
                    obj["latent_int8"].to(device=device),
                    obj["latent_scale"].to(device=device, dtype=dtype),
                    dtype,
                )
            else:
                z = obj["latent"].to(device=device, dtype=dtype)
            layer_idx = int(obj.get("layer_idx", 0))
            ckpt_dir = obj.get(
                "ckpt_dir",
                os.environ.get(
                    "HY_LAYERWISE_RAW_V_AE_CKPT_DIR",
                    "kv_ae_outputs/ae16x_v2/layerwise_raw_v_12000",
                ),
            )
            model_key = (
                f"layerwise::{Path(str(ckpt_dir)).resolve()}::"
                f"{obj.get('model_variant', 'dcae_d128_8x')}_layer_{layer_idx:02d}"
            )
            if model_key not in _LAYERWISE_AE_MODELS:
                variant = obj.get("model_variant", "dcae_d128_8x")
                ckpt_path = _layerwise_ckpt_path(ckpt_dir, layer_idx, variant)
                if ckpt_path is None:
                    return obj
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                cfg = ckpt_data.get("model_config", ckpt_data.get("config", obj.get("model_config", {})))
                model = build_pre_kv_ae(cfg)
                model.load_state_dict(ckpt_data["model"])
                model.to(device=device, dtype=dtype)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                _LAYERWISE_AE_MODELS[model_key] = model
            model = _LAYERWISE_AE_MODELS[model_key]
            first_parameter = next(model.parameters(), None)
            already_placed = (
                first_parameter is not None
                and first_parameter.device == torch.device(device)
                and first_parameter.dtype == dtype
            )
            if not (
                _env_flag("HY_KV_EFF_OPT_SKIP_REDUNDANT_TO", "0")
                and already_placed
            ):
                model = model.to(device=device, dtype=dtype)
            start = time.perf_counter()
            if obj.get("decode_prefix_hoisted"):
                out_norm = model.decode_prepared(
                    z, tuple(obj["shape"]), tuple(obj["thw"])
                )
            else:
                out_norm = model.decode(z, tuple(obj["shape"]), tuple(obj["thw"]))
            if obj.get("int8_code_ae"):
                restored_norm = _round_normalized_code_ste(out_norm)
                scale = obj["int8_scale"].to(device=device, dtype=dtype)
                out = (restored_norm.float() * 127.0 * scale.float()).to(dtype=dtype)
            else:
                out = out_norm
            if "oracle_clean" in obj:
                clean = obj["oracle_clean"].to(device=device, dtype=dtype)
                error_scale = float(obj.get("oracle_error_scale", 1.0))
                chunk_spec = os.environ.get(
                    "HY_LAYERWISE_RAW_V_AE_ERROR_SCALE_CHUNKS", ""
                ).strip()
                if chunk_spec:
                    active_chunks = _parse_layers(chunk_spec)
                    current_chunk = int(
                        os.environ.get("HY_KV_MULTIPOS_AR_CHUNK_ID", "-1")
                    )
                    if current_chunk not in active_chunks:
                        error_scale = 1.0
                oracle_layers_spec = os.environ.get(
                    "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_LAYERS", ""
                ).strip()
                oracle_steps_spec = os.environ.get(
                    "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_STEPS", ""
                ).strip()
                oracle_chunks_spec = os.environ.get(
                    "HY_LAYERWISE_RAW_V_AE_CLEAN_ORACLE_CHUNKS", ""
                ).strip()
                if oracle_layers_spec and oracle_steps_spec:
                    current_layer = int(obj.get("layer_idx", -1))
                    current_step = int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1"))
                    current_chunk = int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1"))
                    layer_active = current_layer in _parse_layers(oracle_layers_spec)
                    step_active = current_step in _parse_layers(oracle_steps_spec)
                    chunk_active = (
                        not oracle_chunks_spec
                        or current_chunk in _parse_layers(oracle_chunks_spec)
                    )
                    if layer_active and step_active and chunk_active:
                        error_scale = 0.0
                out = clean + error_scale * (out - clean)
            if device.type == "cuda" and _timing_profile_enabled():
                torch.cuda.synchronize(device)
            decode_elapsed = time.perf_counter() - start
            _STATS["decoder_time_s"] += decode_elapsed
            target = obj.get("target")
            if target in {"k", "v"}:
                _STATS[f"layerwise_{target}_decoder_time_s"] += decode_elapsed
                _STATS[f"layerwise_{target}_decoder_calls"] += 1
            _STATS["decompress_calls"] += 1
            return out
        return obj
    z = obj["latent"].to(device=device, dtype=dtype)
    checkpoint_path = Path(obj.get("checkpoint_path") or os.environ.get(
        "HY_KV_AE3D_CKPT", "kv_ae_outputs/ae3d/prekv_ae3d.pt"
    ))
    model = _model(device, dtype, checkpoint_path)
    if device.type == "cuda" and _timing_profile_enabled():
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    out = model.decode(z, tuple(obj["shape"]), tuple(obj["thw"]))
    if obj.get("residual_codec"):
        from hyvideo.models.transformers.modules.kv_codecs import (
            dequantize_kv_tensor,
            unpack_quantized_codes,
        )

        residual_std = obj.get("residual_std")
        if residual_std is not None:
            out = out * residual_std
        base_codes = unpack_quantized_codes(
            obj["base_packed_codes"].to(device=device),
            int(obj["base_bits"]),
            int(obj["base_numel"]),
            tuple(obj["base_shape"]),
        )
        base = dequantize_kv_tensor(
            base_codes,
            obj["base_scale"].to(device=device),
            dtype,
        )
        out = (base + out).to(dtype)
    else:
        normalization = obj.get("normalization")
        if normalization is not None:
            mean, std = normalization
            out = out * std + mean
    if device.type == "cuda" and _timing_profile_enabled():
        torch.cuda.synchronize(device)
    _STATS["decoder_time_s"] += time.perf_counter() - start
    _STATS["decompress_calls"] += 1
    return out


def write_summary() -> None:
    if not (
        _env_flag("HY_KV_AE3D_PROFILE", "0")
        or enabled()
        or _corruption_enabled()
        or _k_corruption_enabled()
        or bool(_STATS.get("layerwise_enabled"))
    ):
        return
    out_path = Path(os.environ.get("HY_KV_AE3D_SUMMARY", "kv_ae_outputs/ae3d/online_summary.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(_STATS)
    if _STATS["resident_ae_bytes"]:
        out["resident_ratio"] = _STATS["resident_bf16_bytes"] / max(1, _STATS["resident_ae_bytes"])
    if _STATS["resident_effective_bytes"]:
        out["effective_resident_ratio"] = _STATS["resident_original_total_bytes"] / max(
            1, _STATS["resident_effective_bytes"]
        )
    if _STATS["resident_estimated_effective_bytes"]:
        out["estimated_effective_resident_ratio"] = _STATS[
            "resident_original_total_bytes"
        ] / max(1.0, _STATS["resident_estimated_effective_bytes"])
    if _STATS["resident_estimated_codec_bytes"]:
        out["estimated_target_codec_ratio"] = _STATS["resident_bf16_bytes"] / max(
            1.0, _STATS["resident_estimated_codec_bytes"]
        )
    if _STATS["layerwise_payload_bytes"]:
        out["layerwise_effective_ratio"] = _STATS["layerwise_original_bytes"] / max(
            1, _STATS["layerwise_payload_bytes"]
        )
    if _STATS["layerwise_k_payload_bytes"]:
        out["layerwise_k_effective_ratio"] = _STATS["layerwise_k_original_bytes"] / max(
            1, _STATS["layerwise_k_payload_bytes"]
        )
    if _STATS["layerwise_v_payload_bytes"]:
        out["layerwise_v_effective_ratio"] = _STATS["layerwise_v_original_bytes"] / max(
            1, _STATS["layerwise_v_payload_bytes"]
        )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if _corruption_enabled() or _k_corruption_enabled():
        corr_path = Path(
            os.environ.get(
                "HY_KV_CORRUPTION_SUMMARY",
                "kv_ae_outputs/ae16x_v2/phase4kv_calib/corruption_summary.json",
            )
        )
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        corr = dict(_K_CORRUPTION_STATS if _k_corruption_enabled() else _CORRUPTION_STATS)
        calls = max(1, int(corr.get("calls", 0)))
        for key in (
            "actual_rel_l2",
            "cosine",
            "l1",
            "dc_rel_l2",
            "ac_rel_l2",
            "ac_error_ratio",
        ):
            corr[key + "_mean"] = corr.get("sum_" + key, 0.0) / calls
        for layer, row in corr.get("per_layer", {}).items():
            layer_calls = max(1, int(row.get("calls", 0)))
            for key in (
                "actual_rel_l2",
                "cosine",
                "l1",
                "dc_rel_l2",
                "ac_rel_l2",
                "ac_error_ratio",
            ):
                row[key + "_mean"] = row.get("sum_" + key, 0.0) / layer_calls
        corr_path.write_text(json.dumps(corr, indent=2), encoding="utf-8")
