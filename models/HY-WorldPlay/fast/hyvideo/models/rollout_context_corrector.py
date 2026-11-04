from __future__ import annotations

import math

import torch
from torch import nn


class RolloutContextCorrector(nn.Module):
    """Correct retrieved rollout latents with shared per-pixel temporal modeling."""

    def __init__(
        self,
        latent_channels: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        max_action_id: int = 32,
        use_gate: bool = True,
        use_pose: bool = False,
        channel_basis: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.use_gate = use_gate
        self.use_pose = use_pose
        if channel_basis is not None:
            if channel_basis.ndim != 2 or channel_basis.shape[0] != latent_channels:
                raise ValueError(
                    "channel_basis must have shape [latent_channels, rank], got "
                    f"{tuple(channel_basis.shape)}"
                )
            self.register_buffer("channel_basis", channel_basis.float(), persistent=True)
            residual_channels = int(channel_basis.shape[1])
        else:
            self.channel_basis = None
            residual_channels = latent_channels
        self.input_proj = nn.Linear(latent_channels, hidden_dim)
        self.meta_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_embed = nn.Embedding(2, hidden_dim)
        self.action_embed = nn.Embedding(max_action_id, hidden_dim)
        if use_pose:
            self.pose_proj = nn.Sequential(
                nn.Linear(16, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.residual_head = nn.Conv2d(hidden_dim, residual_channels, kernel_size=1)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def _pose_embedding(
        self,
        context_viewmats: torch.Tensor,
        context_Ks: torch.Tensor,
        query_viewmats: torch.Tensor,
        query_Ks: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context-camera geometry relative to the next query frame."""
        context_w2c = context_viewmats.float()
        query_w2c = query_viewmats[:, :1].float()
        context_c2w = torch.linalg.inv(context_w2c)
        context_to_query = torch.matmul(query_w2c, context_c2w)
        rotation = context_to_query[..., :3, :3].flatten(-2)
        translation = context_to_query[..., :3, 3]

        context_intrinsics = torch.stack(
            [
                context_Ks[..., 0, 0],
                context_Ks[..., 1, 1],
                context_Ks[..., 0, 2],
                context_Ks[..., 1, 2],
            ],
            dim=-1,
        ).float()
        query_K = query_Ks[:, :1]
        query_intrinsics = torch.stack(
            [
                query_K[..., 0, 0],
                query_K[..., 1, 1],
                query_K[..., 0, 2],
                query_K[..., 1, 2],
            ],
            dim=-1,
        ).float()
        intrinsic_delta = context_intrinsics - query_intrinsics
        return self.pose_proj(
            torch.cat([rotation, translation, intrinsic_delta], dim=-1)
        )

    def _metadata_embedding(
        self,
        frame_indices: torch.Tensor,
        distance_to_query: torch.Tensor,
        anchor_mask: torch.Tensor,
        action: torch.Tensor,
        boundary_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames = frame_indices.shape
        retrieval_position = torch.arange(
            frames, device=frame_indices.device, dtype=torch.float32
        ).expand(batch, -1)
        boundary = boundary_chunk.float().view(batch, 1).expand(-1, frames)
        metadata = torch.stack(
            [
                frame_indices.float() / 64.0,
                distance_to_query.float() / 64.0,
                retrieval_position / max(1, frames - 1),
                boundary / 16.0,
            ],
            dim=-1,
        )
        return (
            self.meta_proj(metadata)
            + self.anchor_embed(anchor_mask.long())
            + self.action_embed(action.long().clamp(0, self.action_embed.num_embeddings - 1))
        )

    def forward(
        self,
        context: torch.Tensor,
        frame_indices: torch.Tensor,
        distance_to_query: torch.Tensor,
        anchor_mask: torch.Tensor,
        action: torch.Tensor,
        boundary_chunk: torch.Tensor,
        context_viewmats: torch.Tensor | None = None,
        context_Ks: torch.Tensor | None = None,
        query_viewmats: torch.Tensor | None = None,
        query_Ks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(context.shape)}")
        batch, channels, frames, height, width = context.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}"
            )

        tokens = context.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, frames, channels
        )
        hidden = self.input_proj(tokens)
        metadata = self._metadata_embedding(
            frame_indices,
            distance_to_query,
            anchor_mask,
            action,
            boundary_chunk,
        )
        if self.use_pose:
            pose_values = (
                context_viewmats,
                context_Ks,
                query_viewmats,
                query_Ks,
            )
            if any(value is None for value in pose_values):
                raise ValueError("Pose-aware corrector requires camera matrices and intrinsics")
            metadata = metadata + self._pose_embedding(*pose_values)
        metadata = (
            metadata[:, None, None]
            .expand(-1, height, width, -1, -1)
            .reshape(batch * height * width, frames, self.hidden_dim)
        )
        hidden = self.temporal_norm(self.temporal(hidden + metadata))
        hidden_2d = hidden.reshape(batch, height, width, frames, self.hidden_dim)
        hidden_2d = hidden_2d.permute(0, 3, 4, 1, 2).reshape(
            batch * frames, self.hidden_dim, height, width
        )
        hidden_2d = hidden_2d + self.spatial(hidden_2d)

        raw_residual = self.residual_head(hidden_2d).reshape(
            batch, frames, -1, height, width
        )
        raw_residual = raw_residual.permute(0, 2, 1, 3, 4)
        if self.channel_basis is not None:
            raw_residual = torch.einsum(
                "cr,brthw->bcthw", self.channel_basis.to(raw_residual.dtype), raw_residual
            )
        pooled = hidden_2d.mean(dim=(-2, -1)).reshape(
            batch, frames, self.hidden_dim
        )
        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(pooled)).view(
                batch, 1, frames, 1, 1
            )
        else:
            gate = torch.ones(
                (batch, 1, frames, 1, 1),
                device=raw_residual.device,
                dtype=raw_residual.dtype,
            )
        correction = gate * raw_residual
        return context + correction, correction, gate

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
