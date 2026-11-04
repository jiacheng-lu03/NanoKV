"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch

from position_utils import rel_l2


def unpack_weight(entry: dict, name: str, device: torch.device) -> torch.Tensor:
    packed = entry[name]
    qweight = packed["qweight"].to(device)
    scale = packed["scale"].to(device)
    group = int(packed["group_size"])
    return (
        qweight.reshape(qweight.shape[0], -1, group).to(torch.bfloat16)
        * scale[:, :, None]
    ).reshape_as(qweight).float()


def load_operator(container: Path, layer: int, target: str, device: torch.device) -> dict:
    payload = torch.load(container, map_location="cpu", weights_only=False)
    rank = int(payload["rank"])
    if rank not in {256, 512}:
        raise ValueError(f"codec-native fitting supports rank256/rank512, got {rank}")
    entry = payload["entries"][f"{target}:{layer:02d}"]
    return {
        "mean": entry["mean"].to(device=device, dtype=torch.float32),
        "encoder": unpack_weight(entry, "encoder", device),
        "decoder": unpack_weight(entry, "decoder", device),
        "rank": rank,
    }


def rope_channel_indices(dim: int, device: torch.device) -> torch.Tensor:
    if dim <= 0 or dim > 128 or dim % 4:
        raise ValueError(f"invalid reduced RoPE dimension {dim}")
    temporal = int(round((dim * 16 / 128) / 2) * 2)
    temporal = max(2, min(16, temporal))
    remaining = dim - temporal
    height = (remaining // 4) * 2
    targets = (temporal, height, remaining - height)
    channels = []
    for (start, end), target in zip(((0, 16), (16, 72), (72, 128)), targets):
        source_pairs = (end - start) // 2
        pair_ids = torch.linspace(0, source_pairs - 1, steps=target // 2).round().long()
        pair_channels = torch.stack((2 * pair_ids, 2 * pair_ids + 1), dim=-1).reshape(-1)
        channels.append(pair_channels + start)
    return torch.cat(channels).to(device=device)


def reduced_rope(x: torch.Tensor, payload: dict, suffix: str) -> torch.Tensor:
    from position_utils import rope_selected

    indices = rope_channel_indices(x.shape[-1], payload[f"freqs_cos_{suffix}"].device)
    local = dict(payload)
    local[f"freqs_cos_{suffix}"] = payload[f"freqs_cos_{suffix}"].index_select(-1, indices)
    local[f"freqs_sin_{suffix}"] = payload[f"freqs_sin_{suffix}"].index_select(-1, indices)
    return rope_selected(x, local, suffix)


def normalized_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.float() - y.float()).square().mean() / y.float().square().mean().clamp_min(1e-8)

