"""Artifact construction helpers; these functions do not collect or train data."""
from __future__ import annotations

import torch


def initialize_reader_state(k_decoder, v_decoder, groups, *, heads=24, native_dim=128):
    """Create a decoder-slice initialization for later teacher calibration.

    ``groups`` is a list of ``(head_ids, native_dimension_indices)`` tensors.
    The returned state is intentionally initialization-only: the manifest must
    remain ``calibrated:false`` until held-out and closed-loop gates pass.
    """
    rank = int(k_decoder.shape[1])
    if tuple(k_decoder.shape) != (heads * native_dim, rank) or tuple(v_decoder.shape) != tuple(k_decoder.shape):
        raise ValueError("decoder shape does not match MG3.5 joint K/V layout")
    k_decoder = k_decoder.float().reshape(heads, native_dim, rank)
    v_decoder = v_decoder.float().reshape(heads, native_dim, rank)
    payload = {"heads": heads, "native_dim": native_dim, "rank": rank, "groups": []}
    for head_ids, indices in groups:
        head_ids, indices = head_ids.long(), indices.long()
        if indices.shape[0] != head_ids.numel():
            raise ValueError("each selected head needs one native-dimension map")
        width = indices.shape[1]
        identity = torch.zeros((head_ids.numel(), native_dim, width))
        identity.scatter_(1, indices[:, None, :], 1.0)
        gather = indices[..., None].expand(-1, -1, rank)
        payload["groups"].append({
            "heads": head_ids, "indices": indices,
            "q_reduce": identity.clone(),
            "k_read": k_decoder.index_select(0, head_ids).gather(1, gather),
            "v_read": v_decoder.index_select(0, head_ids).gather(1, gather),
            "k_affine": identity.clone(), "v_affine": identity.clone(),
            "lift": identity.transpose(1, 2).contiguous(),
        })
    return payload
