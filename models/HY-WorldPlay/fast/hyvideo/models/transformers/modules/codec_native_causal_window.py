"""Training-only formal Teacher views for codec-native causal windows.

The deployed dictionary-low-D cache deliberately drops the full C64 centre
table when the recent-native tail is captured raw.  That is correct for
inference, but it prevents a functional-FT Teacher from decoding the *same*
rank-512 INT4 payload to native H16D128.

This module keeps the boundary explicit and dormant by default:

* the resident Student payload remains the ordinary direct-codec payload;
* only selected training layers retain one full C64 centre table per K/V;
* INT4 codes, token scales, and semantic IDs are not duplicated;
* a short-lived formal-codec dict view is assembled only for the Teacher.

The Student writer is the sole numeric encoder.  Before that encode this
module captures only raw-tensor identity and statically audited decoder
metadata; the Teacher then aliases the Student's exact codes/scales/IDs/C64
table.  Accidental source, codec, or operator drift remains a hard failure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


_SIDECAR_KEY = "_codec_native_causal_formal_meta"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enabled() -> bool:
    return _flag("HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER")


def selected_layers() -> set[int]:
    text = os.environ.get(
        "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS",
        "27,28,29,30,31,32,33,34,35",
    ).strip()
    if not text:
        raise ValueError(
            "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS must not be empty"
        )
    try:
        layers = {int(item.strip()) for item in text.split(",") if item.strip()}
    except ValueError as error:
        raise ValueError(
            "HY_CODEC_NATIVE_CAUSAL_FORMAL_TEACHER_LAYERS must be a "
            "comma-separated integer list"
        ) from error
    if not layers or min(layers) < 0:
        raise ValueError(f"invalid formal Teacher layers: {sorted(layers)}")
    return layers


def prepare_formal_payloads(
    layer_cache: Mapping[str, Any], layer_idx: int
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Capture source identity/config only; never perform a second K/V encode."""

    if not enabled() or int(layer_idx) not in selected_layers():
        return None
    from hyvideo.models.transformers.modules.direct_codec_latent_attention import (
        formal_teacher_metadata,
    )

    result: Dict[str, Dict[str, Any]] = {}
    for cache_name, target in (("k_vision", "k"), ("v_vision", "v")):
        raw = layer_cache.get(cache_name)
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(
                "formal Teacher source identity must be captured before direct "
                f"compression: L{int(layer_idx)} {cache_name}={type(raw)}"
            )
        result[target] = {
            "source_identity": {
                "data_ptr": int(raw.data_ptr()),
                "version": int(raw._version),
                "shape": tuple(int(value) for value in raw.shape),
                "stride": tuple(int(value) for value in raw.stride()),
                "storage_offset": int(raw.storage_offset()),
                "dtype": str(raw.dtype),
                "device": str(raw.device),
            },
            "metadata": formal_teacher_metadata(
                int(layer_idx), target, shape=raw.shape, source_dtype=raw.dtype
            ),
        }
    return result


def _validate_single_payload(
    direct: Mapping[str, Any], prepared: Mapping[str, Any], layer_idx: int, target: str
) -> Dict[str, Any]:
    if direct.get("payload_mode") != "dictionary_lowkv":
        raise RuntimeError(
            f"L{int(layer_idx)} {target} is not dictionary_lowkv: "
            f"{direct.get('payload_mode')!r}"
        )
    metadata = prepared.get("metadata")
    captured_identity = prepared.get("source_identity")
    encoded_identity = direct.get("_formal_source_identity")
    if not isinstance(metadata, Mapping) or not isinstance(
        captured_identity, Mapping
    ) or not isinstance(encoded_identity, Mapping):
        raise RuntimeError(
            f"L{int(layer_idx)} {target} missing single-encode identity metadata"
        )
    if dict(captured_identity) != dict(encoded_identity):
        raise RuntimeError(
            f"L{int(layer_idx)} {target} raw source identity changed between "
            "descriptor capture and direct encode"
        )
    if tuple(direct.get("shape", ())) != tuple(metadata.get("shape", ())):
        raise RuntimeError(f"L{int(layer_idx)} {target} source shape mismatch")
    expected = {
        "kv_codec": True,
        "codec": "global_pca",
        "coeff_quant": "int4",
        "rank": 512,
        "semantic_smooth": True,
        "layer_idx": int(layer_idx),
        "tensor_name": target,
    }
    mismatches = {
        name: (metadata.get(name), expected_value)
        for name, expected_value in expected.items()
        if metadata.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"L{int(layer_idx)} {target} formal decoder metadata mismatch: "
            f"{mismatches}"
        )
    source_shape = tuple(int(value) for value in direct.get("shape", ()))
    if len(source_shape) != 4 or source_shape[1] != 16 or source_shape[3] != 128:
        raise RuntimeError(
            f"L{int(layer_idx)} {target} invalid native source shape={source_shape}"
        )
    latent_shape = tuple(int(value) for value in direct.get("latent_shape", ()))
    latent_numel = 1
    for value in latent_shape:
        latent_numel *= value
    if (
        len(latent_shape) != 3
        or latent_shape[:2] != (source_shape[0], source_shape[2])
        or latent_shape[-1] != 512
        or int(direct.get("latent_numel", -1)) != latent_numel
    ):
        raise RuntimeError(f"L{int(layer_idx)} {target} invalid rank512 latent shape")
    for name in ("latent_int4", "latent_scale", "semantic_ids"):
        if not isinstance(direct.get(name), torch.Tensor):
            raise RuntimeError(
                f"L{int(layer_idx)} {target} direct payload missing tensor {name}"
            )
    codes = direct["latent_int4"]
    scales = direct["latent_scale"]
    ids = direct["semantic_ids"]
    if codes.dtype != torch.uint8 or codes.numel() != (latent_numel + 1) // 2:
        raise RuntimeError(
            f"L{int(layer_idx)} {target} malformed packed INT4 storage"
        )
    if tuple(scales.shape) != (*latent_shape[:2], 1) or scales.dtype not in {
        torch.bfloat16,
        torch.float16,
        torch.float32,
    }:
        raise RuntimeError(
            f"L{int(layer_idx)} {target} malformed token-scale storage"
        )
    if tuple(ids.shape) != latent_shape[:2] or ids.dtype != torch.uint8:
        raise RuntimeError(
            f"L{int(layer_idx)} {target} malformed semantic-ID storage"
        )
    direct_centers = direct.get("semantic_centers")
    if not isinstance(direct_centers, torch.Tensor):
        raise RuntimeError(
            f"L{int(layer_idx)} {target} missing direct C64 centers"
        )
    if tuple(direct_centers.shape) != (source_shape[0], 64, 2048):
        raise RuntimeError(
            f"L{int(layer_idx)} {target} malformed C64 centers: "
            f"{tuple(direct_centers.shape)}"
        )
    return dict(metadata)


def attach_formal_metadata(
    layer_cache: Dict[str, Any],
    formal_payloads: Optional[Mapping[str, Mapping[str, Any]]],
    layer_idx: int,
) -> Dict[str, Any]:
    """Attach centre/metadata-only sidecars to a compressed direct cache."""

    if formal_payloads is None:
        return layer_cache
    if not layer_cache.get("direct_codec_latent"):
        raise RuntimeError(
            f"L{int(layer_idx)} formal metadata must attach after direct compression"
        )
    # A formal-anchor Student also needs the full C64 table to decode its
    # sparse native rows.  Keep a second Python reference to the exact same
    # tensor storage in that case; this does not clone or allocate the table.
    # With anchors disabled, preserve the established sidecar-only ownership.
    from hyvideo.models.transformers.modules.direct_codec_latent_attention import (
        _formal_anchor_policy_for_layer,
    )

    retain_student_centers = any(_formal_anchor_policy_for_layer(int(layer_idx)))
    sidecars: Dict[str, Dict[str, Any]] = {}
    for cache_name, target in (("k_vision", "k"), ("v_vision", "v")):
        direct = layer_cache.get(cache_name)
        prepared = formal_payloads.get(target)
        if not isinstance(direct, dict) or not isinstance(prepared, Mapping):
            raise RuntimeError(f"L{int(layer_idx)} missing {target} payload pair")
        metadata = _validate_single_payload(
            direct, prepared, int(layer_idx), target
        )
        # The Teacher owns the exact table used by the deployed direct encode.
        # Remove it from the Student payload after moving its sole reference to
        # the training-only sidecar.
        centers = direct["semantic_centers"]
        if not retain_student_centers:
            direct.pop("semantic_centers")
        # Never retain a second copy/reference to the large per-token payload.
        # The Teacher view below aliases the deployed direct codes/scale/IDs.
        direct.pop("_formal_source_identity")
        sidecars[target] = metadata
        sidecars[target]["semantic_centers"] = centers.detach()
    layer_cache[_SIDECAR_KEY] = sidecars
    layer_cache["_codec_native_causal_payload_verified"] = True
    layer_cache["_codec_native_causal_single_encode_verified"] = True
    return layer_cache


def _formal_view(
    direct: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    view = dict(metadata)
    view["latent_int8"] = direct["latent_int4"]
    view["latent_scale"] = direct["latent_scale"]
    view["semantic_ids"] = direct["semantic_ids"]
    view["latent_shape"] = tuple(int(value) for value in direct["latent_shape"])
    view["latent_numel"] = int(direct["latent_numel"])
    view["shape"] = tuple(int(value) for value in direct["shape"])
    return view


def build_formal_teacher_cache(
    kv_cache: Optional[Iterable[Mapping[str, Any]]],
    target_layers: Iterable[int],
) -> Optional[list[Dict[str, Any]]]:
    """Return a shallow cache whose selected layers use formal D128 decode.

    Non-selected layers continue to use the deployed Student cache.  This is
    the same-state group teacher required by prefix-reused physical windows.
    """

    if kv_cache is None:
        return None
    requested = {int(layer) for layer in target_layers}
    configured = selected_layers()
    if not requested.issubset(configured):
        raise RuntimeError(
            "formal Teacher requested layers outside retained sidecars: "
            f"requested={sorted(requested)} configured={sorted(configured)}"
        )
    out: list[Dict[str, Any]] = []
    for layer_idx, source in enumerate(kv_cache):
        copied = dict(source)
        if layer_idx in requested:
            sidecars = source.get(_SIDECAR_KEY)
            if not isinstance(sidecars, Mapping):
                raise RuntimeError(
                    f"L{layer_idx} has no verified formal Teacher sidecar"
                )
            for cache_name, target in (("k_vision", "k"), ("v_vision", "v")):
                direct = source.get(cache_name)
                metadata = sidecars.get(target)
                if not isinstance(direct, Mapping) or not isinstance(
                    metadata, Mapping
                ):
                    raise RuntimeError(f"L{layer_idx} malformed {target} sidecar")
                copied[cache_name] = _formal_view(direct, metadata)
            copied["direct_codec_latent"] = False
            copied["_codec_native_force_formal_teacher"] = True
            copied.pop("_direct_codec_positioned_history", None)
            copied.pop("_direct_codec_recent_native_pre", None)
        out.append(copied)
    return out


def sidecar_key() -> str:
    """Expose the stable key for static/unit audits without duplicating it."""

    return _SIDECAR_KEY
