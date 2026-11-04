"""Direct low-dimensional history attention over LingBot's frozen KV codec.

The feature is dormant unless ``LINGBOT_CODEC_NATIVE_ENABLE=1``.  Persistent
history remains exclusively the formal INT4+C64 codec payload.  The current
chunk and policy-protected history use native D128; old history is read from
the codec latent through calibrated, operator-bound per-head matrices.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .lingbot_v2_kv_codec import (
    decode_tensor_rows, direct_latent_rows, decode_tensor_slice,
    direct_latent_slice, encode_pair, _load_operator,
)
from .lingbot_v2_cache_runtime import (
    rope_pre_key_by_positions, _prune_segments, maybe_dump_committed_source,
)


def _enabled() -> bool:
    return os.environ.get("LINGBOT_CODEC_NATIVE_ENABLE", "0").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=8)
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge(a, la, b, lb):
    if os.environ.get("LINGBOT_CODEC_NATIVE_FUSED_MERGE", "0") == "1" and all(x.is_cuda and x.is_contiguous() for x in (a,la,b,lb)):
        from .codec_native_merge_triton import merge
        return merge(a,la,b,lb)
    lse = torch.logaddexp(la, lb)
    origin = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    out = a.float() * (la-origin).exp()[..., None] + b.float() * (lb-origin).exp()[..., None]
    return out.to(a.dtype), lse


def _attention_lse(q, k, v, *, scale: float, q_tile: int = 128, k_tile: int = 1024):
    if q.shape[-1] != v.shape[-1]:
        # Quality-only asymmetric prototype. Padding performs the larger
        # kernel width: never count nominal K/V MAC savings as measured speed.
        width=max(q.shape[-1], v.shape[-1]); vd=v.shape[-1]
        out,lse=_attention_lse(
            torch.nn.functional.pad(q,(0,width-q.shape[-1])),
            torch.nn.functional.pad(k,(0,width-k.shape[-1])),
            torch.nn.functional.pad(v,(0,width-v.shape[-1])),scale=scale,
            q_tile=q_tile,k_tile=k_tile)
        return out[...,:vd],lse
    if k.shape[-2] == 0:
        return torch.zeros((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=q.dtype), torch.full(q.shape[:-1], -math.inf, device=q.device)
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        if os.environ.get("LINGBOT_CODEC_NATIVE_BACKEND", "aten") == "fa2_varlen":
            from flash_attn import flash_attn_varlen_func
            batch, heads, nq, dim = q.shape
            nk = k.shape[2]
            cq = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * nq
            ck = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * nk
            out, lse, _ = flash_attn_varlen_func(
                q.transpose(1, 2).reshape(batch*nq, heads, dim),
                k.transpose(1, 2).reshape(batch*nk, heads, dim),
                v.transpose(1, 2).reshape(batch*nk, heads, dim),
                cq, ck, nq, nk, dropout_p=0.0, softmax_scale=scale,
                causal=False, return_attn_probs=True,
            )
            return out.reshape(batch, nq, heads, dim).transpose(1, 2), lse.reshape(heads, batch, nq).transpose(0, 1)
        if os.environ.get("LINGBOT_CODEC_NATIVE_BACKEND", "aten") == "fa2":
            from flash_attn.flash_attn_interface import _flash_attn_forward
            out, lse, _, _ = _flash_attn_forward(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                0.0, scale, False, -1, -1, 0.0, None, False,
            )
            return out.transpose(1, 2), lse
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q.contiguous(), k.contiguous(), v.contiguous(), 0.0, False, False, scale=scale
        )
        return result[0], result[1]
    outputs, lses = [], []
    for qi in range(0, q.shape[-2], q_tile):
        qq = q[..., qi:qi+q_tile, :]
        out = torch.zeros((*qq.shape[:-1], v.shape[-1]), device=q.device, dtype=q.dtype)
        lse = torch.full(qq.shape[:-1], -math.inf, device=q.device)
        for ki in range(0, k.shape[-2], k_tile):
            logits = (qq.float() @ k[..., ki:ki+k_tile, :].float().transpose(-1, -2)) * scale
            ll = torch.logsumexp(logits, -1)
            probabilities = (logits - ll[..., None]).exp()
            oo = (probabilities @ v[..., ki:ki+k_tile, :].float()).to(q.dtype)
            out, lse = _merge(out, lse, oo, ll)
        outputs.append(out); lses.append(lse)
    return torch.cat(outputs, -2), torch.cat(lses, -1)


class _Group(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.register_buffer("heads", state["heads"].long())
        self.register_buffer("indices", state["indices"].long())
        for name in ("q_reduce", "k_read", "v_read", "k_affine", "v_affine", "lift"):
            self.register_parameter(name, nn.Parameter(state[name].float(), requires_grad=False))

    def query(self, value):
        return torch.einsum("bhsd,hde->bhse", value, self.q_reduce.to(value.dtype))

    def _constant_read_bias(self, mean, affine, target, dtype):
        name="_cached_"+target+"_read_bias"
        sources=getattr(self,"_read_bias_sources",{})
        old=sources.get(target)
        value=getattr(self,name,None)
        version=None if mean.is_inference() else mean._version
        if value is None or value.device!=mean.device or value.dtype!=dtype or old is None or old[0].data_ptr()!=mean.data_ptr() or old[1]!=version:
            selected=mean.reshape(-1,self.q_reduce.shape[1]).index_select(0,self.heads).to(dtype)
            value=torch.einsum("hd,hde->he",selected,affine).reshape(-1)
            if hasattr(self,name):setattr(self,name,value)
            else:self.register_buffer(name,value,persistent=False)
            sources[target]=(mean,version)
            self._read_bias_sources=sources
        return value

    def _project_centers(self, centers, affine, target, dtype):
        d128=self.q_reduce.shape[1]
        selected=centers.reshape(centers.shape[0],centers.shape[1],-1,d128).index_select(2,self.heads)
        if not getattr(self,"_runtime_center_cache",False):
            return torch.einsum("bchd,hde->bhce",selected.to(dtype),affine)
        cache=getattr(self,"_center_projection_cache",{})
        key=(target,dtype,centers.data_ptr(),None if centers.is_inference() else centers._version)
        value=cache.get(key)
        if value is None:
            value=torch.einsum("bchd,hde->bhce",selected.to(dtype),affine)
            # A generation has at most eight immutable source chunks.  Bound
            # derived resident state even if this module is reused repeatedly.
            cache[key]=(centers,value)
            if len(cache)>16:
                cache.pop(next(iter(cache)))
            self._center_projection_cache=cache
        else:
            value=value[1]
        return value

    def _joined_centers(self, descriptors, affine, target, dtype):
        key=tuple(
            (
                centers.data_ptr(),
                None if centers.is_inference() else centers._version,
                ids.data_ptr(),
                None if ids.is_inference() else ids._version,
                int(ids.shape[1]),
            )
            for _,(centers,ids),_ in descriptors
        )
        cache=getattr(self,"_joined_center_cache",{})
        old=cache.get(target)
        if old is not None and old[0]==key and old[1]==dtype:
            return old[3],old[4]
        # Retain only the latest history layout for K and V.  This keeps the
        # derived cache bounded as a rollout appends immutable chunks.
        cache.pop(target,None)
        d128=self.q_reduce.shape[1]
        banks=[];labels=[];references=[];offset=0
        for _,(centers,ids),_ in descriptors:
            selected=centers.reshape(
                centers.shape[0],centers.shape[1],-1,d128
            ).index_select(2,self.heads)
            center=torch.einsum("bchd,hde->bhce",selected.to(dtype),affine)
            banks.append(center)
            labels.append(ids+offset)
            references.append((centers,ids))
            offset+=int(center.shape[2])
        bank=torch.cat(banks,dim=2)
        joined_ids=torch.cat(labels,dim=1)
        cache[target]=(key,dtype,tuple(references),bank,joined_ids)
        self._joined_center_cache=cache
        return bank,joined_ids

    def read(self, descriptor, target: str, dtype: torch.dtype):
        latent, (centers, ids), mean = descriptor
        d128 = self.q_reduce.shape[1]
        affine = getattr(self, target + "_affine").to(dtype)
        weight = getattr(self, target + "_read").to(dtype)
        if os.environ.get("LINGBOT_CODEC_NATIVE_READ_BIAS", "separate") == "fused":
            if os.environ.get("LINGBOT_CODEC_NATIVE_CACHE_READ_BIAS", "0") == "1":
                bias=self._constant_read_bias(mean,affine,target,dtype)
            else:
                selected_mean=mean.reshape(-1,d128).index_select(0,self.heads).to(dtype)
                bias=torch.einsum("hd,hde->he",selected_mean,affine).reshape(-1)
            flat_weight = weight.reshape(-1, weight.shape[-1])
            result = torch.nn.functional.linear(latent.to(dtype), flat_weight, bias)
            result = result.reshape(*latent.shape[:2], len(self.heads), weight.shape[1]).permute(0, 2, 1, 3)
        else:
            selected_mean = mean.reshape(-1, d128).index_select(0, self.heads).to(dtype)
            result = torch.einsum("bsr,hdr->bhsd", latent.to(dtype), weight)
            result += torch.einsum("hd,hde->he", selected_mean, affine)[None, :, None]
        center = self._project_centers(centers,affine,target,dtype)
        result += center.gather(2, ids[:, None, :, None].expand(-1, len(self.heads), -1, center.shape[-1]))
        return result

    def lift_value(self, value):
        return torch.einsum("bhse,hed->bhsd", value, self.lift.to(value.dtype))

    def read_many(
        self, descriptors, target: str, dtype: torch.dtype,
        joined_latent: torch.Tensor | None = None,
    ):
        # Same operator/mean across immutable chunks; only C64 centers differ.
        # Keep each center projection unchanged; batch the shared latent GEMM.
        mean = descriptors[0][2]
        # Very narrow GEMMs can switch accumulation algorithms when the token
        # dimension is batched. Retain the established path for these groups.
        narrow = getattr(self, target + "_read").shape[0] * getattr(self, target + "_read").shape[1] < 128
        if narrow or os.environ.get("LINGBOT_CODEC_NATIVE_READ_BIAS", "separate") != "fused" or any(d[2].data_ptr() != mean.data_ptr() for d in descriptors):
            return torch.cat([self.read(d, target, dtype) for d in descriptors], dim=2)
        affine = getattr(self, target + "_affine").to(dtype)
        weight = getattr(self, target + "_read").to(dtype)
        d128 = self.q_reduce.shape[1]
        if os.environ.get("LINGBOT_CODEC_NATIVE_CACHE_READ_BIAS", "0") == "1":
            bias=self._constant_read_bias(mean,affine,target,dtype)
        else:
            selected_mean = mean.reshape(-1, d128).index_select(0, self.heads).to(dtype)
            bias = torch.einsum("hd,hde->he", selected_mean, affine).reshape(-1)
        if joined_latent is None:
            latent = torch.cat([d[0] for d in descriptors], dim=1).to(dtype)
        else:
            expected = sum(int(d[0].shape[1]) for d in descriptors)
            if joined_latent.shape[1] != expected:
                raise ValueError("joined latent does not match descriptor lengths")
            latent = joined_latent.to(dtype)
        result = torch.nn.functional.linear(latent, weight.reshape(-1, weight.shape[-1]), bias)
        result = result.reshape(*latent.shape[:2], len(self.heads), weight.shape[1]).permute(0, 2, 1, 3)
        if getattr(self,"_runtime_joined_centers",False):
            center,ids=self._joined_centers(descriptors,affine,target,dtype)
            result.add_(center.gather(
                2,ids[:,None,:,None].expand(
                    -1,len(self.heads),-1,center.shape[-1]
                )
            ))
            return result
        offset = 0
        for lat, (centers, ids), _ in descriptors:
            center = self._project_centers(centers,affine,target,dtype)
            length = lat.shape[1]
            result[:, :, offset:offset+length].add_(center.gather(2, ids[:, None, :, None].expand(-1, len(self.heads), -1, center.shape[-1])))
            offset += length
        return result


class _Layer(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.heads = int(state["heads"]); self.native_dim = int(state["native_dim"])
        self.rank = int(state["rank"])
        self.groups = nn.ModuleList([_Group(item) for item in state["groups"]])
        assigned = sorted(torch.cat([group.heads for group in self.groups]).tolist())
        if assigned != list(range(self.heads)):
            raise ValueError("reader groups must partition the local head shard")
        for group in self.groups:
            count, width = int(group.heads.numel()), int(group.indices.shape[1])
            value_width = int(group.v_affine.shape[-1])
            if not torch.equal(group.indices[:, 1::2], group.indices[:, ::2] + 1) or bool((group.indices[:, ::2] % 2).any()):
                raise ValueError("low-dimensional RoPE indices must preserve native complex pairs")
            expected = {
                "q_reduce": (count, self.native_dim, width),
                "k_read": (count, width, self.rank),
                "v_read": (count, value_width, self.rank),
                "k_affine": (count, self.native_dim, width),
                "v_affine": (count, self.native_dim, value_width),
                "lift": (count, value_width, self.native_dim),
            }
            for name, shape in expected.items():
                if tuple(getattr(group, name).shape) != shape:
                    raise ValueError(f"bad {name} shape; expected {shape}")

    def join_group_read_weights_(self):
        """Pack group reader rows once without retaining duplicate storage."""
        if getattr(self, "_group_read_weights_joined", False):
            return self
        for target in ("k", "v"):
            name = target + "_read"
            shapes = [tuple(getattr(group, name).shape) for group in self.groups]
            rows = [getattr(group, name).reshape(-1, self.rank) for group in self.groups]
            joint = nn.Parameter(torch.cat(rows, dim=0), requires_grad=False)
            self.register_parameter(name + "_joint", joint)
            offset = 0
            for group, shape in zip(self.groups, shapes):
                count = math.prod(shape[:-1])
                # The group view shares the joint parameter storage.  It is an
                # ordinary runtime attribute because reader assets are frozen
                # and have already been moved to their final device/dtype.
                del group._parameters[name]
                object.__setattr__(
                    group, name, joint[offset:offset + count].view(shape)
                )
                offset += count
        self._group_read_weights_joined = True
        return self

    def read_many_groups(self, descriptors, target, dtype, joined_latent):
        """Read every width group with one shared-latent GEMM."""
        mean = descriptors[0][2]
        if any(item[2].data_ptr() != mean.data_ptr() for item in descriptors):
            return [
                group.read_many(descriptors, target, dtype, joined_latent)
                for group in self.groups
            ]
        weight = getattr(self, target + "_read_joint").to(dtype)
        biases = []
        for group in self.groups:
            affine = getattr(group, target + "_affine").to(dtype)
            biases.append(group._constant_read_bias(mean, affine, target, dtype))
        latent = joined_latent.to(dtype)
        joint = torch.nn.functional.linear(latent, weight, torch.cat(biases))
        outputs = []
        offset = 0
        for group in self.groups:
            width = int(getattr(group, target + "_read").shape[1])
            count = int(group.heads.numel()) * width
            value = joint[..., offset:offset + count].reshape(
                *latent.shape[:2], int(group.heads.numel()), width
            ).permute(0, 2, 1, 3)
            affine = getattr(group, target + "_affine").to(dtype)
            if getattr(group, "_runtime_joined_centers", False):
                center, ids = group._joined_centers(
                    descriptors, affine, target, dtype
                )
                value.add_(center.gather(
                    2, ids[:, None, :, None].expand(
                        -1, int(group.heads.numel()), -1, center.shape[-1]
                    )
                ))
            else:
                cursor = 0
                for latent_part, (centers, ids), _ in descriptors:
                    center = group._project_centers(
                        centers, affine, target, dtype
                    )
                    length = int(latent_part.shape[1])
                    value[:, :, cursor:cursor + length].add_(center.gather(
                        2, ids[:, None, :, None].expand(
                            -1, int(group.heads.numel()), -1, center.shape[-1]
                        )
                    ))
                    cursor += length
            outputs.append(value)
            offset += count
        return outputs


_BANK: dict[tuple[str, str], tuple[dict[str, Any], dict[tuple[int, int], _Layer]]] = {}


def _reader(cache, *, heads: int, native_dim: int, device: torch.device):
    if not _enabled():
        return None, None
    manifest_path = Path(os.environ["LINGBOT_CODEC_NATIVE_MANIFEST"]).resolve()
    operator_path = Path(os.environ["LINGBOT_V2_KV_CODEC_PREPACKED_OPERATOR_PATH"]).resolve()
    key = (str(manifest_path), str(device))
    if key not in _BANK:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "codec_native_manifest_v1" or manifest.get("model") != "lingbot-world-v2":
            raise ValueError("wrong LingBot codec-native manifest")
        if manifest.get("codec_operator_sha256") != _sha256(str(operator_path)):
            raise ValueError("reader was not calibrated for the deployed codec operator")
        _BANK[key] = (manifest, {})
    manifest, loaded = _BANK[key]
    layer = int(cache["codec_layer"]); rank = int(cache.get("codec_sp_rank", 0))
    entry = manifest.get("layers", {}).get(f"{rank}:{layer}")
    if entry is None:
        return None, manifest
    if not entry.get("calibrated", False) and os.environ.get("LINGBOT_CODEC_NATIVE_ALLOW_UNCALIBRATED") != "1":
        raise RuntimeError("initial reader weights are not deployable; calibrate them first")
    identity = (rank, layer)
    if identity not in loaded:
        artifact = (manifest_path.parent / entry["path"]).resolve()
        if _sha256(str(artifact)) != entry["sha256"]:
            raise ValueError("reader checkpoint checksum mismatch")
        loaded[identity] = _Layer(torch.load(artifact, map_location="cpu", weights_only=True)).to(
            device=device, dtype=torch.bfloat16 if os.environ.get("LINGBOT_CODEC_NATIVE_BF16_WEIGHTS", "0") == "1" else torch.float32)
        loaded[identity].join_group_read_weights_()
    reader = loaded[identity]
    if reader.heads != heads or reader.native_dim != native_dim:
        raise ValueError("reader does not match this sequence-parallel head shard")
    return reader, manifest


def preload_readers(*, layers, heads, native_dim, sp_rank, device):
    """Reader checkpoints are static model assets, just like codec operators."""
    for layer in range(layers):
        reader,_=_reader({"codec_layer":layer,"codec_sp_rank":sp_rank},heads=heads,native_dim=native_dim,device=device)
        if reader is None:raise RuntimeError(f"Missing reader at rank {sp_rank}, layer {layer}")


def _policy(manifest: dict[str, Any], layer: int) -> dict[str, Any]:
    policy = dict(manifest.get("policy", {}))
    policy.update(manifest.get("layer_policy", {}).get(str(layer), {}))
    return policy


@lru_cache(maxsize=64)
def _cached_arange(device: str, start: int, stop: int) -> torch.Tensor:
    """Share immutable position/index tensors across layers and denoise calls."""
    return torch.arange(start, stop, device=torch.device(device), dtype=torch.long)


def _selected_multiplier(positions, grid_sizes, freqs, frame_seqlen, indices):
    _, height, width = map(int, grid_sizes[0].tolist())
    half = indices.shape[1] // 2
    original_half = freqs.shape[1]
    split = [original_half - 2 * (original_half // 3), original_half // 3, original_half // 3]
    temporal, vertical, horizontal = freqs.split(split, dim=1)
    frame = torch.div(positions, frame_seqlen, rounding_mode="floor")
    spatial = positions.remainder(frame_seqlen)
    row = torch.div(spatial, width, rounding_mode="floor")
    column = spatial.remainder(width)
    multiplier = torch.cat((temporal[frame], vertical[row], horizontal[column]), 1)
    pair = indices[:, ::2] // 2
    return multiplier[:, pair].permute(1, 0, 2)


def _rope_low(value, positions, grid_sizes, freqs, frame_seqlen, group):
    indices = group.indices
    if os.environ.get("LINGBOT_CODEC_NATIVE_TRITON_ROPE", "0") == "1":
        from .codec_native_rope_triton import rope_selected
        return rope_selected(value, positions, grid_sizes, freqs, frame_seqlen, indices)
    # Pair validation happens once on CPU in _Layer, not synchronously per call.
    multiplier = _selected_multiplier(positions, grid_sizes, freqs, frame_seqlen, indices)
    complex_value = torch.view_as_complex(value.double().reshape(*value.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(complex_value * multiplier[None]).flatten(-2).to(value.dtype)


def _active_pieces(cache, *, capacity: int, sink_tokens: int, history_budget: int):
    committed = int(cache.get("codec_committed_end", 0))
    budget = min(int(capacity), max(0, int(history_budget)))
    sink_stop = min(sink_tokens, committed, budget)
    recent_start = max(sink_stop, committed - max(budget - sink_stop, 0))
    intervals = [(0, sink_stop), (recent_start, committed)]
    result = []
    for segment in cache.get("codec_segments", []):
        start, end = int(segment["global_start"]), int(segment["global_end"])
        for left, right in intervals:
            begin, stop = max(start, left), min(end, right)
            if begin < stop:
                result.append((segment, begin, stop))
    return result


def _sample_axis(length: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(int(count), int(length))
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def _compressed_value_mean(low, device, cache=None):
    """Transient V mean from latent sums and center counts, never full token KV.

    Uses the formal BF16 dequantized decoder in FP32 arithmetic. This is a
    first-moment correction, not bitwise mean of BF16-rounded decoded rows.
    """
    cache_key = (
        int(cache.get("codec_committed_end", 0)),
        tuple(item.get("bounds", ()) for item in low),
    ) if cache is not None else None
    cached = cache.get("_codec_native_value_mean") if cache is not None else None
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    latent_sum = center_sum = None
    count = 0
    for item in low:
        latent, (centers, ids), mean = item["v"]
        n = latent.shape[1]
        counts = torch.zeros(centers.shape[:2], device=device, dtype=torch.float32)
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        ls = latent.float().sum(1)
        cs = torch.einsum("bc,bcd->bd", counts, centers.float())
        latent_sum = ls if latent_sum is None else latent_sum + ls
        center_sum = cs if center_sum is None else center_sum + cs
        count += n
    payload = low[0]["v_payload"]
    batch, _, heads, dim = map(int, payload["shape"])
    op = _load_operator(payload["basis_root"], target="v", sp_rank=int(payload["sp_rank"]),
        layer=int(payload["layer"]), joint_dim=heads*dim, rank=int(payload["rank"]),
        device=device, linear_root=payload.get("linear_root") or None)
    decoder=op["decoder"]
    weight=decoder if torch.is_tensor(decoder) else decoder["qweight"].to(torch.bfloat16)*decoder["scale"]
    value=torch.nn.functional.linear(latent_sum/count, weight.float(), mean.to(torch.bfloat16).float())+center_sum/count
    value = value.reshape(batch, heads, 1, dim)
    if cache is not None:
        cache["_codec_native_value_mean"] = (cache_key, value)
    return value


@torch.no_grad()
def _maybe_dump_training_case(
    *, reader: _Layer, q_pre: torch.Tensor, q_native: torch.Tensor,
    low: list[dict[str, Any]], cache: dict[str, Any], current_start: int,
    grid_sizes: torch.Tensor, freqs: torch.Tensor, frame_seqlen: int,
    key_current: torch.Tensor | None = None, value_current: torch.Tensor | None = None,
) -> None:
    """Sample real A1-codec Q/K/V for offline reader calibration.

    Collection is deliberately restricted to an exact D128 reader.  Thus the
    stored K/V are reconstructed from the deployed INT4 latent and A1/QAT
    operator, making the formal codec path—not clean KV—the teacher.
    """
    root = os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_ROOT", "").strip()
    if not root or not low:
        return
    if len(reader.groups) != 1 or int(reader.groups[0].indices.shape[1]) != reader.native_dim:
        raise RuntimeError("training dumps require the exact D128 reader")
    chunk = int(cache.get("codec_chunk_id", 0))
    chunks = {int(x) for x in os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_CHUNKS", "4,6,8").split(",") if x}
    seen = int(cache.get("_codec_native_dump_chunk", -1))
    if seen != chunk:
        cache["_codec_native_dump_chunk"] = chunk
        cache["_codec_native_dump_call"] = 0
    call = int(cache.get("_codec_native_dump_call", 0))
    cache["_codec_native_dump_call"] = call + 1
    calls = {int(x) for x in os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_CALLS", "0,3").split(",") if x}
    if chunk not in chunks or call not in calls:
        return

    layer = int(cache["codec_layer"])
    target = Path(root) / f"chunk_{chunk:02d}" / f"call_{call:02d}" / f"layer_{layer:02d}.pt"
    if target.exists():
        return
    group = reader.groups[0]
    k_pre = torch.cat([group.read(item["k"], "k", q_pre.dtype) for item in low], dim=2)
    value = torch.cat([group.read(item["v"], "v", q_pre.dtype) for item in low], dim=2)
    positions = torch.cat([item["positions"] for item in low])
    k_index = _sample_axis(k_pre.shape[2], int(os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_K", "512")), q_pre.device)
    q_index = _sample_axis(q_pre.shape[1], int(os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_Q", "64")), q_pre.device)
    k_pre = k_pre.index_select(2, k_index)
    value = value.index_select(2, k_index)
    positions = positions.index_select(0, k_index)
    key_native = rope_pre_key_by_positions(
        k_pre.permute(0, 2, 1, 3), positions,
        grid_sizes=grid_sizes, freqs=freqs, frame_seqlen=frame_seqlen,
    ).to(q_native.dtype).permute(0, 2, 1, 3)
    q_positions = torch.arange(current_start, current_start + q_pre.shape[1], device=q_pre.device)
    payload = {
        "format": "lingbot_codec_native_attention_sample_v1",
        "layer": layer, "chunk": chunk, "call": call,
        "q_pre": q_pre.permute(0, 2, 1, 3).index_select(2, q_index).cpu(),
        "q_native": q_native.permute(0, 2, 1, 3).index_select(2, q_index).cpu(),
        "q_positions": q_positions.index_select(0, q_index).cpu(),
        "k_pre": k_pre.cpu(), "k_native": key_native.cpu(),
        "v": value.cpu(), "k_positions": positions.cpu(),
        "grid_sizes": grid_sizes.cpu(), "freqs": freqs.cpu(),
        "frame_seqlen": int(frame_seqlen),
    }
    if key_current is not None:
        current_positions = torch.arange(current_start, current_start + key_current.shape[1], device=q_pre.device)
        current_key = rope_pre_key_by_positions(key_current, current_positions,
            grid_sizes=grid_sizes, freqs=freqs, frame_seqlen=frame_seqlen).to(q_native.dtype)
        qo = q_native.permute(0, 2, 1, 3).index_select(2, q_index)
        co, cl = _attention_lse(qo, current_key.permute(0, 2, 1, 3),
            value_current.permute(0, 2, 1, 3), scale=128**-0.5)
        payload["current_out"] = co.cpu()
        payload["current_lse"] = cl.cpu()
        payload["history_token_count"] = sum(int(item["positions"].numel()) for item in low)
        payload["teacher_note"] = "D128 fused-bias joint rollout; exact current branch; sampled full-history including sink"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    # Collection-only early stop: retain the exact 125-frame trajectory setup,
    # but avoid generating unused later chunks. Never enabled in deployment.
    stop_chunk=int(os.environ.get("LINGBOT_CODEC_NATIVE_DUMP_STOP_CHUNK", "0"))
    if stop_chunk and chunk==stop_chunk and call==4 and layer==39:
        if os.environ.get("LINGBOT_CODEC_NATIVE_D128_JOINT")!="1":
            raise RuntimeError("Early collection stop requires exact D128 joint teacher")
        raise SystemExit(0)


@torch.no_grad()
def attend(
    q_pre: torch.Tensor, q_native: torch.Tensor, key_current: torch.Tensor,
    value_current: torch.Tensor, cache: dict[str, Any], *, current_start: int,
    capacity: int, sink_tokens: int, max_attention_size: int,
    grid_sizes: torch.Tensor, freqs: torch.Tensor, frame_seqlen: int,
) -> torch.Tensor | None:
    """Return [B,S,H,D], or None when this layer is a native fallback."""
    reader, manifest = _reader(
        cache, heads=int(q_pre.shape[2]), native_dim=int(q_pre.shape[3]), device=q_pre.device
    )
    if reader is None:
        return None
    if cache.get("a1_controller") is not None:
        raise RuntimeError("codec-native deployment cannot share the legacy A1 training cache")
    layer = int(cache["codec_layer"]); policy = _policy(manifest, layer)
    runtime_options=manifest.get("runtime_optimizations",{})
    if int(policy.get("native_prefix_chunks", 0)) > 0 and int(cache.get("codec_chunk_id", 0)) <= int(policy["native_prefix_chunks"]):
        return None
    native_calls=policy.get("native_denoise_calls", [])
    if native_calls:
        seen=cache.get("_codec_native_schedule_start")
        index=0 if seen != int(current_start) else int(cache.get("_codec_native_schedule_index", -1))+1
        cache["_codec_native_schedule_start"]=int(current_start)
        cache["_codec_native_schedule_index"]=index
        if not cache.get("codec_commit", False) and index in native_calls:
            return None
    if policy.get("native_commit", False) and cache.get("codec_commit", False):
        return None
    maybe_dump_committed_source(cache, key_current, value_current)
    recent_tokens = int(policy.get("recent_native_frames", 0)) * int(frame_seqlen)
    anchor_stride = int(policy.get("anchor_stride", 0))
    default_sink_frames = int(sink_tokens) // int(frame_seqlen)
    native_sink_tokens = min(
        int(sink_tokens),
        int(policy.get("sink_native_frames", default_sink_frames)) * int(frame_seqlen),
    )
    committed = int(cache.get("codec_committed_end", 0))
    native_k, native_v, native_pos, low = [], [], [], []
    history_budget = max(0, int(max_attention_size) - int(key_current.shape[1]))
    for segment, begin, stop in _active_pieces(
        cache, capacity=capacity, sink_tokens=sink_tokens,
        history_budget=history_budget,
    ):
        if os.environ.get("LINGBOT_CODEC_NATIVE_HOST_POLICY", "0") == "1" and not anchor_stride:
            # Prefix/suffix protection is a union of intervals. Resolve bounds
            # from CPU metadata, without synchronizing a GPU boolean mask.
            lo = max(begin, min(stop, native_sink_tokens))
            hi = min(stop, max(begin, committed - recent_tokens)) if recent_tokens else stop
            if runtime_options.get("contiguous_slices",False) and not recent_tokens:
                base=int(segment["global_start"])
                if lo>begin:
                    native_k.append(decode_tensor_slice(segment["k"],begin-base,lo-base,dtype=q_pre.dtype,device=q_pre.device))
                    native_v.append(decode_tensor_slice(segment["v"],begin-base,lo-base,dtype=q_pre.dtype,device=q_pre.device))
                    native_pos.append(_cached_arange(str(q_pre.device), begin, lo))
                if hi>lo:
                    low_positions_part=_cached_arange(str(q_pre.device), lo, hi)
                    low.append({"k":direct_latent_slice(segment["k"],lo-base,hi-base,device=q_pre.device),
                                "v":direct_latent_slice(segment["v"],lo-base,hi-base,device=q_pre.device),
                                "v_payload":segment["v"],"positions":low_positions_part,
                                "bounds":(lo,hi)})
                continue
            if hi <= lo:
                native_positions = _cached_arange(str(q_pre.device), begin, stop)
                low_positions_part = torch.empty(0, dtype=torch.long, device=q_pre.device)
            else:
                native_positions = torch.cat((
                    _cached_arange(str(q_pre.device), begin, lo),
                    _cached_arange(str(q_pre.device), hi, stop),
                ))
                low_positions_part = _cached_arange(str(q_pre.device), lo, hi)
            if native_positions.numel():
                selected = native_positions - int(segment["global_start"])
                native_k.append(decode_tensor_rows(segment["k"], selected, dtype=q_pre.dtype, device=q_pre.device))
                native_v.append(decode_tensor_rows(segment["v"], selected, dtype=q_pre.dtype, device=q_pre.device))
                native_pos.append(native_positions)
            if low_positions_part.numel():
                selected = low_positions_part - int(segment["global_start"])
                low.append({"k": direct_latent_rows(segment["k"], selected, device=q_pre.device), "v": direct_latent_rows(segment["v"], selected, device=q_pre.device), "v_payload": segment["v"], "positions": low_positions_part, "bounds": (lo, hi)})
            continue
        positions = _cached_arange(str(q_pre.device), begin, stop)
        protect = positions < native_sink_tokens
        if recent_tokens:
            protect |= positions >= max(0, committed - recent_tokens)
        if anchor_stride:
            protect |= torch.div(positions, frame_seqlen, rounding_mode="floor").remainder(anchor_stride) == 0
        local = positions - int(segment["global_start"])
        if bool(protect.any()):
            selected = local[protect]
            native_k.append(decode_tensor_rows(segment["k"], selected, dtype=q_pre.dtype, device=q_pre.device))
            native_v.append(decode_tensor_rows(segment["v"], selected, dtype=q_pre.dtype, device=q_pre.device))
            native_pos.append(positions[protect])
        if bool((~protect).any()):
            selected = local[~protect]
            low.append({
                "k": direct_latent_rows(segment["k"], selected, device=q_pre.device),
                "v": direct_latent_rows(segment["v"], selected, device=q_pre.device),
                "v_payload": segment["v"],
                "positions": positions[~protect],
                "bounds": (begin, stop),
            })

    # Preserve the exact frozen path for the seed/all-protected case.
    if not low:
        return None

    _maybe_dump_training_case(
        reader=reader, q_pre=q_pre, q_native=q_native, low=low, cache=cache,
        current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
        frame_seqlen=frame_seqlen,
        key_current=key_current, value_current=value_current,
    )

    current_positions = _cached_arange(
        str(q_pre.device), current_start, current_start + key_current.shape[1]
    )
    native_k.append(key_current); native_v.append(value_current); native_pos.append(current_positions)
    key_native = torch.cat(native_k, 1); value_native = torch.cat(native_v, 1)
    positions_native = torch.cat(native_pos)
    if os.environ.get("LINGBOT_CODEC_NATIVE_TRITON_NATIVE_ROPE", "0") == "1":
        from .codec_native_rope_triton import rope_selected
        indices=_cached_arange(
            str(key_native.device), 0, key_native.shape[-1]
        )[None].expand(key_native.shape[2],-1)
        key_native=rope_selected(key_native.permute(0,2,1,3),positions_native,grid_sizes,freqs,frame_seqlen,indices).to(q_native.dtype)
    else:
        key_native = rope_pre_key_by_positions(
            key_native, positions_native, grid_sizes=grid_sizes, freqs=freqs,
            frame_seqlen=frame_seqlen,
        ).to(q_native.dtype).permute(0, 2, 1, 3)
    value_native = value_native.permute(0, 2, 1, 3)
    qh = q_native.permute(0, 2, 1, 3)
    scale = reader.native_dim ** -0.5
    native_out, native_lse = _attention_lse(qh, key_native, value_native, scale=scale)
    # Every group writes its complete, disjoint head set before this tensor is
    # returned; clearing tens of MiB on every layer call is redundant.
    result = torch.empty_like(qh)
    q_pre_h = q_pre.permute(0, 2, 1, 3)
    q_positions = current_positions
    value_mean = _compressed_value_mean(low, q_pre.device, cache) if os.environ.get("LINGBOT_CODEC_NATIVE_VALUE_MEAN_RESIDUAL", "0") == "1" else None
    low_positions = torch.cat([segment["positions"] for segment in low])
    batch_read=(runtime_options.get("batch_read",False)
                or os.environ.get("LINGBOT_CODEC_NATIVE_BATCH_READ", "0") == "1")
    # K/V codec latents do not depend on the head-width group.  Materialize
    # each joined view once per layer call instead of repeating the same large
    # concatenation in every reader group.
    joined_k_latent = (
        torch.cat([segment["k"][0] for segment in low], dim=1)
        if batch_read else None
    )
    joined_v_latent = (
        torch.cat([segment["v"][0] for segment in low], dim=1)
        if batch_read else None
    )
    fused_finish = (runtime_options.get("fused_finish", False)
                    or os.environ.get("LINGBOT_CODEC_NATIVE_FUSED_FINISH", "0") == "1")
    joint_group_read = bool(
        batch_read and runtime_options.get("joint_group_read", False)
    )
    for group in reader.groups:
        group._runtime_center_cache=bool(runtime_options.get("center_cache",False))
        group._runtime_joined_centers=bool(runtime_options.get("joined_centers",False))
    grouped_k = reader.read_many_groups(
        [segment["k"] for segment in low], "k", q_pre.dtype, joined_k_latent
    ) if joint_group_read else None
    grouped_v = reader.read_many_groups(
        [segment["v"] for segment in low], "v", q_pre.dtype, joined_v_latent
    ) if joint_group_read else None
    for group_index, group in enumerate(reader.groups):
        heads = group.heads
        query = _rope_low(group.query(q_pre_h.index_select(1, heads)), q_positions, grid_sizes, freqs, frame_seqlen, group)
        if not fused_finish:
            out = native_out.index_select(1, heads); lse = native_lse.index_select(1, heads)
        # Immutable chunks have different C64 codebooks, but after their
        # per-chunk affine read they share one head space.  Concatenating here
        # turns N segment FlashAttention launches and N LSE merges into one
        # mathematically equivalent low-history launch.
        if joint_group_read:
            kk = grouped_k[group_index]
            vv = grouped_v[group_index]
        elif batch_read:
            kk = group.read_many(
                [segment["k"] for segment in low], "k", q_pre.dtype,
                joined_k_latent,
            )
            vv = group.read_many(
                [segment["v"] for segment in low], "v", q_pre.dtype,
                joined_v_latent,
            )
        else:
            kk = torch.cat([group.read(segment["k"], "k", q_pre.dtype) for segment in low], dim=2)
            vv = torch.cat([group.read(segment["v"], "v", q_pre.dtype) for segment in low], dim=2)
        kk = _rope_low(kk, low_positions, grid_sizes, freqs, frame_seqlen, group)
        if os.environ.get("LINGBOT_CODEC_NATIVE_D128_JOINT", "0") == "1":
            if len(reader.groups) != 1 or group.indices.shape[1] != 128:
                raise RuntimeError("D128 joint control requires a single identity D128 reader")
            from .attention import attention
            order = torch.argsort(torch.cat((positions_native, low_positions)))
            all_k = torch.cat((key_native, kk), 2).index_select(2, order)
            all_v = torch.cat((value_native, vv), 2).index_select(2, order)
            exact = attention(qh.transpose(1, 2), all_k.transpose(1, 2), all_v.transpose(1, 2))
            result.copy_(exact.transpose(1, 2))
            continue
        oo, ll = _attention_lse(query, kk, vv, scale=scale)
        lifted = group.lift_value(oo)
        projected_mean = None
        if value_mean is not None and group.lift.shape[1] < reader.native_dim:
            projected_mean = torch.einsum("bhse,hed->bhsd", vv.float().mean(2, keepdim=True), group.lift.to(q_pre.dtype).float())
            if not fused_finish:
                lifted = lifted.float() + value_mean.index_select(1, heads) - projected_mean
        if fused_finish:
            from .codec_native_finish_triton import finish
            finish(native_out,native_lse,lifted,ll,heads,result,
                   value_mean if projected_mean is not None else None,projected_mean)
            continue
        out, lse = _merge(out, lse, lifted, ll)
        result.index_copy_(1, heads, out)

    current_end = current_start + int(key_current.shape[1])
    if cache.get("codec_commit", False):
        if current_start != committed:
            raise RuntimeError("non-causal codec-native commit")
        cache["codec_commit_start"] = current_start
        cache.setdefault("codec_segments", []).append(encode_pair(key_current, value_current, cache=cache))
        cache["codec_committed_end"] = current_end
        cache["codec_segments"] = _prune_segments(
            cache["codec_segments"], committed_end=current_end, capacity=capacity, sink_tokens=sink_tokens
        )
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(min(current_end, capacity))
    return result.permute(0, 2, 1, 3)
