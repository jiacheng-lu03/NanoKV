"""Causal block-output alignment (K-A1) for LingBot-World 2.0.

The implementation intentionally mirrors the frozen WorldPlay recipe:

* the DiT, K encoder/C64 path, and V-QAT path stay frozen;
* Teacher and Student see the same current latent, timestep, condition and
  compressed prefix; only historical K of the active layer group is clean in
  the Teacher;
* supervision is collected at block outputs on the final noisy denoising step;
* only a per-layer K decoder delta is optimized.

LingBot has 40 blocks, therefore the paper path uses five causal groups of
eight blocks.  The runtime object is deliberately not an ``nn.Module`` child
of the DiT: this prevents accidental checkpoint/model-state pollution.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .lingbot_v2_kv_codec import _linear, _load_operator, _unpack_int4


FORMAT = "lingbot_v2_k_a1_decoder_delta_v1"


def _fixed_indices(length: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    count = min(int(length), int(count))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(length), generator=generator)[:count].sort().values.to(device)


class LingBotV2KA1:
    """Own trainable decoder deltas and the matched Teacher/Student state."""

    def __init__(
        self,
        *,
        basis_root: str | Path,
        rank: int = 640,
        layers: Iterable[int],
        sp_rank: int = 0,
        chunks: Iterable[int] = (2, 4, 6, 8),
        inner_steps: int = 1,
        learning_rate: float = 5e-6,
        token_samples: int = 256,
        channel_samples: int = 256,
        reconstruction_weight: float = 0.2,
        cosine_weight: float = 0.02,
        grad_clip: float = 1.0,
        seed: int = 2027,
        source: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.basis_root = str(Path(basis_root).resolve())
        self.rank = int(rank)
        self.layers = tuple(sorted({int(x) for x in layers}))
        self.sp_rank = int(sp_rank)
        self.chunks = frozenset(int(x) for x in chunks)
        self.inner_steps = int(inner_steps)
        self.learning_rate = float(learning_rate)
        self.token_samples = int(token_samples)
        self.channel_samples = int(channel_samples)
        self.reconstruction_weight = float(reconstruction_weight)
        self.cosine_weight = float(cosine_weight)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.output_root = None if output_root is None else Path(output_root)
        if not self.layers or min(self.layers) < 0 or max(self.layers) >= 40:
            raise ValueError(f"invalid LingBot A1 layers: {self.layers}")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        self.pass_name = ""
        self.chunk_id = -1
        self.step_id = -1
        self.trajectory_id = ""
        self.teacher_targets: dict[tuple[int, int, int], torch.Tensor] = {}
        self.student_losses: list[torch.Tensor] = []
        self.metrics: list[dict[str, float]] = []
        self.deltas: dict[int, torch.nn.Parameter] = {}
        self.optimizers: dict[int, torch.optim.Optimizer] = {}
        self.source = "" if source is None else str(Path(source).resolve())
        self._source_payload = None
        if source is not None:
            self._source_payload = torch.load(source, map_location="cpu", weights_only=False)
            if self._source_payload.get("format") != FORMAT:
                raise RuntimeError(f"bad LingBot K-A1 checkpoint: {source}")
            if str(Path(self._source_payload["basis_root"]).resolve()) != self.basis_root:
                raise RuntimeError("K-A1 source/PCA basis provenance mismatch")
            if int(self._source_payload["rank"]) != self.rank:
                raise RuntimeError("K-A1 source rank mismatch")

    def active(self, *, chunk_id: int, step_id: int) -> bool:
        return int(chunk_id) in self.chunks and int(step_id) == 3

    def prepare(self, device: torch.device, joint_dim: int = 5120) -> None:
        """Create FP32 masters/optimizers outside any inference context."""
        for layer in self.layers:
            self._ensure_delta(layer, device, joint_dim)

    def begin_trajectory(self, trajectory_id: str) -> None:
        self.trajectory_id = str(trajectory_id)
        self.teacher_targets.clear()
        self.student_losses.clear()
        self.metrics.clear()
        self.pass_name = ""

    def set_window(self, *, chunk_id: int, step_id: int, pass_name: str) -> None:
        self.chunk_id = int(chunk_id)
        self.step_id = int(step_id)
        self.pass_name = str(pass_name)
        if pass_name == "teacher":
            self.student_losses.clear()
            self.metrics.clear()

    def wants_clean_k(self, layer: int) -> bool:
        return self.pass_name == "teacher" and int(layer) in self.layers

    def wants_trainable_k(self, layer: int) -> bool:
        # After an update, the new decoder must govern the remainder of the
        # same causal rollout (including non-supervised steps and commits).
        return self.pass_name != "teacher" and int(layer) in self.layers

    def should_detach_after(self, layer: int) -> bool:
        """The suffix cannot affect a local block-output objective."""
        return self.pass_name == "student" and int(layer) == max(self.layers)

    def _ensure_delta(
        self, layer: int, device: torch.device, joint_dim: int
    ) -> torch.nn.Parameter:
        layer = int(layer)
        existing = self.deltas.get(layer)
        if existing is not None:
            if existing.device != device:
                raise RuntimeError("A1 decoder delta cannot migrate after optimizer creation")
            return existing
        base = _load_operator(
            self.basis_root, target="k", sp_rank=self.sp_rank, layer=layer,
            joint_dim=int(joint_dim), rank=self.rank, device=device,
        )["decoder"]
        delta = torch.zeros_like(base, dtype=torch.float32, device=device)
        if self._source_payload is not None:
            state = self._source_payload.get("decoder_delta", {})
            saved = state.get(str(layer), state.get(layer))
            if saved is not None:
                if tuple(saved.shape) != tuple(delta.shape):
                    raise RuntimeError(f"A1 source shape mismatch at L{layer}")
                delta.copy_(saved.float().to(device))
        parameter = torch.nn.Parameter(delta)
        self.deltas[layer] = parameter
        self.optimizers[layer] = torch.optim.AdamW(
            [parameter], lr=self.learning_rate, weight_decay=0.0,
        )
        if self._source_payload is not None:
            optimizer_states = self._source_payload.get("optimizer", {})
            saved_optimizer = optimizer_states.get(str(layer), optimizer_states.get(layer))
            if saved_optimizer is not None:
                self.optimizers[layer].load_state_dict(saved_optimizer)
        return parameter

    def decode_student(self, payload: dict[str, Any], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Differentiably decode an immutable INT4 K payload."""
        batch, sequence, heads, head_dim = map(int, payload["shape"])
        layer = int(payload["layer"])
        operator = _load_operator(
            payload["basis_root"], target="k", sp_rank=int(payload["sp_rank"]),
            layer=layer, joint_dim=heads * head_dim, rank=int(payload["rank"]),
            device=device,
        )
        delta = self._ensure_delta(layer, device, heads * head_dim)
        codes = _unpack_int4(
            payload["latent_packed"].to(device), int(payload["latent_count"]),
            tuple(map(int, payload["latent_shape"])),
        )
        latent = codes.to(torch.bfloat16) * payload["latent_scale"].to(device, torch.bfloat16)
        # Exact zero-delta gate plus an FP32-master update path.  The tied-PCA
        # term is evaluated by the identical BF16 kernel as ordinary codec
        # inference; the delta contribution is accumulated in FP32 before the
        # single output cast.  Therefore delta=0 is bit-identical to no-A1.
        baseline = _linear(
            latent, operator["decoder"], operator["mean"].to(torch.bfloat16)
        )
        # Payload tensors are produced inside the generation inference path.
        # Materialize a normal tensor explicitly; otherwise PyTorch can keep
        # inference-tensor semantics and silently omit the weight graph even
        # after ``torch.enable_grad()``.
        with torch.inference_mode(False), torch.enable_grad():
            latent_for_delta = latent.detach().clone().float()
            contribution = F.linear(latent_for_delta, delta)
            rows = (baseline.detach().float() + contribution).to(torch.bfloat16)
        if torch.is_grad_enabled() and not rows.requires_grad:
            raise RuntimeError(
                f"A1 decoder graph missing at L{layer}: "
                f"delta_requires_grad={delta.requires_grad} "
                f"delta_is_inference={delta.is_inference()} "
                f"latent_is_inference={latent.is_inference()} "
                f"materialized_is_inference={latent_for_delta.is_inference()} "
                f"contribution_requires_grad={contribution.requires_grad} "
                f"inference_mode={torch.is_inference_mode_enabled()}"
            )
        centers = payload["centers"].to(device=device, dtype=torch.bfloat16)
        ids = payload["ids"].to(device=device, dtype=torch.long)
        rows = rows + torch.gather(
            centers, 1, ids[..., None].expand(-1, -1, heads * head_dim)
        )
        return rows.reshape(batch, sequence, heads, head_dim).to(dtype)

    def record_block(self, layer: int, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        layer = int(layer)
        if layer not in self.layers or self.pass_name not in {"teacher", "student"}:
            return
        token_idx = _fixed_indices(
            block_output.shape[1], self.token_samples,
            self.seed + layer * 1009, block_output.device,
        )
        channel_idx = _fixed_indices(
            block_output.shape[2], self.channel_samples,
            self.seed + layer * 2003 + 17, block_output.device,
        )
        sketch = block_output[:, token_idx][:, :, channel_idx]
        key = (self.chunk_id, self.step_id, layer)
        if self.pass_name == "teacher":
            self.teacher_targets[key] = sketch.detach()
            return
        if not block_output.requires_grad:
            raise RuntimeError(
                f"A1 block graph missing at L{layer}: "
                f"grad_enabled={torch.is_grad_enabled()} "
                f"input_requires_grad={block_input.requires_grad}"
            )
        target = self.teacher_targets.get(key)
        if target is None:
            raise RuntimeError(f"A1 Student has no matched Teacher target for {key}")
        pred = sketch.float()
        ref = target.to(pred).float()
        entry = block_input[:, token_idx][:, :, channel_idx].detach().float()
        diff = pred - ref
        denom = (ref - entry).pow(2).sum().clamp_min(1e-8)
        relative_mse = diff.pow(2).sum() / denom
        cosine = F.cosine_similarity(pred.reshape(1, -1), ref.reshape(1, -1), dim=1).mean()
        loss = relative_mse + self.cosine_weight * (1.0 - cosine)
        self.student_losses.append(loss)
        self.metrics.append({
            "layer": float(layer),
            "block_rel_l2": float(relative_mse.detach().sqrt().cpu()),
            "block_cosine": float(cosine.detach().cpu()),
        })

    def reconstruction_anchor(self, caches: list[dict[str, Any]]) -> torch.Tensor | None:
        if self.reconstruction_weight <= 0:
            return None
        terms = []
        for layer in self.layers:
            for segment in caches[layer].get("codec_segments", []):
                clean = segment.get("_a1_clean_k")
                if not torch.is_tensor(clean):
                    continue
                restored = self.decode_student(segment["k"], dtype=clean.dtype, device=clean.device)
                flat_clean = clean.detach().float().reshape(-1)
                flat_restored = restored.float().reshape(-1)
                count = min(65536, flat_clean.numel())
                stride = max(1, flat_clean.numel() // count)
                flat_clean = flat_clean[::stride][:count]
                flat_restored = flat_restored[::stride][:count]
                terms.append((flat_restored - flat_clean).norm() / flat_clean.norm().clamp_min(1e-8))
        return None if not terms else torch.stack(terms).mean() * self.reconstruction_weight

    def update(self, caches: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.student_losses:
            raise RuntimeError("A1 update has no block loss")
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        block_loss = torch.stack(self.student_losses).mean()
        anchor = self.reconstruction_anchor(caches)
        loss = block_loss if anchor is None else block_loss + anchor
        loss.backward()
        grad_sq = 0.0
        for parameter in self.deltas.values():
            if parameter.grad is not None:
                grad_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
        parameters = [self.deltas[layer] for layer in self.layers if layer in self.deltas]
        if self.grad_clip > 0 and parameters:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        for optimizer in self.optimizers.values():
            optimizer.step()
        result = {
            "trajectory": self.trajectory_id,
            "chunk": self.chunk_id,
            "step": self.step_id,
            "layers": list(self.layers),
            "loss": float(loss.detach().cpu()),
            "block_loss": float(block_loss.detach().cpu()),
            "anchor_loss": 0.0 if anchor is None else float(anchor.detach().cpu()),
            "grad_norm": grad_sq ** 0.5,
            "metrics": list(self.metrics),
        }
        self.student_losses.clear()
        self.metrics.clear()
        return result

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        payload = {
            "format": FORMAT,
            "basis_root": self.basis_root,
            "rank": self.rank,
            "sp_rank": self.sp_rank,
            "layers": list(self.layers),
            "decoder_delta": {
                str(layer): parameter.detach().float().cpu()
                for layer, parameter in self.deltas.items()
            },
            "optimizer": {
                str(layer): optimizer.state_dict()
                for layer, optimizer in self.optimizers.items()
            },
            "source": self.source,
            "metadata": dict(metadata or {}),
            "created_at": time.time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        temporary.replace(path)

    def append_log(self, row: dict[str, Any]) -> None:
        if self.output_root is None:
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def merge_group_checkpoints(paths: Iterable[str | Path], output: str | Path) -> None:
    """Merge five disjoint 8-layer group checkpoints into one inference artifact."""
    merged: dict[str, torch.Tensor] = {}
    provenance = []
    basis_root = None
    rank = None
    for raw in paths:
        path = Path(raw)
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != FORMAT:
            raise RuntimeError(f"bad group checkpoint: {path}")
        basis_root = basis_root or item["basis_root"]
        rank = rank or int(item["rank"])
        if basis_root != item["basis_root"] or rank != int(item["rank"]):
            raise RuntimeError("incompatible K-A1 group checkpoints")
        overlap = set(merged) & set(item["decoder_delta"])
        if overlap:
            raise RuntimeError(f"overlapping K-A1 layers: {sorted(overlap)}")
        merged.update(item["decoder_delta"])
        provenance.append(str(path.resolve()))
    if set(map(int, merged)) != set(range(40)):
        raise RuntimeError(f"merged K-A1 artifact does not cover 40 layers: {sorted(map(int, merged))}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT, "basis_root": basis_root, "rank": rank, "sp_rank": 0,
        "layers": list(range(40)), "decoder_delta": merged,
        "source_groups": provenance, "created_at": time.time(),
    }, output)
