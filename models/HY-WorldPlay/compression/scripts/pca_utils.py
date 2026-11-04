"""Shared NanoKV fitting and tensor utilities."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


CANDIDATE_DIMS = {
    "spatiotemporal": (1024, 512),
    "spatial": (512, 256),
    "temporal": (256, 128),
    "channel": (128, 64),
    # The actual dimensions of the two head modes are checked dynamically.
    "global_head": (None, None),
    "grouped_head_contiguous": (512, 256),
    "grouped_head_contiguous_per_group": (512, 256),
    "head_axis_shared": (None, None),
    "head_axis_per_channel": (None, None),
}


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def load_config(path: Path) -> Dict:
    config = json.loads(path.read_text())
    required = {
        "data_root",
        "layers",
        "candidates",
        "max_blocks_per_atom",
        "seed",
        "train_atoms",
        "val_atoms",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    unknown = set(config["candidates"]) - set(CANDIDATE_DIMS)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    return config


def atom_path(data_root: Path, split: str, atom: Dict, layer: int) -> Path:
    case_id = str(atom["case_id"])
    position = int(atom["position"])
    return (
        data_root
        / split
        / case_id
        / f"position_{position:02d}"
        / "kv"
        / (
            f"case-{case_id}_position{position:02d}_layer{layer:02d}"
            "_target-v_branch-main_rope_kvstage-pre_v_vision.pt"
        )
    )


def load_raw_bf16(path: Path) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "tensor" not in payload:
        raise ValueError(f"{path}: expected raw BF16 payload with tensor")
    tensor = payload["tensor"]
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{path}: expected BF16, found {tensor.dtype}")
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"{path}: unexpected tensor shape {tuple(tensor.shape)}")
    thw = tuple(int(value) for value in payload["thw"])
    t, h, w = thw
    _, heads, sequence, dim = tensor.shape
    if sequence != t * h * w or dim != 128:
        raise ValueError(f"{path}: shape/thw mismatch {tuple(tensor.shape)} vs {thw}")
    # [B, heads, sequence, dim] -> [T, H, W, heads, dim].
    tensor = (
        tensor[0]
        .reshape(heads, t, h, w, dim)
        .permute(1, 2, 3, 0, 4)
        .contiguous()
    )
    return tensor, thw


def sampled_linear_indices(total: int, count: int, seed: int) -> torch.Tensor:
    count = min(int(count), int(total))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:count]


def unravel(linear: torch.Tensor, sizes: Sequence[int]) -> List[torch.Tensor]:
    values: List[torch.Tensor] = []
    remaining = linear
    for size in reversed(sizes):
        values.append(remaining.remainder(int(size)))
        remaining = torch.div(remaining, int(size), rounding_mode="floor")
    return list(reversed(values))


def pack_samples(
    tensor: torch.Tensor,
    thw: Tuple[int, int, int],
    mode: str,
    count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Return packed FP32 rows, row head/group ids, and packing metadata."""
    t, h, w = thw
    heads = int(tensor.shape[3])
    dim = int(tensor.shape[4])
    if dim != 128:
        raise ValueError(f"head_dim must be 128, found {dim}")

    if mode == "spatiotemporal":
        sizes = (t // 2, h // 2, w // 2, heads)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi, head = unravel(linear, sizes)
        offsets = torch.tensor(
            [(dt, dh, dw) for dt in range(2) for dh in range(2) for dw in range(2)],
            dtype=torch.long,
        )
        values = tensor[
            2 * ti[:, None] + offsets[None, :, 0],
            2 * hi[:, None] + offsets[None, :, 1],
            2 * wi[:, None] + offsets[None, :, 2],
            head[:, None],
            :,
        ]
        rows = values.reshape(values.shape[0], -1).float()
        return rows, head, {"tokens_per_row": 8, "heads": heads, "head_dim": dim}

    if mode == "spatial":
        sizes = (t, h // 2, w // 2, heads)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi, head = unravel(linear, sizes)
        offsets = torch.tensor(
            [(dh, dw) for dh in range(2) for dw in range(2)], dtype=torch.long
        )
        values = tensor[
            ti[:, None],
            2 * hi[:, None] + offsets[None, :, 0],
            2 * wi[:, None] + offsets[None, :, 1],
            head[:, None],
            :,
        ]
        rows = values.reshape(values.shape[0], -1).float()
        return rows, head, {"tokens_per_row": 4, "heads": heads, "head_dim": dim}

    if mode == "temporal":
        sizes = (t // 2, h, w, heads)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi, head = unravel(linear, sizes)
        offsets = torch.tensor([0, 1], dtype=torch.long)
        values = tensor[
            2 * ti[:, None] + offsets[None, :],
            hi[:, None],
            wi[:, None],
            head[:, None],
            :,
        ]
        rows = values.reshape(values.shape[0], -1).float()
        return rows, head, {"tokens_per_row": 2, "heads": heads, "head_dim": dim}

    if mode == "channel":
        sizes = (t, h, w, heads)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi, head = unravel(linear, sizes)
        rows = tensor[ti, hi, wi, head, :].float()
        return rows, head, {"tokens_per_row": 1, "heads": heads, "head_dim": dim}

    if mode == "global_head":
        sizes = (t, h, w)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi = unravel(linear, sizes)
        values = tensor[ti, hi, wi, :, :]
        rows = values.reshape(values.shape[0], -1).float()
        row_ids = torch.full((rows.shape[0],), -1, dtype=torch.long)
        return rows, row_ids, {
            "tokens_per_row": 1,
            "heads": heads,
            "heads_per_row": heads,
            "head_dim": dim,
        }

    if mode in {
        "grouped_head_contiguous",
        "grouped_head_contiguous_per_group",
    }:
        group_size = 4
        if heads % group_size:
            raise ValueError(f"{heads} heads cannot be divided into groups of {group_size}")
        groups = heads // group_size
        sizes = (t, h, w, groups)
        linear = sampled_linear_indices(math.prod(sizes), count, seed)
        ti, hi, wi, group = unravel(linear, sizes)
        head_offsets = torch.arange(group_size, dtype=torch.long)
        values = tensor[
            ti[:, None],
            hi[:, None],
            wi[:, None],
            group[:, None] * group_size + head_offsets[None, :],
            :,
        ]
        rows = values.reshape(values.shape[0], -1).float()
        return rows, group, {
            "tokens_per_row": 1,
            "heads": heads,
            "heads_per_row": group_size,
            "group_size": group_size,
            "groups": groups,
            "head_dim": dim,
        }

    if mode in {"head_axis_shared", "head_axis_per_channel"}:
        if heads % 2:
            raise ValueError(
                f"head_axis_shared requires an even head count, found {heads}"
            )
        # At every token/channel coordinate, compress only the head axis.
        values = tensor.reshape(t * h * w, heads, dim).permute(0, 2, 1)
        total = int(values.shape[0] * values.shape[1])
        linear = sampled_linear_indices(total, count, seed)
        token = torch.div(linear, dim, rounding_mode="floor")
        channel = linear.remainder(dim)
        rows = values[token, channel].float()
        return rows, channel, {
            "tokens_per_row": 1,
            "heads": heads,
            "heads_per_row": heads,
            "head_dim": dim,
            "axis": "head",
            "shared_across_channels": True,
        }

    raise ValueError(mode)


def collect_samples(
    config: Dict,
    layer: int,
    split: str,
    atoms: Sequence[Dict],
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Dict],
    Dict[str, List[int]],
    List[Dict],
]:
    data_root = Path(config["data_root"])
    max_blocks = int(config["max_blocks_per_atom"])
    base_seed = int(config["seed"])
    rows: Dict[str, List[torch.Tensor]] = {mode: [] for mode in config["candidates"]}
    ids: Dict[str, List[torch.Tensor]] = {mode: [] for mode in config["candidates"]}
    counts: Dict[str, List[int]] = {mode: [] for mode in config["candidates"]}
    metadata: Dict[str, Dict] = {}
    audit: List[Dict] = []

    for atom_index, atom in enumerate(atoms):
        path = atom_path(data_root, split, atom, layer)
        if not path.is_file():
            raise FileNotFoundError(path)
        tensor, thw = load_raw_bf16(path)
        atom_record = {
            "split": split,
            "case_id": atom["case_id"],
            "position": int(atom["position"]),
            "layer": int(layer),
            "path": str(path),
            "shape": [int(value) for value in tensor.shape],
            "thw": list(thw),
            "dtype": str(tensor.dtype),
        }
        audit.append(atom_record)
        for mode in config["candidates"]:
            mode_seed = stable_seed(base_seed, split, atom_index, atom["case_id"], atom["position"], mode)
            packed, row_ids, mode_meta = pack_samples(
                tensor, thw, mode, max_blocks, mode_seed
            )
            rows[mode].append(packed)
            ids[mode].append(row_ids)
            counts[mode].append(int(packed.shape[0]))
            if mode in metadata and metadata[mode] != mode_meta:
                raise ValueError(f"inconsistent {mode} metadata: {metadata[mode]} vs {mode_meta}")
            metadata[mode] = mode_meta
        del tensor

    return (
        {mode: torch.cat(parts, dim=0) for mode, parts in rows.items()},
        {mode: torch.cat(parts, dim=0) for mode, parts in ids.items()},
        metadata,
        counts,
        audit,
    )


def fit_pca(train: torch.Tensor, latent_dim: int, device: torch.device):
    x = train.to(device=device, dtype=torch.float32)
    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.T @ centered / max(1, x.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    components = eigenvectors[:, -latent_dim:].T.contiguous()
    explained = float(
        eigenvalues[-latent_dim:].sum().div(eigenvalues.sum().clamp_min(1e-20)).item()
    )
    return mean, components, explained


def fit_per_id_pca(
    train: torch.Tensor,
    row_ids: torch.Tensor,
    num_ids: int,
    latent_dim: int,
    device: torch.device,
):
    means = []
    components = []
    explained = []
    counts = []
    for group_id in range(int(num_ids)):
        selected = train[row_ids == group_id]
        if selected.shape[0] < selected.shape[1]:
            raise ValueError(
                f"group {group_id} has too few rows: {selected.shape[0]}"
            )
        mean, basis, ratio = fit_pca(selected, latent_dim, device)
        means.append(mean.detach().cpu())
        components.append(basis.detach().cpu())
        explained.append(float(ratio))
        counts.append(int(selected.shape[0]))
    total = max(1, sum(counts))
    weighted = sum(ratio * count for ratio, count in zip(explained, counts)) / total
    return torch.stack(means), torch.stack(components), weighted


def reconstruct_per_id(
    rows: torch.Tensor,
    row_ids: torch.Tensor,
    means: torch.Tensor,
    components: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty_like(rows, dtype=torch.float32)
    for group_id in range(means.shape[0]):
        mask = row_ids == group_id
        if not mask.any():
            continue
        output[mask] = reconstruct(
            rows[mask],
            means[group_id].to(device),
            components[group_id].to(device),
            device,
        )
    return output


def reconstruct(
    rows: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    device: torch.device,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    outputs = []
    for start in range(0, rows.shape[0], chunk_rows):
        x = rows[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        centered = x - mean
        decoded = (centered @ components.T) @ components + mean
        outputs.append(decoded.cpu())
    return torch.cat(outputs, dim=0)


def rel_l2(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    numerator = (reconstructed - original).square().sum()
    denominator = original.square().sum().clamp_min(1e-20)
    return float(torch.sqrt(numerator / denominator).item())


def cosine_mean(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(original.float(), reconstructed.float(), dim=1, eps=1e-8)
        .mean()
        .item()
    )


def per_head_rel(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    row_ids: torch.Tensor,
    mode: str,
    metadata: Dict,
) -> List[float]:
    heads = int(metadata["heads"])
    dim = int(metadata["head_dim"])
    numerator = torch.zeros(heads, dtype=torch.float64)
    denominator = torch.zeros(heads, dtype=torch.float64)
    error = (reconstructed - original).double()
    source = original.double()

    if mode in {"spatiotemporal", "spatial", "temporal", "channel"}:
        for head in range(heads):
            mask = row_ids == head
            if mask.any():
                numerator[head] += error[mask].square().sum()
                denominator[head] += source[mask].square().sum()
    elif mode == "global_head":
        e = error.reshape(error.shape[0], heads, dim)
        x = source.reshape(source.shape[0], heads, dim)
        numerator += e.square().sum(dim=(0, 2))
        denominator += x.square().sum(dim=(0, 2))
    elif mode in {
        "grouped_head_contiguous",
        "grouped_head_contiguous_per_group",
    }:
        group_size = int(metadata["group_size"])
        e = error.reshape(error.shape[0], group_size, dim)
        x = source.reshape(source.shape[0], group_size, dim)
        for group in range(int(metadata["groups"])):
            mask = row_ids == group
            if not mask.any():
                continue
            start = group * group_size
            numerator[start : start + group_size] += e[mask].square().sum(dim=(0, 2))
            denominator[start : start + group_size] += x[mask].square().sum(dim=(0, 2))
    elif mode in {"head_axis_shared", "head_axis_per_channel"}:
        # Each row is [heads] for one token/channel coordinate.
        numerator += error.square().sum(dim=0)
        denominator += source.square().sum(dim=0)
    else:
        raise ValueError(mode)

    values = torch.sqrt(numerator / denominator.clamp_min(1e-20))
    return [float(value) for value in values]


def dc_ac_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    tokens_per_row: int,
    head_dim: int,
) -> Tuple[float, float]:
    x = original.reshape(original.shape[0], tokens_per_row, head_dim).double()
    y = reconstructed.reshape(reconstructed.shape[0], tokens_per_row, head_dim).double()
    x_dc = x.mean(dim=1, keepdim=True)
    y_dc = y.mean(dim=1, keepdim=True)
    dc = torch.sqrt(
        (y_dc - x_dc).square().sum() / x_dc.square().sum().clamp_min(1e-20)
    )
    x_ac = x - x_dc
    y_ac = y - y_dc
    ac = torch.sqrt(
        (y_ac - x_ac).square().sum() / x_ac.square().sum().clamp_min(1e-20)
    )
    return float(dc.item()), float(ac.item())


def evaluate_rows(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    row_ids: torch.Tensor,
    mode: str,
    metadata: Dict,
) -> Dict:
    head_values = per_head_rel(original, reconstructed, row_ids, mode, metadata)
    sorted_heads = sorted(head_values)
    p95_index = min(len(sorted_heads) - 1, math.ceil(0.95 * len(sorted_heads)) - 1)
    result = {
        "rel_l2": rel_l2(original, reconstructed),
        "cosine": cosine_mean(original, reconstructed),
        "per_head_rel_l2": head_values,
        "per_head_rel_l2_mean": sum(head_values) / len(head_values),
        "per_head_rel_l2_max": max(head_values),
        "per_head_rel_l2_p95": sorted_heads[p95_index],
    }
    if mode in {"spatiotemporal", "spatial"}:
        dc, ac = dc_ac_metrics(
            original,
            reconstructed,
            int(metadata["tokens_per_row"]),
            int(metadata["head_dim"]),
        )
        result["dc_rel_l2"] = dc
        result["ac_rel_l2"] = ac
    return result


def apply_overrides(
    config: Dict,
    *,
    seed_override: int | None,
    max_blocks_override: int | None,
    candidates_override: str | None,
) -> Dict:
    config = json.loads(json.dumps(config))
    if seed_override is not None:
        config["seed"] = int(seed_override)
    if max_blocks_override is not None:
        config["max_blocks_per_atom"] = int(max_blocks_override)
    if candidates_override:
        candidates = [value.strip() for value in candidates_override.split(",") if value.strip()]
        unknown = set(candidates) - set(CANDIDATE_DIMS)
        if unknown:
            raise ValueError(f"unknown override candidates: {sorted(unknown)}")
        config["candidates"] = candidates
    return config


def run_layer(
    config_path: Path,
    output_dir: Path,
    layer: int,
    device_name: str,
    *,
    seed_override: int | None = None,
    max_blocks_override: int | None = None,
    candidates_override: str | None = None,
) -> Path:
    config = load_config(config_path)
    config = apply_overrides(
        config,
        seed_override=seed_override,
        max_blocks_override=max_blocks_override,
        candidates_override=candidates_override,
    )
    if layer not in [int(value) for value in config["layers"]]:
        raise ValueError(f"layer {layer} not configured")
    device = torch.device(device_name)
    if device.type == "cuda":
        # Workers are isolated with CUDA_VISIBLE_DEVICES, so an unindexed
        # "cuda" refers to logical device 0 inside that worker.
        torch.cuda.set_device(0 if device.index is None else device.index)

    started = time.time()
    train_rows, train_ids, train_meta, train_counts, train_audit = collect_samples(
        config, layer, "train", config["train_atoms"]
    )
    val_rows, val_ids, val_meta, val_counts, val_audit = collect_samples(
        config, layer, "val", config["val_atoms"]
    )
    if train_meta != val_meta:
        raise ValueError("train/val metadata mismatch")

    layer_dir = output_dir / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in config["candidates"]:
        input_dim = int(train_rows[mode].shape[1])
        if mode in {
            "global_head",
            "head_axis_shared",
            "head_axis_per_channel",
        }:
            latent_dim = input_dim // 2
        else:
            expected_input, expected_latent = CANDIDATE_DIMS[mode]
            if input_dim != expected_input:
                raise ValueError(f"{mode}: expected input {expected_input}, found {input_dim}")
            latent_dim = int(expected_latent)
        if input_dim % 2 or latent_dim * 2 != input_dim:
            raise ValueError(f"{mode}: not an exact 2x bottleneck")

        candidate_started = time.time()
        if mode in {
            "head_axis_per_channel",
            "grouped_head_contiguous_per_group",
        }:
            num_ids = (
                int(train_meta[mode]["head_dim"])
                if mode == "head_axis_per_channel"
                else int(train_meta[mode]["groups"])
            )
            mean, components, explained = fit_per_id_pca(
                train_rows[mode],
                train_ids[mode],
                num_ids,
                latent_dim,
                device,
            )
            train_recon = reconstruct_per_id(
                train_rows[mode], train_ids[mode], mean, components, device
            )
            val_recon = reconstruct_per_id(
                val_rows[mode], val_ids[mode], mean, components, device
            )
        else:
            mean, components, explained = fit_pca(
                train_rows[mode], latent_dim, device
            )
            train_recon = reconstruct(train_rows[mode], mean, components, device)
            val_recon = reconstruct(val_rows[mode], mean, components, device)
        train_metrics = evaluate_rows(
            train_rows[mode], train_recon, train_ids[mode], mode, train_meta[mode]
        )
        val_metrics = evaluate_rows(
            val_rows[mode], val_recon, val_ids[mode], mode, val_meta[mode]
        )
        if mode in {
            "head_axis_per_channel",
            "grouped_head_contiguous_per_group",
        }:
            mean_only = torch.empty_like(val_rows[mode])
            for group_id in range(mean.shape[0]):
                mask = val_ids[mode] == group_id
                mean_only[mask] = mean[group_id]
        else:
            mean_only = mean.detach().cpu()[None, :].expand_as(val_rows[mode])
        val_metrics["mean_only_rel_l2"] = rel_l2(val_rows[mode], mean_only)
        atom_metrics = []
        start = 0
        for atom, count in zip(config["val_atoms"], val_counts[mode]):
            stop = start + int(count)
            atom_metrics.append(
                {
                    "case_id": atom["case_id"],
                    "position": int(atom["position"]),
                    "rows": int(count),
                    "rel_l2": rel_l2(val_rows[mode][start:stop], val_recon[start:stop]),
                    "cosine": cosine_mean(
                        val_rows[mode][start:stop], val_recon[start:stop]
                    ),
                }
            )
            start = stop
        artifact = {
            "mean": mean.detach().cpu().to(torch.float32),
            "components": components.detach().cpu().to(torch.float32),
            "candidate": mode,
            "layer": layer,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        }
        artifact_path = layer_dir / f"{mode}_pca.pt"
        torch.save(artifact, artifact_path)
        results.append(
            {
                "layer": layer,
                "candidate": mode,
                "compression_ratio": input_dim / latent_dim,
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "train_rows": int(train_rows[mode].shape[0]),
                "val_rows": int(val_rows[mode].shape[0]),
                "explained_variance_ratio": explained,
                "train": train_metrics,
                "val": val_metrics,
                "val_atoms": atom_metrics,
                "train_val_gap_rel_l2": val_metrics["rel_l2"] - train_metrics["rel_l2"],
                "packing": train_meta[mode],
                "pca_artifact": str(artifact_path),
                "elapsed_sec": time.time() - candidate_started,
            }
        )
        print(
            f"[VAxis] L{layer:02d} {mode:24s} "
            f"val_rel={val_metrics['rel_l2']:.6f} "
            f"cos={val_metrics['cosine']:.6f} "
            f"worst_head={val_metrics['per_head_rel_l2_max']:.6f}",
            flush=True,
        )

    manifest = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "layer": layer,
        "train_atoms": train_audit,
        "val_atoms": val_audit,
        "input_storage": "raw_bf16",
        "latent_storage": "bf16_oracle_no_quantization",
        "sampling_seed": int(config["seed"]),
        "max_blocks_per_atom": int(config["max_blocks_per_atom"]),
        "effective_candidates": list(config["candidates"]),
    }
    (layer_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = {
        "layer": layer,
        "device": str(device),
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    result_path = layer_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    return result_path


def summarize(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    rows = []
    for layer in config["layers"]:
        result_path = output_dir / f"layer_{int(layer):02d}" / "results.json"
        payload = json.loads(result_path.read_text())
        for result in payload["results"]:
            val = result["val"]
            rows.append(
                {
                    "layer": int(layer),
                    "candidate": result["candidate"],
                    "compression_ratio": result["compression_ratio"],
                    "input_dim": result["input_dim"],
                    "latent_dim": result["latent_dim"],
                    "val_rel_l2": val["rel_l2"],
                    "val_cosine": val["cosine"],
                    "worst_head": val["per_head_rel_l2_max"],
                    "p95_head": val["per_head_rel_l2_p95"],
                    "explained_variance": result["explained_variance_ratio"],
                    "train_val_gap": result["train_val_gap_rel_l2"],
                    "dc_rel_l2": val.get("dc_rel_l2", ""),
                    "ac_rel_l2": val.get("ac_rel_l2", ""),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for candidate in config["candidates"]:
        selected = [row for row in rows if row["candidate"] == candidate]
        aggregate.append(
            {
                "candidate": candidate,
                "layers": len(selected),
                "mean_val_rel_l2": sum(row["val_rel_l2"] for row in selected) / len(selected),
                "mean_val_cosine": sum(row["val_cosine"] for row in selected) / len(selected),
                "mean_worst_head": sum(row["worst_head"] for row in selected) / len(selected),
                "max_layer_rel_l2": max(row["val_rel_l2"] for row in selected),
                "mean_explained_variance": sum(row["explained_variance"] for row in selected)
                / len(selected),
            }
        )
    aggregate.sort(key=lambda row: row["mean_val_rel_l2"])
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# V axis Stage-1 static PCA ranking",
        "",
        "Raw BF16 input; exact 2x structural bottleneck; no latent quantization.",
        "",
        "| rank | candidate | mean rel-L2 | mean cosine | mean worst-head | worst layer rel-L2 | explained variance |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate, 1):
        lines.append(
            f"| {rank} | {row['candidate']} | {row['mean_val_rel_l2']:.6f} | "
            f"{row['mean_val_cosine']:.6f} | {row['mean_worst_head']:.6f} | "
            f"{row['max_layer_rel_l2']:.6f} | {row['mean_explained_variance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a static reconstruction ranking only. It must not be used to",
            "select a deployable codec before same-state attention/block evaluation.",
        ]
    )
    (output_dir / "STATIC_RANKING.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def validate_only(config_path: Path, layer: int) -> None:
    config = load_config(config_path)
    atom = config["train_atoms"][0]
    path = atom_path(Path(config["data_root"]), "train", atom, layer)
    tensor, thw = load_raw_bf16(path)
    report = {}
    for mode in config["candidates"]:
        rows, row_ids, metadata = pack_samples(
            tensor,
            thw,
            mode,
            min(32, int(config["max_blocks_per_atom"])),
            stable_seed(config["seed"], "gate", mode),
        )
        input_dim = rows.shape[1]
        latent_dim = input_dim // 2
        if input_dim % 2:
            raise AssertionError(f"{mode}: odd input dimension {input_dim}")
        if not torch.isfinite(rows).all():
            raise AssertionError(f"{mode}: non-finite packed values")
        report[mode] = {
            "rows": int(rows.shape[0]),
            "input_dim": int(input_dim),
            "latent_dim": int(latent_dim),
            "compression_ratio": input_dim / latent_dim,
            "row_id_min": int(row_ids.min().item()),
            "row_id_max": int(row_ids.max().item()),
            "packing": metadata,
        }
    print(json.dumps({"layer": layer, "path": str(path), "thw": thw, "gate": report}, indent=2))






