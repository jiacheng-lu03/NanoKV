#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from kv_dataset_int8_utils import load_kv_dataset_payload


DEFAULT_LAYERS = "0,1,2,4,6,8,10,13,15,18,20,23,25,27,29,32,34,36,38,40,42,45,47,49,50,52,53"
DEFAULT_OUT = "data/codec_samples"


def parse_int_list(value: str):
    items = []
    for part in re.split(r"[\s,]+", value.strip()):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            items.extend(range(int(start), int(end) + 1))
        else:
            items.append(int(part))
    return sorted(dict.fromkeys(items))


def load_cases(split_json: Path):
    data = json.loads(split_json.read_text(encoding="utf-8"))
    # A standalone *_cases.json is convenient for a small held-out dump.  Keep
    # those samples under ``val`` so they can never be mixed with fitting data
    # by consumers that discover atoms from the split directory.
    if isinstance(data, list):
        data = {"train": [], "val": data, "test": []}
    cases = []
    for split in ("train", "val", "test"):
        for item in data.get(split, []):
            case = dict(item)
            case["split"] = split
            cases.append(case)
    return data, cases


def select_cases(cases, spec: str):
    selected = []
    for idx in parse_int_list(spec):
        if idx < 0 or idx >= len(cases):
            raise ValueError(f"case index {idx} out of range [0,{len(cases)-1}]")
        selected.append((idx, cases[idx]))
    return selected


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_file_info(path: Path):
    item = torch.load(path, map_location="cpu")
    tensor, meta = load_kv_dataset_payload(path)
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "metadata": meta if isinstance(item, dict) else {},
    }


def build_command(args, case, video_dir: Path):
    return [
        sys.executable,
        "hyvideo/generate.py",
        "--prompt",
        case["prompt"],
        "--image_path",
        case["image"],
        "--resolution",
        "480p",
        "--aspect_ratio",
        args.aspect_ratio,
        "--video_length",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--rewrite",
        "false",
        "--sr",
        "false",
        "--save_pre_sr_video",
        "--pose",
        case["pose"],
        "--output_path",
        str(video_dir),
        "--model_path",
        args.model_path,
        "--action_ckpt",
        args.action_ckpt,
        "--few_step",
        "true",
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--model_type",
        "ar",
        "--use_vae_parallel",
        "false",
        "--use_sageattn",
        "false",
        "--use_fp8_gemm",
        "false",
        "--transformer_resident_ar_rollout",
        "true" if args.optimized_cache_lifecycle else "false",
        "--offloading",
        "false",
        "--group_offloading",
        "false",
    ]


def collect_position(args, case, position: int):
    position_root = Path(args.output_dir) / case["split"] / case["id"] / f"position_{position:02d}"
    kv_dir = position_root / "kv"
    targets = ["k", "v"] if args.target == "kv" else [args.target]
    tensor_names = {target: f"{target}_vision" for target in targets}
    files = []
    for tensor_name in tensor_names.values():
        files.extend(sorted(kv_dir.glob(f"*_{tensor_name}.pt")))
    layer_infos = {}
    warnings = []
    total_bytes = 0
    for path in files:
        info = tensor_file_info(path)
        meta = info["metadata"]
        layer = int(meta.get("layer_idx", -1))
        file_target = str(meta.get("target", ""))
        tensor_name = tensor_names.get(file_target, f"{file_target}_vision")
        total_bytes += int(info["file_size_bytes"])
        layer_infos[f"{file_target}:{layer}"] = info
        if meta.get("tensor_name") != tensor_name:
            warnings.append(f"{path.name}: tensor_name is not {tensor_name}")
        if meta.get("branch") != "main_rope":
            warnings.append(f"{path.name}: branch is not main_rope")
        if meta.get("kv_stage") != "pre":
            warnings.append(f"{path.name}: kv_stage is not pre")
    sample_meta = next(iter(layer_infos.values()), {}).get("metadata", {})
    actual_context_len = int(sample_meta.get("actual_context_len", -1))
    expected_context_len = int(sample_meta.get("expected_context_len", 5))
    if actual_context_len != expected_context_len:
        warnings.append(
            f"actual_context_len={actual_context_len}, expected_context_len={expected_context_len}"
        )
    meta_json = {
        "video_id": case["id"],
        "case_id": case["id"],
        "split": case["split"],
        "prompt": case["prompt"],
        "image": case["image"],
        "pose": case["pose"],
        "seed": args.seed,
        "num_frames": args.num_frames,
        "chunk_position": position,
        "requested_position": position,
        "actual_context_start": sample_meta.get("actual_context_start"),
        "actual_context_end": sample_meta.get("actual_context_end"),
        "actual_context_len": actual_context_len,
        "actual_context_latent_count": sample_meta.get("actual_context_latent_count"),
        "selected_frame_indices": sample_meta.get("selected_frame_indices", []),
        "expected_context_len": expected_context_len,
        "layers": sorted({int(k.split(':', 1)[1]) for k in layer_infos if ":" in k}),
        "target": args.target,
        "branch": "main_rope",
        "kv_stage": "pre",
        "dtype": args.dtype,
        "storage_format": args.storage_format,
        "tensor_shape_per_layer": {
            layer: info["shape"] for layer, info in layer_infos.items()
        },
        "file_path_per_layer": {
            layer: info["path"] for layer, info in layer_infos.items()
        },
        "file_size_per_layer": {
            layer: info["file_size_bytes"] for layer, info in layer_infos.items()
        },
        "dump_hook_name": "dump_multipos_pre_kv",
        "dump_module_path": "hyvideo.models.transformers.modules.kv_ae3d",
        "whether_model_offload_disabled": True,
        "warnings": warnings,
        "notes": [
            "Dumped from cache_vision=True recache forward.",
            "The transformer stores K/V before RoPE/ProPE when HY_PRE_POSITION_VISION_KV_CACHE=1.",
            f"This run saves target={args.target.upper()}; post-position KV is not saved.",
        ],
    }
    write_json(position_root / "meta.json", meta_json)
    return meta_json, total_bytes, files


def parse_peak_from_log(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\[Peak CUDA Memory\] allocated=([0-9.]+) GB reserved=([0-9.]+) GB", text)
    if not match:
        return {}
    return {
        "peak_allocated_gb": float(match.group(1)),
        "peak_reserved_gb": float(match.group(2)),
    }


def find_free_local_port():
    """Ask the OS for a currently unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def is_rendezvous_port_collision(log_path: Path):
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "EADDRINUSE" in text or "address already in use" in text.lower()




def run_case(args, case_index, case):
    case_root = Path(args.output_dir) / case["split"] / case["id"]
    video_dir = case_root / "video"
    log_path = case_root / "run.log"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    # generate.py initializes a one-rank distributed process group directly.
    # Give every case a deterministic, distinct rendezvous port so an unrelated
    # job using the framework default cannot cause EADDRINUSE mid-shard.
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(int(args.master_port) + int(case_index))

    # Keep the generation path to baseline + pre-position KV cache only.
    # These assignments prevent stale shell variables from silently enabling
    # AE, codec, attention dumps, or other online compression experiments.
    forced_off_env = {
        "HY_KV_AE3D_ENABLE": "0",
        "HY_KV_AE3D_DUMP_ENABLE": "0",
        "HY_KV_AE3D_ATTN_DUMP_ENABLE": "0",
        "HY_LAYERWISE_RAW_V_AE": "0",
        "HY_KV_CODEC_ENABLE": "0",
        "HY_KV_CODEC_MODE": "none",
        "HY_KV_CODEC_PROFILE": "0",
        "HY_PREKV_AE_ENABLE": "0",
        "HY_QVG_ENABLE": "0",
        "HY_KVTC_ENABLE": "0",
    }
    env.update(forced_off_env)
    if args.optimized_cache_lifecycle:
        # Match the lossless pre-RoPE lifecycle used by the frozen formal
        # WorldPlay runners.  These flags change temporary-buffer ownership,
        # not the clean generation path or the tensors captured below.
        env.update(
            {
                "HY_WORLDPLAY_TEMPORAL_CONTEXT_SIZE": "24",
                "HY_WORLDPLAY_MEMORY_FRAMES": "32",
                "HY_POSITION_ENCODING_PROFILE": "0",
                "HY_PRE_POSITION_RESTORE_SINGLE_ROPE": "1",
                "HY_PRE_POSITION_RESTORE_PROPE_KV_ONLY": "1",
                "HY_PRE_POSITION_PROPE_PREPARE_CACHE": "1",
                "HY_PRE_POSITION_ROPE_FREQ_CACHE": "1",
                "HY_PRE_POSITION_ROPE_WORK_REUSE": "1",
                "HY_PRE_POSITION_DIRECT_FINAL_CAT": "1",
                "HY_FUSED_ROPE_ONE": "0",
                "HY_FUSED_ROPE_ONE_EXACT": "0",
                "HY_PRE_POSITION_TRITON_FINAL_CAT": "0",
                "HY_PRE_POSITION_PROPE_JOINT_KV": "0",
                "HY_KV_EFF_OPT_FUSED_CACHE_CAT": "1",
                "HY_KV_EFF_OPT_PROPE_SINGLE_BLOCK": "1",
                "HY_KV_EFF_OPT_EMPTY_CACHE_INTERVAL_LAYERS": "0",
                "HY_KV_DROP_OLD_BEFORE_REBUILD": "1",
                "HY_KV_DROP_OLD_EMPTY_CACHE": "0",
                "HY_LAYERWISE_PREDECODE_NEXT": "0",
            }
        )
    env["HY_PRE_POSITION_VISION_KV_CACHE"] = "1"
    env["HY_PRE_POSITION_VISION_KV_PROFILE"] = "1"
    env["HY_KV_MULTIPOS_DUMP_ENABLE"] = "0" if args.kvtc_sampled_only else "1"
    if args.kvtc_sampled_only:
        env["HY_KVTC_CALIB_DUMP_ENABLE"] = "1"
        env["HY_KVTC_CALIB_DUMP_DIR"] = str(Path(args.output_dir))
        env["HY_KVTC_CALIB_DUMP_POSITIONS"] = ",".join(str(v) for v in args.positions)
        env["HY_KVTC_CALIB_TOKENS_PER_SNAPSHOT"] = str(args.kvtc_tokens_per_snapshot)
    env["HY_KV_MULTIPOS_DUMP_DIR"] = str(Path(args.output_dir))
    env["HY_KV_MULTIPOS_DUMP_POSITIONS"] = ",".join(str(v) for v in args.positions)
    env["HY_KV_MULTIPOS_DUMP_LAYERS"] = ",".join(str(v) for v in args.layers)
    env["HY_KV_MULTIPOS_TARGET"] = args.target
    env["HY_KV_MULTIPOS_BRANCH"] = args.branch
    env["HY_KV_MULTIPOS_KV_STAGE"] = args.kv_stage
    env["HY_KV_MULTIPOS_DTYPE"] = args.dtype
    env["HY_KV_MULTIPOS_STORAGE_FORMAT"] = args.storage_format
    env["HY_KV_MULTIPOS_FORCE"] = "1" if args.force else "0"
    env["HY_KV_MULTIPOS_CASE_ID"] = case["id"]
    env["HY_KV_MULTIPOS_SPLIT"] = case["split"]
    env["HY_KV_MULTIPOS_VIDEO_ID"] = f"video_{case_index:03d}"
    env["HY_KV_MULTIPOS_SEED"] = str(args.seed)
    env["HY_KV_MULTIPOS_NUM_FRAMES"] = str(args.num_frames)
    env["HY_STOP_AFTER_AR_CHUNK"] = str(max(args.positions)) if args.smoke else env.get("HY_STOP_AFTER_AR_CHUNK", "-1")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = build_command(args, case, video_dir)
    start = time.time()
    proc = None
    max_attempts = max(1, int(args.port_retries) + 1)
    for attempt in range(max_attempts):
        if args.dynamic_master_port:
            env["MASTER_PORT"] = str(find_free_local_port())
        mode = "w" if attempt == 0 else "a"
        with log_path.open(mode, encoding="utf-8") as log:
            if attempt:
                log.write(f"\n[port-retry] attempt={attempt + 1}/{max_attempts}\n")
            log.write(" ".join(cmd) + "\n\n")
            log.write("Environment summary:\n")
            log.write(f"MASTER_ADDR={env['MASTER_ADDR']}\n")
            log.write(f"MASTER_PORT={env['MASTER_PORT']}\n")
            logged_keys = set(forced_off_env) | {
                "HY_PRE_POSITION_VISION_KV_CACHE",
                "HY_PRE_POSITION_VISION_KV_PROFILE",
                "HY_STOP_AFTER_AR_CHUNK",
            }
            for key in sorted(
                k for k in env if k.startswith("HY_KV_MULTIPOS") or k in logged_keys
            ):
                log.write(f"{key}={env[key]}\n")
            log.write("\n")
            log.flush()
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            break
        if not is_rendezvous_port_collision(log_path) or attempt + 1 >= max_attempts:
            break
        time.sleep(min(1.0 + attempt, 3.0))
    elapsed = time.time() - start

    peak = parse_peak_from_log(log_path)
    records = []
    for position in args.positions:
        meta, total_bytes, files = collect_position(args, case, position)
        layer_info = {}
        for layer, path in meta["file_path_per_layer"].items():
            layer_info[layer] = tensor_file_info(Path(path))
        records.append(
            {
                "case_index": case_index,
                "case_id": case["id"],
                "split": case["split"],
                "position": position,
                "seed": args.seed,
                "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "log": str(log_path),
                "meta": meta,
                "meta_path": str(
                    Path(args.output_dir)
                    / case["split"]
                    / case["id"]
                    / f"position_{position:02d}"
                    / "meta.json"
                ),
                "total_bytes": total_bytes,
                "files": [str(p) for p in files],
                "layer_info": layer_info,
                **peak,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True, help="JSON manifest containing input images, prompts and split labels")
    parser.add_argument("--cases", default="0")
    parser.add_argument("--positions", nargs="+", default=["5"])
    parser.add_argument("--layers", nargs="+", default=["13", "27", "45"])
    parser.add_argument("--num_frames", type=int, default=253)
    parser.add_argument("--target", default="v", choices=["v", "k", "kv"])
    parser.add_argument("--branch", default="main_rope", choices=["main_rope"])
    parser.add_argument("--kv_stage", default="pre", choices=["pre"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--storage_format", default="bf16", choices=["bf16", "int8"])
    parser.add_argument("--kvtc_sampled_only", action="store_true")
    parser.add_argument("--kvtc_tokens_per_snapshot", type=int, default=32)
    parser.add_argument("--disable_model_offload", action="store_true")
    parser.add_argument(
        "--optimized_cache_lifecycle",
        action="store_true",
        help="Use the frozen lossless pre-RoPE cache lifecycle and resident rollout.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--master_port", default=os.environ.get("HY_MASTER_PORT", "29671"))
    parser.add_argument("--dynamic_master_port", action="store_true")
    parser.add_argument("--port_retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--aspect_ratio", default="16:9")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--action_ckpt", required=True)
    args = parser.parse_args()

    args.positions = parse_int_list(",".join(args.positions))
    args.layers = parse_int_list(",".join(args.layers))
    if not args.disable_model_offload:
        raise SystemExit("Collection requires --disable_model_offload")
    if args.target not in {"k", "v", "kv"} or args.branch != "main_rope" or args.kv_stage != "pre":
        raise SystemExit("Collection only supports target=k/v/kv, branch=main_rope, kv_stage=pre")

    split_data, all_cases = load_cases(Path(args.split_json))
    selected = select_cases(all_cases, args.cases)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "dataset_plan_snapshot.json", split_data)

    run_records = []
    total_bytes = 0
    for case_index, case in selected:
        positions_text = ",".join(str(v) for v in args.positions)
        print(f"[multipos-dump] case={case['id']} positions={positions_text}", flush=True)
        records = run_case(args, case_index, case)
        for rec in records:
            run_records.append(rec)
            total_bytes += rec["total_bytes"]
            if rec["returncode"] != 0:
                raise RuntimeError(f"generation failed; see {rec['log']}")

    summary = {
        "config": vars(args),
        "runs": run_records,
        "total_bytes": total_bytes,
    }
    write_json(Path(args.output_dir) / "dataset_summary.json", summary)
    print(Path(args.output_dir) / "dataset_summary.json")


if __name__ == "__main__":
    main()
