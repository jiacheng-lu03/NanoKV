#!/usr/bin/env python3
# Modified for NanoKV: KV-cache codec integration or evaluation configuration.
"""Internal one-GPU runner used by the public ``infer_distilled.py`` CLI."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import accelerate
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core.data.unified_dataset import _derive_seed
from distilled_config import load_inference_config
from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args
from examples.wanvideo.pipeline.mosaic.causal_inference import (
    run_distilled_inference,
)
from examples.wanvideo.pipeline.mosaic.datasets import (
    build_mosaic_validation_dataset,
)
from examples.wanvideo.pipeline.mosaic.pipeline_module import (
    WanMosaicPipelineModule,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--wan-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--memory-cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_paths(args):
    for label, path, kind in (
        ("config", args.config, "file"),
        ("checkpoint", args.checkpoint, "file"),
        ("dataset index", args.dataset_index, "file"),
        ("workspace", args.workspace, "dir"),
        ("Wan model", args.wan_dir, "dir"),
        ("tokenizer", args.tokenizer_dir, "dir"),
    ):
        ok = path.is_file() if kind == "file" else path.is_dir()
        if not ok:
            raise FileNotFoundError(f"{label} {kind} does not exist: {path}")


def build_model(args, accelerator):
    """Construct the existing public Matrix model with inference settings."""
    return WanMosaicPipelineModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        trained_dit=args.trained_dit,
        task="sft",
        device=accelerator.device,
        enable_mosaic=args.enable_mosaic,
        preview_dataset=False,
        num_preview=1,
        use_prope=args.use_prope,
        prope_attention_interval=args.prope_attention_interval,
        prope_disable_native_rope=args.prope_disable_native_rope,
        prope_disable_t_rope=args.prope_disable_t_rope,
        prope_camera_layout=args.prope_camera_layout,
        only_prope=args.only_prope,
        vae_decode_tiled=args.vae_decode_tiled,
        mosaic_use_revgrid_rope=args.mosaic_use_revgrid_rope,
        mosaic_view_change_prope=args.mosaic_view_change_prope,
        mosaic_interval=args.mosaic_interval,
        mosaic_fuse_mode=args.mosaic_fuse_mode_train,
        mosaic_fuse_block_size=args.mosaic_fuse_block_size,
        mosaic_return_source_frame_ids=args.mosaic_return_source_frame_ids,
        mosaic_drop_holes=args.mosaic_drop_holes,
        mosaic_no_mosaic_sync_mode=args.mosaic_no_mosaic_sync_mode,
        mosaic_sequence_balance_mode=args.mosaic_sequence_balance_mode,
        recent_first_photo_refine=args.recent_first_photo_refine,
        recent_first_dedupe_tiles=args.recent_first_dedupe_tiles,
        candidates_per_query_group_train=args.candidates_per_query_group_train,
        recent_first_geometry_device=args.recent_first_geometry_device,
        recent_first_backend=args.recent_first_backend,
        recent_first_warp_dtype=args.recent_first_warp_dtype,
        recent_first_enable_tf32=args.recent_first_enable_tf32,
        recent_first_zbuffer=args.recent_first_zbuffer,
        recent_first_n_candidates_keyframe=args.recent_first_n_candidates_keyframe,
        recent_first_keyframe_neighbour_window=(
            args.recent_first_keyframe_neighbour_window
        ),
        mosaic_query_reference_frame=args.mosaic_query_reference_frame,
        memory_vae_encode_input_frames=args.memory_vae_encode_input_frames,
        train_anchor_block_prob=0.0,
        train_context_count_max=args.train_context_count_max,
        train_context_selection=args.train_context_selection,
        loose_model_loading=args.loose_model_loading,
        trans_scale=args.trans_scale,
        allow_no_prompt=args.allow_no_prompt,
        debug=args.debug,
    )


def main(argv=None):
    cli = build_parser().parse_args(argv)
    validate_paths(cli)
    config = load_inference_config(cli.config)
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Accelerate may otherwise leave the inter-op pool at the full host core
    # count even when OMP_NUM_THREADS is explicitly bounded.  Pin both pools
    # before model construction so concurrent one-GPU launches do not
    # oversubscribe this two-socket host during the 5B module build.
    cpu_threads = int(
        os.environ.get("MG35_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "16"))
    )
    interop_threads = int(os.environ.get("MG35_CPU_INTEROP_THREADS", "8"))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch only permits this before its first inter-op task.  Report the
        # actual value below if an embedding process initialized it earlier.
        pass
    startup_trace = os.environ.get("MG35_STARTUP_TRACE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    startup_t0 = time.monotonic()
    if startup_trace:
        print(
            "[startup] thread pools "
            f"intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()} "
            f"omp={os.environ.get('OMP_NUM_THREADS')} mkl={os.environ.get('MKL_NUM_THREADS')}",
            flush=True,
        )

    args = build_runtime_args(
        config,
        checkpoint=cli.checkpoint,
        dataset_index=cli.dataset_index,
        workspace=cli.workspace,
        wan_dir=cli.wan_dir,
        tokenizer_dir=cli.tokenizer_dir,
        memory_cache_dir=cli.memory_cache_dir,
    )
    args.dataset_pass_id = 0
    args.start_epoch = 0
    args.start_global_step = 0

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )
    if startup_trace:
        print(
            f"[startup] accelerator device={accelerator.device} "
            f"cuda_available={torch.cuda.is_available()} "
            f"cuda_devices={torch.cuda.device_count()}",
            flush=True,
        )
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "MG3.5 inference requires CUDA; refusing Accelerate CPU fallback "
            f"(device={accelerator.device}, count={torch.cuda.device_count()})"
        )
    args.rank = accelerator.process_index
    seed = _derive_seed(args.seed, "trainer", accelerator.process_index, 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)

    input_dataset = build_mosaic_validation_dataset(args, train_dataset=None)
    if len(input_dataset) != 1:
        raise RuntimeError(
            f"explicit input index must contain exactly one sample, got {len(input_dataset)}"
        )
    if startup_trace:
        print(
            f"[startup] dataset_ready elapsed={time.monotonic() - startup_t0:.3f}s; "
            "begin build_model",
            flush=True,
        )
    model = build_model(args, accelerator)
    if startup_trace:
        print(
            f"[startup] model_ready elapsed={time.monotonic() - startup_t0:.3f}s",
            flush=True,
        )
    if startup_trace:
        print("[startup] begin model.eval", flush=True)
    model.eval()
    if startup_trace:
        print(
            f"[startup] model.eval complete elapsed={time.monotonic() - startup_t0:.3f}s; "
            "begin distilled inference",
            flush=True,
        )
    torch.cuda.synchronize(accelerator.device)
    model_ready_seconds = time.monotonic() - startup_t0
    model_ready_allocated = torch.cuda.memory_allocated(accelerator.device)
    model_ready_reserved = torch.cuda.memory_reserved(accelerator.device)
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    inference_t0 = time.monotonic()
    inference_status = "complete"
    inference_error = None
    try:
        run_distilled_inference(
            accelerator,
            input_dataset,
            model,
            cli.output,
            args,
        )
    except BaseException as exc:
        inference_status = "failed"
        inference_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        torch.cuda.synchronize(accelerator.device)
        if os.environ.get("MG35_KVTC_PROFILE_PATH", "").strip():
            from diffsynth.models.worldplay_kvtc import write_profile
            write_profile()
        telemetry_path = os.environ.get("MG35_RUNTIME_SUMMARY", "").strip()
        if telemetry_path:
            free_bytes, total_bytes = torch.cuda.mem_get_info(accelerator.device)
            telemetry = {
                "status": inference_status,
                "error": inference_error,
                "mode": os.environ.get("MG35_KVCODEC_MODE", "off"),
                "rung": os.environ.get("MG35_WORLDPLAY_RUNG"),
                "device": str(accelerator.device),
                "device_name": torch.cuda.get_device_name(accelerator.device),
                "model_ready_seconds": model_ready_seconds,
                "inference_seconds": time.monotonic() - inference_t0,
                "model_ready_allocated_bytes": model_ready_allocated,
                "model_ready_reserved_bytes": model_ready_reserved,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(accelerator.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(accelerator.device),
                "ending_allocated_bytes": torch.cuda.memory_allocated(accelerator.device),
                "ending_reserved_bytes": torch.cuda.memory_reserved(accelerator.device),
                "device_free_bytes_at_end": free_bytes,
                "device_total_bytes": total_bytes,
                "peak_scope": "after model construction through completed inference",
            }
            path = Path(telemetry_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(telemetry, indent=2) + "\n")
    if startup_trace:
        print(
            f"[startup] distilled inference complete elapsed={time.monotonic() - startup_t0:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
