# Modified for NanoKV: KV-cache codec integration or evaluation configuration.
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
        "image":
            "examples/04/image.jpg",
    },
}


def _configure_deterministic_inference(seed: int) -> None:
    """Opt-in deterministic execution for cross-machine reproducibility gates.

    This is deliberately disabled on the frozen default path.  The launcher
    must also set CUBLAS_WORKSPACE_CONFIG before Python starts.
    """
    if os.environ.get("LINGBOT_DETERMINISTIC_INFERENCE", "0") != "1":
        return
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="causal_fast",
        choices=["causal_fast", "causal_pretrain"],
        help="Inference mode: 'causal_fast' for the distilled few-step model, "
             "'causal_pretrain' for the pretrained causal model with 40-step CFG sampling.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="The chunk size."),
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument("--kv_codec_enable", action="store_true")
    parser.add_argument("--kv_pre_rope_clean", action="store_true")
    parser.add_argument("--kv_codec_basis_root_k", type=str, default=None)
    parser.add_argument("--kv_codec_basis_root_v", type=str, default=None)
    parser.add_argument("--kv_codec_linear_root_v", type=str, default=None)
    parser.add_argument("--kv_codec_linear_root_k", type=str, default=None)
    parser.add_argument("--kv_codec_rank", type=int, default=640)
    parser.add_argument("--kv_codec_clusters", type=int, default=64)
    parser.add_argument("--kv_codec_iters", type=int, default=2)
    parser.add_argument("--kv_dump_root", type=str, default=None)
    parser.add_argument("--kv_dump_split", choices=("train", "dev"), default="train")
    parser.add_argument("--kv_dump_case_id", type=str, default="")
    parser.add_argument("--kv_dump_chunks", type=str, default="1,3,5,7")
    parser.add_argument("--kv_dump_tokens", type=int, default=1024)
    parser.add_argument("--kvtc_enable", action="store_true")
    parser.add_argument("--kvtc_root_k", type=str, default=None)
    parser.add_argument("--kvtc_root_v", type=str, default=None)
    parser.add_argument("--kvtc_group_size", type=int, default=8)
    parser.add_argument("--kvtc_sink_tokens", type=int, default=4)
    parser.add_argument("--kvtc_window_tokens", type=int, default=128)
    parser.add_argument(
        "--kv_baseline", choices=("qvg", "turboquant"), default=None,
        help="Training-free pre-RoPE KV baseline.",
    )
    parser.add_argument(
        "--qvg_root", type=str,
        default=os.environ.get("LINGBOT_V2_QVG_ROOT", "external/Quant-VideoGen"),
    )
    parser.add_argument(
        "--worldplay_root", type=str,
        default=str(Path(__file__).resolve().parents[2] / "HY-WorldPlay/compression"),
    )
    parser.add_argument("--kv_runtime_summary", type=str, default=None)
    parser.add_argument("--k_a1_train", action="store_true")
    parser.add_argument("--k_a1_layers", type=str, default="")
    parser.add_argument("--k_a1_chunks", type=str, default="2,4,6,8")
    parser.add_argument("--k_a1_inner_steps", type=int, default=1)
    parser.add_argument("--k_a1_lr", type=float, default=5e-6)
    parser.add_argument("--k_a1_source", type=str, default=None)
    parser.add_argument("--k_a1_output", type=str, default=None)
    parser.add_argument("--k_a1_log_root", type=str, default=None)
    parser.add_argument("--k_a1_trajectory_id", type=str, default="")
    parser.add_argument("--k_a1_discard_video", action="store_true")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def run_causal(args, cfg, img, device, rank, mode="causal_fast"):
    """Run inference with the distilled few-step model (infer_mode='causal_fast')."""
    logging.info(f"Creating WanI2VCausal pipeline (infer_mode={mode}).")
    codec_requested = bool(
        args.kv_pre_rope_clean or args.kv_codec_enable
        or args.kv_dump_root or args.kvtc_enable
        or args.kv_baseline
        or args.kv_runtime_summary
    )
    if args.kv_codec_enable and (not args.kv_codec_basis_root_k or not args.kv_codec_basis_root_v):
        raise ValueError("--kv_codec_enable requires both K and V basis roots")
    if args.kvtc_enable and (not args.kvtc_root_k or not args.kvtc_root_v):
        raise ValueError("--kvtc_enable requires both K and V artifact roots")
    active_compressed = sum(
        (bool(args.kv_codec_enable), bool(args.kvtc_enable), bool(args.kv_baseline))
    )
    if active_compressed > 1:
        raise ValueError("codec, KVTC, QVG and TurboQuant are mutually exclusive")
    if args.kv_baseline == "qvg" and not Path(args.qvg_root).is_dir():
        raise FileNotFoundError(args.qvg_root)
    kv_codec_options = None
    if codec_requested:
        kv_codec_options = {
            "pre_rope_clean": bool(args.kv_pre_rope_clean),
            "codec_enabled": bool(args.kv_codec_enable),
            "codec_basis_root_k": args.kv_codec_basis_root_k,
            "codec_basis_root_v": args.kv_codec_basis_root_v,
            "codec_linear_root_k": args.kv_codec_linear_root_k,
            "codec_linear_root_v": args.kv_codec_linear_root_v,
            "codec_rank": int(args.kv_codec_rank),
            "codec_clusters": int(args.kv_codec_clusters),
            "codec_iters": int(args.kv_codec_iters),
            "codec_dump_root": args.kv_dump_root,
            "codec_dump_split": args.kv_dump_split,
            "codec_case_id": args.kv_dump_case_id,
            "codec_dump_chunks": args.kv_dump_chunks,
            "codec_dump_tokens": int(args.kv_dump_tokens),
            "kvtc_enabled": bool(args.kvtc_enable),
            "kvtc_root_k": args.kvtc_root_k,
            "kvtc_root_v": args.kvtc_root_v,
            "kvtc_group_size": int(args.kvtc_group_size),
            "kvtc_sink_tokens": int(args.kvtc_sink_tokens),
            "kvtc_window_tokens": int(args.kvtc_window_tokens),
            "baseline_enabled": bool(args.kv_baseline),
            "baseline_mode": args.kv_baseline,
            "runtime_summary": args.kv_runtime_summary,
        }
        if args.kv_baseline:
            os.environ["LINGBOT_V2_QVG_ROOT"] = str(args.qvg_root)
            os.environ["LINGBOT_V2_WORLDPLAY_ROOT"] = str(args.worldplay_root)
    wan_i2v = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=mode,
        kv_codec_options=kv_codec_options,
    )
    a1_controller = None
    if args.k_a1_train:
        if not args.kv_codec_enable:
            raise ValueError("--k_a1_train requires --kv_codec_enable")
        if not args.k_a1_layers or not args.k_a1_output:
            raise ValueError("--k_a1_train requires --k_a1_layers and --k_a1_output")
        from wan.modules.lingbot_v2_k_a1 import LingBotV2KA1
        a1_controller = LingBotV2KA1(
            basis_root=args.kv_codec_basis_root_k,
            rank=int(args.kv_codec_rank),
            layers=[int(x) for x in args.k_a1_layers.split(",") if x.strip()],
            chunks=[int(x) for x in args.k_a1_chunks.split(",") if x.strip()],
            inner_steps=int(args.k_a1_inner_steps),
            learning_rate=float(args.k_a1_lr),
            source=args.k_a1_source,
            output_root=args.k_a1_log_root,
        )
        wan_i2v.kv_codec_options["a1_trajectory_id"] = args.k_a1_trajectory_id
        wan_i2v.set_k_a1_controller(a1_controller)
    # Static KV-compression operators are model assets.  Load them in the
    # model-ready setup phase so end-to-end generation timing measures codec
    # computation rather than one-time disk I/O.  Their resident bytes remain
    # included in CUDA peaks and in the strict system-KV accounting.
    if args.kv_codec_enable and not args.k_a1_train:
        from wan.modules.lingbot_v2_kv_codec import preload_operators
        preload_operators(
            basis_root_k=args.kv_codec_basis_root_k,
            basis_root_v=args.kv_codec_basis_root_v,
            linear_root_k=args.kv_codec_linear_root_k,
            linear_root_v=args.kv_codec_linear_root_v,
            rank=int(args.kv_codec_rank), layers=int(wan_i2v.model.config.num_layers),
            joint_dim=int(wan_i2v.model.config.dim), sp_rank=int(rank),
            device=torch.device(f"cuda:{device}"),
        )
    elif args.kvtc_enable:
        from wan.modules.lingbot_v2_kvtc import preload as preload_kvtc
        preload_kvtc(
            root_k=args.kvtc_root_k, root_v=args.kvtc_root_v,
            group_size=int(args.kvtc_group_size),
            num_layers=int(wan_i2v.model.config.num_layers), sp_rank=int(rank),
            device=torch.device(f"cuda:{device}"),
        )
    elif args.kv_baseline:
        from wan.modules.lingbot_v2_training_free_baselines import preload
        preload(
            str(args.kv_baseline), device=torch.device(f"cuda:{device}"),
            layers=int(wan_i2v.model.config.num_layers),
            head_dim=int(wan_i2v.model.config.dim // wan_i2v.model.config.num_heads),
        )
    logging.info("Generating video ...")
    # Unified WorldPlay-style protocol: model construction/loading is excluded;
    # input preparation, online generation and VAE decode are included.  Reset
    # only the allocator counter, without changing the cache lifecycle.
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        profile_start_allocated = torch.cuda.memory_allocated(device)
        profile_start_reserved = torch.cuda.memory_reserved(device)
    profile_started = time.perf_counter()
    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    profile_seconds = time.perf_counter() - profile_started
    if args.kv_runtime_summary and rank == 0:
        summary_path = Path(args.kv_runtime_summary)
        payload = json.loads(summary_path.read_text())
        payload["runtime_profile"] = {
            "protocol": "model-ready through input preparation, generation and VAE decode",
            "seconds": profile_seconds,
            "start_allocated_bytes": profile_start_allocated,
            "start_reserved_bytes": profile_start_reserved,
            "end_allocated_bytes": torch.cuda.memory_allocated(device),
            "end_reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        stage_profile = getattr(wan_i2v, "_last_runtime_stage_profile", None)
        if stage_profile is not None:
            payload["runtime_profile"]["stages"] = stage_profile
        if os.environ.get("LINGBOT_V2_CUDA_LIVE_TENSOR_AUDIT", "0") == "1":
            import gc

            storages = {}
            for obj in gc.get_objects():
                try:
                    if not torch.is_tensor(obj) or not obj.is_cuda:
                        continue
                    storage = obj.untyped_storage()
                    key = (str(obj.device), int(storage.data_ptr()))
                    entry = storages.setdefault(key, {
                        "bytes": int(storage.nbytes()), "views": [],
                    })
                    if len(entry["views"]) < 4:
                        entry["views"].append({
                            "shape": list(obj.shape), "dtype": str(obj.dtype),
                        })
                except (ReferenceError, RuntimeError, TypeError):
                    continue
            largest = sorted(
                storages.values(), key=lambda item: item["bytes"], reverse=True
            )
            payload["cuda_live_tensor_audit"] = {
                "unique_storage_bytes": int(sum(x["bytes"] for x in largest)),
                "largest_storages": largest[:100],
            }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    if a1_controller is not None:
        a1_controller.save(
            args.k_a1_output,
            metadata={
                "trajectory_id": args.k_a1_trajectory_id,
                "chunks": args.k_a1_chunks,
                "inner_steps": int(args.k_a1_inner_steps),
                "learning_rate": float(args.k_a1_lr),
            },
        )
    return video


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _configure_deterministic_inference(args.base_seed)
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        init_distributed_group()

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    video = run_causal(args, cfg, img, device, rank, mode=args.infer_mode)

    if rank == 0 and not args.k_a1_discard_video:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            args.save_file = f"lingbot-world-v2-{args.infer_mode}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
