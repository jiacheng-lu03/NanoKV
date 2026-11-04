#!/usr/bin/env python3
"""One-GPU, resumable, causal LingBot-World 2.0 K-A1 trainer.

Profiles are protocol declarations, not heuristic aliases:

* small: 8 trajectories by default (optionally 12), inner=1;
* medium: 12 trajectories, inner=2;
* worldplay: 48 trajectories, inner=5.

Every trajectory follows the rollout length declared by its manifest. Updates
happen at an explicit, audited chunk schedule and denoiser step 3. Groups advance from L0-7 to L32-39;
already completed groups are used in the compressed prefix of later groups.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(sys.executable)
PROFILES = {
    "small": {"trajectories": 8, "inner_steps": 1},
    "medium": {"trajectories": 12, "inner_steps": 2},
    "worldplay": {"trajectories": 48, "inner_steps": 5},
    # Same complete calibration coverage as WorldPlay, but with an update
    # budget normalized to the empirically stable LingBot regime.
    "full48_normalized": {"trajectories": 48, "inner_steps": 1},
    # Constant-prefix scale study: an intermediate coverage point between
    # small/medium and the complete 48-trajectory calibration set.
    "half24_normalized": {"trajectories": 24, "inner_steps": 1},
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=None,
                        help="Only small may be overridden, and only to 8 or 12.")
    parser.add_argument("--lr", type=float, default=2.5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=(2, 4, 6, 8),
        help="One-based cache-commit chunks that receive A1 updates.",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-one-atom", action="store_true",
                        help="Run only group0/trajectory0 and write SMOKE_COMPLETE.json.")
    args = parser.parse_args()

    profile = dict(PROFILES[args.profile])
    if args.trajectories is not None:
        if args.profile != "small" or args.trajectories not in (8, 12):
            raise ValueError("--trajectories override is only small={8,12}")
        profile["trajectories"] = int(args.trajectories)
    chunks = tuple(int(value) for value in args.chunks)
    if not chunks or tuple(sorted(set(chunks))) != chunks:
        raise ValueError("--chunks must be a non-empty, sorted, unique sequence")
    manifest = json.loads(args.manifest.read_text())
    total_chunks = int(manifest.get("total_chunks", 8))
    if chunks[0] < 2 or chunks[-1] > total_chunks:
        raise ValueError(
            f"--chunks must lie in the historical-cache range [2, {total_chunks}]"
        )
    declared_commits = tuple(int(value) for value in manifest.get("sampled_commits", ()))
    if declared_commits and chunks != declared_commits:
        raise RuntimeError(
            f"requested chunks {chunks} do not match manifest commits {declared_commits}"
        )
    rows = manifest["train"][: profile["trajectories"]]
    if len(rows) != profile["trajectories"]:
        raise RuntimeError("manifest has too few training trajectories")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate trajectory IDs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "lingbot_v2_k_a1_serial_protocol_v1",
        "profile": args.profile,
        "trajectories": profile["trajectories"],
        "rollout_chunks": total_chunks,
        "chunks": list(chunks),
        "denoiser_step": 3,
        "inner_steps": profile["inner_steps"],
        "groups": [list(range(start, start + 8)) for start in range(0, 40, 8)],
        "teacher": "same compressed state; active-group historical K clean",
        "student": "causal shared compressed history",
        "trainable": "K decoder delta only",
        "manifest": str(args.manifest.resolve()),
        "artifact_root": str(args.artifact_root.resolve()),
        "learning_rate": args.lr,
        "seed": args.seed,
    }
    atomic_json(args.output_root / "PROTOCOL.json", protocol)
    if args.audit_only:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return

    # Mask before importing torch/wan so the requested physical GPU becomes
    # logical cuda:0.  The model is loaded exactly once for all five groups.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.pop("LINGBOT_V2_KV_CODEC_PREPACKED_OPERATOR_PATH", None)
    root_string = str(ROOT)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    import torch
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    from wan.modules.lingbot_v2_k_a1 import LingBotV2KA1

    cfg = WAN_CONFIGS["i2v-A14B"]
    codec_options = {
        "codec_enabled": True,
        "codec_basis_root_k": str(args.artifact_root / "k_pca"),
        "codec_basis_root_v": str(args.artifact_root / "v_pca"),
        "codec_linear_root_k": None,
        "codec_linear_root_v": str(args.artifact_root / "v_qat"),
        "codec_rank": 640, "codec_clusters": 64, "codec_iters": 2,
    }
    pipe = WanI2VCausal(
        config=cfg, checkpoint_dir=str(args.weights), device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=18, sink_size=6,
        infer_mode="causal_fast", kv_codec_options=codec_options,
    )

    cumulative = None
    for group_id, start in enumerate(range(0, 40, 8)):
        layers = list(range(start, start + 8))
        group_root = args.output_root / f"group_{group_id:02d}_L{start:02d}_{start + 7:02d}"
        group_root.mkdir(parents=True, exist_ok=True)
        latest = group_root / "checkpoint_latest.pt"
        group_state_path = group_root / "STATE.json"
        group_state = (
            json.loads(group_state_path.read_text())
            if group_state_path.is_file() and latest.is_file()
            else {"last_trajectory": -1}
        )
        last_complete = int(group_state.get("last_trajectory", -1))
        rolling = latest if last_complete >= 0 else None
        if args.smoke_one_atom and group_id == 0 and last_complete >= 0:
            print(latest)
            return
        for trajectory_index, row in enumerate(rows):
            if trajectory_index <= last_complete:
                continue
            controller = LingBotV2KA1(
                basis_root=args.artifact_root / "k_pca", rank=640,
                layers=layers, chunks=chunks,
                inner_steps=profile["inner_steps"], learning_rate=args.lr,
                source=rolling, output_root=group_root,
            )
            pipe.kv_codec_options["codec_linear_root_k"] = (
                None if cumulative is None else str(cumulative)
            )
            pipe.kv_codec_options["a1_trajectory_id"] = row["id"]
            pipe.set_k_a1_controller(controller)
            log = group_root / f"atom_{trajectory_index:03d}_{row['id']}.log"
            with log.open("a", encoding="utf-8") as handle:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                try:
                    sys.stdout = sys.stderr = handle
                    pipe.generate(
                        row["prompt"], Image.open(row["image"]).convert("RGB"),
                        action_path=row["action_path"], chunk_size=4,
                        max_area=MAX_AREA_CONFIGS["480*832"],
                        frame_num=int(row.get("frame_num", 125)),
                        shift=cfg.sample_shift, seed=args.seed,
                        offload_model=False,
                    )
                    controller.save(latest, metadata={
                        "trajectory_id": row["id"], "chunks": list(chunks),
                        "inner_steps": profile["inner_steps"], "group": group_id,
                        "trajectory_index": trajectory_index,
                    })
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    torch.cuda.empty_cache()
            rolling = latest
            atomic_json(group_state_path, {
                "last_trajectory": trajectory_index,
                "trajectory_id": row["id"], "checkpoint": str(latest),
            })
            atomic_json(args.output_root / "STATE.json", {
                **protocol, "last_group": group_id,
                "last_trajectory": trajectory_index, "last_checkpoint": str(latest),
            })
            if args.smoke_one_atom:
                atomic_json(args.output_root / "SMOKE_COMPLETE.json", {
                    **protocol, "status": "complete", "checkpoint": str(latest),
                    "scope": "group0 trajectory0 full eight-chunk gradient smoke",
                })
                print(latest)
                return
        if rolling is None:
            raise RuntimeError(f"group {group_id} produced no checkpoint")
        stage = args.output_root / f"stage_after_group_{group_id:02d}.pt"
        inputs = [rolling] if cumulative is None else [cumulative, rolling]
        subprocess.run([
            str(args.python), str(ROOT / "scripts/merge_key.py"),
            "--inputs", *map(str, inputs), "--output", str(stage),
        ], cwd=ROOT, check=True)
        cumulative = stage

    final = args.output_root / "k_a1_final.pt"
    subprocess.run([
        str(args.python), str(ROOT / "scripts/merge_key.py"),
        "--inputs", str(cumulative), "--output", str(final), "--require-complete",
    ], cwd=ROOT, check=True)
    atomic_json(args.output_root / "COMPLETE.json", {
        **protocol, "status": "complete", "final_checkpoint": str(final),
    })
    print(final)


if __name__ == "__main__":
    main()
