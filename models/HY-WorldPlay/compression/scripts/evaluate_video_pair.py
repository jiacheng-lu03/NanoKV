#!/usr/bin/env python3
"""Direct frame-aligned PSNR/SSIM/LPIPS for one generated video pair."""

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch

from video_metrics import metric_triplet, read_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = read_video(args.reference, device)
    candidate = read_video(args.candidate, device)
    if reference.shape != candidate.shape or reference.shape[0] == 0:
        raise ValueError(f'Video shapes must match and be nonempty: {reference.shape} vs {candidate.shape}')
    frames = reference.shape[0]
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    with torch.no_grad():
        for index in range(frames):
            psnr, ssim, perceptual = metric_triplet(
                reference[index : index + 1],
                candidate[index : index + 1],
                loss_fn,
            )
            rows.append(
                {
                    "frame": index,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": perceptual,
                }
            )
    values = np.asarray(
        [[row["psnr"], row["ssim"], row["lpips"]] for row in rows],
        dtype=np.float64,
    )
    # Frame zero is the shared input image and can dominate a short canary.
    generated = values[1:] if frames > 1 else values
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "frames": frames,
        "all_frames": {
            "psnr": float(values[:, 0].mean()),
            "ssim": float(values[:, 1].mean()),
            "lpips": float(values[:, 2].mean()),
        },
        "generated_frames_1_onward": {
            "psnr": float(generated[:, 0].mean()),
            "ssim": float(generated[:, 1].mean()),
            "lpips": float(generated[:, 2].mean()),
        },
        "per_frame": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["generated_frames_1_onward"], indent=2))


if __name__ == "__main__":
    main()
