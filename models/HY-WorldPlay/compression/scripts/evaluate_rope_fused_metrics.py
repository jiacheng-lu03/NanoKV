import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_pt
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_pt
from tqdm import tqdm


def read_video(path: Path, device: torch.device, target_hw=None) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_hw is not None and frame.shape[:2] != target_hw:
            frame = cv2.resize(frame, (target_hw[1], target_hw[0]))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to read video: {path}")
    arr = np.asarray(frames)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).to(device)


def safe_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    val = psnr_pt(a, b, data_range=1.0).item()
    return 99.0 if math.isinf(val) else float(val)


@torch.no_grad()
def metric_triplet(a: torch.Tensor, b: torch.Tensor, loss_fn) -> tuple[float, float, float]:
    return (
        safe_psnr(a, b),
        float(ssim_pt(a, b, kernel_size=7, data_range=1.0).item()),
        float(loss_fn(a.mul(2.0).sub(1.0), b.mul(2.0).sub(1.0)).item()),
    )


@torch.no_grad()
def direct_eval(ref: torch.Tensor, test: torch.Tensor, loss_fn) -> dict:
    n = min(ref.shape[0], test.shape[0])
    vals = [metric_triplet(ref[i : i + 1], test[i : i + 1], loss_fn) for i in tqdm(range(n), desc="direct")]
    vals_np = np.asarray(vals, dtype=np.float64)
    return {
        "alignment": "direct",
        "frames": n,
        "psnr": float(vals_np[:, 0].mean()),
        "ssim": float(vals_np[:, 1].mean()),
        "lpips": float(vals_np[:, 2].mean()),
    }


@torch.no_grad()
def mutual_eval(ref: torch.Tensor, test: torch.Tensor, loss_fn, window: int) -> dict:
    vals = []
    trace = []
    for i in tqdm(range(ref.shape[0]), desc="mutual"):
        start = max(0, i - window)
        end = min(test.shape[0], i + window + 1)
        candidates = test[start:end]
        mse = torch.mean((candidates - ref[i : i + 1]) ** 2, dim=(1, 2, 3))
        best = int(torch.argmin(mse).item()) + start
        vals.append(metric_triplet(ref[i : i + 1], test[best : best + 1], loss_fn))
        trace.append((i, best))
    vals_np = np.asarray(vals, dtype=np.float64)
    return {
        "alignment": f"mutual_window_{window}",
        "frames": ref.shape[0],
        "psnr": float(vals_np[:, 0].mean()),
        "ssim": float(vals_np[:, 1].mean()),
        "lpips": float(vals_np[:, 2].mean()),
        "trace_mean_abs_offset": float(np.mean([abs(i - j) for i, j in trace])),
        "trace_max_abs_offset": int(max(abs(i - j) for i, j in trace)),
    }


@torch.no_grad()
def self_eval(video: torch.Tensor, loss_fn, pivot: int, end: int, window: int) -> dict:
    small = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    limit = min(int(pivot * 0.9), video.shape[0])
    vals = []
    trace = []
    for i in tqdm(range(limit), desc="self"):
        ideal = min(end - i, video.shape[0] - 1)
        start = max(pivot, ideal - window)
        stop = min(video.shape[0] - 1, ideal + window)
        if start > stop:
            continue
        best_score = -1.0
        best_idx = start
        for j in range(start, stop + 1):
            score = float(ssim_pt(small[i : i + 1], small[j : j + 1], kernel_size=3, data_range=1.0).item())
            if score > best_score:
                best_score = score
                best_idx = j
        vals.append(metric_triplet(video[i : i + 1], video[best_idx : best_idx + 1], loss_fn))
        trace.append((i, best_idx))
    vals_np = np.asarray(vals, dtype=np.float64)
    return {
        "alignment": f"self_window_{window}",
        "frames": len(vals),
        "psnr": float(vals_np[:, 0].mean()),
        "ssim": float(vals_np[:, 1].mean()),
        "lpips": float(vals_np[:, 2].mean()),
        "trace_mean_abs_offset_from_ideal": float(np.mean([abs((end - i) - j) for i, j in trace])),
        "trace_max_abs_offset_from_ideal": int(max(abs((end - i) - j) for i, j in trace)),
    }


def write_outputs(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "quality_metrics.csv"
    keys = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "quality_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    lines = ["# Fused RoPE Quality Metrics", ""]
    lines.append("| case | type | PSNR | SSIM | LPIPS | notes |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows:
        notes = []
        for key in ("trace_mean_abs_offset", "trace_max_abs_offset", "trace_mean_abs_offset_from_ideal", "trace_max_abs_offset_from_ideal"):
            if key in row:
                notes.append(f"{key}={row[key]}")
        lines.append(
            f"| {row['case']} | {row['alignment']} | {row['psnr']:.3f} | "
            f"{row['ssim']:.4f} | {row['lpips']:.4f} | {'; '.join(notes)} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--nofused", required=True, type=Path)
    parser.add_argument("--fused", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--mutual_window", type=int, default=30)
    parser.add_argument("--self_pivot", type=int, default=125)
    parser.add_argument("--self_end", type=int, default=250)
    parser.add_argument("--self_window", type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    baseline = read_video(args.baseline, device)
    h, w = int(baseline.shape[2]), int(baseline.shape[3])
    nofused = read_video(args.nofused, device, target_hw=(h, w))
    fused = read_video(args.fused, device, target_hw=(h, w))

    rows = []
    for case, vid in (("prekv_nofused_vs_baseline", nofused), ("prekv_fused_vs_baseline", fused)):
        direct = direct_eval(baseline, vid, loss_fn)
        direct.update({"case": case})
        rows.append(direct)
        mutual = mutual_eval(baseline, vid, loss_fn, args.mutual_window)
        mutual.update({"case": case})
        rows.append(mutual)

    for case, vid in (("baseline_self", baseline), ("prekv_nofused_self", nofused), ("prekv_fused_self", fused)):
        self_row = self_eval(vid, loss_fn, args.self_pivot, args.self_end, args.self_window)
        self_row.update({"case": case})
        rows.append(self_row)

    write_outputs(args.out_dir, rows)
    print(f"saved metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
