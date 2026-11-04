"""Frame-aligned PSNR, SSIM and LPIPS for a generated/reference pair."""
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    script=root/'models/HY-WorldPlay/compression/scripts/evaluate_video_pair_direct.py'
    raise SystemExit(subprocess.call([sys.executable,str(script),*sys.argv[1:]],env=os.environ.copy()))
