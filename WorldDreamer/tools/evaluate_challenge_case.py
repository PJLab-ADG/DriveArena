#!/usr/bin/env python
"""Evaluate camera-first challenge predictions against GT future images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.metrics import evaluate_camera_first


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--camera-first", type=str2bool, default=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.camera_first:
        raise ValueError("Only camera-first layout is supported.")
    report = evaluate_camera_first(args.gt_dir, args.pred_dir, args.num_frames, out_json=args.out_json)
    overall = report["overall"]
    print(f"PSNR: {overall['psnr']}")
    print(f"SSIM: {overall['ssim']}")
    print(f"Missing frame rate: {overall['missing_frame_rate']:.6f}")
    print(f"Invalid image rate: {overall['invalid_image_rate']:.6f}")
    print(f"Wrote metrics: {args.out_json}")


if __name__ == "__main__":
    main()
