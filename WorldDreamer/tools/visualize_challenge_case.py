#!/usr/bin/env python
"""Create 6-camera grids and preview media for a challenge case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.camera_io import CAMERA_ORDER, camera_frame_path, load_camera_first_images, make_6view_grid, make_grid, read_json


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _grid_for(root: Path, frame_id: int) -> Image.Image:
    return make_6view_grid(load_camera_first_images(root, frame_id))


def _save_preview_gif(grids: List[Image.Image], out_path: Path, duration_ms: int = 120) -> None:
    if not grids:
        return
    grids[0].save(out_path, save_all=True, append_images=grids[1:], duration=duration_ms, loop=0)


def visualize(args: argparse.Namespace) -> None:
    case_dir = Path(args.case_dir)
    manifest = read_json(case_dir / "input" / "manifest.json")
    pred_dir = Path(args.pred_dir) if args.pred_dir else case_dir / "outputs" / "basedreamer_limsim_no_leakage"
    diag = case_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    history_root = case_dir / "input" / "history_images"
    gt_root = case_dir / "gt_future_for_eval_only" / "images"
    history_last = int(manifest["num_history_frames"]) - 1
    _grid_for(history_root, history_last).save(diag / "history_last_grid.jpg", quality=95)

    preview_ids = [0, max(0, args.num_frames // 2), args.num_frames - 1]
    gt_grids = [_grid_for(gt_root, frame_id) for frame_id in preview_ids if camera_frame_path(gt_root, CAMERA_ORDER[0], frame_id).exists()]
    pred_grids = [_grid_for(pred_dir, frame_id) for frame_id in preview_ids if camera_frame_path(pred_dir, CAMERA_ORDER[0], frame_id).exists()]
    if gt_grids:
        make_grid(gt_grids, cols=1).save(diag / "gt_preview_grid.jpg", quality=95)
    if pred_grids:
        make_grid(pred_grids, cols=1).save(diag / "pred_preview_grid.jpg", quality=95)
    compare = [Image.open(diag / "history_last_grid.jpg").convert("RGB")]
    if gt_grids:
        compare.append(gt_grids[0])
    if pred_grids:
        compare.append(pred_grids[0])
    make_grid(compare, cols=1).save(diag / "compare_history_gt_pred.jpg", quality=95)

    gif_ids = list(range(0, args.num_frames, max(1, args.preview_stride)))
    gif_grids = [_grid_for(pred_dir, frame_id) for frame_id in gif_ids if camera_frame_path(pred_dir, CAMERA_ORDER[0], frame_id).exists()]
    _save_preview_gif(gif_grids, diag / "pred_preview.gif")
    print(f"Wrote visualizations to {diag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--camera-first", type=str2bool, default=True)
    parser.add_argument("--preview-stride", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.camera_first:
        raise ValueError("Only camera-first layout is supported.")
    visualize(args)


if __name__ == "__main__":
    main()
