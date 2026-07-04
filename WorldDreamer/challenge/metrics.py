"""Basic image metrics for camera-first challenge predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from .camera_io import CAMERA_ORDER, camera_frame_path, write_json


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(gt.astype(np.uint8), pred.astype(np.uint8), channel_axis=2, data_range=255))
    except Exception:
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu_x = float(gt.mean())
        mu_y = float(pred.mean())
        var_x = float(gt.var())
        var_y = float(pred.var())
        cov = float(((gt - mu_x) * (pred - mu_y)).mean())
        return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))


def evaluate_camera_first(
    gt_dir: Path | str,
    pred_dir: Path | str,
    num_frames: int = 180,
    cameras: Sequence[str] = CAMERA_ORDER,
    out_json: Path | str | None = None,
) -> Mapping[str, object]:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    per_camera: Dict[str, object] = {}
    all_psnr = []
    all_ssim = []
    missing = 0
    invalid = 0
    total = num_frames * len(cameras)
    for camera in cameras:
        cam_psnr = []
        cam_ssim = []
        cam_missing = 0
        cam_invalid = 0
        for frame_id in range(num_frames):
            gt_path = camera_frame_path(gt_dir, camera, frame_id)
            pred_path = camera_frame_path(pred_dir, camera, frame_id)
            if not gt_path.is_file() or not pred_path.is_file():
                missing += 1
                cam_missing += 1
                continue
            try:
                gt = _read_rgb(gt_path)
                pred = _read_rgb(pred_path)
                if gt.shape != pred.shape:
                    pred_img = Image.fromarray(np.clip(pred, 0, 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]))
                    pred = np.asarray(pred_img, dtype=np.float32)
                p = psnr(gt, pred)
                s = ssim(gt, pred)
                cam_psnr.append(p)
                cam_ssim.append(s)
                all_psnr.append(p)
                all_ssim.append(s)
            except Exception:
                invalid += 1
                cam_invalid += 1
        per_camera[camera] = {
            "psnr": float(np.mean(cam_psnr)) if cam_psnr else None,
            "ssim": float(np.mean(cam_ssim)) if cam_ssim else None,
            "missing_frame_rate": cam_missing / max(1, num_frames),
            "invalid_image_rate": cam_invalid / max(1, num_frames),
            "valid_frames": len(cam_psnr),
        }
    report = {
        "overall": {
            "psnr": float(np.mean(all_psnr)) if all_psnr else None,
            "ssim": float(np.mean(all_ssim)) if all_ssim else None,
            "lpips": None,
            "lpips_note": "LPIPS dependency is optional and not computed by this lightweight evaluator.",
            "missing_frame_rate": missing / max(1, total),
            "invalid_image_rate": invalid / max(1, total),
            "valid_images": len(all_psnr),
            "total_images": total,
        },
        "per_camera": per_camera,
    }
    if out_json is not None:
        write_json(report, out_json)
    return report
