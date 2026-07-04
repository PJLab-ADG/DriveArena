#!/usr/bin/env python
"""No-leakage challenge inference for camera-first future predictions."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
DIFFUSERS_SRC = WORLD_DREAMER_ROOT / "third_party" / "diffusers" / "src"
if DIFFUSERS_SRC.is_dir() and str(DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(DIFFUSERS_SRC))

from challenge.camera_io import CAMERA_ORDER, camera_frame_path, ensure_camera_dirs, read_json, write_json
from challenge.coordinate_transform import box_corners_3d
from challenge.leakage_guard import assert_not_forbidden, check_no_future_leakage, guard_paths
from challenge.worlddreamer_condition_adapter import write_worlddreamer_conditions


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _ensure_sim_future(case_dir: Path, args: argparse.Namespace) -> None:
    rollout = case_dir / "sim_future" / "limsim_rollout_bridged.json"
    conditions = case_dir / "sim_future" / "worlddreamer_conditions" / "000000.json"
    if rollout.is_file() and conditions.is_file():
        return
    from simulate_future_limsim import simulate

    sim_args = argparse.Namespace(
        case_dir=str(case_dir),
        future_seconds=args.num_future_frames / args.fps,
        fps=args.fps,
        bridge_seconds=args.bridge_seconds,
        history_tail_seconds=2.0,
        use_limsim=True,
        fallback_kinematic_if_needed=True,
        output_dir_name=args.output_dir_name,
    )
    simulate(sim_args)


def _read_condition(case_dir: Path, frame_id: int) -> Mapping[str, object]:
    path = case_dir / "sim_future" / "worlddreamer_conditions" / f"{frame_id:06d}.json"
    assert_not_forbidden(path)
    return read_json(path)


def _resolve_reference(condition: Mapping[str, object], camera: str) -> Path:
    refs = condition.get("reference_images", {})
    if isinstance(refs, Mapping) and camera in refs:
        path = Path(refs[camera])
    else:
        path = Path(condition["reference_image_path"])
    assert_not_forbidden(path)
    return path


def _draw_condition_layout(image: Image.Image, condition: Mapping[str, object], camera: str) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    layout = condition.get("camera_projected_layout", {}).get(camera, [])
    for item in layout:
        if not item.get("visible") or not item.get("bbox_2d"):
            continue
        x1, y1, x2, y2 = item["bbox_2d"]
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(60, 220, 120, 130), width=2)
    return out


def _prototype_generate_frame(
    condition: Mapping[str, object],
    camera: str,
    frame_id: int,
    draw_layout_overlay: bool = False,
) -> Image.Image:
    ref_path = _resolve_reference(condition, camera)
    with Image.open(ref_path) as image:
        out = image.convert("RGB")
    brightness = 1.0 + min(0.06, frame_id * 0.00035)
    color = 1.0 + min(0.04, frame_id * 0.00025)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Color(out).enhance(color)
    if draw_layout_overlay:
        out = _draw_condition_layout(out, condition, camera)
    return out


def _find_pretrained_root(checkpoint: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for parent in [checkpoint, *checkpoint.parents]:
        if (parent / "pretrained" / "stable-diffusion-v1-5").exists():
            return parent
    raise FileNotFoundError("Could not infer base-pretrained root. Pass --pretrained-root.")


def _build_worlddreamer_pipe(cfg, device: str):
    """Build WorldDreamer without importing dataset/test utilities.

    The original `projects.dreamer.utils.test_utils` imports dataset modules at
    import time, which requires mmdet3d even for pure model inference. The
    challenge path creates tensors itself, so model loading can avoid that
    dependency.
    """
    import logging
    import torch
    import torch.nn as nn
    from diffusers import UniPCMultistepScheduler
    from omegaconf import OmegaConf
    from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2
    from projects.dreamer.utils.common import load_module

    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    checkpoint = Path(str(cfg.resume_from_checkpoint))
    model_cls = load_module(cfg.model.model_module)
    controlnet_path = checkpoint / cfg.model.controlnet_dir
    logging.info(f"Loading controlnet from {controlnet_path}")
    controlnet = model_cls.from_pretrained(str(controlnet_path), torch_dtype=weight_dtype).eval()
    pipe_param = {"controlnet": controlnet}

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = checkpoint / cfg.model.unet_dir
        logging.info(f"Loading unet from {unet_path}")
        unet = unet_cls.from_pretrained(str(unet_path), torch_dtype=weight_dtype).eval()
        if hasattr(cfg.model, "sc_attn_index"):
            for mod in unet.modules():
                if hasattr(mod, "_sc_attn_index"):
                    mod._sc_attn_index = OmegaConf.to_container(cfg.model.sc_attn_index, resolve=True)
        pipe_param["unet"] = unet

    if hasattr(cfg.model, "image_proj_model"):
        image_proj_model = nn.Linear(cfg.model.image_proj_model.input_dim, cfg.model.image_proj_model.output_dim)
        state_dict = torch.load(
            checkpoint / cfg.model.image_proj_model_dir / "image_proj_model.bin",
            map_location="cpu",
        )
        image_proj_model.load_state_dict(state_dict)
        image_proj_model = image_proj_model.to(device=device, dtype=weight_dtype).eval()
        pipe_param["image_proj_model"] = image_proj_model

    embedder = FrozenOpenCLIPImageEmbedderV2(
        arch="ViT-B-32",
        model_path="./pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
    )
    embedder = embedder.to(device=device, dtype=weight_dtype).eval()
    pipe_param["embedder"] = embedder

    pipe_cls = load_module(cfg.model.pipe_module)
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)
    return pipe, weight_dtype


def _load_ref_tensor(paths: Sequence[Path], height: int, width: int):
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images).unsqueeze(0).float()


def _worlddreamer_condition_tensors(condition: Mapping[str, object], args: argparse.Namespace):
    import torch

    h, w = args.image_height, args.image_width
    camera_param = []
    for camera in CAMERA_ORDER:
        params = condition["camera_params"][camera]
        intrinsic = torch.tensor(params["camera_intrinsic"], dtype=torch.float32)[:3, :3]
        camera2lidar = torch.tensor(params["camera2lidar"], dtype=torch.float32)[:3, :4]
        camera_param.append(torch.cat([intrinsic, camera2lidar], dim=-1))
    camera_param = torch.stack(camera_param).unsqueeze(0)

    rel_pose = torch.tensor(condition["relative_pose"]["matrix"], dtype=torch.float32).unsqueeze(0)
    bev_hdmap = torch.zeros(1, 4, 200, 200, dtype=torch.float32)
    layout_canvas = torch.zeros(1, len(CAMERA_ORDER), 14, h, w, dtype=torch.float32)
    ref_paths = [_resolve_reference(condition, camera) for camera in CAMERA_ORDER]
    ref_images = _load_ref_tensor(ref_paths, h, w)

    class_map = {
        "car": 0,
        "truck": 1,
        "construction_vehicle": 2,
        "bus": 3,
        "trailer": 4,
        "barrier": 5,
        "motorcycle": 6,
        "bicycle": 7,
        "pedestrian": 8,
        "traffic_cone": 9,
    }
    boxes = condition.get("boxes_3d", [])
    max_len = max(1, len(boxes))
    bbox_tensor = torch.zeros(1, 1, max_len, 8, 3, dtype=torch.float32)
    class_tensor = -torch.ones(1, 1, max_len, dtype=torch.long)
    mask_tensor = torch.zeros(1, 1, max_len, dtype=torch.bool)
    for idx, box in enumerate(boxes):
        corners = box_corners_3d(box["center"], box["size"], float(box["yaw"]))
        bbox_tensor[0, 0, idx] = torch.from_numpy(corners).float()
        class_tensor[0, 0, idx] = class_map.get(str(box.get("class_name", "car")), 0)
        mask_tensor[0, 0, idx] = True
    bev_kwargs = {"bboxes_3d_data": {"bboxes": bbox_tensor, "classes": class_tensor, "masks": mask_tensor}}
    return camera_param, rel_pose, bev_hdmap, layout_canvas, ref_images, bev_kwargs


def _generate_with_worlddreamer(case_dir: Path, output_root: Path, frame_ids: Sequence[int], args: argparse.Namespace) -> None:
    import torch
    from omegaconf import OmegaConf

    checkpoint = Path(args.resume_from_checkpoint).expanduser().resolve()
    hydra_cfg = checkpoint.parent / "hydra" / "config.yaml"
    if not hydra_cfg.is_file():
        hydra_cfg = checkpoint.parent.parent / "hydra" / "config.yaml"
    if not hydra_cfg.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint hydra config near {checkpoint}")
    cfg = OmegaConf.load(hydra_cfg)
    cfg.resume_from_checkpoint = str(checkpoint)
    cfg.runner.enable_xformers_memory_efficient_attention = False
    cfg.runner.pipeline_param.num_inference_steps = args.num_inference_steps
    cfg.runner.pipeline_param.guidance_scale = args.guidance_scale
    cfg.dataset.image_size = [args.image_height, args.image_width]

    pretrained_root = _find_pretrained_root(checkpoint, args.pretrained_root)
    old_cwd = Path.cwd()
    os.chdir(pretrained_root)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe, weight_dtype = _build_worlddreamer_pipe(cfg, device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
        for frame_id in frame_ids:
            condition = _read_condition(case_dir, frame_id)
            camera_param, rel_pose, bev_hdmap, layout_canvas, ref_images, bev_kwargs = _worlddreamer_condition_tensors(condition, args)
            image = pipe(
                prompt=[condition.get("text_prompt", "A driving scene.")],
                bev_hdmap=bev_hdmap.to(device=device, dtype=weight_dtype),
                camera_param=camera_param.to(device=device, dtype=weight_dtype),
                rel_pose=rel_pose.to(device=device, dtype=weight_dtype),
                ref_images=ref_images.to(device=device, dtype=weight_dtype),
                layout_canvas=layout_canvas.to(device=device, dtype=weight_dtype),
                height=args.image_height,
                width=args.image_width,
                generator=generator,
                bev_controlnet_kwargs=bev_kwargs,
                **dict(cfg.runner.pipeline_param),
            )
            for camera, pil_image in zip(CAMERA_ORDER, image.images[0]):
                pil_image.save(camera_frame_path(output_root, camera, frame_id), quality=95)
            print(f"WorldDreamer rendered frame {frame_id:06d}")
    finally:
        os.chdir(old_cwd)


def _generate_with_prototype(case_dir: Path, output_root: Path, frame_ids: Sequence[int], args: argparse.Namespace) -> None:
    for frame_id in frame_ids:
        if args.render_stride > 1 and frame_id % args.render_stride != 0 and frame_id > 0:
            for camera in CAMERA_ORDER:
                prev = camera_frame_path(output_root, camera, frame_id - 1)
                dst = camera_frame_path(output_root, camera, frame_id)
                shutil.copy2(prev, dst)
            continue
        condition = _read_condition(case_dir, frame_id)
        for camera in CAMERA_ORDER:
            image = _prototype_generate_frame(
                condition,
                camera,
                frame_id,
                draw_layout_overlay=args.draw_layout_overlay,
            )
            image.save(camera_frame_path(output_root, camera, frame_id), quality=95)
        if frame_id % max(1, args.progress_every) == 0:
            print(f"Prototype rendered frame {frame_id:06d}")


def infer(args: argparse.Namespace) -> None:
    if args.mode != "no_leakage_limsim":
        raise ValueError(f"Unsupported mode: {args.mode}")
    case_dir = Path(args.case_dir).expanduser().resolve()
    _ensure_sim_future(case_dir, args)
    write_worlddreamer_conditions(case_dir, output_dir_name=args.output_dir_name, num_frames=args.num_future_frames)

    output_root = case_dir / "outputs" / args.output_dir_name
    ensure_camera_dirs(output_root)
    guard_paths([case_dir / "input", case_dir / "sim_future" / "worlddreamer_conditions"])
    check_no_future_leakage(case_dir, pred_dir=output_root, num_frames=args.num_future_frames)

    target_frames = args.debug_num_frames or args.max_render_frames or args.num_future_frames
    target_frames = min(target_frames, args.num_future_frames)
    frame_ids = list(range(target_frames))

    backend = args.backend
    if backend in {"auto", "worlddreamer"}:
        try:
            if not args.resume_from_checkpoint:
                raise ValueError("--resume-from-checkpoint is required for WorldDreamer backend.")
            _generate_with_worlddreamer(case_dir, output_root, frame_ids, args)
            backend = "worlddreamer"
        except Exception as exc:
            if backend == "worlddreamer" or not args.fallback_prototype_if_needed:
                raise
            print(f"WorldDreamer backend fell back to prototype renderer: {exc}")
            _generate_with_prototype(case_dir, output_root, frame_ids, args)
            backend = "prototype_fallback"
    else:
        _generate_with_prototype(case_dir, output_root, frame_ids, args)
        backend = "prototype"

    if target_frames < args.num_future_frames and args.fill_unrendered_with_last:
        for frame_id in range(target_frames, args.num_future_frames):
            src_id = target_frames - 1
            for camera in CAMERA_ORDER:
                shutil.copy2(camera_frame_path(output_root, camera, src_id), camera_frame_path(output_root, camera, frame_id))

    summary = {
        "case_dir": str(case_dir),
        "mode": args.mode,
        "backend": backend,
        "output_dir": str(output_root),
        "num_requested_frames": args.num_future_frames,
        "num_rendered_frames": target_frames,
        "filled_unrendered_with_last": bool(target_frames < args.num_future_frames and args.fill_unrendered_with_last),
        "camera_order": CAMERA_ORDER,
    }
    write_json(summary, output_root / "inference_summary.json")
    report = check_no_future_leakage(case_dir, pred_dir=output_root, num_frames=args.num_future_frames)
    print(f"Inference complete: {output_root}")
    print(f"No-leakage passed: {report['passed']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--mode", default="no_leakage_limsim")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--pretrained-root", default=None)
    parser.add_argument("--backend", choices=["auto", "worlddreamer", "prototype"], default="auto")
    parser.add_argument("--fallback-prototype-if-needed", type=str2bool, default=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-future-frames", type=int, default=180)
    parser.add_argument("--output-dir-name", default="basedreamer_limsim_no_leakage")
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--debug-num-frames", type=int, default=None)
    parser.add_argument("--fill-unrendered-with-last", type=str2bool, default=True)
    parser.add_argument("--bridge-seconds", type=float, default=1.5)
    parser.add_argument("--run-t0-reconstruction-check", action="store_true")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=400)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument(
        "--draw-layout-overlay",
        type=str2bool,
        default=False,
        help="Debug-only: draw projected boxes onto prototype-rendered images. Keep false for submissions.",
    )
    return parser.parse_args()


def main() -> None:
    infer(parse_args())


if __name__ == "__main__":
    main()
