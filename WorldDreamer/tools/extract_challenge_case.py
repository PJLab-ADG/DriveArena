#!/usr/bin/env python
"""Extract one camera-first 20s challenge case from 12Hz nuScenes infos."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.camera_io import CAMERA_ORDER, copy_image_to_camera_first, ensure_camera_dirs, write_json
from challenge.coordinate_transform import (
    quaternion_to_rotation_matrix,
    transform_matrix,
    wrap_angle,
    yaw_from_quaternion,
)


def _load_pickle(path: Path) -> Mapping[str, object]:
    with path.open("rb") as f:
        return pickle.load(f)


def _default_info_pkl(nusc_root: Path, split: str) -> Path:
    base = nusc_root.parent
    candidates = [
        base / "nuscenes_mmdet3d-12Hz_description" / f"nuscenes_interp_12Hz_updated_description_{split}.pkl",
        base / "nuscenes_mmdet3d-12Hz_description" / f"nuscenes_interp_12Hz_updated_description_{split}.pickle",
        base / "nuscenes_mmdet3d-12Hz" / f"nuscenes_interp_12Hz_infos_{split}.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a 12Hz nuScenes info pkl near {nusc_root}. Pass --info-pkl.")


def _resolve_data_path(raw_path: str, nusc_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    parts = path.parts
    if "nuscenes" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("nuscenes")
        candidate = nusc_root.joinpath(*parts[idx + 1 :])
        if candidate.is_file():
            return candidate
    text = str(raw_path)
    marker = "data/nuscenes/"
    if marker in text:
        candidate = nusc_root / text.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        candidate = nusc_root.parent / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image path not found: {raw_path} remapped under {nusc_root}")


def _load_json_table(nusc_root: Path, version: str, name: str) -> List[Mapping[str, object]]:
    path = nusc_root / version / f"{name}.json"
    if not path.is_file() and version != "v1.0-trainval":
        path = nusc_root / "v1.0-trainval" / f"{name}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_interp_suffix(token: str) -> str:
    return token[:32] if len(token) >= 32 else token


def _scene_lookup(nusc_root: Path, version: str) -> Tuple[Dict[str, Mapping[str, object]], Dict[str, str]]:
    scenes = _load_json_table(nusc_root, version, "scene")
    samples = _load_json_table(nusc_root, "v1.0-trainval", "sample")
    scene_by_token = {str(scene["token"]): scene for scene in scenes}
    sample_to_scene = {str(sample["token"]): str(sample["scene_token"]) for sample in samples}
    return scene_by_token, sample_to_scene


def _build_info_index(infos: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(info["token"]): info for info in infos}


def _select_scene_tokens(
    data: Mapping[str, object],
    nusc_root: Path,
    version: str,
    scene_token: Optional[str],
    scene_name: Optional[str],
    sample_token: Optional[str],
    total_frames: int,
) -> Tuple[int, List[str], Dict[str, str]]:
    scene_tokens_list = data.get("scene_tokens", [])
    scene_by_token, sample_to_scene = _scene_lookup(nusc_root, version)
    name_to_scene = {str(scene.get("name")): token for token, scene in scene_by_token.items()}

    target_scene_token = scene_token or (name_to_scene.get(scene_name) if scene_name else None)
    for idx, tokens in enumerate(scene_tokens_list):
        if len(tokens) < total_frames:
            continue
        first_base = _strip_interp_suffix(str(tokens[0]))
        current_scene_token = sample_to_scene.get(first_base)
        if target_scene_token and current_scene_token != target_scene_token:
            continue
        if sample_token:
            sample_base = _strip_interp_suffix(sample_token)
            if sample_token not in tokens and sample_base not in [_strip_interp_suffix(str(t)) for t in tokens]:
                continue
        scene = scene_by_token.get(current_scene_token or "", {})
        return idx, list(tokens), {
            "source_scene_token": current_scene_token or target_scene_token or "unknown",
            "source_scene_name": str(scene.get("name", scene_name or f"scene_index_{idx:03d}")),
        }

    for idx, tokens in enumerate(scene_tokens_list):
        if len(tokens) >= total_frames:
            first_base = _strip_interp_suffix(str(tokens[0]))
            current_scene_token = sample_to_scene.get(first_base, "unknown")
            scene = scene_by_token.get(current_scene_token, {})
            return idx, list(tokens), {
                "source_scene_token": current_scene_token,
                "source_scene_name": str(scene.get("name", f"scene_index_{idx:03d}")),
            }
    raise RuntimeError(f"No scene has at least {total_frames} frames in info pkl.")


def _matrix_to_list(mat: np.ndarray) -> List[List[float]]:
    return [[float(v) for v in row] for row in mat.tolist()]


def _camera_params_for_info(info: Mapping[str, object], nusc_root: Path) -> Dict[str, object]:
    ego_mat = transform_matrix(info["ego2global_translation"], info["ego2global_rotation"])
    lidar2ego = transform_matrix(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    lidar2global = ego_mat @ lidar2ego
    cameras = {}
    for camera in CAMERA_ORDER:
        cam = info["cams"][camera]
        cam2ego = transform_matrix(cam["sensor2ego_translation"], cam["sensor2ego_rotation"])
        cam2lidar = transform_matrix(cam["sensor2lidar_translation"], _rotation_matrix_to_quaternion(np.asarray(cam["sensor2lidar_rotation"], dtype=float)))
        lidar2camera = np.linalg.inv(cam2lidar)
        path = _resolve_data_path(cam["data_path"], nusc_root)
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width, height = 1600, 900
        cameras[camera] = {
            "camera_intrinsic": cam["camera_intrinsics"],
            "sensor2ego_translation": cam["sensor2ego_translation"],
            "sensor2ego_rotation": cam["sensor2ego_rotation"],
            "camera2ego": _matrix_to_list(cam2ego),
            "ego2camera": _matrix_to_list(np.linalg.inv(cam2ego)),
            "camera2lidar": _matrix_to_list(cam2lidar),
            "lidar2camera": _matrix_to_list(lidar2camera),
            "lidar2image": _matrix_to_list(np.asarray(cam["camera_intrinsics"], dtype=float) @ lidar2camera[:3, :]),
            "sample_data_token": cam.get("sample_data_token"),
            "timestamp": float(cam.get("timestamp", info["timestamp"])),
            "data_path": str(path),
            "width": width,
            "height": height,
        }
    return {
        "frame_id": None,
        "timestamp": float(info["timestamp"]) / 1e6,
        "ego_pose": {
            "translation": info["ego2global_translation"],
            "rotation": info["ego2global_rotation"],
            "x": float(info["ego2global_translation"][0]),
            "y": float(info["ego2global_translation"][1]),
            "z": float(info["ego2global_translation"][2]),
            "yaw": yaw_from_quaternion(info["ego2global_rotation"]),
            "ego2global": _matrix_to_list(ego_mat),
            "lidar2ego": _matrix_to_list(lidar2ego),
            "lidar2global": _matrix_to_list(lidar2global),
        },
        "cameras": cameras,
    }


def _rotation_matrix_to_quaternion(rot: np.ndarray) -> List[float]:
    # Returns [w, x, y, z]. This is only used to build a homogeneous transform
    # from an existing sensor2lidar rotation matrix.
    m = np.asarray(rot, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _box_records(info: Mapping[str, object]) -> List[Dict[str, object]]:
    boxes = np.asarray(info.get("gt_boxes", []), dtype=float)
    names = list(info.get("gt_names", []))
    velocities = np.asarray(info.get("gt_velocity", np.zeros((len(boxes), 2))), dtype=float)
    obj_ids = list(info.get("obj_ids", [f"obj_{i:04d}" for i in range(len(boxes))]))
    ego_mat = transform_matrix(info["ego2global_translation"], info["ego2global_rotation"])
    lidar2ego = transform_matrix(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    lidar2global = ego_mat @ lidar2ego
    lidar_global_yaw = math.atan2(lidar2global[1, 0], lidar2global[0, 0])
    records = []
    for idx, box in enumerate(boxes):
        center_lidar = np.array([box[0], box[1], box[2], 1.0], dtype=float)
        center_global = lidar2global @ center_lidar
        width, length, height = float(box[3]), float(box[4]), float(box[5])
        yaw_lidar = float(box[6])
        records.append(
            {
                "track_id": str(obj_ids[idx]) if idx < len(obj_ids) else f"obj_{idx:04d}",
                "class_name": str(names[idx]) if idx < len(names) else "car",
                "center_lidar": [float(v) for v in box[:3]],
                "global_center": [float(center_global[0]), float(center_global[1]), float(center_global[2])],
                "yaw": yaw_lidar,
                "global_yaw": wrap_angle(yaw_lidar + lidar_global_yaw),
                "size": [width, length, height],
                "length": length,
                "width": width,
                "height": height,
                "velocity_lidar": [float(v) for v in velocities[idx].tolist()] if idx < len(velocities) else [0.0, 0.0],
            }
        )
    return records


def _ego_records(infos: Sequence[Mapping[str, object]], fps: float) -> List[Dict[str, object]]:
    records = []
    for frame_id, info in enumerate(infos):
        translation = info["ego2global_translation"]
        yaw = yaw_from_quaternion(info["ego2global_rotation"])
        records.append(
            {
                "frame_id": frame_id,
                "timestamp": frame_id / fps,
                "source_timestamp_us": float(info["timestamp"]),
                "translation": translation,
                "rotation": info["ego2global_rotation"],
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
                "yaw": yaw,
            }
        )
    for idx, item in enumerate(records):
        if idx == 0:
            nxt = records[min(1, len(records) - 1)]
            dt = max(nxt["timestamp"] - item["timestamp"], 1e-3)
            vx = (nxt["x"] - item["x"]) / dt
            vy = (nxt["y"] - item["y"]) / dt
        else:
            prev = records[idx - 1]
            dt = max(item["timestamp"] - prev["timestamp"], 1e-3)
            vx = (item["x"] - prev["x"]) / dt
            vy = (item["y"] - prev["y"]) / dt
        item["vx"] = float(vx)
        item["vy"] = float(vy)
        item["v"] = float(math.hypot(vx, vy))
    return records


def _write_map_placeholder(case_input_dir: Path, location: str, description: str) -> None:
    static_map = {
        "source": "nuScenes metadata",
        "location": location,
        "summary": {
            "description": description,
            "map_bev": "input/map_bev.png",
            "note": "Prototype stores a placeholder BEV map; WorldDreamer map-cache integration can replace this file.",
        },
        "map_bound": {"x": [-50.0, 50.0, 0.5], "y": [-50.0, 50.0, 0.5]},
    }
    write_json(static_map, case_input_dir / "static_map.json")
    img = Image.new("RGB", (200, 200), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    draw.line((100, 0, 100, 199), fill=(70, 70, 70), width=1)
    draw.line((0, 100, 199, 100), fill=(70, 70, 70), width=1)
    draw.ellipse((96, 96, 104, 104), fill=(0, 180, 255))
    img.save(case_input_dir / "map_bev.png")


def extract_case(args: argparse.Namespace) -> Path:
    nusc_root = Path(args.nusc_root).expanduser().resolve()
    info_pkl = Path(args.info_pkl).expanduser().resolve() if args.info_pkl else _default_info_pkl(nusc_root, args.split)
    data = _load_pickle(info_pkl)
    infos = data["infos"]
    info_index = _build_info_index(infos)
    num_history = int(round(args.history_seconds * args.fps))
    num_future = int(round(args.future_seconds * args.fps))
    total = num_history + num_future
    scene_index, scene_tokens, scene_meta = _select_scene_tokens(
        data,
        nusc_root,
        args.version,
        args.scene_token,
        args.scene_name,
        args.sample_token,
        total,
    )
    start_idx = args.start_frame_index
    if args.sample_token:
        for idx, token in enumerate(scene_tokens):
            if token == args.sample_token or _strip_interp_suffix(str(token)) == _strip_interp_suffix(args.sample_token):
                start_idx = idx
                break
    if start_idx + total > len(scene_tokens):
        raise RuntimeError(f"Requested {total} frames from start {start_idx}, but selected scene has {len(scene_tokens)} frames.")
    selected_tokens = scene_tokens[start_idx : start_idx + total]
    selected_infos = [info_index[token] for token in selected_tokens]

    case_dir = Path(args.out_dir).expanduser().resolve() / args.case_id
    input_dir = case_dir / "input"
    history_img_dir = input_dir / "history_images"
    gt_img_dir = case_dir / "gt_future_for_eval_only" / "images"
    ensure_camera_dirs(history_img_dir)
    ensure_camera_dirs(gt_img_dir)
    (case_dir / "sim_future" / "worlddreamer_conditions").mkdir(parents=True, exist_ok=True)
    (case_dir / "outputs" / "basedreamer_limsim_no_leakage").mkdir(parents=True, exist_ok=True)
    (case_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

    for frame_id, info in enumerate(selected_infos[:num_history]):
        for camera in CAMERA_ORDER:
            src = _resolve_data_path(info["cams"][camera]["data_path"], nusc_root)
            copy_image_to_camera_first(src, history_img_dir, camera, frame_id)
    for frame_id, info in enumerate(selected_infos[num_history:]):
        for camera in CAMERA_ORDER:
            src = _resolve_data_path(info["cams"][camera]["data_path"], nusc_root)
            copy_image_to_camera_first(src, gt_img_dir, camera, frame_id)

    history_ego = {
        "fps": args.fps,
        "num_frames": num_history,
        "frames": _ego_records(selected_infos[:num_history], args.fps),
    }
    gt_ego = {
        "fps": args.fps,
        "num_frames": num_future,
        "frames": _ego_records(selected_infos[num_history:], args.fps),
    }
    history_boxes = {
        "fps": args.fps,
        "num_frames": num_history,
        "frames": [
            {
                "frame_id": frame_id,
                "timestamp": frame_id / args.fps,
                "source_token": str(info["token"]),
                "boxes": _box_records(info),
            }
            for frame_id, info in enumerate(selected_infos[:num_history])
        ],
    }
    gt_boxes = {
        "fps": args.fps,
        "num_frames": num_future,
        "frames": [
            {
                "frame_id": frame_id,
                "timestamp": frame_id / args.fps,
                "source_token": str(info["token"]),
                "boxes": _box_records(info),
            }
            for frame_id, info in enumerate(selected_infos[num_history:])
        ],
    }
    camera_frames = []
    for frame_id, info in enumerate(selected_infos[:num_history]):
        params = _camera_params_for_info(info, nusc_root)
        params["frame_id"] = frame_id
        camera_frames.append(params)
    camera_params = {
        "camera_order": CAMERA_ORDER,
        "num_history_frames": num_history,
        "frames": camera_frames,
    }

    description = str(selected_infos[0].get("description", "A driving scene."))
    location = str(selected_infos[0].get("location", "unknown"))
    write_json(history_ego, input_dir / "history_ego_states.json")
    write_json(history_boxes, input_dir / "history_boxes.json")
    write_json(camera_params, input_dir / "camera_params.json")
    _write_map_placeholder(input_dir, location, description)
    (input_dir / "scene_description.txt").write_text(description, encoding="utf-8")
    write_json(gt_ego, case_dir / "gt_future_for_eval_only" / "ego_states.json")
    write_json(gt_boxes, case_dir / "gt_future_for_eval_only" / "boxes.json")
    write_json({"fps": args.fps, "frames": [], "note": "CAN bus GT is not copied into input and is reserved for eval only."}, case_dir / "gt_future_for_eval_only" / "can_bus.json")

    manifest = {
        "case_id": args.case_id,
        "dataset": "nuScenes",
        "source_scene_token": scene_meta["source_scene_token"],
        "source_scene_name": scene_meta["source_scene_name"],
        "source_scene_index": scene_index,
        "start_sample_token": str(selected_infos[0]["token"]),
        "history_seconds": args.history_seconds,
        "future_seconds": args.future_seconds,
        "fps": args.fps,
        "num_history_frames": num_history,
        "num_future_frames": num_future,
        "num_total_frames": total,
        "camera_order": CAMERA_ORDER,
        "image_layout": "camera_first",
        "history_image_pattern": "input/history_images/{camera}/{frame_id:06d}.jpg",
        "future_gt_image_pattern": "gt_future_for_eval_only/images/{camera}/{frame_id:06d}.jpg",
        "prediction_image_pattern": "outputs/basedreamer_limsim_no_leakage/{camera}/{frame_id:06d}.jpg",
        "inference_allowed_sources": [
            "input/history_images",
            "input/history_ego_states.json",
            "input/history_boxes.json",
            "input/camera_params.json",
            "input/static_map.json",
            "input/map_bev.png",
            "input/map_bev.npz",
            "input/scene_description.txt",
            "sim_future/limsim_rollout_bridged.json",
            "sim_future/worlddreamer_conditions",
        ],
        "inference_forbidden_sources": [
            "gt_future_for_eval_only/images",
            "gt_future_for_eval_only/ego_states.json",
            "gt_future_for_eval_only/boxes.json",
            "gt_future_for_eval_only/can_bus.json",
        ],
        "scene_description": description,
        "notes": "GT future is only for local evaluation and must not be accessed during inference.",
        "info_pkl": str(info_pkl),
    }
    write_json(manifest, input_dir / "manifest.json")
    return case_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nusc-root", required=True, help="nuScenes root, e.g. /path/Nuscenes-data/nuscenes")
    parser.add_argument("--info-pkl", default=None, help="Prepared 12Hz WorldDreamer info pkl.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Default info split to search.")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes metadata version for scene lookup.")
    parser.add_argument("--case-id", default="case_000001")
    parser.add_argument("--scene-token", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--start-frame-index", type=int, default=0)
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument("--future-seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out-dir", default="./challenge_data")
    return parser.parse_args()


def main() -> None:
    case_dir = extract_case(parse_args())
    print(f"Extracted challenge case to {case_dir}")


if __name__ == "__main__":
    main()
