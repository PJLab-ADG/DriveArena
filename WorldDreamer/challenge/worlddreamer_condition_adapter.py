"""Convert bridged future rollout into per-frame WorldDreamer conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

from .camera_io import CAMERA_ORDER, camera_frame_path, read_json, write_json
from .coordinate_transform import (
    ego_to_global,
    project_box_to_camera_layout,
    transform_box_2d_to_3d,
    wrap_angle,
)


DEFAULT_CLASS_DIMS = {
    "car": (4.6, 1.9, 1.6),
    "truck": (7.0, 2.5, 3.0),
    "bus": (10.0, 2.6, 3.2),
    "pedestrian": (0.8, 0.8, 1.7),
    "bicycle": (1.8, 0.6, 1.5),
    "motorcycle": (2.2, 0.8, 1.5),
}


def _with_default_dims(agent: Mapping[str, object]) -> Dict[str, object]:
    class_name = str(agent.get("class_name", "car"))
    length, width, height = DEFAULT_CLASS_DIMS.get(class_name, DEFAULT_CLASS_DIMS["car"])
    out = dict(agent)
    out["length"] = float(out.get("length", length))
    out["width"] = float(out.get("width", width))
    out["height"] = float(out.get("height", height))
    return out


def _history_reference_paths(case_dir: Path, frame_id: int = 59) -> Dict[str, str]:
    root = case_dir / "input" / "history_images"
    return {camera: str(camera_frame_path(root, camera, frame_id)) for camera in CAMERA_ORDER}


def _prediction_reference_paths(case_dir: Path, output_dir_name: str, frame_id: int) -> Dict[str, str]:
    root = case_dir / "outputs" / output_dir_name
    return {camera: str(camera_frame_path(root, camera, frame_id)) for camera in CAMERA_ORDER}


def build_worlddreamer_conditions(
    case_dir: Path | str,
    output_dir_name: str = "basedreamer_limsim_no_leakage",
    num_frames: int | None = None,
) -> List[Dict[str, object]]:
    case_dir = Path(case_dir)
    manifest = read_json(case_dir / "input" / "manifest.json")
    camera_params_all = read_json(case_dir / "input" / "camera_params.json")
    static_map = read_json(case_dir / "input" / "static_map.json")
    rollout = read_json(case_dir / "sim_future" / "limsim_rollout_bridged.json")
    scene_description_path = case_dir / "input" / "scene_description.txt"
    scene_description = scene_description_path.read_text(encoding="utf-8").strip() if scene_description_path.exists() else manifest.get("scene_description", "")
    frames = rollout["frames"][:num_frames]

    # The cameras are rigidly mounted; future conditions use the history-last
    # calibration and future ego pose from the no-leakage rollout.
    last_camera_frame = camera_params_all["frames"][-1]
    camera_params = {camera: last_camera_frame["cameras"][camera] for camera in CAMERA_ORDER}
    history_last_ego = camera_params_all["frames"][-1].get("ego_pose", {})
    previous_ego = history_last_ego

    conditions: List[Dict[str, object]] = []
    for frame in frames:
        frame_id = int(frame["frame_id"])
        ego = frame["ego"]
        boxes_3d = []
        for agent in frame.get("agents", []):
            agent = _with_default_dims(agent)
            if float(agent.get("existence_score", 1.0)) <= 0.05:
                continue
            boxes_3d.append(transform_box_2d_to_3d(agent, ego))
        layout = project_box_to_camera_layout(boxes_3d, camera_params, image_size=(1600, 900))

        curr_ego_pose = {
            "x": ego["x"],
            "y": ego["y"],
            "z": ego.get("z", 0.0),
            "yaw": ego["yaw"],
        }
        rel_dx = float(ego["x"]) - float(previous_ego.get("x", previous_ego.get("translation", [ego["x"], ego["y"]])[0]))
        rel_dy = float(ego["y"]) - float(previous_ego.get("y", previous_ego.get("translation", [ego["x"], ego["y"]])[1]))
        rel_pose = {
            "dx": rel_dx,
            "dy": rel_dy,
            "dyaw": wrap_angle(float(ego["yaw"]) - float(previous_ego.get("yaw", ego["yaw"]))),
            "matrix": [
                [1.0, 0.0, 0.0, rel_dx],
                [0.0, 1.0, 0.0, rel_dy],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        if frame_id == 0:
            reference_images = _history_reference_paths(case_dir, int(manifest["num_history_frames"]) - 1)
            reference_source = "input/history_images"
        else:
            reference_images = _prediction_reference_paths(case_dir, output_dir_name, frame_id - 1)
            reference_source = "outputs/previous_prediction"

        condition = {
            "frame_id": frame_id,
            "timestamp": frame["timestamp"],
            "ego_pose": curr_ego_pose,
            "camera_params": camera_params,
            "relative_pose": rel_pose,
            "bev_map_condition": {
                "static_map_path": "input/static_map.json",
                "map_bev_png": "input/map_bev.png",
                "summary": static_map.get("summary", {}),
            },
            "boxes_3d": boxes_3d,
            "camera_projected_layout": layout,
            "text_prompt": scene_description,
            "reference_image_path": reference_images.get("CAM_FRONT"),
            "reference_images": reference_images,
            "reference_source": reference_source,
            "condition_source": "sim_future/limsim_rollout_bridged.json",
            "no_future_gt_used": True,
        }
        conditions.append(condition)
        previous_ego = curr_ego_pose
    return conditions


def write_worlddreamer_conditions(
    case_dir: Path | str,
    output_dir_name: str = "basedreamer_limsim_no_leakage",
    num_frames: int | None = None,
) -> List[Path]:
    condition_dir = Path(case_dir) / "sim_future" / "worlddreamer_conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for condition in build_worlddreamer_conditions(case_dir, output_dir_name=output_dir_name, num_frames=num_frames):
        path = condition_dir / f"{int(condition['frame_id']):06d}.json"
        write_json(condition, path)
        paths.append(path)
    return paths
