"""Coordinate transforms for nuScenes, LimSim, WorldDreamer, and cameras.

Yaw angles are radians. In nuScenes/global and this prototype's ego frame,
positive yaw rotates counter-clockwise around +z. WorldDreamer conditions here
use an ego-aligned lidar-like frame: +x forward, +y left, +z up.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_angles(angles: Sequence[float]) -> np.ndarray:
    return np.unwrap(np.asarray(angles, dtype=float))


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a nuScenes quaternion [w, x, y, z]."""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def yaw_from_quaternion(q: Sequence[float]) -> float:
    rot = quaternion_to_rotation_matrix(q)
    return math.atan2(rot[1, 0], rot[0, 0])


def transform_matrix(translation: Sequence[float], rotation_q: Sequence[float], inverse: bool = False) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = quaternion_to_rotation_matrix(rotation_q)
    mat[:3, 3] = np.asarray(translation, dtype=float)[:3]
    if inverse:
        return np.linalg.inv(mat)
    return mat


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def global_to_ego(point_xy: Sequence[float], ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Transform a global xy point into the ego frame."""
    delta = np.asarray(point_xy, dtype=float)[:2] - np.asarray(ego_xy, dtype=float)[:2]
    return yaw_matrix(-ego_yaw) @ delta


def ego_to_global(point_xy: Sequence[float], ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Transform an ego-frame xy point into the global frame."""
    return yaw_matrix(ego_yaw) @ np.asarray(point_xy, dtype=float)[:2] + np.asarray(ego_xy, dtype=float)[:2]


def nusc_to_limsim(state: Mapping[str, float]) -> Dict[str, float]:
    """Convert a nuScenes/global state to the LimSim adapter state.

    The prototype keeps the same global xy/yaw convention for LimSim, which
    makes bridge diagnostics easier and avoids a hidden frame flip.
    """
    return dict(state)


def limsim_to_nusc(state: Mapping[str, float]) -> Dict[str, float]:
    """Convert the LimSim adapter state back to nuScenes/global convention."""
    return dict(state)


def limsim_to_worlddreamer(
    state: Mapping[str, float],
    ego_state: Mapping[str, float],
) -> Dict[str, float]:
    """Convert a global LimSim state into a WorldDreamer ego-frame state."""
    xy = global_to_ego((state["x"], state["y"]), (ego_state["x"], ego_state["y"]), ego_state["yaw"])
    return {
        **dict(state),
        "x": float(xy[0]),
        "y": float(xy[1]),
        "yaw": wrap_angle(float(state.get("yaw", 0.0)) - float(ego_state.get("yaw", 0.0))),
        "frame": "worlddreamer_ego",
    }


def transform_box_2d_to_3d(
    state: Mapping[str, float],
    ego_state: Mapping[str, float],
    default_z: float = 0.0,
) -> Dict[str, object]:
    """Create an ego-frame 3D box from a 2D actor state.

    Input state is global/LimSim xy/yaw. Output box is centered in the
    WorldDreamer ego frame with dimensions length, width, height.
    """
    wd = limsim_to_worlddreamer(state, ego_state)
    length = float(state.get("length", 4.6))
    width = float(state.get("width", 1.9))
    height = float(state.get("height", 1.6))
    return {
        "track_id": state.get("track_id", "unknown"),
        "class_name": state.get("class_name", "car"),
        "center": [float(wd["x"]), float(wd["y"]), default_z + height / 2.0],
        "size": [length, width, height],
        "yaw": float(wd["yaw"]),
        "frame": "worlddreamer_ego",
    }


def box_corners_3d(center: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    """Return 8 corners for a box in an ego/lidar-like frame."""
    length, width, height = [float(v) for v in size]
    x = length / 2.0
    y = width / 2.0
    z = height / 2.0
    corners = np.array(
        [
            [x, y, z],
            [x, -y, z],
            [-x, -y, z],
            [-x, y, z],
            [x, y, -z],
            [x, -y, -z],
            [-x, -y, -z],
            [-x, y, -z],
        ],
        dtype=float,
    )
    rot = np.eye(3)
    rot[:2, :2] = yaw_matrix(yaw)
    return corners @ rot.T + np.asarray(center, dtype=float)[:3]


def _camera2ego_matrix(camera_param: Mapping[str, object]) -> np.ndarray:
    if "camera2ego" in camera_param:
        return np.asarray(camera_param["camera2ego"], dtype=float)
    return transform_matrix(camera_param["sensor2ego_translation"], camera_param["sensor2ego_rotation"])


def project_box_to_camera(
    box: Mapping[str, object],
    camera_param: Mapping[str, object],
    image_size: Sequence[int] | None = None,
) -> Dict[str, object]:
    """Project an ego-frame 3D box to one camera image.

    Camera extrinsics are stored as camera-to-ego transforms, so projection uses
    ego_to_camera = inverse(camera2ego). This function never reads future GT.
    """
    corners_ego = box_corners_3d(box["center"], box["size"], float(box["yaw"]))
    camera2ego = _camera2ego_matrix(camera_param)
    ego2camera = np.linalg.inv(camera2ego)
    corners_h = np.concatenate([corners_ego, np.ones((len(corners_ego), 1))], axis=1)
    corners_cam = (ego2camera @ corners_h.T).T[:, :3]
    valid_z = corners_cam[:, 2] > 1e-3
    intrinsic = np.asarray(camera_param["camera_intrinsic"], dtype=float)
    uvw = (intrinsic @ corners_cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    if image_size is None:
        width = camera_param.get("width")
        height = camera_param.get("height")
    else:
        width, height = image_size[0], image_size[1]
    visible = bool(np.any(valid_z))
    bbox_2d = None
    if visible:
        front_uv = uv[valid_z]
        x1, y1 = np.min(front_uv, axis=0)
        x2, y2 = np.max(front_uv, axis=0)
        if width is not None and height is not None:
            visible = x2 >= 0 and y2 >= 0 and x1 <= float(width) and y1 <= float(height)
            x1, x2 = np.clip([x1, x2], 0, float(width))
            y1, y2 = np.clip([y1, y2], 0, float(height))
        bbox_2d = [float(x1), float(y1), float(x2), float(y2)]
    return {
        "track_id": box.get("track_id"),
        "class_name": box.get("class_name"),
        "visible": visible,
        "bbox_2d": bbox_2d,
        "corners_2d": uv.tolist(),
        "corners_camera_z": corners_cam[:, 2].tolist(),
    }


def project_box_to_camera_layout(
    boxes: Iterable[Mapping[str, object]],
    camera_params: Mapping[str, Mapping[str, object]],
    image_size: Sequence[int] | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    layout: Dict[str, List[Dict[str, object]]] = {}
    for camera, params in camera_params.items():
        layout[camera] = [project_box_to_camera(box, params, image_size=image_size) for box in boxes]
    return layout
