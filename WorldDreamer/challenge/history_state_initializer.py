"""Build LimSim initialization states from history-only challenge inputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .camera_io import read_json, write_json
from .coordinate_transform import unwrap_angles, wrap_angle


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    times = times - times[-1]
    try:
        return float(np.polyfit(times, values, deg=1)[0])
    except Exception:
        return float((values[-1] - values[0]) / max(times[-1] - times[0], 1e-3))


def _estimate_motion(samples: Sequence[Mapping[str, float]], fps: float, tail_seconds: float) -> Dict[str, float]:
    if not samples:
        raise ValueError("No samples available for motion estimation.")
    tail_count = max(3, int(round(tail_seconds * fps)))
    tail = list(samples)[-tail_count:]
    t0 = float(tail[0].get("timestamp", 0.0))
    times = np.asarray([float(item.get("timestamp", idx / fps)) - t0 for idx, item in enumerate(tail)], dtype=float)
    xs = np.asarray([float(item["x"]) for item in tail], dtype=float)
    ys = np.asarray([float(item["y"]) for item in tail], dtype=float)
    yaws = unwrap_angles([float(item.get("yaw", 0.0)) for item in tail])

    vx = _linear_slope(times, xs)
    vy = _linear_slope(times, ys)
    yaw_rate = _linear_slope(times, yaws)
    speed = math.hypot(vx, vy)

    if len(tail) >= 4:
        dt = np.diff(times)
        dt = np.maximum(dt, 1e-3)
        step_speeds = np.hypot(np.diff(xs) / dt, np.diff(ys) / dt)
        accel = _linear_slope(times[1:], step_speeds)
    else:
        accel = 0.0

    return {
        "vx": _clip(vx, -45.0, 45.0),
        "vy": _clip(vy, -45.0, 45.0),
        "v": _clip(speed, 0.0, 45.0),
        "a": _clip(accel, -8.0, 8.0),
        "yaw_rate": _clip(yaw_rate, -1.2, 1.2),
    }


def _normalize_ego_frames(history_ego: Mapping[str, object]) -> List[Dict[str, float]]:
    frames = []
    for frame in history_ego.get("frames", []):
        translation = frame.get("translation", [frame.get("x", 0.0), frame.get("y", 0.0), 0.0])
        frames.append(
            {
                "frame_id": int(frame["frame_id"]),
                "timestamp": float(frame.get("timestamp", frame["frame_id"])),
                "x": float(frame.get("x", translation[0])),
                "y": float(frame.get("y", translation[1])),
                "z": float(frame.get("z", translation[2] if len(translation) > 2 else 0.0)),
                "yaw": float(frame.get("yaw", 0.0)),
            }
        )
    return frames


def _normalize_agent_frame(frame: Mapping[str, object]) -> List[Dict[str, object]]:
    boxes = []
    for box in frame.get("boxes", []):
        center = box.get("global_center") or box.get("center") or [0.0, 0.0, 0.0]
        size = box.get("size") or [box.get("width", 1.9), box.get("length", 4.6), box.get("height", 1.6)]
        class_name = str(box.get("class_name", box.get("name", "car")))
        boxes.append(
            {
                "frame_id": int(frame["frame_id"]),
                "timestamp": float(frame.get("timestamp", frame["frame_id"])),
                "track_id": str(box.get("track_id", box.get("instance_token", f"box_{len(boxes):03d}"))),
                "class_name": class_name,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]) if len(center) > 2 else 0.0,
                "yaw": float(box.get("global_yaw", box.get("yaw", 0.0))),
                "length": float(box.get("length", size[1] if len(size) > 1 else 4.6)),
                "width": float(box.get("width", size[0] if len(size) > 0 else 1.9)),
                "height": float(box.get("height", size[2] if len(size) > 2 else 1.6)),
            }
        )
    return boxes


def estimate_initial_state(
    history_ego_states: Mapping[str, object],
    history_boxes: Mapping[str, object],
    fps: float = 12.0,
    tail_seconds: float = 2.0,
) -> Dict[str, object]:
    ego_frames = _normalize_ego_frames(history_ego_states)
    if not ego_frames:
        raise ValueError("history_ego_states.json has no frames.")

    last_ego = ego_frames[-1]
    ego_motion = _estimate_motion(ego_frames, fps=fps, tail_seconds=tail_seconds)
    ego_state = {
        **last_ego,
        **ego_motion,
        "class_name": "ego",
        "track_id": "ego",
    }

    tracks: Dict[str, List[Dict[str, object]]] = {}
    for frame in history_boxes.get("frames", []):
        for box in _normalize_agent_frame(frame):
            tracks.setdefault(str(box["track_id"]), []).append(box)

    last_frame_id = int(last_ego["frame_id"])
    agents = []
    for track_id, samples in sorted(tracks.items()):
        samples = sorted(samples, key=lambda item: (item["timestamp"], item["frame_id"]))
        last = samples[-1]
        if int(last["frame_id"]) < last_frame_id - max(2, int(round(0.5 * fps))):
            continue
        motion = _estimate_motion(samples, fps=fps, tail_seconds=tail_seconds)
        agents.append(
            {
                "track_id": track_id,
                "class_name": last.get("class_name", "car"),
                "x": float(last["x"]),
                "y": float(last["y"]),
                "z": float(last.get("z", 0.0)),
                "yaw": wrap_angle(float(last.get("yaw", 0.0))),
                "v": motion["v"],
                "vx": motion["vx"],
                "vy": motion["vy"],
                "a": motion["a"],
                "yaw_rate": motion["yaw_rate"],
                "length": float(last.get("length", 4.6)),
                "width": float(last.get("width", 1.9)),
                "height": float(last.get("height", 1.6)),
                "last_history_frame_id": int(last["frame_id"]),
            }
        )

    return {
        "source": "history_only",
        "fps": fps,
        "tail_seconds": tail_seconds,
        "history_last_frame_id": last_frame_id,
        "timestamp": float(last_ego.get("timestamp", 0.0)),
        "ego": ego_state,
        "agents": agents,
    }


def initialize_from_case(case_dir: Path | str, fps: float = 12.0, tail_seconds: float = 2.0) -> Dict[str, object]:
    case_dir = Path(case_dir)
    history_ego = read_json(case_dir / "input" / "history_ego_states.json")
    history_boxes = read_json(case_dir / "input" / "history_boxes.json")
    init_state = estimate_initial_state(history_ego, history_boxes, fps=fps, tail_seconds=tail_seconds)
    return init_state


def write_initial_state(case_dir: Path | str, init_state: Mapping[str, object]) -> Path:
    return write_json(init_state, Path(case_dir) / "sim_future" / "history_init_state.json")
