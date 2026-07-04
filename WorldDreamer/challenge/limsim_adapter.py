"""LimSim/TrafficManager adapter with a no-leakage kinematic fallback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping

from .camera_io import write_json
from .coordinate_transform import wrap_angle


def _step_state(state: Mapping[str, float], dt: float) -> Dict[str, float]:
    yaw = float(state.get("yaw", 0.0))
    v = max(0.0, float(state.get("v", 0.0)))
    a = float(state.get("a", 0.0))
    yaw_rate = float(state.get("yaw_rate", 0.0))
    x = float(state.get("x", 0.0))
    y = float(state.get("y", 0.0))
    nx = x + v * math.cos(yaw) * dt + 0.5 * a * math.cos(yaw) * dt * dt
    ny = y + v * math.sin(yaw) * dt + 0.5 * a * math.sin(yaw) * dt * dt
    nv = max(0.0, v + a * dt)
    return {
        **dict(state),
        "x": nx,
        "y": ny,
        "yaw": wrap_angle(yaw + yaw_rate * dt),
        "v": nv,
        "a": a,
        "yaw_rate": yaw_rate,
    }


def _roll_state(initial: Mapping[str, float], frame_id: int, fps: float) -> Dict[str, float]:
    state = dict(initial)
    dt = 1.0 / fps
    for _ in range(frame_id):
        state = _step_state(state, dt)
    return state


def run_fallback_kinematic_rollout(
    init_state: Mapping[str, object],
    future_seconds: float = 15.0,
    fps: float = 12.0,
) -> Dict[str, object]:
    num_frames = int(round(future_seconds * fps))
    frames: List[Dict[str, object]] = []
    for frame_id in range(num_frames):
        timestamp = frame_id / fps
        ego = _roll_state(init_state["ego"], frame_id, fps)
        agents = [_roll_state(agent, frame_id, fps) for agent in init_state.get("agents", [])]
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "ego": ego,
                "agents": agents,
            }
        )
    return {
        "fps": fps,
        "future_seconds": future_seconds,
        "num_frames": num_frames,
        "source": "fallback_kinematic",
        "fallback_reason": "LimSim direct Python adapter is unavailable in this prototype environment.",
        "frames": frames,
    }


def run_limsim_rollout(
    init_state: Mapping[str, object],
    future_seconds: float = 15.0,
    fps: float = 12.0,
    use_limsim: bool = True,
    fallback_kinematic_if_needed: bool = True,
) -> Dict[str, object]:
    if use_limsim:
        try:
            # The repository ships LimSim sources, but not a stable callable
            # Python API for nuScenes challenge states. Keep the adapter seam
            # explicit so a real TrafficManager rollout can replace fallback.
            import TrafficManager.LimSim  # noqa: F401

            raise NotImplementedError("TrafficManager LimSim API bridge is not implemented for this case format yet.")
        except Exception as exc:
            if not fallback_kinematic_if_needed:
                raise
            rollout = run_fallback_kinematic_rollout(init_state, future_seconds=future_seconds, fps=fps)
            rollout["requested_source"] = "limsim"
            rollout["fallback_reason"] = str(exc)
            return rollout
    return run_fallback_kinematic_rollout(init_state, future_seconds=future_seconds, fps=fps)


def write_rollout(case_dir: Path | str, rollout: Mapping[str, object], filename: str = "limsim_rollout_raw.json") -> Path:
    return write_json(rollout, Path(case_dir) / "sim_future" / filename)
