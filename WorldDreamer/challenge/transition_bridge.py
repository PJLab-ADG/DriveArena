"""Bridge history states and future rollout states to remove t=0 gaps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .camera_io import write_json
from .coordinate_transform import wrap_angle


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def blend_angle(a: float, b: float, alpha: float) -> float:
    return wrap_angle(a + wrap_angle(b - a) * alpha)


def _extrapolate_state(state: Mapping[str, float], t: float) -> Dict[str, float]:
    yaw0 = float(state.get("yaw", 0.0))
    yaw_rate = float(state.get("yaw_rate", 0.0))
    v0 = max(0.0, float(state.get("v", 0.0)))
    a = float(state.get("a", 0.0))
    yaw = wrap_angle(yaw0 + yaw_rate * t)
    distance = v0 * t + 0.5 * a * t * t
    return {
        **dict(state),
        "x": float(state.get("x", 0.0)) + distance * math.cos(yaw0),
        "y": float(state.get("y", 0.0)) + distance * math.sin(yaw0),
        "yaw": yaw,
        "v": max(0.0, v0 + a * t),
    }


def extrapolate_from_history_tail(init_state: Mapping[str, object], t: float) -> Dict[str, object]:
    return {
        "ego": _extrapolate_state(init_state["ego"], t),
        "agents": [_extrapolate_state(agent, t) for agent in init_state.get("agents", [])],
    }


def _blend_state(history_state: Mapping[str, float], rollout_state: Mapping[str, float], alpha: float) -> Dict[str, float]:
    blended = dict(rollout_state)
    for key in ("x", "y", "v", "a", "yaw_rate"):
        blended[key] = float(history_state.get(key, 0.0)) * (1.0 - alpha) + float(rollout_state.get(key, 0.0)) * alpha
    blended["yaw"] = blend_angle(float(history_state.get("yaw", 0.0)), float(rollout_state.get("yaw", 0.0)), alpha)
    return blended


def _state_gap(a: Mapping[str, float], b: Mapping[str, float]) -> Tuple[float, float, float]:
    pos = math.hypot(float(a.get("x", 0.0)) - float(b.get("x", 0.0)), float(a.get("y", 0.0)) - float(b.get("y", 0.0)))
    yaw = abs(wrap_angle(float(a.get("yaw", 0.0)) - float(b.get("yaw", 0.0)))) * 180.0 / math.pi
    speed = abs(float(a.get("v", 0.0)) - float(b.get("v", 0.0)))
    return pos, yaw, speed


def _limit_step(prev: Mapping[str, float], curr: Mapping[str, float], dt: float, max_speed: float = 45.0, max_yaw_rate: float = 1.2) -> Dict[str, float]:
    out = dict(curr)
    dx = float(curr.get("x", 0.0)) - float(prev.get("x", 0.0))
    dy = float(curr.get("y", 0.0)) - float(prev.get("y", 0.0))
    max_dist = max_speed * dt
    dist = math.hypot(dx, dy)
    if dist > max_dist and dist > 1e-6:
        scale = max_dist / dist
        out["x"] = float(prev.get("x", 0.0)) + dx * scale
        out["y"] = float(prev.get("y", 0.0)) + dy * scale
    dyaw = wrap_angle(float(curr.get("yaw", 0.0)) - float(prev.get("yaw", 0.0)))
    max_dyaw = max_yaw_rate * dt
    if abs(dyaw) > max_dyaw:
        out["yaw"] = wrap_angle(float(prev.get("yaw", 0.0)) + math.copysign(max_dyaw, dyaw))
    return out


def apply_transition_bridge(
    init_state: Mapping[str, object],
    raw_rollout: Mapping[str, object],
    bridge_seconds: float = 1.5,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    fps = float(raw_rollout.get("fps", init_state.get("fps", 12.0)))
    dt = 1.0 / fps
    init_agents = {str(agent["track_id"]): agent for agent in init_state.get("agents", [])}
    prev_ego = None
    prev_agents: Dict[str, Mapping[str, float]] = {}
    bridged_frames: List[Dict[str, object]] = []

    raw_first = raw_rollout["frames"][0]
    ego_pos_gap, ego_yaw_gap, ego_speed_gap = _state_gap(init_state["ego"], raw_first["ego"])
    agent_gaps = []

    for frame in raw_rollout.get("frames", []):
        frame_id = int(frame["frame_id"])
        t = frame_id / fps
        alpha = smoothstep(t / max(bridge_seconds, 1e-6))
        hist_extra = extrapolate_from_history_tail(init_state, t)
        ego = _blend_state(hist_extra["ego"], frame["ego"], alpha)
        if frame_id == 0:
            ego.update({k: init_state["ego"][k] for k in ("x", "y", "yaw", "v", "a", "yaw_rate") if k in init_state["ego"]})
        if prev_ego is not None:
            ego = _limit_step(prev_ego, ego, dt)
        prev_ego = ego

        raw_agents = {str(agent["track_id"]): agent for agent in frame.get("agents", [])}
        hist_agents = {str(agent["track_id"]): agent for agent in hist_extra.get("agents", [])}
        all_ids = sorted(set(raw_agents) | set(hist_agents) | set(prev_agents))
        agents = []
        for track_id in all_ids:
            if track_id in raw_agents and track_id in hist_agents:
                agent = _blend_state(hist_agents[track_id], raw_agents[track_id], alpha)
                if frame_id == 0 and track_id in init_agents:
                    agent.update({k: init_agents[track_id][k] for k in ("x", "y", "yaw", "v", "a", "yaw_rate") if k in init_agents[track_id]})
                if frame_id == 0:
                    gap = _state_gap(init_agents.get(track_id, hist_agents[track_id]), raw_agents[track_id])[0]
                    agent_gaps.append(gap)
            elif track_id in hist_agents and t <= bridge_seconds:
                agent = dict(hist_agents[track_id])
                agent["existence_score"] = 1.0 - alpha
            elif track_id in raw_agents:
                agent = dict(raw_agents[track_id])
                agent["existence_score"] = alpha
            elif track_id in prev_agents and t <= bridge_seconds + 0.5:
                agent = _extrapolate_state(prev_agents[track_id], dt)
                agent["existence_score"] = max(0.0, float(prev_agents[track_id].get("existence_score", 1.0)) - dt / 0.5)
            else:
                continue
            if track_id in prev_agents:
                limited = _limit_step(prev_agents[track_id], agent, dt)
                for key in ("x", "y", "v"):
                    limited[key] = 0.65 * float(agent.get(key, 0.0)) + 0.35 * float(limited.get(key, 0.0))
                limited["yaw"] = blend_angle(float(prev_agents[track_id].get("yaw", 0.0)), float(agent.get("yaw", 0.0)), 0.65)
                agent = limited
            agents.append(agent)
        prev_agents = {str(agent["track_id"]): agent for agent in agents}
        bridged_frames.append({"frame_id": frame_id, "timestamp": frame["timestamp"], "ego": ego, "agents": agents})

    diagnostics = {
        "bridge_seconds": bridge_seconds,
        "ego_position_gap_m": ego_pos_gap,
        "ego_yaw_gap_deg": ego_yaw_gap,
        "ego_speed_gap_mps": ego_speed_gap,
        "num_history_agents": len(init_agents),
        "num_limsim_agents": len(raw_first.get("agents", [])),
        "num_bridged_agents": len(bridged_frames[0].get("agents", [])) if bridged_frames else 0,
        "mean_agent_position_gap_m": float(np.mean(agent_gaps)) if agent_gaps else 0.0,
        "max_agent_position_gap_m": float(np.max(agent_gaps)) if agent_gaps else 0.0,
        "agent_id_match_rate": len(set(init_agents) & {str(a["track_id"]) for a in raw_first.get("agents", [])}) / max(1, len(init_agents)),
    }
    bridged = {
        **dict(raw_rollout),
        "source": f"{raw_rollout.get('source', 'rollout')}_bridged",
        "bridge_seconds": bridge_seconds,
        "frames": bridged_frames,
    }
    return bridged, diagnostics


def write_bridged_rollout(case_dir: Path | str, bridged: Mapping[str, object], diagnostics: Mapping[str, object]) -> None:
    case_dir = Path(case_dir)
    write_json(bridged, case_dir / "sim_future" / "limsim_rollout_bridged.json")
    write_json(diagnostics, case_dir / "sim_future" / "diagnostics.json")
    write_json(diagnostics, case_dir / "diagnostics" / "gap_metrics.json")
