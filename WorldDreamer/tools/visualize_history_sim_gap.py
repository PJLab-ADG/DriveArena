#!/usr/bin/env python
"""Visualize history-to-rollout BEV gaps for a challenge case."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.camera_io import read_json, write_json


def _to_canvas(x: float, y: float, origin: Mapping[str, float], scale: float = 3.0) -> tuple[float, float]:
    return 300 + (x - origin["x"]) * scale, 300 - (y - origin["y"]) * scale


def _draw_actor(draw: ImageDraw.ImageDraw, state: Mapping[str, float], origin: Mapping[str, float], color: tuple[int, int, int], radius: int = 4) -> None:
    cx, cy = _to_canvas(float(state["x"]), float(state["y"]), origin)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    yaw = float(state.get("yaw", 0.0))
    draw.line((cx, cy, cx + math.cos(yaw) * radius * 3, cy - math.sin(yaw) * radius * 3), fill=color, width=2)


def _draw_frame(frame: Mapping[str, object], origin: Mapping[str, float], title: str, path: Path) -> None:
    img = Image.new("RGB", (600, 600), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.line((300, 0, 300, 600), fill=(210, 210, 210))
    draw.line((0, 300, 600, 300), fill=(210, 210, 210))
    _draw_actor(draw, frame["ego"], origin, (20, 80, 220), radius=6)
    for agent in frame.get("agents", []):
        _draw_actor(draw, agent, origin, (220, 80, 20), radius=3)
    draw.text((12, 12), title, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def visualize(args: argparse.Namespace) -> None:
    case_dir = Path(args.case_dir)
    init_state = read_json(case_dir / "sim_future" / "history_init_state.json")
    raw = read_json(case_dir / "sim_future" / "limsim_rollout_raw.json")
    bridged = read_json(case_dir / "sim_future" / "limsim_rollout_bridged.json")
    diagnostics = read_json(case_dir / "sim_future" / "diagnostics.json")
    diag_dir = case_dir / "diagnostics"
    origin = init_state["ego"]
    history_frame = {"ego": init_state["ego"], "agents": init_state.get("agents", [])}
    _draw_frame(history_frame, origin, "history last state", diag_dir / "bev_t0_history_vs_limsim_init.png")
    _draw_frame(raw["frames"][0], origin, "raw rollout t0", diag_dir / "bev_t0_to_t1_bridge.png")

    img = Image.new("RGB", (600, 600), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.line((300, 0, 300, 600), fill=(210, 210, 210))
    draw.line((0, 300, 600, 300), fill=(210, 210, 210))
    for frame in bridged["frames"][: int(round(3 * bridged.get("fps", 12)))]:
        _draw_actor(draw, frame["ego"], origin, (20, 80, 220), radius=3)
        for agent in frame.get("agents", [])[:80]:
            _draw_actor(draw, agent, origin, (220, 80, 20), radius=2)
    draw.text((12, 12), "bridged rollout first 3s", fill=(20, 20, 20))
    img.save(diag_dir / "bev_rollout_first_3s.png")
    write_json(diagnostics, diag_dir / "gap_metrics.json")
    print(f"Wrote gap visualizations to {diag_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    return parser.parse_args()


def main() -> None:
    visualize(parse_args())


if __name__ == "__main__":
    main()
