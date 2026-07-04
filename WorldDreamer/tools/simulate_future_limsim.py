#!/usr/bin/env python
"""Generate no-leakage future rollout and WorldDreamer conditions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.history_state_initializer import initialize_from_case, write_initial_state
from challenge.limsim_adapter import run_limsim_rollout, write_rollout
from challenge.transition_bridge import apply_transition_bridge, write_bridged_rollout
from challenge.worlddreamer_condition_adapter import write_worlddreamer_conditions


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def simulate(args: argparse.Namespace) -> None:
    case_dir = Path(args.case_dir)
    init_state = initialize_from_case(case_dir, fps=args.fps, tail_seconds=args.history_tail_seconds)
    write_initial_state(case_dir, init_state)
    raw = run_limsim_rollout(
        init_state,
        future_seconds=args.future_seconds,
        fps=args.fps,
        use_limsim=args.use_limsim,
        fallback_kinematic_if_needed=args.fallback_kinematic_if_needed,
    )
    write_rollout(case_dir, raw, "limsim_rollout_raw.json")
    bridged, diagnostics = apply_transition_bridge(init_state, raw, bridge_seconds=args.bridge_seconds)
    write_bridged_rollout(case_dir, bridged, diagnostics)
    condition_paths = write_worlddreamer_conditions(case_dir, output_dir_name=args.output_dir_name)
    print(f"Wrote raw rollout: {case_dir / 'sim_future' / 'limsim_rollout_raw.json'}")
    print(f"Wrote bridged rollout: {case_dir / 'sim_future' / 'limsim_rollout_bridged.json'}")
    print(f"Wrote {len(condition_paths)} WorldDreamer conditions.")
    print(f"Bridge ego gap: {diagnostics['ego_position_gap_m']:.3f} m, {diagnostics['ego_yaw_gap_deg']:.3f} deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--future-seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--bridge-seconds", type=float, default=1.5)
    parser.add_argument("--history-tail-seconds", type=float, default=2.0)
    parser.add_argument("--use-limsim", type=str2bool, default=True)
    parser.add_argument("--fallback-kinematic-if-needed", type=str2bool, default=True)
    parser.add_argument("--output-dir-name", default="basedreamer_limsim_no_leakage")
    return parser.parse_args()


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
