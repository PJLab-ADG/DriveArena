#!/usr/bin/env python
"""Check that challenge inference artifacts do not reference future GT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DREAMER_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, WORLD_DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge.leakage_guard import check_no_future_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_no_future_leakage(args.case_dir, pred_dir=args.pred_dir, num_frames=args.num_frames)
    print(f"No-leakage passed: {report['passed']}")
    print(f"Report: {Path(args.case_dir) / 'sim_future' / 'no_leakage_report.json'}")


if __name__ == "__main__":
    main()
