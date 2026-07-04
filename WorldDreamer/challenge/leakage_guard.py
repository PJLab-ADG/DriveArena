"""No-future-GT leakage checks for challenge inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .camera_io import CAMERA_ORDER, list_condition_files, read_json, write_json


FORBIDDEN_SOURCE_PATTERNS = [
    "gt_future_for_eval_only/images",
    "gt_future_for_eval_only/ego_states.json",
    "gt_future_for_eval_only/boxes.json",
    "gt_future_for_eval_only/can_bus.json",
    "gt_future/images",
    "gt_future/boxes",
    "gt_future/ego_states",
]


def _norm(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def is_forbidden_path(path: str | Path, forbidden_patterns: Sequence[str] = FORBIDDEN_SOURCE_PATTERNS) -> bool:
    text = _norm(path)
    return any(pattern in text for pattern in forbidden_patterns)


def assert_not_forbidden(path: str | Path) -> None:
    if is_forbidden_path(path):
        raise RuntimeError(f"Future GT leakage risk: forbidden path accessed during inference: {path}")


def guard_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        assert_not_forbidden(path)


def _scan_condition(condition: Mapping[str, object]) -> List[str]:
    problems: List[str] = []
    condition_source = str(condition.get("condition_source", ""))
    if "gt_future" in condition_source:
        problems.append(f"condition_source uses GT future: {condition_source}")
    refs = []
    if "reference_image_path" in condition:
        refs.append(condition["reference_image_path"])
    if isinstance(condition.get("reference_images"), Mapping):
        refs.extend(condition["reference_images"].values())
    for ref in refs:
        if is_forbidden_path(str(ref)):
            problems.append(f"reference image uses GT future: {ref}")
    text = str(condition)
    if "gt_future_for_eval_only" in text and "reference" not in text:
        problems.append("condition contains gt_future_for_eval_only text")
    if condition.get("no_future_gt_used") is not True:
        problems.append("condition no_future_gt_used is not true")
    return problems


def check_no_future_leakage(
    case_dir: Path | str,
    pred_dir: Path | str | None = None,
    num_frames: int | None = None,
    raise_on_fail: bool = True,
) -> Mapping[str, object]:
    case_dir = Path(case_dir)
    manifest = read_json(case_dir / "input" / "manifest.json")
    num_frames = int(num_frames or manifest.get("num_future_frames", 180))
    condition_dir = case_dir / "sim_future" / "worlddreamer_conditions"
    condition_files = list_condition_files(condition_dir, num_frames)

    missing_conditions = [str(path) for path in condition_files if not path.is_file()]
    condition_problems: List[str] = []
    reference_problems: List[str] = []
    for path in condition_files:
        if not path.is_file():
            continue
        if is_forbidden_path(path):
            condition_problems.append(f"condition file path forbidden: {path}")
        condition = read_json(path)
        problems = _scan_condition(condition)
        condition_problems.extend(problems)
        for problem in problems:
            if "reference" in problem:
                reference_problems.append(problem)

    pred_forbidden = bool(pred_dir and is_forbidden_path(pred_dir))
    report = {
        "case_id": manifest.get("case_id", case_dir.name),
        "no_future_image_access": not pred_forbidden and not any("images" in p for p in condition_problems),
        "no_future_box_access": not any("box" in p.lower() and "gt_future" in p for p in condition_problems),
        "no_future_ego_access": not any("ego" in p.lower() and "gt_future" in p for p in condition_problems),
        "no_future_can_bus_access": not any("can_bus" in p for p in condition_problems),
        "conditions_from_limsim": not condition_problems and not missing_conditions,
        "references_from_history_or_prediction_only": not reference_problems,
        "missing_conditions": missing_conditions,
        "problems": condition_problems + (["prediction dir is forbidden"] if pred_forbidden else []),
    }
    report["passed"] = all(
        bool(report[key])
        for key in [
            "no_future_image_access",
            "no_future_box_access",
            "no_future_ego_access",
            "no_future_can_bus_access",
            "conditions_from_limsim",
            "references_from_history_or_prediction_only",
        ]
    )
    write_json(report, case_dir / "sim_future" / "no_leakage_report.json")
    if raise_on_fail and not report["passed"]:
        raise RuntimeError(f"No-leakage check failed: {report['problems']}")
    return report
