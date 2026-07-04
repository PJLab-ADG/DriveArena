"""Camera-first image IO utilities for the DriveArena challenge prototype."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from PIL import Image


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def frame_name(frame_id: int, ext: str = ".jpg") -> str:
    return f"{frame_id:06d}{ext}"


def ensure_camera_dirs(root: Path | str, cameras: Sequence[str] = CAMERA_ORDER) -> None:
    root = Path(root)
    for camera in cameras:
        (root / camera).mkdir(parents=True, exist_ok=True)


def camera_frame_path(root: Path | str, camera: str, frame_id: int, ext: str = ".jpg") -> Path:
    return Path(root) / camera / frame_name(frame_id, ext)


def save_image(image: Image.Image, root: Path | str, camera: str, frame_id: int, ext: str = ".jpg") -> Path:
    ensure_camera_dirs(root, [camera])
    path = camera_frame_path(root, camera, frame_id, ext)
    image.convert("RGB").save(path, quality=95)
    return path


def copy_image_to_camera_first(
    src: Path | str,
    dst_root: Path | str,
    camera: str,
    frame_id: int,
    ext: str = ".jpg",
) -> Path:
    ensure_camera_dirs(dst_root, [camera])
    dst = camera_frame_path(dst_root, camera, frame_id, ext)
    src = Path(src)
    try:
        shutil.copy2(src, dst)
    except Exception:
        with Image.open(src) as image:
            image.convert("RGB").save(dst, quality=95)
    return dst


def save_camera_first_images(
    images_by_camera: Mapping[str, Image.Image | Path | str],
    dst_root: Path | str,
    frame_id: int,
    cameras: Sequence[str] = CAMERA_ORDER,
    ext: str = ".jpg",
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    ensure_camera_dirs(dst_root, cameras)
    for camera in cameras:
        image_or_path = images_by_camera[camera]
        if isinstance(image_or_path, Image.Image):
            out = save_image(image_or_path, dst_root, camera, frame_id, ext)
        else:
            out = copy_image_to_camera_first(image_or_path, dst_root, camera, frame_id, ext)
        paths[camera] = str(out)
    return paths


def load_camera_first_images(
    root: Path | str,
    frame_id: int,
    cameras: Sequence[str] = CAMERA_ORDER,
    ext: str = ".jpg",
) -> Dict[str, Image.Image]:
    root = Path(root)
    images = {}
    for camera in cameras:
        path = camera_frame_path(root, camera, frame_id, ext)
        images[camera] = Image.open(path).convert("RGB")
    return images


def get_frame_paths(
    root: Path | str,
    num_frames: int,
    cameras: Sequence[str] = CAMERA_ORDER,
    ext: str = ".jpg",
) -> Dict[str, List[Path]]:
    root = Path(root)
    return {
        camera: [camera_frame_path(root, camera, frame_id, ext) for frame_id in range(num_frames)]
        for camera in cameras
    }


def get_history_frame_paths(case_dir: Path | str, num_frames: int = 60) -> Dict[str, List[Path]]:
    return get_frame_paths(Path(case_dir) / "input" / "history_images", num_frames)


def get_prediction_frame_paths(
    case_dir: Path | str,
    output_dir_name: str = "basedreamer_limsim_no_leakage",
    num_frames: int = 180,
) -> Dict[str, List[Path]]:
    return get_frame_paths(Path(case_dir) / "outputs" / output_dir_name, num_frames)


def validate_camera_first_structure(
    root: Path | str,
    num_frames: int,
    cameras: Sequence[str] = CAMERA_ORDER,
    ext: str = ".jpg",
    require_valid_images: bool = False,
) -> Dict[str, object]:
    root = Path(root)
    missing: List[str] = []
    invalid: List[str] = []
    for camera in cameras:
        camera_dir = root / camera
        if not camera_dir.is_dir():
            missing.append(str(camera_dir))
            continue
        for frame_id in range(num_frames):
            path = camera_frame_path(root, camera, frame_id, ext)
            if not path.is_file():
                missing.append(str(path))
                continue
            if require_valid_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    invalid.append(str(path))
    return {
        "root": str(root),
        "layout": "camera_first",
        "num_frames": num_frames,
        "cameras": list(cameras),
        "missing": missing,
        "invalid": invalid,
        "passed": not missing and not invalid,
    }


def read_json(path: Path | str) -> MutableMapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(data: object) -> object:
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, np.generic):
            return data.item()
    except Exception:
        pass
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, Mapping):
        return {str(k): _json_safe(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(v) for v in data]
    return data


def write_json(data: object, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def list_condition_files(condition_dir: Path | str, num_frames: Optional[int] = None) -> List[Path]:
    condition_dir = Path(condition_dir)
    if num_frames is None:
        return sorted(condition_dir.glob("*.json"))
    return [condition_dir / f"{frame_id:06d}.json" for frame_id in range(num_frames)]


def make_grid(
    images: Sequence[Image.Image],
    cols: int = 3,
    pad: int = 4,
    bg: tuple[int, int, int] = (18, 18, 18),
) -> Image.Image:
    if not images:
        raise ValueError("No images to make a grid.")
    widths, heights = zip(*(img.size for img in images))
    cell_w, cell_h = max(widths), max(heights)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w + (cols - 1) * pad, rows * cell_h + (rows - 1) * pad), bg)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        x = col * (cell_w + pad)
        y = row * (cell_h + pad)
        grid.paste(image.convert("RGB"), (x, y))
    return grid


def make_6view_grid(images_by_camera: Mapping[str, Image.Image], cameras: Sequence[str] = CAMERA_ORDER) -> Image.Image:
    return make_grid([images_by_camera[camera] for camera in cameras], cols=3)
