"""Image and experiment result I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SUMMARY_FIELDS = [
    "image",
    "source",
    "method",
    "output_width",
    "output_height",
    "target_size_source",
    "aspect_error_percent",
    "transfer_rmse_vs_truth_px",
    "transfer_max_vs_truth_px",
    "target_mismatch_rmse_px",
    "matrix_error_vs_truth",
    "fit_residual_rmse_px",
    "fit_residual_max_px",
    "solve_ms",
    "solve_ms_std",
    "warp_ms",
    "warp_ms_std",
    "image_mae_vs_opencv",
]


def require_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("image I/O requires OpenCV; install dependencies with: python -m pip install -r requirements.txt") from exc
    return cv2


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2 = require_opencv()
    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV could not write image: {path}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ground_truth(image_path: Path) -> dict | None:
    """Load `<name>.groundtruth.json` sitting next to an image, if it exists.

    Returns ``None`` when there is no sidecar, so callers can treat ground truth
    as optional.  Raises on a malformed file, because silently ignoring a broken
    ground truth would hide a real problem.
    """

    sidecar = image_path.with_suffix(".groundtruth.json")
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    required = ("homography_source_to_destination", "destination_corners_tl_tr_br_bl")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{sidecar} is missing required key(s): {', '.join(missing)}")
    return payload


def save_summary(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to CSV, using the union of all keys as the header.

    Note this *overwrites* the file rather than appending.  Mixing a single-image
    run with a later batch run therefore discards the earlier rows -- copy the
    file aside first if you need to keep them.
    """

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object, digits: int = 4) -> str:
    """Format a number for console tables; pass through empty/None placeholders."""

    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if value != 0.0 and abs(value) < 1e-4:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)
