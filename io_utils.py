"""Image and experiment result I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Identity of a summary row: the same image, from the same source, solved by the same
# method for the same target rectangle.  ``source`` has to be part of the key because
# ``image`` is only a file name -- data/public and data/phone can both contain a
# ``fig1.jpg``, and without the source the later run would silently replace the
# earlier one.  Two runs at different target sizes are different rows, which is what
# keeps both variants of the aspect-ratio experiment on record.
SUMMARY_KEY_FIELDS = (
    "source", "image", "method", "target_size_source", "output_width", "output_height",
)

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
    """Image files under ``root``, skipping the corner sidecar directories.

    ``corners/`` holds the four-corner JSON files, which describe images rather than
    being images themselves.  A stray preview picture dropped in there must not be
    picked up as an input.
    """

    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "corners" not in path.parts
    )


def read_image(path: Path, flags: int | None = None) -> np.ndarray:
    """Read an image through numpy instead of ``cv2.imread``.

    ``cv2.imread`` goes through the C runtime's narrow-character file API, so it
    fails on any path that the active code page cannot represent -- every Chinese
    filename on a Windows machine, for instance.  ``np.fromfile`` uses the Python
    filesystem encoding and ``cv2.imdecode`` then decodes the bytes, which works
    regardless of the path.
    """

    cv2 = require_opencv()
    if flags is None:
        flags = cv2.IMREAD_COLOR
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"file is empty or unreadable: {path}")
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write an image through numpy for the same reason as :func:`read_image`."""

    cv2 = require_opencv()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    parameters: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, buffer = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise IOError(f"OpenCV could not encode image: {path}")
    buffer.tofile(str(path))


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


def load_summary_rows(path: Path) -> list[dict]:
    """Read an existing summary CSV back into dicts (empty list if absent)."""

    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def save_summary(
    path: Path,
    rows: Iterable[dict],
    *,
    merge_existing: bool = False,
    key_fields: Sequence[str] = SUMMARY_KEY_FIELDS,
) -> int:
    """Write rows to CSV, using the union of all keys as the header.

    With ``merge_existing=True`` the rows already in the file are kept, except for
    those whose ``key_fields`` tuple is produced again by this run -- that is how a
    single-image run and a later batch run can coexist, and how two runs of the
    same image at *different* target rectangles both stay on record.  Without it
    the file is replaced, which is the old behaviour and loses earlier rows.
    """

    rows = list(rows)
    if merge_existing:
        superseded = {tuple(str(row.get(field, "")) for field in key_fields) for row in rows}
        kept = [
            previous for previous in load_summary_rows(path)
            if tuple(str(previous.get(field, "")) for field in key_fields) not in superseded
        ]
        rows = kept + rows

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


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
