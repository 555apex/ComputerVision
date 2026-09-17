"""Generate a deterministic slanted document-like sample image.

Besides the image itself, this script writes a *ground-truth sidecar* next to it
(`sample_slanted.groundtruth.json`).  Because the slanted image is produced by
projecting a known document through a known homography, the true rectification
matrix is exactly recoverable -- which is what makes a quantitative closed-loop
experiment possible (see `gt_experiment.py`).

Sidecar naming convention: for an image `data/.../name.ext`, the ground truth is
`data/.../name.groundtruth.json`.  `run.py` picks it up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homography import destination_corners, solve_homography
from io_utils import save_image
from warping import inverse_warp_bilinear

DOCUMENT_WIDTH = 640
DOCUMENT_HEIGHT = 420
CANVAS_WIDTH = 920
CANVAS_HEIGHT = 700
SLANTED_CORNERS = np.array(
    [[155.0, 95.0], [765.0, 70.0], [835.0, 590.0], [85.0, 625.0]], dtype=np.float64
)
OUTPUT_IMAGE = Path("data/sample/sample_slanted.png")


def make_document(width: int = DOCUMENT_WIDTH, height: int = DOCUMENT_HEIGHT) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    image[18:-18, 18:-18] = (252, 252, 252)
    image[35:105, 45:width - 45] = (220, 235, 250)
    image[125:185, 45:width - 45] = (250, 224, 190)
    image[205:265, 45:width - 45] = (215, 242, 215)
    for x in range(45, width - 45, 40):
        image[290:height - 45, x:x + 3] = (70, 120, 180)
    for y in range(290, height - 45, 35):
        image[y:y + 3, 45:width - 45] = (70, 120, 180)
    image[:18, :] = image[-18:, :] = (30, 70, 120)
    image[:, :18] = image[:, -18:] = (30, 70, 120)
    return image


def ground_truth_sidecar_path(image_path: Path) -> Path:
    """Where the ground truth for `image_path` lives: `name.groundtruth.json`."""
    return image_path.with_suffix(".groundtruth.json")


def main() -> None:
    base = make_document()
    base_corners = destination_corners(base.shape[1], base.shape[0])

    # Document -> slanted canvas.  `inverse_warp_bilinear` consumes exactly this
    # direction, so the same matrix both generates the sample and defines truth.
    document_to_slanted, _, _ = solve_homography(base_corners, SLANTED_CORNERS)
    slanted = inverse_warp_bilinear(base, document_to_slanted, CANVAS_WIDTH, CANVAS_HEIGHT)
    save_image(OUTPUT_IMAGE, slanted)
    print(f"Wrote {OUTPUT_IMAGE}")

    # The rectification a student should recover: slanted canvas -> document.
    slanted_to_document = np.linalg.inv(document_to_slanted)
    sidecar = {
        "generator": "generate_sample.py",
        "note": (
            "Exact ground truth for the synthetic sample. "
            "`homography_source_to_destination` maps the slanted image to the true document "
            "rectangle; it is the reference used by the ground-truth metrics in summary.csv."
        ),
        "document_size": {"width": int(base.shape[1]), "height": int(base.shape[0])},
        "source_size": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
        "source_corners_tl_tr_br_bl": SLANTED_CORNERS.tolist(),
        "destination_corners_tl_tr_br_bl": base_corners.tolist(),
        "homography_source_to_destination": slanted_to_document.tolist(),
        "homography_document_to_source": document_to_slanted.tolist(),
    }
    sidecar_path = ground_truth_sidecar_path(OUTPUT_IMAGE)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {sidecar_path}  (true document size {base.shape[1]}x{base.shape[0]})")


if __name__ == "__main__":
    main()
