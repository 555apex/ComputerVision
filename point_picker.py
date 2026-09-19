"""Interactive OpenCV point picker, plus a CLI for saving the picked corners.

The saved JSON is what makes a later batch run reproducible: click once per image,
store ``corners/<image stem>.json``, and ``run.py --batch --no-interactive`` will
find it without any window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation import validate_corners


def select_four_corners(image: np.ndarray) -> np.ndarray:
    """Select TL, TR, BR, BL with mouse clicks; R resets and Enter confirms."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("interactive point selection requires OpenCV; install requirements.txt") from exc

    points: list[tuple[int, int]] = []
    window = "Select corners: TL -> TR -> BR -> BL"
    canvas = image.copy()

    def redraw() -> None:
        nonlocal canvas
        canvas = image.copy()
        for index, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(canvas, str(index + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, "Click TL, TR, BR, BL | R reset | Enter confirm | Esc cancel", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise RuntimeError(
            "cannot open a window to pick corners -- OpenCV was built without GUI "
            "support or there is no display. Supply corners from a JSON file instead "
            "(run.py --points-file) or run with --no-interactive."
        ) from exc
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    try:
        while True:
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt("point selection cancelled")
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key in (13, 10, 32) and len(points) == 4:
                valid, message = validate_corners(np.asarray(points, dtype=np.float64))
                if valid:
                    return np.asarray(points, dtype=np.float64)
                print(f"Invalid corners: {message}. Press R to select again.")
    finally:
        cv2.destroyWindow(window)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick four corners interactively and optionally save them as JSON")
    parser.add_argument("--input", type=Path, required=True, help="image to pick corners on")
    parser.add_argument(
        "--output", type=Path,
        help=("where to write the four corners. Defaults to <image dir>/corners/<image stem>.json, "
              "which is exactly where run.py --batch looks for them."),
    )
    args = parser.parse_args()

    from io_utils import read_image, save_json

    output = args.output if args.output is not None else args.input.parent / "corners" / f"{args.input.stem}.json"
    image = read_image(args.input)
    corners = select_four_corners(image)
    valid, message = validate_corners(corners)
    if not valid:
        raise SystemExit(f"picked corners are invalid: {message}")

    payload = [[round(float(x), 2), round(float(y), 2)] for x, y in corners]
    save_json(output, payload)
    print(f"image {args.input.name}  {image.shape[1]}x{image.shape[0]}")
    print(f"corners {json.dumps(payload)}")
    print(f"saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
