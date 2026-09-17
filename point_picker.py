"""Interactive OpenCV point picker."""

from __future__ import annotations

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

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
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
