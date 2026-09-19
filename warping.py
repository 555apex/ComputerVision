"""Manual inverse-mapped image warping and optional OpenCV comparison.

The resampling runs in horizontal strips.  A single-shot implementation has to keep
the full-size coordinate array, the projected array, the per-pixel weights and the
index lists alive at the same time, which costs several hundred megabytes for a
12 MP image; restricting the work to a strip keeps the peak bounded while producing
bit-identical output, because every pixel is computed by exactly the same
expressions as before.
"""

from __future__ import annotations

import numpy as np

# Rows per strip.  Measured on a 12 MP image (4000x3000) with tracemalloc, peak
# allocation drops from 2424 MB to 136 MB (grayscale) and from 3576 MB to 358 MB
# (colour), with runtime unchanged within noise.  See .audit/verify_warp_blocking.py.
BLOCK_ROWS = 64


def _cast_image(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Round and clip in place before casting.

    Doing this with ``np.clip(np.rint(x), ...)`` allocates two more full-size
    float64 buffers; on a 12 MP colour image that alone costs ~580 MB.  The in-place
    forms produce the same values with no extra allocation.

    The rounding only applies to floating-point input: feed in an array that is
    already integer and the in-place ufunc would fail on the cast.
    """

    if np.issubdtype(dtype, np.integer) and np.issubdtype(values.dtype, np.floating):
        info = np.iinfo(dtype)
        np.rint(values, out=values)
        np.clip(values, info.min, info.max, out=values)
    return values.astype(dtype, copy=False)


def _strip_homogeneous(y_from: int, y_to: int, width: int) -> np.ndarray:
    """Homogeneous target coordinates for rows [y_from, y_to), C order."""

    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(y_from, y_to, dtype=np.float64)
    strip = np.empty((3, (y_to - y_from) * width), dtype=np.float64)
    strip[0] = np.tile(columns, y_to - y_from)
    strip[1] = np.repeat(rows, width)
    strip[2] = 1.0
    return strip


def inverse_warp_bilinear(
    image: np.ndarray,
    homography: np.ndarray,
    width: int,
    height: int,
    *,
    block_rows: int = BLOCK_ROWS,
) -> np.ndarray:
    """Warp source image to a target canvas with inverse mapping.

    H maps source coordinates to target coordinates. Pixels whose inverse-mapped
    source coordinate falls outside the source image are black.
    """

    source = np.asarray(image)
    if source.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D color array")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if block_rows < 1:
        raise ValueError(f"block_rows must be positive, got {block_rows}")
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    channels = 1 if source.ndim == 2 else source.shape[2]
    output_shape = (height, width) if channels == 1 else (height, width, channels)
    output = np.zeros(output_shape, dtype=np.float64)

    try:
        inverse_h = np.linalg.inv(h)
    except np.linalg.LinAlgError as exc:
        raise ValueError("homography is singular and cannot be inverted for warping") from exc

    src_height, src_width = source.shape[:2]
    eps = np.finfo(float).eps

    for y_from in range(0, height, block_rows):
        y_to = min(y_from + block_rows, height)
        strip = _strip_homogeneous(y_from, y_to, width)
        source_h = inverse_h @ strip
        denominator = source_h[2]

        source_x = np.full_like(denominator, np.nan)
        source_y = np.full_like(denominator, np.nan)
        valid_denominator = np.abs(denominator) > eps
        source_x[valid_denominator] = source_h[0, valid_denominator] / denominator[valid_denominator]
        source_y[valid_denominator] = source_h[1, valid_denominator] / denominator[valid_denominator]

        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0.0)
            & (source_x <= src_width - 1)
            & (source_y >= 0.0)
            & (source_y <= src_height - 1)
        )
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            continue

        x = source_x[indices]
        y = source_y[indices]
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, src_width - 1)
        y1 = np.minimum(y0 + 1, src_height - 1)
        wx = x - x0
        wy = y - y0

        if source.ndim == 2:
            top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
            bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
            output[y_from:y_to].ravel()[indices] = top * (1.0 - wy) + bottom * wy
        else:
            top = source[y0, x0, :] * (1.0 - wx[:, None]) + source[y0, x1, :] * wx[:, None]
            bottom = source[y1, x0, :] * (1.0 - wx[:, None]) + source[y1, x1, :] * wx[:, None]
            values = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
            output[y_from:y_to].reshape(-1, channels)[indices, :] = values

    return _cast_image(output, source.dtype)


def opencv_warp(image: np.ndarray, homography: np.ndarray, width: int, height: int) -> np.ndarray:
    """Use OpenCV's warpPerspective, imported lazily for optional dependency support."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the library baseline; install requirements.txt") from exc
    return cv2.warpPerspective(image, np.asarray(homography, dtype=np.float64), (width, height), borderValue=0)
