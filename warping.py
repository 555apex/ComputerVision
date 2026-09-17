"""Manual inverse-mapped image warping and optional OpenCV comparison."""

from __future__ import annotations

import numpy as np


def _cast_image(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


def inverse_warp_bilinear(image: np.ndarray, homography: np.ndarray, width: int, height: int) -> np.ndarray:
    """Warp source image to a target canvas with inverse mapping.

    H maps source coordinates to target coordinates. Pixels whose inverse-mapped
    source coordinate falls outside the source image are black.
    """

    source = np.asarray(image)
    if source.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D color array")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    channels = 1 if source.ndim == 2 else source.shape[2]
    output_shape = (height, width) if channels == 1 else (height, width, channels)
    output = np.zeros(output_shape, dtype=np.float64)

    yy, xx = np.indices((height, width), dtype=np.float64)
    target_h = np.stack([xx.ravel(), yy.ravel(), np.ones(width * height)], axis=0)
    try:
        inverse_h = np.linalg.inv(h)
    except np.linalg.LinAlgError as exc:
        raise ValueError("homography is singular and cannot be inverted for warping") from exc
    source_h = inverse_h @ target_h
    denominator = source_h[2]
    valid_denominator = np.abs(denominator) > np.finfo(float).eps
    source_x = np.full_like(denominator, np.nan)
    source_y = np.full_like(denominator, np.nan)
    source_x[valid_denominator] = source_h[0, valid_denominator] / denominator[valid_denominator]
    source_y[valid_denominator] = source_h[1, valid_denominator] / denominator[valid_denominator]

    src_height, src_width = source.shape[:2]
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
        return _cast_image(output, source.dtype)

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
        values = top * (1.0 - wy) + bottom * wy
        flat_output = output.ravel()
        flat_output[indices] = values
    else:
        top = source[y0, x0, :] * (1.0 - wx[:, None]) + source[y0, x1, :] * wx[:, None]
        bottom = source[y1, x0, :] * (1.0 - wx[:, None]) + source[y1, x1, :] * wx[:, None]
        values = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
        flat_output = output.reshape(-1, channels)
        flat_output[indices, :] = values
    return _cast_image(output, source.dtype)


def opencv_warp(image: np.ndarray, homography: np.ndarray, width: int, height: int) -> np.ndarray:
    """Use OpenCV's warpPerspective, imported lazily for optional dependency support."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the library baseline; install requirements.txt") from exc
    return cv2.warpPerspective(image, np.asarray(homography, dtype=np.float64), (width, height), borderValue=0)
