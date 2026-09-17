"""Geometry validation for interactive and automated point selection."""

from __future__ import annotations

import numpy as np


def polygon_signed_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - b
    return float(first[0] * second[1] - first[1] * second[0])


def validate_corners(points: np.ndarray, *, min_area: float = 1.0) -> tuple[bool, str]:
    """Validate four corners in TL, TR, BR, BL order."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        return False, "exactly four points are required"
    if not np.all(np.isfinite(pts)):
        return False, "points must be finite"
    if len(np.unique(pts, axis=0)) != 4:
        return False, "duplicate points are not allowed"
    area = abs(polygon_signed_area(pts))
    if area < min_area:
        return False, "the selected quadrilateral has near-zero area"

    crosses = np.array([_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)])
    if np.any(np.abs(crosses) < np.finfo(float).eps):
        return False, "three consecutive points are collinear"
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False, "points are not a convex quadrilateral in the required order"
    return True, "ok"


def validate_correspondences(src: np.ndarray, dst: np.ndarray) -> tuple[bool, str]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        return False, "source and destination points must have the same shape (n, 2)"
    if len(src) < 4:
        return False, "at least four correspondences are required"
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        return False, "correspondences must be finite"
    if len(np.unique(src, axis=0)) < 4 or len(np.unique(dst, axis=0)) < 4:
        return False, "at least four unique points are required"
    return True, "ok"
