"""Homography estimation with basic and Hartley-normalized DLT.

The implementation deliberately keeps the DLT matrix construction visible so it
can be used as the algorithm section of the course assignment.  NumPy is used
for the numerical SVD; no OpenCV homography solver is used here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


Array = np.ndarray


def _as_points(points: Iterable[Iterable[float]], name: str) -> Array:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {result.shape}")
    if result.shape[0] < 4:
        raise ValueError(f"at least four point correspondences are required for {name}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def build_dlt_matrix(src_points: Iterable[Iterable[float]], dst_points: Iterable[Iterable[float]]) -> Array:
    """Build the 2n x 9 homogeneous DLT matrix for source -> destination points."""

    src = _as_points(src_points, "src_points")
    dst = _as_points(dst_points, "dst_points")
    if src.shape != dst.shape:
        raise ValueError(f"source and destination point arrays must have the same shape, got {src.shape} and {dst.shape}")

    rows = []
    for (x, y), (xp, yp) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp, -xp])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp, -yp])
    return np.asarray(rows, dtype=np.float64)


def normalize_points(points: Iterable[Iterable[float]]) -> tuple[Array, Array]:
    """Hartley-normalize 2D points and return (normalized_points, transform)."""

    pts = _as_points(points, "points")
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    distances = np.linalg.norm(centered, axis=1)
    mean_distance = float(np.mean(distances))
    if mean_distance <= np.finfo(float).eps:
        raise ValueError("cannot normalize a point set with zero spread")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    normalized_h = (transform @ homogeneous.T).T
    return normalized_h[:, :2] / normalized_h[:, 2:3], transform


def _canonicalize_homography(matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = float(np.linalg.norm(matrix))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("homography solution has zero or invalid norm")
    result = matrix / norm
    pivot = np.argmax(np.abs(result))
    if result.flat[pivot] < 0:
        result = -result
    return result


def solve_homography(
    src_points: Iterable[Iterable[float]],
    dst_points: Iterable[Iterable[float]],
    *,
    normalize: bool = False,
) -> tuple[Array, Array, Array]:
    """Solve H with DLT and SVD.

    Returns (H, A, singular_values).  H maps source homogeneous points to
    destination homogeneous points.  The returned H is Frobenius-normalized;
    its scale is therefore not interpreted as a physical quantity.
    """

    src = _as_points(src_points, "src_points")
    dst = _as_points(dst_points, "dst_points")
    if src.shape != dst.shape:
        raise ValueError(f"source and destination point arrays must have the same shape, got {src.shape} and {dst.shape}")

    if normalize:
        src_work, src_transform = normalize_points(src)
        dst_work, dst_transform = normalize_points(dst)
        matrix = build_dlt_matrix(src_work, dst_work)
    else:
        src_transform = np.eye(3, dtype=np.float64)
        dst_transform = np.eye(3, dtype=np.float64)
        matrix = build_dlt_matrix(src, dst)

    rank = np.linalg.matrix_rank(matrix)
    if rank < 8:
        raise ValueError(f"correspondences are degenerate: DLT matrix rank is {rank}, expected at least 8")

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    normalized_h = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(dst_transform) @ normalized_h @ src_transform
    return _canonicalize_homography(homography), matrix, singular_values


def project_points(homography: Array, points: Iterable[Iterable[float]]) -> Array:
    """Project 2D points through a homography."""

    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"homography must have shape (3, 3), got {h.shape}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    projected_h = (h @ homogeneous.T).T
    denominator = projected_h[:, 2:3]
    result = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.abs(denominator[:, 0]) > np.finfo(float).eps
    result[valid] = projected_h[valid, :2] / denominator[valid]
    return result


def reprojection_errors(homography: Array, src_points: Iterable[Iterable[float]], dst_points: Iterable[Iterable[float]]) -> Array:
    """Return Euclidean reprojection error for each correspondence."""

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    projected = project_points(homography, src)
    errors = np.linalg.norm(projected - dst, axis=1)
    return errors


def aligned_matrix_error(estimated: Array, reference: Array) -> float:
    """Compare two homographies after the optimal scalar alignment."""

    estimate = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != (3, 3) or reference.shape != (3, 3):
        raise ValueError("both homographies must have shape (3, 3)")
    denominator = float(np.sum(estimate * estimate))
    if denominator <= np.finfo(float).eps:
        raise ValueError("estimated homography is all zeros; scale alignment is undefined")
    scale = float(np.sum(estimate * reference) / denominator)
    return float(np.linalg.norm(scale * estimate - reference))


def quad_grid(corners: Iterable[Iterable[float]], steps: int = 25) -> Array:
    """Bilinear grid of points inside a quad ordered TL, TR, BR, BL.

    Used as held-out evaluation points: they are *not* the correspondences the
    homography was fitted from, so comparing two homographies on them is free of
    the "fit residual is identically zero" degeneracy.
    """

    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {pts.shape}")
    steps = max(2, int(steps))
    u, v = np.meshgrid(np.linspace(0.0, 1.0, steps), np.linspace(0.0, 1.0, steps))
    top = pts[0][None, :] * (1.0 - u.ravel())[:, None] + pts[1][None, :] * u.ravel()[:, None]
    bottom = pts[3][None, :] * (1.0 - u.ravel())[:, None] + pts[2][None, :] * u.ravel()[:, None]
    return top * (1.0 - v.ravel())[:, None] + bottom * v.ravel()[:, None]


def transfer_errors(homography: Array, reference: Array, points: Iterable[Iterable[float]]) -> dict[str, float]:
    """Transfer error between two homographies, measured on held-out points.

    Projects `points` through both matrices and reports the distance between the
    two results in the target plane, in input units (pixels if `points` are in
    pixels).  Unlike :func:`reprojection_errors` this does not vanish for a
    four-point fit, because the evaluation points are independent of the fit.
    """

    estimate = project_points(homography, points)
    reference_points = project_points(reference, points)
    finite = np.isfinite(estimate).all(axis=1) & np.isfinite(reference_points).all(axis=1)
    if not np.any(finite):
        return {"transfer_rmse_vs_truth_px": float("nan"),
                "transfer_max_vs_truth_px": float("nan"),
                "transfer_eval_count": 0.0}
    distances = np.linalg.norm(estimate[finite] - reference_points[finite], axis=1)
    return {
        "transfer_rmse_vs_truth_px": float(np.sqrt(np.mean(distances ** 2))),
        "transfer_max_vs_truth_px": float(np.max(distances)),
        "transfer_eval_count": float(finite.sum()),
    }


def ground_truth_errors(
    homography: Array,
    source_points: Iterable[Iterable[float]],
    reference_homography: Array,
    reference_destination_points: Iterable[Iterable[float]],
    evaluation_points: Iterable[Iterable[float]] | None = None,
) -> dict[str, float]:
    """Metrics measured against known truth rather than against the fitted points.

    Why this exists: :func:`reprojection_errors` measures the *fit residual* -- how
    well ``H`` explains the very points it was fitted from.  With four
    correspondences that residual is exactly zero by construction (eight equations
    for eight degrees of freedom), so it degenerates into floating-point noise and
    carries no information about the quality of the solution.

    Three distinct quantities are reported instead:

    ``matrix_error_vs_truth``
        Frobenius distance to the reference homography after optimal scale
        alignment (the two matrices live in a homogeneous class, so they must be
        aligned first).  Sensitive to the parameterisation, so use it as a
        secondary check rather than a headline number.

    ``target_mismatch_rmse_px``
        How far the fitted map sends the source corners from the *true* target
        corners.  This isolates the "wrong target rectangle" error: it is large
        when the output rectangle's aspect ratio is wrong, and -- importantly --
        it is **identically zero when the chosen rectangle equals the truth**,
        because then the fit reproduces the target corners exactly.  So it answers
        "did I pick the right rectangle?", not "did I estimate H well?".

    ``transfer_rmse_vs_truth_px``
        Transfer error on held-out ``evaluation_points``.  This is the only metric
        here that responds to *both* error sources (bad rectangle and noisy
        corners), because the evaluation points are independent of the fit.  Pass
        :func:`quad_grid` of the source corners to keep evaluation inside the
        region of interest.
    """

    reference = np.asarray(reference_homography, dtype=np.float64)
    if reference.shape != (3, 3):
        raise ValueError(f"reference homography must have shape (3, 3), got {reference.shape}")

    errors = reprojection_errors(homography, source_points, reference_destination_points)
    finite = errors[np.isfinite(errors)]
    result: dict[str, float] = {
        "matrix_error_vs_truth": aligned_matrix_error(homography, reference),
        "target_mismatch_rmse_px": float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan"),
        "target_mismatch_max_px": float(np.max(finite)) if finite.size else float("nan"),
    }
    if evaluation_points is not None:
        result.update(transfer_errors(homography, reference, evaluation_points))
    return result


def estimate_output_size(corners: Iterable[Iterable[float]]) -> tuple[int, int]:
    """Estimate (width, height) from corners ordered TL, TR, BR, BL."""

    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")
    width = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
    height = max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
    width_i = max(1, int(round(width)))
    height_i = max(1, int(round(height)))
    return width_i, height_i


def destination_corners(width: int, height: int) -> Array:
    if width < 2 or height < 2:
        raise ValueError("output width and height must be at least 2")
    return np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
