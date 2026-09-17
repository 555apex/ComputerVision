"""Tests for the ground-truth metrics and the summary CSV writer.

The central behaviour under test: a four-point homography has an identically zero
fit residual, so metrics have to be evaluated on held-out points to say anything.
The key regression is that a *wrong target rectangle* produces a large transfer
error while the fit residual stays at machine precision.
"""

import numpy as np

from homography import (
    destination_corners,
    estimate_output_size,
    ground_truth_errors,
    quad_grid,
    reprojection_errors,
    solve_homography,
    transfer_errors,
)
from io_utils import load_ground_truth, save_summary

SLANTED = np.array([[155.0, 95.0], [765.0, 70.0], [835.0, 590.0], [85.0, 625.0]])


def test_quad_grid_is_inside_the_quad():
    grid = quad_grid(SLANTED, steps=9)
    assert grid.shape == (81, 2)
    assert np.all(grid[:, 0] >= SLANTED[:, 0].min())
    assert np.all(grid[:, 0] <= SLANTED[:, 0].max())
    assert np.all(grid[:, 1] >= SLANTED[:, 1].min())
    assert np.all(grid[:, 1] <= SLANTED[:, 1].max())
    # corners of the grid must coincide with the quad corners
    assert np.allclose(grid[0], SLANTED[0])
    assert np.allclose(grid[-1], SLANTED[2])


def test_transfer_error_is_zero_for_identical_homographies():
    matrix = np.array([[1.12, 0.08, 80.0], [0.035, 1.05, 55.0], [0.00035, -0.00022, 1.0]])
    metrics = transfer_errors(matrix, matrix, quad_grid(SLANTED, 7))
    assert metrics["transfer_rmse_vs_truth_px"] == 0.0
    assert metrics["transfer_max_vs_truth_px"] == 0.0
    assert metrics["transfer_eval_count"] == 49.0


def test_fit_residual_is_degenerate_but_transfer_error_is_not():
    """The whole reason the ground-truth metrics exist."""

    truth_size = (640, 420)
    truth_corners = destination_corners(*truth_size)
    # rectification direction: slanted image -> document rectangle
    truth, _, _ = solve_homography(SLANTED, truth_corners)
    evaluation = quad_grid(SLANTED, 15)

    wrong_size = estimate_output_size(SLANTED)
    assert wrong_size != truth_size

    wrong, _, _ = solve_homography(SLANTED, destination_corners(*wrong_size), normalize=True)
    right, _, _ = solve_homography(SLANTED, destination_corners(*truth_size), normalize=True)

    # fit residual claims both solutions are perfect
    wrong_residual = float(np.sqrt(np.mean(reprojection_errors(wrong, SLANTED, destination_corners(*wrong_size)) ** 2)))
    right_residual = float(np.sqrt(np.mean(reprojection_errors(right, SLANTED, truth_corners) ** 2)))
    assert wrong_residual < 1e-6
    assert right_residual < 1e-6

    # held-out transfer error separates them by many orders of magnitude
    wrong_metrics = ground_truth_errors(wrong, SLANTED, truth, truth_corners, evaluation_points=evaluation)
    right_metrics = ground_truth_errors(right, SLANTED, truth, truth_corners, evaluation_points=evaluation)

    assert wrong_metrics["transfer_rmse_vs_truth_px"] > 50.0
    assert right_metrics["transfer_rmse_vs_truth_px"] < 1e-6
    assert wrong_metrics["transfer_rmse_vs_truth_px"] > right_metrics["transfer_rmse_vs_truth_px"] * 1e6


def test_target_mismatch_is_zero_when_rectangle_matches_truth():
    """Documents the one case where the corner-based metric says nothing."""

    truth_size = (640, 420)
    truth_corners = destination_corners(*truth_size)
    truth, _, _ = solve_homography(SLANTED, truth_corners)
    estimate, _, _ = solve_homography(SLANTED, truth_corners)

    metrics = ground_truth_errors(estimate, SLANTED, truth, truth_corners)
    assert metrics["target_mismatch_rmse_px"] < 1e-6
    assert "transfer_rmse_vs_truth_px" not in metrics


def test_aligned_matrix_error_rejects_zero_matrix():
    import pytest

    from homography import aligned_matrix_error

    with pytest.raises(ValueError):
        aligned_matrix_error(np.zeros((3, 3)), np.eye(3))


def test_save_summary_unions_columns_across_rows(tmp_path):
    path = tmp_path / "summary.csv"
    save_summary(path, [
        {"image": "a.png", "method": "basic_dlt", "solve_ms": 0.4},
        {"image": "a.png", "method": "opencv", "warp_ms": 1.2, "extra": "x"},
    ])
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    for column in ("image", "method", "solve_ms", "warp_ms", "extra"):
        assert column in header
    assert len(text.strip().splitlines()) == 3


def test_load_ground_truth_returns_none_without_sidecar(tmp_path):
    assert load_ground_truth(tmp_path / "missing.png") is None
