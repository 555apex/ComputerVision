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
from io_utils import (
    iter_images,
    load_ground_truth,
    load_summary_rows,
    read_image,
    save_image,
    save_summary,
)

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


def _row(image, method, width, height, **extra):
    return {"image": image, "method": method, "target_size_source": "estimated",
            "output_width": width, "output_height": height, **extra}


def test_summary_keeps_same_file_name_from_different_sources(tmp_path):
    """`image` is only a file name, so the source has to be part of the row identity.

    data/public and data/phone can both hold a fig1.jpg. Without the source in the
    key the second run silently replaced the first one's rows.
    """

    path = tmp_path / "summary.csv"
    save_summary(path, [_row("fig1.jpg", "basic_dlt", 676, 594, source="phone")])
    save_summary(path, [_row("fig1.jpg", "basic_dlt", 676, 594, source="public")], merge_existing=True)

    rows = load_summary_rows(path)
    assert len(rows) == 2, "同名不同来源的图片必须各占一行"
    assert {row["source"] for row in rows} == {"phone", "public"}


def test_summary_merge_leaves_a_single_number_format_per_column(tmp_path):
    """Numbers must not end up as a mix of '0.07' and '0.0700' in one column."""

    path = tmp_path / "summary.csv"
    save_summary(path, [_row("a.png", "basic_dlt", 10, 10, solve_ms=0.07)])
    save_summary(path, [_row("b.png", "basic_dlt", 10, 10, solve_ms=0.09)], merge_existing=True)

    formats = {len(row["solve_ms"].split(".")[-1]) for row in load_summary_rows(path)}
    assert len(formats) == 1, f"一列里出现了多种小数位格式: {formats}"


def test_iter_images_skips_corner_sidecar_directories(tmp_path):
    """corners/ describes images, so a picture dropped in there is not an input."""

    (tmp_path / "corners").mkdir()
    (tmp_path / "corners" / "preview.png").write_bytes(b"not really an image")
    (tmp_path / "photo.jpg").write_bytes(b"not really an image")

    found = [path.name for path in iter_images(tmp_path)]
    assert found == ["photo.jpg"], found


def test_save_summary_merges_without_losing_earlier_runs(tmp_path):
    """A batch run must not silently drop the rows a single-image run wrote."""

    path = tmp_path / "summary.csv"
    save_summary(path, [_row("sample.png", "basic_dlt", 640, 420, solve_ms=0.4)])
    save_summary(path, [_row("board.jpg", "basic_dlt", 554, 1357)], merge_existing=True)

    rows = load_summary_rows(path)
    assert {(r["image"], r["method"]) for r in rows} == {("sample.png", "basic_dlt"), ("board.jpg", "basic_dlt")}
    assert rows[0]["solve_ms"] == "0.4"
    assert rows[1]["image"] == "board.jpg"


def test_save_summary_replaces_only_the_same_configuration(tmp_path):
    """Re-running one configuration updates it in place; other variants survive."""

    path = tmp_path / "summary.csv"
    save_summary(path, [
        _row("sample.png", "basic_dlt", 751, 535, solve_ms=0.07),
        _row("sample.png", "basic_dlt", 640, 420, target_size_source="cli", solve_ms=0.04),
    ])
    save_summary(path, [_row("sample.png", "basic_dlt", 640, 420, target_size_source="cli", solve_ms=0.09)],
                 merge_existing=True)

    rows = load_summary_rows(path)
    assert len(rows) == 2, "the auto-estimated variant must survive"
    by_size = {int(r["output_width"]): r["solve_ms"] for r in rows}
    assert by_size == {751: "0.07", 640: "0.09"}


def test_save_summary_overwrite_flag_still_replaces_everything(tmp_path):
    path = tmp_path / "summary.csv"
    save_summary(path, [_row("a.png", "basic_dlt", 10, 10)])
    save_summary(path, [_row("b.png", "basic_dlt", 20, 20)])

    assert [r["image"] for r in load_summary_rows(path)] == ["b.png"]


def test_image_io_round_trips_through_a_chinese_path(tmp_path):
    """cv2.imread cannot open non-ASCII paths on Windows; read_image must."""

    image = (np.arange(12 * 9 * 3) % 251).astype(np.uint8).reshape(12, 9, 3)
    path = tmp_path / "手机拍摄_斜拍文档.png"
    save_image(path, image)
    assert path.is_file()

    restored = read_image(path)
    assert restored.shape == image.shape
    assert np.array_equal(restored, image)
