import numpy as np

from educational_svd import jacobi_svd
from homography import (
    aligned_matrix_error,
    project_points,
    reprojection_errors,
    solve_homography,
)
from validation import validate_corners
from warping import inverse_warp_bilinear


def test_four_point_dlt_recovers_known_mapping():
    source = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], dtype=float)
    expected = np.array([[1.2, 0.08, 20], [0.03, 1.1, 15], [0.0005, -0.0008, 1]], dtype=float)
    target = project_points(expected, source)
    estimated, matrix, _ = solve_homography(source, target)
    assert matrix.shape == (8, 9)
    assert np.max(reprojection_errors(estimated, source, target)) < 1e-8
    assert aligned_matrix_error(estimated, expected) < 1e-8


def test_normalized_dlt_handles_large_coordinates_and_extra_points():
    rng = np.random.default_rng(4)
    source = rng.uniform(-2e6, 2e6, size=(30, 2))
    expected = np.array([[1.1, 0.03, 4e5], [-0.02, 0.9, -2e5], [2e-7, -1e-7, 1]], dtype=float)
    target = project_points(expected, source)
    target += rng.normal(0, 0.05, size=target.shape)
    basic, _, _ = solve_homography(source, target, normalize=False)
    normalized, _, _ = solve_homography(source, target, normalize=True)
    basic_error = float(np.sqrt(np.mean(reprojection_errors(basic, source, target) ** 2)))
    normalized_error = float(np.sqrt(np.mean(reprojection_errors(normalized, source, target) ** 2)))
    assert np.isfinite(normalized_error)
    assert normalized_error <= basic_error * 1.05


def test_validation_rejects_duplicate_and_crossed_points():
    valid, _ = validate_corners(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float))
    assert valid
    duplicate, _ = validate_corners(np.array([[0, 0], [10, 0], [10, 10], [10, 10]], dtype=float))
    assert not duplicate
    crossed, _ = validate_corners(np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float))
    assert not crossed


def test_manual_identity_warp_preserves_pixels():
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    warped = inverse_warp_bilinear(image, np.eye(3), 7, 5)
    assert np.array_equal(warped, image)


def test_jacobi_svd_has_correct_smallest_right_vector():
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=float)
    _, singular_values, v_t = jacobi_svd(matrix)
    assert np.all(np.diff(singular_values) <= 1e-10)
    assert abs(np.linalg.norm(matrix @ v_t[-1])) < 1e-8
