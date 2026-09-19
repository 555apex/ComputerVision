"""Tests for the strip-based resampling.

Two things are guarded here.  First, that splitting the work into strips really does
bound the peak allocation -- the single-shot version needed gigabytes on a
multi-megapixel image, which is the defect this rewrite removed.  Second, that the
split changes nothing numerically: every block size must reproduce the same output,
because the strips only reorganise the work.
"""

import tracemalloc

import numpy as np
import pytest

from warping import BLOCK_ROWS, _cast_image, inverse_warp_bilinear

SLANT = np.array([[1.09, 0.07, 40.0], [0.03, 1.04, 22.0], [3.1e-4, -2.2e-4, 1.0]])


def _textured(height, width, channels=None):
    rng = np.random.default_rng(11)
    shape = (height, width) if channels is None else (height, width, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_strip_size_does_not_change_the_result():
    """A strip is a reorganisation, not an approximation."""

    for image in (_textured(97, 131), _textured(97, 131, 3)):
        reference = inverse_warp_bilinear(image, SLANT, 111, 83, block_rows=4096)
        for block in (1, 5, 64, 100):
            candidate = inverse_warp_bilinear(image, SLANT, 111, 83, block_rows=block)
            assert np.array_equal(candidate, reference), f"block_rows={block} diverged"


def test_default_strip_size_is_used_when_not_given():
    image = _textured(97, 131, 3)
    assert np.array_equal(
        inverse_warp_bilinear(image, SLANT, 111, 83),
        inverse_warp_bilinear(image, SLANT, 111, 83, block_rows=BLOCK_ROWS),
    )


STRONG = np.array([[1.7, 0.35, -60.0], [-0.22, 1.4, 45.0], [7.0e-4, -5.0e-4, 1.0]])


def _single_shot(image, homography, width, height):
    """Reference implementation: one full-size coordinate grid, no strips."""

    source = np.asarray(image)
    channels = 1 if source.ndim == 2 else source.shape[2]
    output = np.zeros((height, width) if channels == 1 else (height, width, channels), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    strip = np.stack([xx.ravel(), yy.ravel(), np.ones(width * height)], axis=0)
    projected = np.linalg.inv(np.asarray(homography, dtype=np.float64)) @ strip
    denominator = projected[2]
    ok = np.abs(denominator) > np.finfo(float).eps
    sx = np.full_like(denominator, np.nan)
    sy = np.full_like(denominator, np.nan)
    sx[ok] = projected[0, ok] / denominator[ok]
    sy[ok] = projected[1, ok] / denominator[ok]
    sh, sw = source.shape[:2]
    valid = (np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= sw - 1) & (sy >= 0) & (sy <= sh - 1))
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return _cast_image(output, source.dtype)
    x, y = sx[indices], sy[indices]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, sw - 1); y1 = np.minimum(y0 + 1, sh - 1)
    wx = x - x0; wy = y - y0
    if source.ndim == 2:
        top = source[y0, x0] * (1 - wx) + source[y0, x1] * wx
        bottom = source[y1, x0] * (1 - wx) + source[y1, x1] * wx
        output.ravel()[indices] = top * (1 - wy) + bottom * wy
    else:
        top = source[y0, x0, :] * (1 - wx[:, None]) + source[y0, x1, :] * wx[:, None]
        bottom = source[y1, x0, :] * (1 - wx[:, None]) + source[y1, x1, :] * wx[:, None]
        output.reshape(-1, channels)[indices, :] = top * (1 - wy[:, None]) + bottom * wy[:, None]
    return _cast_image(output, source.dtype)


@pytest.mark.parametrize(
    "label, shape, transform, out_size",
    [
        ("输出大于源图", (40, 60), STRONG, (120, 90)),
        ("输出小于源图", (80, 90), STRONG, (31, 24)),
        ("强透视大部分越界", (30, 30), np.array([[6.0, 1.2, -300.0], [0.8, 5.0, -200.0], [2e-3, 3e-3, 1.0]]), (50, 50)),
        ("灰度", (27, 41), SLANT, (63, 39)),
        ("彩色", (27, 41, 3), SLANT, (63, 39)),
    ],
)
def test_every_block_size_matches_the_single_shot_result(label, shape, transform, out_size):
    """The strips must reproduce the one-shot result at every block boundary.

    Block sizes above, below and equal to the output height are all exercised: an
    off-by-one in the strip split would only show up on the edge cases.
    """

    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    width, height = out_size
    reference = _single_shot(image, transform, width, height)
    for block in (1, 2, 63, 64, 65, height - 1, height, height + 1):
        if block < 1:
            continue
        candidate = inverse_warp_bilinear(image, transform, width, height, block_rows=block)
        assert np.array_equal(candidate, reference), f"{label}: block_rows={block} 与单发实现不一致"

    candidate = inverse_warp_bilinear(image, transform, width, height)
    assert candidate.dtype == image.dtype, f"{label}: 输出 dtype 应与源图一致"


def test_strip_processing_bounds_the_peak_allocation():
    """A 6 MP grayscale warp must not allocate anything close to the full-size
    coordinate arrays the previous implementation used.

    The single-shot version needed about 1.2 GB here; the strip version is well
    under 100 MB.  The threshold is deliberately loose so it fails only on a real
    regression rather than on allocator variation.
    """

    image = np.zeros((2000, 3000), dtype=np.uint8)
    tracemalloc.start()
    try:
        inverse_warp_bilinear(image, np.eye(3), 3000, 2000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 200e6, f"peak allocation {peak / 1e6:.0f} MB is too high for a 6 MP warp"


def test_out_of_range_strip_size_is_rejected():
    image = _textured(9, 9)
    for block in (0, -3):
        try:
            inverse_warp_bilinear(image, np.eye(3), 9, 9, block_rows=block)
        except ValueError:
            continue
        raise AssertionError(f"block_rows={block} should have been rejected")
