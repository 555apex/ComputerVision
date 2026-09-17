"""Tests for the paired ablation design.

The regressions guarded here are the reason P0 existed: the earlier unpaired design
produced a spurious "normalization is worse" at four correspondences, where the two
methods are mathematically identical.  These tests fail if that ever comes back, or
if failures start being dropped from the statistics again.
"""

import numpy as np

from visualize_experiments import (
    PRACTICAL_FLOOR_PX,
    SWEEPS,
    _summarise,
    paired_sweep,
    paired_trial,
)

NOISE_SWEEP, COUNT_SWEEP, SCALE_SWEEP = SWEEPS


def _sweep_with(key: str, values, *, point_count: int, noise: float, scale: float,
                repeats: int = 48) -> list[dict]:
    sweep = {"key": key, "values": values,
             "fixed": {"point_count": point_count, "noise": noise, "scale": scale}}
    return paired_sweep(sweep, repeats=repeats, seed=7)


def test_paired_trial_is_reproducible_and_returns_both_methods():
    first = paired_trial(np.random.default_rng([1, 12, 100, 1]), 12, 1.0, 1.0)
    second = paired_trial(np.random.default_rng([1, 12, 100, 1]), 12, 1.0, 1.0)
    assert set(first) == {"basic_dlt", "normalized_dlt"}
    assert np.isclose(first["basic_dlt"], second["basic_dlt"])
    assert np.isclose(first["normalized_dlt"], second["normalized_dlt"])


def test_four_points_show_no_method_difference():
    """The headline regression: four correspondences determine H exactly.

    Both methods must therefore recover the same matrix, so the paired difference is
    at the floating-point level (about 1e-12 px) and must be reported as negligible.
    The old unpaired design reported a +0.175 pixel "normalization is worse" here.
    """

    rows = _sweep_with("point_count", [4], point_count=4, noise=1.0, scale=1.0)
    row = rows[0]
    assert row["paired_count"] == 48
    assert abs(row["diff_mean"]) < PRACTICAL_FLOOR_PX
    assert row["practically_negligible"] is True
    assert not row["significant"]
    # the two methods' own means must agree to many digits
    assert np.isclose(row["basic_mean"], row["normalized_mean"], rtol=1e-9)


def test_practical_floor_beats_a_meaningless_t_statistic():
    """A reproducible 1e-12 offset must not be reported as a real difference.

    With enough repeats the floating-point offset at n=4 is perfectly consistent, so
    a t-test alone would call it significant.  The effect-size floor is what stops
    that from being reported as a finding.
    """

    rows = _sweep_with("point_count", [4], point_count=4, noise=1.0, scale=1.0, repeats=200)
    row = rows[0]
    assert abs(row["diff_mean"]) < PRACTICAL_FLOOR_PX
    assert not row["significant"], "t alone is not a sufficient criterion"


def test_normalization_benefit_is_significant_with_many_points():
    rows = _sweep_with("point_count", [30], point_count=30, noise=1.0, scale=1.0)
    row = rows[0]
    assert row["diff_mean"] < 0.0, "normalized DLT should be the more accurate one"
    assert row["significant"]
    assert row["t_stat"] < -1.96


def test_failures_are_counted_not_dropped():
    """At a huge coordinate scale basic DLT stops working; this must be visible."""

    rows = _sweep_with("scale", [1e5], point_count=12, noise=0.0, scale=1e5, repeats=8)
    row = rows[0]
    assert row["repeats"] == 8
    assert row["basic_success_rate"] == 0.0
    assert row["normalized_success_rate"] == 1.0
    assert row["paired_count"] == 0
    assert not np.isfinite(row["basic_mean"])
    assert np.isfinite(row["normalized_mean"]), "the normalized fit still works here"
    assert not np.isfinite(row["diff_mean"]), "no paired sample means no difference"


def test_scale_sweep_degrades_monotonically_for_basic_dlt():
    scales = [1.0, 10.0, 100.0, 1000.0]
    rows = _sweep_with("scale", scales, point_count=12, noise=0.0, scale=1.0, repeats=24)
    means = [row["basic_mean"] for row in rows]
    assert all(np.isfinite(means))
    assert all(later > earlier for earlier, later in zip(means, means[1:])), means
    # normalized must stay essentially flat by comparison
    normalized = [row["normalized_mean"] for row in rows]
    assert max(normalized) / min(normalized) < 1e4
    assert means[-1] / normalized[-1] > 1e6


def test_summarise_keeps_failures_in_the_denominator():
    row = _summarise(1.0, [1.0, 2.0, np.nan, np.nan], [1.0, 2.0, 3.0, 4.0])
    assert row["repeats"] == 4
    assert row["basic_success_rate"] == 0.5
    assert row["normalized_success_rate"] == 1.0
    assert row["paired_count"] == 2


def test_summarise_handles_total_failure():
    row = _summarise(1.0, [np.nan, np.nan], [np.nan, np.nan])
    assert row["paired_count"] == 0
    assert row["basic_success_rate"] == 0.0
    assert not np.isfinite(row["basic_mean"])
    assert not row["significant"]
