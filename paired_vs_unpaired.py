"""Quantify the standard-error inflation that the paired ablation design removed.

The ablation's first version drew one dataset for Basic DLT and a different dataset
for Normalized DLT, then treated the two as independent samples.  The point estimate
of the difference stays unbiased either way; what changes is its standard error --
sqrt(var_a/n + var_b/n) for the unpaired treatment instead of sd(d)/sqrt(n) for the
paired one.  Because the two methods are looking at correlated data, var(d) is far
smaller than var_a + var_b, so the unpaired formula overstates the uncertainty and
suppresses the t statistic.

Both forms can be computed from the SAME draws, which is what this script does.  The
inflation factors quoted in the report's 9.6 come from here, so they can be checked
rather than taken on trust.

Writes outputs/ablation/paired_vs_unpaired.csv.

Usage: python paired_vs_unpaired.py [--repeats 300] [--seed 20260918]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from visualize_experiments import SWEEPS, paired_trial  # noqa: E402

DEFAULT_REPEATS = 300
DEFAULT_SEED = 20260918
OUT_DIR = ROOT / "outputs" / "ablation"


def collect(sweep: dict, value: float, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired draws for one configuration, keeping repeats where both succeeded."""

    config = dict(sweep["fixed"])
    config[sweep["key"]] = value
    rng = np.random.default_rng(
        [seed, int(config["point_count"]), int(round(config["noise"] * 100)), int(config["scale"])]
    )
    basic, normalized = [], []
    for _ in range(repeats):
        row = paired_trial(rng, config["point_count"], config["noise"], config["scale"])
        if np.isfinite(row["basic_dlt"]) and np.isfinite(row["normalized_dlt"]):
            basic.append(row["basic_dlt"])
            normalized.append(row["normalized_dlt"])
    return np.asarray(basic), np.asarray(normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sweep_key = next(item["key"] for item in SWEEPS if item["key"] == "point_count")
    parser.add_argument("--sweep", default=sweep_key, choices=[item["key"] for item in SWEEPS])
    args = parser.parse_args()

    sweep = next(item for item in SWEEPS if item["key"] == args.sweep)
    rows: list[dict] = []
    print(f"{'n':>4} {'有效次数':>8} {'配对 SE':>11} {'非配对 SE':>11} "
          f"{'SE 放大':>8} {'t 配对':>8} {'t 非配对':>9} {'|t| 缩小':>8}")
    for value in sweep["values"]:
        basic, normalized = collect(sweep, value, args.repeats, args.seed)
        n = len(basic)
        if n < 3:
            print(f"{int(value):>4} {n:>8}   有效样本太少，跳过")
            continue
        difference = normalized - basic
        paired_se = float(np.std(difference, ddof=1) / np.sqrt(n))
        unpaired_se = float(np.sqrt(np.var(basic, ddof=1) / n + np.var(normalized, ddof=1) / n))
        mean_diff = float(np.mean(difference))
        t_paired = mean_diff / paired_se if paired_se > 0 else float("nan")
        t_unpaired = mean_diff / unpaired_se if unpaired_se > 0 else float("nan")
        se_ratio = unpaired_se / paired_se if paired_se > 0 else float("inf")
        t_ratio = abs(t_paired / t_unpaired) if t_unpaired != 0 else float("inf")
        print(f"{int(value):>4} {n:>8} {paired_se:>11.3e} {unpaired_se:>11.3e} "
              f"{se_ratio:>8.1f} {t_paired:>8.2f} {t_unpaired:>9.2f} {t_ratio:>8.1f}")
        rows.append({"value": value, "n": n, "paired_se": paired_se, "unpaired_se": unpaired_se,
                     "se_ratio": se_ratio, "t_paired": t_paired, "t_unpaired": t_unpaired,
                     "t_ratio": t_ratio})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "paired_vs_unpaired.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
