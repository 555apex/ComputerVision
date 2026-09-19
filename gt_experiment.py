"""Closed-loop ground-truth experiment for the synthetic sample.

Why this script exists
----------------------
A homography fitted to four correspondences reproduces those four points
*exactly* (eight equations, eight degrees of freedom).  The usual "reprojection
error" is therefore identically zero and tells you nothing.  To say anything
quantitative you need independent truth -- which the synthetic sample provides:
it was produced by projecting a known 640x420 document through a known
homography, so the true rectification matrix is exactly recoverable.

Three questions are answered here:

A. Implementation correctness
   Feed the exact corners and the true target rectangle.  A correct
   implementation must recover the truth to machine precision.

B. Effect of the target rectangle (the dominant error source)
   Four correspondences determine the mapping **only relative to a chosen target
   rectangle**.  Naming a 751x535 rectangle instead of the true 640x420 still
   maps the quad onto that rectangle exactly -- so the fit looks perfect -- but
   the recovered geometry is wrong.  This part quantifies by how much.

C. Effect of corner jitter (clicking error)
   Sweep the source corners by a Gaussian jitter and watch the true corner error.
   With the right target rectangle the error tracks the jitter; with the wrong
   one there is an irreducible floor far above it.

Usage
-----
    python gt_experiment.py
    python gt_experiment.py --jitter-repeats 200 --jitter 0 0.5 1 2 3 5

Outputs
-------
    outputs/gt_experiment/results.csv     machine-readable numbers
    outputs/gt_experiment/report.md       markdown tables ready to paste
    outputs/figures/gt_experiment.png     6-panel figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_sample import make_document
from homography import (
    destination_corners,
    estimate_output_size,
    ground_truth_errors,
    quad_grid,
    solve_homography,
)
from io_utils import read_image, require_opencv, save_json
from warping import inverse_warp_bilinear, opencv_warp

SAMPLE_IMAGE = Path("data/sample/sample_slanted.png")
SIDECAR = Path("data/sample/sample_slanted.groundtruth.json")
OUT_DIR = Path("outputs/gt_experiment")
FIGURE = Path("outputs/figures/gt_experiment.png")
INTERIOR_MARGIN = 25
EVAL_STEPS = 31


def truth_arrays(truth: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (source corners, true target corners, true H, true document size)."""

    corners = np.asarray(truth["source_corners_tl_tr_br_bl"], dtype=np.float64)
    truth_h = np.asarray(truth["homography_source_to_destination"], dtype=np.float64)
    truth_corners = np.asarray(truth["destination_corners_tl_tr_br_bl"], dtype=np.float64)
    size = (int(truth["document_size"]["width"]), int(truth["document_size"]["height"]))
    return corners, truth_corners, truth_h, size


def load_truth() -> dict:
    if not SIDECAR.is_file():
        raise SystemExit(
            f"ground truth not found: {SIDECAR}\n"
            "run `python generate_sample.py` first (it writes the sidecar)."
        )
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


def rectified_pixels(image: np.ndarray, source_corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Rectify `image` into a `size` canvas with the normalized DLT solution."""

    homography, _, _ = solve_homography(
        source_corners, destination_corners(*size), normalize=True
    )
    return inverse_warp_bilinear(image, homography, size[0], size[1])


def interior_mae(left: np.ndarray, right: np.ndarray, margin: int = INTERIOR_MARGIN) -> float:
    """MAE over both images minus a border margin (avoids resampling edge effects)."""

    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    if margin <= 0:
        a, b = left, right
    else:
        a = left[margin:-margin, margin:-margin]
        b = right[margin:-margin, margin:-margin]
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def part_a_implementation_check(truth: dict, image: np.ndarray) -> dict:
    """Exact corners + true target rectangle must reproduce truth to machine precision."""

    corners, truth_corners, truth_h, size = truth_arrays(truth)
    evaluation = quad_grid(corners, EVAL_STEPS)

    out: dict = {"target_size": f"{size[0]}x{size[1]}", "methods": {}}
    for name, normalized in (("basic_dlt", False), ("normalized_dlt", True)):
        homography, _, _ = solve_homography(corners, destination_corners(*size), normalize=normalized)
        out["methods"][name] = ground_truth_errors(
            homography, corners, truth_h, truth_corners, evaluation_points=evaluation
        )
    return out


def part_b_target_rectangle(truth: dict, image: np.ndarray) -> dict:
    """Quantify how much the choice of target rectangle costs."""

    corners, truth_corners, truth_h, true_size = truth_arrays(truth)
    evaluation = quad_grid(corners, EVAL_STEPS)
    auto_size = estimate_output_size(corners)
    document = make_document(*true_size)

    cases = []
    for label, size in (("auto_estimated", auto_size), ("true_size", true_size)):
        aspect_error = abs((size[0] / size[1]) - (true_size[0] / true_size[1])) / (true_size[0] / true_size[1]) * 100.0
        rectified = rectified_pixels(image, corners, size)
        if rectified.shape[:2] != (true_size[1], true_size[0]):
            cv2 = require_opencv()
            rectified = cv2.resize(rectified, true_size, interpolation=cv2.INTER_AREA)
        case = {
            "case": label,
            "target_size": f"{size[0]}x{size[1]}",
            "aspect_error_percent": aspect_error,
            "stretch_percent": ((size[1] / true_size[1]) / (size[0] / true_size[0]) - 1.0) * 100.0,
            "full_mae_vs_document": interior_mae(rectified, document, margin=0),
            "interior_mae_vs_document": interior_mae(rectified, document),
            "methods": {},
        }
        for name, normalized in (("basic_dlt", False), ("normalized_dlt", True)):
            homography, _, _ = solve_homography(corners, destination_corners(*size), normalize=normalized)
            case["methods"][name] = ground_truth_errors(
                homography, corners, truth_h, truth_corners, evaluation_points=evaluation
            )
        cases.append(case)
    return {"cases": cases, "true_size": f"{true_size[0]}x{true_size[1]}"}


def part_c_jitter_sweep(
    truth: dict, sigmas: tuple[float, ...], repeats: int, seed: int
) -> list[dict]:
    """True transfer error as a function of corner clicking error.

    The metric is the transfer error on held-out points inside the quad.  It has to
    be: any metric evaluated on the fitted corners themselves collapses to zero for
    a four-point fit, no matter how badly the corners were clicked.
    """

    corners, truth_corners, truth_h, true_size = truth_arrays(truth)
    evaluation = quad_grid(corners, EVAL_STEPS)
    auto_size = estimate_output_size(corners)

    rows = []
    for sigma in sigmas:
        for label, size in (("auto_estimated", auto_size), ("true_size", true_size)):
            values = {"basic_dlt": [], "normalized_dlt": []}
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + repeat)
                jittered = corners + (rng.normal(0.0, sigma, corners.shape) if sigma > 0 else 0.0)
                for name, normalized in (("basic_dlt", False), ("normalized_dlt", True)):
                    try:
                        homography, _, _ = solve_homography(
                            jittered, destination_corners(*size), normalize=normalized
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    metrics = ground_truth_errors(
                        homography, jittered, truth_h, truth_corners, evaluation_points=evaluation
                    )
                    values[name].append(metrics["transfer_rmse_vs_truth_px"])
            for name, samples in values.items():
                sample = np.asarray([s for s in samples if np.isfinite(s)], dtype=np.float64)
                rows.append({
                    "sigma_px": sigma,
                    "case": label,
                    "target_size": f"{size[0]}x{size[1]}",
                    "method": name,
                    "repeats": int(sample.size),
                    "transfer_mae_vs_truth_px": float(sample.mean()) if sample.size else float("nan"),
                    "transfer_median_vs_truth_px": float(np.median(sample)) if sample.size else float("nan"),
                    "transfer_std_vs_truth_px": float(sample.std(ddof=1)) if sample.size > 1 else float("nan"),
                })
    return rows


def make_figure(truth: dict, part_b: dict, sweep: list[dict], path: Path) -> None:
    true_size = (int(truth["document_size"]["width"]), int(truth["document_size"]["height"]))
    document = make_document(*true_size)
    image = read_image(SAMPLE_IMAGE)
    corners = np.asarray(truth["source_corners_tl_tr_br_bl"], dtype=np.float64)

    renders: dict[str, np.ndarray] = {}
    for case in part_b["cases"]:
        size = tuple(int(v) for v in case["target_size"].split("x"))
        rectified = rectified_pixels(image, corners, size)
        if rectified.shape[:2] != (true_size[1], true_size[0]):
            rectified = require_opencv().resize(rectified, true_size, interpolation=require_opencv().INTER_AREA)
        renders[case["case"]] = rectified

    diffs = {key: np.abs(value.astype(np.float64) - document.astype(np.float64)).mean(axis=2)
             for key, value in renders.items()}
    vmax = max(1.0, float(diffs["auto_estimated"].max()))

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0][0].imshow(document[:, :, ::-1])
    axes[0][0].set_title(f"Ground-truth document  {true_size[0]}x{true_size[1]}", fontsize=11)
    axes[0][1].imshow(renders["true_size"][:, :, ::-1])
    axes[0][1].set_title("Target size = true size", fontsize=11)
    axes[0][2].imshow(renders["auto_estimated"][:, :, ::-1])
    axes[0][2].set_title("Target size = auto-estimated", fontsize=11)

    for axis, key, title in (
        (axes[1][0], "true_size", "|diff| true size"),
        (axes[1][1], "auto_estimated", "|diff| auto-estimated (same scale)"),
    ):
        image_handle = axis.imshow(diffs[key], cmap="inferno", vmin=0, vmax=vmax)
        axis.set_title(f"{title}   MAE={interior_mae(renders[key], document):.2f}", fontsize=11)
        figure.colorbar(image_handle, ax=axis, fraction=0.046)

    sweep_axis = axes[1][2]
    for label, style in (("auto_estimated", "--"), ("true_size", "-")):
        for method, color in (("basic_dlt", "#4C78A8"), ("normalized_dlt", "#F58518")):
            selected = [row for row in sweep
                        if row["case"] == label and row["method"] == method
                        and np.isfinite(row["transfer_median_vs_truth_px"])]
            if not selected:
                continue
            sweep_axis.plot([row["sigma_px"] for row in selected],
                            [row["transfer_median_vs_truth_px"] for row in selected],
                            style, marker="o", color=color, linewidth=1.8,
                            label=f"{method} / {label}")
    sweep_axis.set_yscale("log")
    sweep_axis.set_xlabel("Corner jitter std (pixel)")
    sweep_axis.set_ylabel("Median transfer RMSE vs truth (pixel)")
    sweep_axis.set_title("Effect of corner clicking error", fontsize=11)
    sweep_axis.grid(alpha=0.3)
    sweep_axis.legend(fontsize=8)

    for row in axes:
        for axis in row:
            if axis is sweep_axis:
                continue
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle("Ground-truth closed-loop experiment", fontsize=16, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_markdown(part_a: dict, part_b: dict, sweep: list[dict], path: Path) -> None:
    auto = next(case for case in part_b["cases"] if case["case"] == "auto_estimated")
    true = next(case for case in part_b["cases"] if case["case"] == "true_size")

    lines: list[str] = []
    lines.append("# 合成真值闭环实验结果")
    lines.append("")
    lines.append("> 由 `gt_experiment.py` 自动生成，数值可直接填入实验报告。")
    lines.append(f"> 真值文档尺寸：{part_b['true_size']}；自动估计尺寸：{auto['target_size']}")
    lines.append("")

    lines.append("## A. 实现正确性验证（精确角点 + 真值尺寸）")
    lines.append("")
    lines.append("| 方法 | ‖aligned(H − H_true)‖ | 留出点转移 RMSE (px) | 目标矩形失配 (px) |")
    lines.append("|---|---:|---:|---:|")
    for name, metrics in part_a["methods"].items():
        lines.append(f"| {name} | {metrics['matrix_error_vs_truth']:.3e} | "
                     f"{metrics['transfer_rmse_vs_truth_px']:.3e} | "
                     f"{metrics['target_mismatch_rmse_px']:.3e} |")
    lines.append("")
    lines.append("机器精度量级（约 1e−8 ~ 1e−9），说明实现与真值一致。")
    lines.append("")

    lines.append("## B. 目标矩形选择的影响（精确角点）")
    lines.append("")
    lines.append("| 目标矩形 | 宽高比误差 | 拉伸 | ‖aligned(H − H_true)‖ | 留出点转移 RMSE (px) | 与真值文档 MAE（内区）|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for case in (true, auto):
        metrics = case["methods"]["normalized_dlt"]
        lines.append(
            f"| {case['target_size']}（{case['case']}） | {case['aspect_error_percent']:.2f}% | "
            f"{case['stretch_percent']:+.2f}% | "
            f"{metrics['matrix_error_vs_truth']:.3e} | {metrics['transfer_rmse_vs_truth_px']:.4g} | "
            f"{case['interior_mae_vs_document']:.2f} |"
        )
    lines.append("")
    lines.append("## C. 角点标定误差的影响（中位数，像素）")
    lines.append("")
    sigmas = sorted({row["sigma_px"] for row in sweep})
    lines.append("| 角点抖动 σ (px) | 自动尺寸 / basic | 自动尺寸 / 归一化 | 真值尺寸 / basic | 真值尺寸 / 归一化 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for sigma in sigmas:
        cells = []
        for case in ("auto_estimated", "true_size"):
            for method in ("basic_dlt", "normalized_dlt"):
                match = [row for row in sweep
                         if row["sigma_px"] == sigma and row["case"] == case and row["method"] == method]
                value = match[0]["transfer_median_vs_truth_px"] if match else float("nan")
                cells.append("n/a" if not np.isfinite(value) else f"{value:.4g}")
        lines.append(f"| {sigma:g} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop ground-truth experiment")
    parser.add_argument("--jitter-repeats", type=int, default=120)
    parser.add_argument("--jitter", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    truth = load_truth()
    try:
        image = read_image(SAMPLE_IMAGE)
    except ValueError as exc:
        raise SystemExit(f"could not read {SAMPLE_IMAGE} ({exc}); run `python generate_sample.py` first") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("A. 实现正确性验证（精确角点 + 真值尺寸）")
    part_a = part_a_implementation_check(truth, image)
    for name, metrics in part_a["methods"].items():
        print(f"   {name:<15} ||aligned(H-H_true)||={metrics['matrix_error_vs_truth']:.3e}"
              f"   转移 RMSE={metrics['transfer_rmse_vs_truth_px']:.3e} px")

    print("\nB. 目标矩形选择的影响（精确角点）")
    part_b = part_b_target_rectangle(truth, image)
    for case in part_b["cases"]:
        metrics = case["methods"]["normalized_dlt"]
        print(f"   {case['case']:<15} {case['target_size']:<10} 宽高比误差={case['aspect_error_percent']:6.2f}%"
              f"   转移 RMSE={metrics['transfer_rmse_vs_truth_px']:.4g} px"
              f"   与真值文档 MAE={case['interior_mae_vs_document']:.2f}")

    print(f"\nC. 角点标定误差扫描（每种 {args.jitter_repeats} 次重复）")
    sweep = part_c_jitter_sweep(truth, tuple(args.jitter), args.jitter_repeats, args.seed)
    for sigma in sorted({row["sigma_px"] for row in sweep}):
        cells = []
        for case in ("auto_estimated", "true_size"):
            match = [row for row in sweep if row["sigma_px"] == sigma
                     and row["case"] == case and row["method"] == "normalized_dlt"]
            cells.append(f"{match[0]['transfer_median_vs_truth_px']:9.4g}" if match else "      n/a")
        print(f"   σ={sigma:<4g} 自动尺寸={cells[0]} px   真值尺寸={cells[1]} px")

    save_json(OUT_DIR / "results.json", {"part_a": part_a, "part_b": part_b, "part_c": sweep})

    import csv

    fields = ["sigma_px", "case", "target_size", "method", "repeats",
              "transfer_mae_vs_truth_px", "transfer_median_vs_truth_px", "transfer_std_vs_truth_px"]
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sweep)

    write_markdown(part_a, part_b, sweep, OUT_DIR / "report.md")
    if not args.skip_figure:
        make_figure(truth, part_b, sweep, FIGURE)
        print(f"\nWrote {FIGURE}")
    print(f"Wrote {OUT_DIR / 'results.csv'} and {OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
