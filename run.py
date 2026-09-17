"""Command-line entry point for the projective rectification experiment.

Metrics reported per method, and why there are several of them
--------------------------------------------------------------

``fit_residual_*``
    How well the fitted homography explains the very points it was fitted from.
    **With four correspondences this is exactly zero by construction** (eight
    equations for eight degrees of freedom), so it degenerates to floating-point
    noise (~1e-9..1e-13) and must not be read as an error.  It is kept only
    because it is informative when n > 4 (redundant constraints).

``matrix_error_vs_truth`` / ``corner_*_vs_truth_px``
    Measured against an independent ground truth.  Present only when a
    ``<name>.groundtruth.json`` sidecar sits next to the image (see
    ``generate_sample.py``).  These are the numbers worth putting in a report.

``aspect_error_percent``
    How far the chosen output rectangle's aspect ratio is from the true one.
    A numerically perfect homography can still be *metrically* wrong: four point
    correspondences only determine the mapping up to the choice of target
    rectangle.  Requires ground truth.

``solve_ms`` / ``warp_ms``
    Timed separately and reported as a median over repeats, because the solver
    costs ~0.4 ms while the hand-written resampling costs ~85 ms at 0.6 MP.
    Summing them (as an earlier version did) measures image resampling speed,
    not algorithm speed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from homography import (
    aligned_matrix_error,
    destination_corners,
    estimate_output_size,
    ground_truth_errors,
    quad_grid,
    reprojection_errors,
    solve_homography,
)
from io_utils import (
    format_number,
    iter_images,
    load_ground_truth,
    require_opencv,
    save_image,
    save_json,
    save_summary,
)
from point_picker import select_four_corners
from validation import validate_corners
from warping import inverse_warp_bilinear, opencv_warp

SOLVE_REPEATS = 20
DEFAULT_WARP_REPEATS = 3

T = TypeVar("T")


def _timed(action: Callable[[], T], repeats: int) -> tuple[T, float, float]:
    """Run `action` `repeats` times; return (last result, median ms, std ms)."""

    repeats = max(1, int(repeats))
    durations: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = action()
        durations.append((time.perf_counter() - started) * 1000.0)
    assert result is not None
    values = np.asarray(durations, dtype=np.float64)
    return result, float(np.median(values)), float(np.std(values))


def _parse_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(payload, dtype=np.float64)
    valid, message = validate_corners(points)
    if not valid:
        raise ValueError(f"invalid points in {path}: {message}")
    return points


def _points_from_ground_truth(image_path: Path) -> np.ndarray:
    """Read the true source corners out of the image's ground-truth sidecar.

    Convenience for reproducible runs on the synthetic sample: it guarantees the
    corners used match the ones the truth was defined with, instead of duplicating
    the constants in a separate JSON file.
    """

    payload = load_ground_truth(image_path)
    if payload is None:
        raise ValueError(
            f"--points-from-ground-truth requires {image_path.with_suffix('.groundtruth.json')}"
        )
    points = np.asarray(payload["source_corners_tl_tr_br_bl"], dtype=np.float64)
    valid, message = validate_corners(points)
    if not valid:
        raise ValueError(f"ground truth contains invalid corners: {message}")
    return points


def _source_label(path: Path, input_root: Path | None) -> str:
    if input_root is not None:
        try:
            relative = path.resolve().relative_to(input_root.resolve())
            return relative.parts[0] if len(relative.parts) > 1 else "unclassified"
        except ValueError:
            pass
    return "single"


def _resolve_output_size(
    source_points: np.ndarray,
    ground_truth: dict | None,
    requested: tuple[int, int] | None,
) -> tuple[int, int, str]:
    if requested is not None:
        width, height = int(requested[0]), int(requested[1])
        if width < 2 or height < 2:
            raise ValueError(f"--output-size must be at least 2x2, got {width}x{height}")
        return width, height, "cli"
    width, height = estimate_output_size(source_points)
    return width, height, "estimated"


def _truth_document_size(ground_truth: dict | None) -> tuple[int, int] | None:
    if ground_truth is None:
        return None
    size = ground_truth.get("document_size")
    if isinstance(size, dict) and "width" in size and "height" in size:
        return int(size["width"]), int(size["height"])
    corners = np.asarray(ground_truth.get("destination_corners_tl_tr_br_bl", []), dtype=np.float64)
    if corners.shape == (4, 2):
        return int(round(corners[:, 0].max())) + 1, int(round(corners[:, 1].max())) + 1
    return None


def _aspect_error_percent(width: int, height: int, truth: tuple[int, int] | None) -> float:
    if truth is None:
        return float("nan")
    true_width, true_height = truth
    true_aspect = true_width / true_height
    return abs((width / height) - true_aspect) / true_aspect * 100.0


def process_image(
    path: Path,
    output_root: Path,
    *,
    input_root: Path | None = None,
    fixed_points: np.ndarray | None = None,
    output_size: tuple[int, int] | None = None,
    warp_repeats: int = DEFAULT_WARP_REPEATS,
    use_ground_truth: bool = True,
) -> list[dict]:
    cv2 = require_opencv()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    source_points = fixed_points if fixed_points is not None else select_four_corners(image)
    valid, message = validate_corners(source_points)
    if not valid:
        raise ValueError(message)

    ground_truth = load_ground_truth(path) if use_ground_truth else None
    truth_homography: np.ndarray | None = None
    truth_corners: np.ndarray | None = None
    if ground_truth is not None:
        truth_homography = np.asarray(ground_truth["homography_source_to_destination"], dtype=np.float64)
        truth_corners = np.asarray(ground_truth["destination_corners_tl_tr_br_bl"], dtype=np.float64)
    truth_size = _truth_document_size(ground_truth)
    # Held-out evaluation points: a bilinear grid inside the source quad.  Using the
    # fitted corners themselves would make the comparison degenerate (a four-point
    # fit reproduces its own inputs exactly).
    evaluation_points = quad_grid(source_points, steps=25)

    width, height, size_source = _resolve_output_size(source_points, ground_truth, output_size)
    target_points = destination_corners(width, height)
    result_dir = output_root / _source_label(path, input_root) / path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, dict] = {}
    rectified_by_method: dict[str, np.ndarray] = {}

    for name, normalized in (("basic_dlt", False), ("normalized_dlt", True)):
        solve = lambda normalized=normalized: solve_homography(  # noqa: E731
            source_points, target_points, normalize=normalized
        )
        (homography, matrix, singular_values), solve_ms, solve_std = _timed(solve, SOLVE_REPEATS)
        rectified, warp_ms, warp_std = _timed(
            lambda: inverse_warp_bilinear(image, homography, width, height), warp_repeats
        )
        rectified_by_method[name] = rectified

        residual = reprojection_errors(homography, source_points, target_points)
        details: dict[str, object] = {
            "H": homography.tolist(),
            "singular_values": singular_values.tolist(),
            "dlt_matrix_shape": list(matrix.shape),
            "fit_residual_rmse_px": float(np.sqrt(np.mean(residual ** 2))),
            "fit_residual_max_px": float(np.max(residual)),
            "solve_ms": solve_ms,
            "solve_ms_std": solve_std,
            "warp_ms": warp_ms,
            "warp_ms_std": warp_std,
        }
        if truth_homography is not None and truth_corners is not None:
            details.update(
                ground_truth_errors(
                    homography, source_points, truth_homography, truth_corners,
                    evaluation_points=evaluation_points,
                )
            )
        methods[name] = details

        save_image(result_dir / f"rectified_{name}.png", rectified)
        np.save(result_dir / f"pixels_{name}.npy", rectified)

    opencv_error: str | None = None
    try:
        opencv_solve = lambda: cv2.getPerspectiveTransform(  # noqa: E731
            source_points.astype(np.float32), target_points.astype(np.float32)
        )
        opencv_h, solve_ms, solve_std = _timed(opencv_solve, SOLVE_REPEATS)
        opencv_rectified, warp_ms, warp_std = _timed(
            lambda: opencv_warp(image, opencv_h, width, height), warp_repeats
        )
        residual = reprojection_errors(opencv_h, source_points, target_points)
        details = {
            "H": opencv_h.tolist(),
            "fit_residual_rmse_px": float(np.sqrt(np.mean(residual ** 2))),
            "fit_residual_max_px": float(np.max(residual)),
            "solve_ms": solve_ms,
            "solve_ms_std": solve_std,
            "warp_ms": warp_ms,
            "warp_ms_std": warp_std,
        }
        if truth_homography is not None and truth_corners is not None:
            details.update(
                ground_truth_errors(
                    opencv_h, source_points, truth_homography, truth_corners,
                    evaluation_points=evaluation_points,
                )
            )
        methods["opencv"] = details

        save_image(result_dir / "rectified_opencv.png", opencv_rectified)
        np.save(result_dir / "pixels_opencv.npy", opencv_rectified)

        # Reuse the already-rendered normalized-DLT result instead of warping again.
        for name in ("basic_dlt", "normalized_dlt"):
            ours = rectified_by_method[name].astype(np.float64)
            their = opencv_rectified.astype(np.float64)
            methods[name]["image_mae_vs_opencv"] = float(np.mean(np.abs(ours - their)))
        methods["opencv"]["normalized_dlt_matrix_error"] = aligned_matrix_error(
            np.asarray(methods["normalized_dlt"]["H"], dtype=np.float64), opencv_h
        )
    except (ImportError, RuntimeError) as exc:
        opencv_error = str(exc)

    aspect_error = _aspect_error_percent(width, height, truth_size)
    metadata = {
        "input": str(path.resolve()),
        "source": _source_label(path, input_root),
        "input_resolution": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "output_resolution": {"width": width, "height": height},
        "output_size_source": size_source,
        "aspect_error_percent": None if not np.isfinite(aspect_error) else aspect_error,
        "source_corners_tl_tr_br_bl": source_points.tolist(),
        "destination_corners_tl_tr_br_bl": target_points.tolist(),
        "ground_truth": {
            "available": ground_truth is not None,
            "document_size": None if truth_size is None else {"width": truth_size[0], "height": truth_size[1]},
            "sidecar": str(path.with_suffix(".groundtruth.json")) if ground_truth is not None else None,
        },
        "timing": {
            "solve_repeats": SOLVE_REPEATS,
            "warp_repeats": max(1, int(warp_repeats)),
            "note": "medians; fit_residual_* is a fit residual, not an error",
        },
        "methods": methods,
        "opencv_error": opencv_error,
    }
    save_json(result_dir / "metadata.json", metadata)

    rows = []
    for method, details in methods.items():
        rows.append({
            "image": path.name,
            "source": metadata["source"],
            "method": method,
            "output_width": width,
            "output_height": height,
            "target_size_source": size_source,
            "fit_residual_rmse_px": details.get("fit_residual_rmse_px", ""),
            "fit_residual_max_px": details.get("fit_residual_max_px", ""),
            "matrix_error_vs_truth": details.get("matrix_error_vs_truth", ""),
            "transfer_rmse_vs_truth_px": details.get("transfer_rmse_vs_truth_px", ""),
            "transfer_max_vs_truth_px": details.get("transfer_max_vs_truth_px", ""),
            "target_mismatch_rmse_px": details.get("target_mismatch_rmse_px", ""),
            "aspect_error_percent": aspect_error if np.isfinite(aspect_error) else "",
            "solve_ms": details.get("solve_ms", ""),
            "solve_ms_std": details.get("solve_ms_std", ""),
            "warp_ms": details.get("warp_ms", ""),
            "warp_ms_std": details.get("warp_ms_std", ""),
            "image_mae_vs_opencv": details.get("image_mae_vs_opencv", ""),
        })

    _print_report(path, width, height, size_source, aspect_error, truth_size, methods)
    return rows


def _print_report(
    path: Path,
    width: int,
    height: int,
    size_source: str,
    aspect_error: float,
    truth_size: tuple[int, int] | None,
    methods: dict[str, dict],
) -> None:
    header = f"{path.name}: {width}x{height} ({size_source})"
    if truth_size is not None:
        header += f"   truth {truth_size[0]}x{truth_size[1]}   aspect error {aspect_error:.2f}%"
    print(header)

    has_truth = all("transfer_rmse_vs_truth_px" in details for details in methods.values())
    if has_truth:
        columns = [("method", 16), ("gt_transfer", 12), ("rect_mismatch", 14),
                   ("fit_residual", 13), ("solve_ms", 10), ("warp_ms", 9), ("mae_vs_cv", 10)]
    else:
        columns = [("method", 16), ("fit_residual", 15), ("solve_ms", 10),
                   ("warp_ms", 9), ("mae_vs_cv", 10)]

    print("    " + "".join(name.ljust(width_) if index == 0 else name.rjust(width_)
                           for index, (name, width_) in enumerate(columns)))
    for name, details in methods.items():
        values = [name]
        if has_truth:
            values += [
                format_number(details.get("transfer_rmse_vs_truth_px")),
                format_number(details.get("target_mismatch_rmse_px")),
            ]
        values.append(format_number(details.get("fit_residual_rmse_px")))
        values += [
            format_number(details.get("solve_ms")),
            format_number(details.get("warp_ms")),
            format_number(details.get("image_mae_vs_opencv")),
        ]
        print("    " + "".join(value.ljust(width_) if index == 0 else value.rjust(width_)
                               for index, (value, (_, width_)) in enumerate(zip(values, columns))))

    if not has_truth:
        print("    (no ground-truth sidecar -- gt_* metrics unavailable; see generate_sample.py)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Projective rectification with basic/normalized DLT and OpenCV comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="one input image")
    group.add_argument("--input-dir", type=Path, help="directory containing image files")
    parser.add_argument("--batch", action="store_true", help="process all images under --input-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--points-file", type=Path, help="JSON file containing four corners for single-image non-interactive runs")
    parser.add_argument(
        "--output-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help=("fix the target rectangle instead of estimating it, e.g. `--output-size 640 420`. "
              "Four point correspondences determine the mapping only relative to the target rectangle, "
              "so supplying the true size removes the aspect-ratio bias."),
    )
    parser.add_argument(
        "--warp-repeats", type=int, default=DEFAULT_WARP_REPEATS,
        help=f"repeats for timing the image warp (default {DEFAULT_WARP_REPEATS}); "
             f"the solver always uses {SOLVE_REPEATS}. Lower this for very large images.",
    )
    parser.add_argument(
        "--points-from-ground-truth", action="store_true",
        help=("take the source corners from the image's <name>.groundtruth.json sidecar, "
              "for reproducible runs on images that ship with truth"),
    )
    parser.add_argument(
        "--ignore-ground-truth", action="store_true",
        help="do not read <name>.groundtruth.json sidecars even if present",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch and args.input_dir is None:
        raise SystemExit("--batch requires --input-dir")
    if args.points_file is not None and args.input is None:
        raise SystemExit("--points-file is only supported with --input")
    if args.points_from_ground_truth and args.input is None:
        raise SystemExit("--points-from-ground-truth is only supported with --input")

    output_size = tuple(args.output_size) if args.output_size is not None else None
    fixed_points = None
    if args.points_from_ground_truth:
        fixed_points = _points_from_ground_truth(args.input)
    elif args.points_file is not None:
        fixed_points = _parse_points(args.points_file)

    rows: list[dict] = []
    if args.input is not None:
        rows.extend(process_image(
            args.input, args.output_dir,
            fixed_points=fixed_points,
            output_size=output_size,
            warp_repeats=args.warp_repeats,
            use_ground_truth=not args.ignore_ground_truth,
        ))
    else:
        images = iter_images(args.input_dir)
        if not images:
            raise SystemExit(f"no supported images found under {args.input_dir}")
        skipped = 0
        for image_path in images:
            try:
                rows.extend(process_image(
                    image_path, args.output_dir,
                    input_root=args.input_dir,
                    output_size=output_size,
                    warp_repeats=args.warp_repeats,
                    use_ground_truth=not args.ignore_ground_truth,
                ))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                skipped += 1
                print(f"SKIP {image_path}: {exc}")
        if skipped:
            print(f"WARNING: skipped {skipped} of {len(images)} image(s)")
    save_summary(args.output_dir / "summary.csv", rows)
    print(f"Saved {len(rows)} method rows to {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
