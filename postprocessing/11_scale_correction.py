"""
Scale and affine correction of the predicted depths.

The scale factor is the median of GT / predicted depth over the matched reference
targets of a 03_validate.py comparison.csv. The affine fit uses the same targets, so
the transect rows of both are an independent check; a second affine fit uses every
matched point deeper than 0.3 m, transect points included. The summary's scale_ratio
is compute_metrics' median predicted / GT depth, the inverse of the scale factor.

Writes scale_correction_analysis.csv (every matched point with its corrected depths),
scale_correction_summary.csv, scale_correction_analysis.txt and
scale_correction_analysis.png to --output_dir. With --bathymetry_csv it also writes
the scaled bathymetry next to that CSV, as <csv stem>_scale_corrected.ply.

Usage example:
    # NeRFrac 50m, outputs next to comparison.csv
    python 11_scale_correction.py \\
        --comparison_csv validation/nerfrac_50m/comparison.csv \\
        --bathymetry_csv bathymetry/bathymetry_nerfrac_50m.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import compute_metrics, write_ply_binary

BATHYMETRY_COLUMNS = {"utm_easting", "utm_northing", "utm_elevation", "depth_m"}
SUMMARY_COLUMNS = [
    "label",
    "N",
    "pearson_r",
    "spearman_rho",
    "rmse",
    "mae",
    "bias",
    "scale_ratio",
]
STRATUM_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
MIN_AFFINE_POINTS = 3
AFFINE_ALL_MIN_DEPTH_M = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale and affine correction of predicted depths."
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="comparison.csv from 03_validate.py",
    )
    parser.add_argument(
        "--bathymetry_csv",
        type=str,
        default=None,
        help="gridded bathymetry CSV from 02_extract_bathymetry.py; the "
        "scale-corrected PLY is written next to it",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output folder (default: the folder of --comparison_csv)",
    )
    return parser.parse_args()


def load_bathymetry(path):
    if not os.path.isfile(path):
        print(f"ERROR: Bathymetry CSV not found: {path}")
        sys.exit(1)
    bathy = pd.read_csv(path)
    missing = BATHYMETRY_COLUMNS - set(bathy.columns)
    if missing:
        print(f"ERROR: Bathymetry CSV missing columns: {', '.join(sorted(missing))}")
        sys.exit(1)
    if bathy.empty:
        print(f"ERROR: Bathymetry CSV has no rows: {path}")
        sys.exit(1)
    return bathy


def n_targets(n):
    return f"{n} ref target" + ("" if n == 1 else "s")


def affine_fit(gt, pred):
    """Least-squares gt = a * pred + b, NaN with fewer than MIN_AFFINE_POINTS."""
    if len(pred) < MIN_AFFINE_POINTS:
        return np.nan, np.nan
    A = np.column_stack([pred, np.ones_like(pred)])
    a, b = np.linalg.lstsq(A, gt, rcond=None)[0]
    return a, b


def metric_rows(name, gt, pred, ref_mask):
    """Metrics of one correction on all points, the reference targets and the
    transect points."""
    return [
        compute_metrics(gt, pred, f"{name} (all)"),
        compute_metrics(gt[ref_mask], pred[ref_mask], f"{name} (ref targets)"),
        compute_metrics(gt[~ref_mask], pred[~ref_mask], f"{name} (transect)"),
    ]


def deep_affine_rows(gt, pred, ref_mask):
    """Metrics of an affine fit on every point deeper than AFFINE_ALL_MIN_DEPTH_M,
    transect points included; none with fewer than MIN_AFFINE_POINTS such points."""
    deep = gt > AFFINE_ALL_MIN_DEPTH_M
    if deep.sum() < MIN_AFFINE_POINTS:
        return []
    a, b = affine_fit(gt[deep], pred[deep])
    print(
        f"Affine fit on all points > {AFFINE_ALL_MIN_DEPTH_M} m (N={deep.sum()}): "
        f"corrected = {a:.4f} * pred + {b:.4f}"
    )
    label = f"Affine-all>{AFFINE_ALL_MIN_DEPTH_M}m"
    return metric_rows(label, gt, a * pred + b, ref_mask)


def print_metrics_table(rows):
    fmt = "{:<35s} {:>5s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s}"
    header = fmt.format(
        "Variant", "N", "Pearson", "Spearman", "RMSE", "MAE", "Bias", "Pred/GT"
    )
    print(header)
    print("-" * len(header))
    columns = [
        ("pearson_r", "{:.4f}"),
        ("spearman_rho", "{:.4f}"),
        ("rmse", "{:.3f}"),
        ("mae", "{:.3f}"),
        ("bias", "{:.3f}"),
        ("scale_ratio", "{:.2f}"),
    ]
    for r in rows:
        values = [
            "n/a" if np.isnan(r[key]) else spec.format(r[key]) for key, spec in columns
        ]
        print(fmt.format(r["label"], str(int(r["N"])), *values))


def write_report(path, comparison_csv, scale, n_scale, a, b, n_ref, key_rows):
    """The text summary: scale factor, affine fit, and RMSE and correlations of each
    correction on all points."""
    lines = ["SCALE / AFFINE CORRECTION ANALYSIS", f"Source: {comparison_csv}", ""]
    if np.isnan(scale):
        lines.append(
            "Scale factor: skipped, no reference target with a predicted depth"
        )
    else:
        lines += [
            f"Scale factor (median gt/pred on {n_targets(n_scale)}): {scale:.4f}",
            f"  => predicted / GT depth at the reference targets: {1 / scale:.2f}",
        ]
    lines.append("")
    if np.isnan(a):
        lines.append(
            f"Affine fit on ref targets: skipped, {n_targets(n_ref)} "
            f"(needs {MIN_AFFINE_POINTS})"
        )
    else:
        lines.append(f"Affine fit on ref targets: corrected = {a:.4f} * pred + {b:.4f}")
    lines += ["", "KEY COMPARISON:"]
    names = ["Uncorrected:", "Scale-corr:", "Affine-corr:"]
    skipped = [False, np.isnan(scale), np.isnan(a)]
    for name, m, skip in zip(names, key_rows, skipped):
        if skip:
            lines.append(f"  {name:13s} skipped")
        else:
            lines.append(
                f"  {name:13s} RMSE={m['rmse']:.3f}m, Pearson={m['pearson_r']:.4f}, "
                f"Spearman={m['spearman_rho']:.4f}"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def plot_corrections(path, gt, ref_mask, panels):
    """Predicted against GT depth, one panel per correction, reference targets as red
    stars."""
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 5))
    for ax, (pred, title) in zip(axes, panels):
        ax.scatter(
            gt[~ref_mask],
            pred[~ref_mask],
            s=15,
            alpha=0.5,
            c="tab:blue",
            label="transect",
        )
        ax.scatter(
            gt[ref_mask],
            pred[ref_mask],
            s=60,
            alpha=0.9,
            c="red",
            marker="*",
            label="ref targets",
            zorder=5,
        )
        lims = [min(gt.min(), pred.min()) - 0.1, max(gt.max(), pred.max()) + 0.1]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlabel("GT depth (m)")
        ax.set_ylabel("Predicted depth (m)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def write_scaled_ply(bathy, scale, bathymetry_csv):
    """The 02 bathymetry with scaled depths, coloured blue (shallow) to red (deep)."""
    depth = bathy["depth_m"].values * scale
    if "wsm_elevation" in bathy.columns:
        surface = bathy["wsm_elevation"].values
    else:
        surface = bathy["utm_elevation"].values + bathy["depth_m"].values
    points = np.column_stack(
        [bathy["utm_easting"].values, bathy["utm_northing"].values, surface - depth]
    )

    d_min, d_max = np.nanmin(depth), np.nanmax(depth)
    if d_max > d_min:
        norm = (depth - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(depth)
    colors = np.zeros((len(depth), 3), dtype=np.uint8)
    colors[:, 0] = (norm * 255).astype(np.uint8)
    colors[:, 2] = ((1 - norm) * 255).astype(np.uint8)

    stem = os.path.splitext(os.path.basename(bathymetry_csv))[0]
    path = os.path.join(os.path.dirname(bathymetry_csv), f"{stem}_scale_corrected.ply")
    write_ply_binary(path, points, colors=colors)
    print(f"Saved: {path} ({len(points):,} points, scale {scale:.4f})")


def main():
    args = parse_args()
    if not os.path.isfile(args.comparison_csv):
        print(f"ERROR: Comparison CSV not found: {args.comparison_csv}")
        sys.exit(1)
    bathy = load_bathymetry(args.bathymetry_csv) if args.bathymetry_csv else None

    df = pd.read_csv(args.comparison_csv)
    df = df[df["matched"] == "yes"]
    # 03 also marks GT points off the WSM as matched, without a GT depth
    finite = np.isfinite(df["gt_depth_m"]) & np.isfinite(df["pred_depth_m"])
    if not finite.all():
        print(
            f"Dropped {(~finite).sum()} matched points without a GT or predicted depth"
        )
    df = df[finite]
    print(f"Loaded {len(df)} matched points from {args.comparison_csv}")
    if df.empty:
        print("ERROR: No matched points")
        sys.exit(1)
    output_dir = args.output_dir or os.path.dirname(args.comparison_csv) or "."
    os.makedirs(output_dir, exist_ok=True)

    gt = df["gt_depth_m"].values
    pred = df["pred_depth_m"].values
    ref_mask = df["type"].values == "reference_target"
    n_ref = int(ref_mask.sum())

    scale_mask = ref_mask & (pred > 0)
    n_scale = int(scale_mask.sum())
    scale = np.median(gt[scale_mask] / pred[scale_mask]) if n_scale else np.nan
    scaled_pred = pred * scale
    if np.isnan(scale):
        print("Scale factor: skipped, no reference target with a predicted depth")
    else:
        print(f"Scale factor (median gt/pred on {n_targets(n_scale)}): {scale:.4f}")

    a, b = affine_fit(gt[ref_mask], pred[ref_mask])
    affine_pred = a * pred + b
    if np.isnan(a):
        print(
            f"Affine fit on ref targets: skipped, {n_targets(n_ref)} "
            f"(needs {MIN_AFFINE_POINTS})"
        )
    else:
        print(f"Affine fit on ref targets: corrected = {a:.4f} * pred + {b:.4f}")

    uncorrected = metric_rows("Uncorrected", gt, pred, ref_mask)
    scale_corrected = metric_rows("Scale-corrected", gt, scaled_pred, ref_mask)
    affine_corrected = metric_rows("Affine-corrected", gt, affine_pred, ref_mask)
    rows = uncorrected + scale_corrected + affine_corrected
    rows += deep_affine_rows(gt, pred, ref_mask)
    print()
    print_metrics_table(rows)

    print("\nDepth-stratified, scale-corrected:")
    print_metrics_table(
        [
            compute_metrics(
                gt[gt > t], scaled_pred[gt > t], f"Scale-corrected >{t:.1f}m"
            )
            for t in STRATUM_THRESHOLDS
        ]
    )

    csv_path = os.path.join(output_dir, "scale_correction_analysis.csv")
    df.assign(
        scale_factor=scale,
        scaled_pred_depth_m=scaled_pred,
        scaled_error_m=scaled_pred - gt,
        affine_a=a,
        affine_b=b,
        affine_pred_depth_m=affine_pred,
        affine_error_m=affine_pred - gt,
    ).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")
    summary_path = os.path.join(output_dir, "scale_correction_summary.csv")
    pd.DataFrame(rows)[SUMMARY_COLUMNS].to_csv(
        summary_path, index=False, float_format="%.4f"
    )
    print(f"Saved: {summary_path}")

    write_report(
        os.path.join(output_dir, "scale_correction_analysis.txt"),
        args.comparison_csv,
        scale,
        n_scale,
        a,
        b,
        n_ref,
        [uncorrected[0], scale_corrected[0], affine_corrected[0]],
    )
    if np.isnan(scale):
        scale_title = "Scale-corrected (skipped, no ref targets)"
    else:
        scale_title = f"Scale-corrected (x{scale:.3f})"
    if np.isnan(a):
        affine_title = f"Affine (skipped, fewer than {MIN_AFFINE_POINTS} ref targets)"
    else:
        affine_title = f"Affine ({a:.3f}*pred + {b:.3f})"
    plot_corrections(
        os.path.join(output_dir, "scale_correction_analysis.png"),
        gt,
        ref_mask,
        [
            (pred, "Uncorrected"),
            (scaled_pred, scale_title),
            (affine_pred, affine_title),
        ],
    )

    if bathy is not None and np.isnan(scale):
        print("No scale-corrected PLY: no reference target with a predicted depth")
    elif bathy is not None:
        write_scaled_ply(bathy, scale, args.bathymetry_csv)


if __name__ == "__main__":
    main()
