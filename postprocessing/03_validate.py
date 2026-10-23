"""
Validate bathymetric depth predictions against ground truth.

Usage examples:
    # Validate NeRFrac 50m
    python 03_validate.py \\
        --pred_csv bathymetry/bathymetry_nerfrac_50m.csv \\
        --wsm_path /path/to/WSM.tif \\
        --ref_csv /path/to/underwater_reference_targets.csv \\
        --transect_csv /path/to/underwater_transect_points.csv \\
        --label "NeRFrac 50m (950k)" \\
        --output_dir validation/nerfrac_50m

    # Validate VGGT 50m v1
    python 03_validate.py \\
        --pred_csv bathymetry/bathymetry_vggt_50m_v1.csv \\
        --wsm_path /path/to/WSM.tif \\
        --ref_csv /path/to/underwater_reference_targets.csv \\
        --transect_csv /path/to/underwater_transect_points.csv \\
        --label "VGGT 50m v1" \\
        --output_dir validation/vggt_50m_v1

    # Compare multiple models (run 03_validate.py once per model,
    # then use the stats JSON files for cross-model comparison)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

from coord_utils import sample_wsm_batch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate bathymetric depth predictions against ground truth."
    )

    # Predicted bathymetry
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="Path to predicted bathymetry CSV from 02_extract_bathymetry.py",
    )

    # Ground truth
    parser.add_argument(
        "--ref_csv",
        type=str,
        required=True,
        help="Path to underwater reference targets CSV (comma-separated)",
    )
    parser.add_argument(
        "--transect_csv",
        type=str,
        required=True,
        help="Path to underwater transect points CSV (comma-separated)",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="Path to Water Surface Model GeoTIFF",
    )

    # Matching
    parser.add_argument(
        "--max_dist",
        type=float,
        default=2.0,
        help="Maximum matching distance in meters (default: 2.0)",
    )

    # Output
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Human-readable model label for plots/reports (e.g. 'NeRFrac 50m')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation",
        help="Output directory for validation results (default: validation/)",
    )

    args = parser.parse_args()
    if not args.max_dist >= 0:
        parser.error("--max_dist must be 0 or more")
    return args


PRED_COLUMNS = {
    "easting": "utm_easting",
    "northing": "utm_northing",
    "elevation": "utm_elevation",
    "depth_m": "depth_m",
}


def load_predicted_grid(csv_path):
    """Load predicted bathymetry grid CSV from 02_extract_bathymetry.py."""
    data = np.genfromtxt(
        csv_path, delimiter=",", names=True, ndmin=1, encoding="utf-8-sig"
    )
    missing = [c for c in PRED_COLUMNS.values() if c not in (data.dtype.names or ())]
    if missing:
        print(f"ERROR: {csv_path} has no column {', '.join(missing)}")
        sys.exit(1)
    if len(data) == 0:
        print(f"ERROR: {csv_path} has no grid cells")
        sys.exit(1)
    result = {key: data[col].copy() for key, col in PRED_COLUMNS.items()}

    print(f"Loaded predicted grid: {len(data):,} cells from {csv_path}")
    print(f"  Easting:  [{result['easting'].min():.2f}, {result['easting'].max():.2f}]")
    print(
        f"  Northing: [{result['northing'].min():.2f}, {result['northing'].max():.2f}]"
    )
    print(
        f"  Depth:    [{result['depth_m'].min():.3f}, {result['depth_m'].max():.3f}] m"
    )
    print(f"  Depth median: {np.median(result['depth_m']):.3f} m")
    return result


def load_gt_reference_targets(csv_path):
    """Load underwater reference target points (U6-U13).

    Format: comma-separated, columns: Point, Easting[m], Northing[m],
    WGS84 Ellip. Height [m], Code

    Handles Unicode right-to-left mark in U12 elevation field.

    :return: list of dicts with keys: name, easting, northing, elevation, type
    """
    points = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader, None)
        for row in reader:
            if not any(field.strip() for field in row):
                continue
            if len(row) < 4:
                print(
                    f"ERROR: {csv_path} line {reader.line_num} has fewer than 4 columns"
                )
                sys.exit(1)
            name = row[0].strip()
            easting = float(row[1].strip())
            northing = float(row[2].strip())
            height_str = row[3].strip().encode("ascii", "ignore").decode()
            elevation = float(height_str)
            points.append(
                {
                    "name": name,
                    "easting": easting,
                    "northing": northing,
                    "elevation": elevation,
                    "type": "reference_target",
                }
            )
    print(f"Loaded {len(points)} reference target points from {csv_path}")
    return points


def load_gt_transect_points(csv_path):
    points = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader, None)
        for row in reader:
            if not any(field.strip() for field in row):
                continue
            if len(row) < 4:
                print(
                    f"ERROR: {csv_path} line {reader.line_num} has fewer than 4 columns"
                )
                sys.exit(1)
            name = row[0].strip()
            easting = float(row[1].strip())
            northing = float(row[2].strip())
            height_str = row[3].strip().encode("ascii", "ignore").decode()
            elevation = float(height_str)
            parts = name.rsplit("_", 1)
            transect = parts[0] if len(parts) == 2 else name
            points.append(
                {
                    "name": name,
                    "easting": easting,
                    "northing": northing,
                    "elevation": elevation,
                    "transect": transect,
                    "type": "transect",
                }
            )
    print(f"Loaded {len(points)} transect points from {csv_path}")
    transects = sorted(set(p["transect"] for p in points))
    print(f"  Transects: {transects}")
    return points


def compute_gt_depths(gt_points, wsm_path):
    eastings = np.array([p["easting"] for p in gt_points])
    northings = np.array([p["northing"] for p in gt_points])
    ws_elevations = sample_wsm_batch(wsm_path, eastings, northings)

    n_valid = 0
    for i, pt in enumerate(gt_points):
        ws = ws_elevations[i]
        pt["ws_elevation"] = float(ws) if np.isfinite(ws) else np.nan
        if np.isfinite(ws):
            pt["gt_depth_m"] = float(ws - pt["elevation"])
            n_valid += 1
        else:
            pt["gt_depth_m"] = np.nan

    print(f"  Computed GT depth for {n_valid}/{len(gt_points)} points")
    return gt_points



def match_gt_to_grid(gt_points, grid, max_dist_m=2.0):
    grid_e = grid["easting"]
    grid_n = grid["northing"]
    results = []

    for pt in gt_points:
        dx = grid_e - pt["easting"]
        dy = grid_n - pt["northing"]
        dists = np.sqrt(dx**2 + dy**2)
        i_min = np.argmin(dists)
        min_dist = float(dists[i_min])

        match = {
            "name": pt["name"],
            "type": pt.get("type", "unknown"),
            "transect": pt.get("transect", ""),
            "gt_easting": pt["easting"],
            "gt_northing": pt["northing"],
            "gt_elevation": pt["elevation"],
            "ws_elevation": pt.get("ws_elevation", np.nan),
            "gt_depth_m": pt.get("gt_depth_m", np.nan),
            "match_dist_m": min_dist,
        }

        if min_dist <= max_dist_m:
            match["matched"] = True
            match["pred_depth_m"] = float(grid["depth_m"][i_min])
            match["pred_easting"] = float(grid_e[i_min])
            match["pred_northing"] = float(grid_n[i_min])
            match["pred_elevation"] = float(grid["elevation"][i_min])
            if "depth_std" in grid:
                match["pred_depth_std"] = float(grid["depth_std"][i_min])
            if "n_observations" in grid:
                match["n_observations"] = int(grid["n_observations"][i_min])
        else:
            match["matched"] = False
            match["pred_depth_m"] = np.nan
            match["pred_easting"] = np.nan
            match["pred_northing"] = np.nan
            match["pred_elevation"] = np.nan

        results.append(match)

    n_matched = sum(1 for r in results if r["matched"])
    print(
        f"  Matched {n_matched}/{len(gt_points)} GT points (max_dist={max_dist_m:.1f}m)"
    )

    if n_matched < len(gt_points):
        unmatched = [r for r in results if not r["matched"]]
        dists_unmatched = [r["match_dist_m"] for r in unmatched]
        print(
            f"  Unmatched nearest distances: "
            f"min={min(dists_unmatched):.2f}m, "
            f"max={max(dists_unmatched):.2f}m"
        )

    return results


def compute_error_statistics(matches):
    valid = [
        m
        for m in matches
        if m["matched"]
        and np.isfinite(m.get("gt_depth_m", np.nan))
        and np.isfinite(m.get("pred_depth_m", np.nan))
    ]

    if len(valid) == 0:
        return {"n_valid": 0}

    gt = np.array([m["gt_depth_m"] for m in valid])
    pred = np.array([m["pred_depth_m"] for m in valid])
    errors = pred - gt
    abs_errors = np.abs(errors)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    # Pearson correlation
    if len(valid) >= 2 and np.std(gt) > 1e-12 and np.std(pred) > 1e-12:
        corr = float(np.corrcoef(gt, pred)[0, 1])
    else:
        corr = np.nan

    return {
        "n_valid": len(valid),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_errors)),
        "bias": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "r2": float(r2),
        "correlation": corr,
        "gt_mean": float(np.mean(gt)),
        "gt_std": float(np.std(gt)),
        "gt_min": float(np.min(gt)),
        "gt_max": float(np.max(gt)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "max_abs_error": float(np.max(abs_errors)),
        "p50_abs_error": float(np.median(abs_errors)),
        "p90_abs_error": float(np.percentile(abs_errors, 90)),
        "scale_factor": float(np.mean(pred) / np.mean(gt))
        if abs(np.mean(gt)) > 1e-12
        else np.nan,
    }


# Write csv
def save_comparison_csv(filepath, matches):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "type",
                "transect",
                "gt_easting",
                "gt_northing",
                "gt_elevation",
                "ws_elevation",
                "gt_depth_m",
                "pred_easting",
                "pred_northing",
                "pred_elevation",
                "pred_depth_m",
                "error_m",
                "match_dist_m",
                "matched",
            ]
        )
        for m in matches:
            gt_d = m.get("gt_depth_m", np.nan)
            pred_d = m.get("pred_depth_m", np.nan)
            error = (
                pred_d - gt_d if (np.isfinite(pred_d) and np.isfinite(gt_d)) else np.nan
            )

            def _fmt(v, prec=4):
                return f"{v:.{prec}f}" if np.isfinite(v) else ""

            writer.writerow(
                [
                    m["name"],
                    m["type"],
                    m.get("transect", ""),
                    _fmt(m["gt_easting"]),
                    _fmt(m["gt_northing"]),
                    _fmt(m["gt_elevation"]),
                    _fmt(m.get("ws_elevation", np.nan)),
                    _fmt(gt_d),
                    _fmt(m.get("pred_easting", np.nan)),
                    _fmt(m.get("pred_northing", np.nan)),
                    _fmt(m.get("pred_elevation", np.nan)),
                    _fmt(pred_d),
                    _fmt(error),
                    _fmt(m["match_dist_m"]),
                    "yes" if m["matched"] else "no",
                ]
            )
    print(f"  Saved comparison CSV: {filepath}")


def format_summary(label, stats_all, stats_ref, stats_trans, n_total, n_matched):
    """Format validation summary as text lines."""
    lines = []
    lines.append(f"Model: {label}")
    lines.append(f"GT points total: {n_total}")
    lines.append(f"  Matched: {n_matched}")
    lines.append(f"  Unmatched: {n_total - n_matched}")
    lines.append("")

    for section_label, stats in [
        ("ALL POINTS", stats_all),
        ("REFERENCE TARGETS ONLY", stats_ref),
        ("TRANSECT POINTS ONLY", stats_trans),
    ]:
        lines.append(f"--- {section_label} ---")
        if stats["n_valid"] == 0:
            lines.append("  No valid points")
            lines.append("")
            continue
        lines.append(f"  N valid:      {stats['n_valid']}")
        lines.append(f"  RMSE:         {stats['rmse']:.4f} m")
        lines.append(f"  MAE:          {stats['mae']:.4f} m")
        lines.append(f"  Bias:         {stats['bias']:+.4f} m")
        lines.append(f"  Std error:    {stats['std_error']:.4f} m")
        lines.append(f"  R2:           {stats['r2']:.4f}")
        lines.append(f"  Correlation:  {stats['correlation']:.4f}")
        lines.append(f"  Scale factor: {stats['scale_factor']:.3f}x")
        lines.append(f"  Max |error|:  {stats['max_abs_error']:.4f} m")
        lines.append(f"  P50 |error|:  {stats['p50_abs_error']:.4f} m")
        lines.append(f"  P90 |error|:  {stats['p90_abs_error']:.4f} m")
        lines.append(
            f"  GT depth:     mean={stats['gt_mean']:.3f}, "
            f"std={stats['gt_std']:.3f}, "
            f"range=[{stats['gt_min']:.3f}, {stats['gt_max']:.3f}] m"
        )
        lines.append(
            f"  Pred depth:   mean={stats['pred_mean']:.3f}, "
            f"std={stats['pred_std']:.3f}, "
            f"range=[{stats['pred_min']:.3f}, {stats['pred_max']:.3f}] m"
        )
        lines.append("")

    return lines


def plot_scatter(matches, stats, label, filepath):
    valid = [
        m
        for m in matches
        if m["matched"]
        and np.isfinite(m.get("gt_depth_m", np.nan))
        and np.isfinite(m.get("pred_depth_m", np.nan))
    ]
    if len(valid) == 0:
        print("  [WARN] No valid points for scatter plot")
        return

    gt = np.array([m["gt_depth_m"] for m in valid])
    pred = np.array([m["pred_depth_m"] for m in valid])
    types = [m["type"] for m in valid]

    fig, ax = plt.subplots(figsize=(8, 8))

    ref_mask = np.array([t == "reference_target" for t in types])
    trans_mask = ~ref_mask

    if np.any(trans_mask):
        ax.scatter(
            gt[trans_mask],
            pred[trans_mask],
            c="steelblue",
            alpha=0.5,
            s=20,
            label=f"Transect ({np.sum(trans_mask)})",
            edgecolors="none",
        )
    if np.any(ref_mask):
        ax.scatter(
            gt[ref_mask],
            pred[ref_mask],
            c="red",
            alpha=0.8,
            s=60,
            marker="^",
            label=f"Reference ({np.sum(ref_mask)})",
            edgecolors="darkred",
            linewidths=0.5,
        )

    all_vals = np.concatenate([gt, pred])
    lim_min = all_vals.min() - 0.1
    lim_max = all_vals.max() + 0.1
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        linewidth=1,
        alpha=0.5,
        label="1:1 line",
    )

    txt = (
        f"N = {stats['n_valid']}\n"
        f"RMSE = {stats['rmse']:.3f} m\n"
        f"MAE = {stats['mae']:.3f} m\n"
        f"Bias = {stats['bias']:+.3f} m\n"
        f"R2 = {stats['r2']:.3f}\n"
        f"r = {stats['correlation']:.3f}\n"
        f"Scale = {stats['scale_factor']:.2f}x"
    )
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Ground Truth Depth (m)", fontsize=12)
    ax.set_ylabel("Predicted Depth (m)", fontsize=12)
    ax.set_title(f"Bathymetry Validation -- {label}", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {filepath}")


def natural_key(name):
    return [int(t) if t.isdecimal() else t for t in re.split(r"(\d+)", name)]


def along_transect(points):
    """The points of a transect run in their order along its main direction.

    The survey order can jump back: QP15_7 lies between QP15_4 and QP15_5.
    """
    xy = np.array([[p["gt_easting"], p["gt_northing"]] for p in points])
    xy -= xy.mean(axis=0)
    position = xy @ np.linalg.svd(xy, full_matrices=False)[2][0]
    if position[-1] < position[0]:
        position = -position
    return [points[i] for i in np.argsort(position, kind="stable")]


def transect_panels(matches):
    """(title, points) of each transect, in natural order of the names.

    A transect name can label separate runs of rows (QP15 has two cross-sections),
    and each run gets its own panel.
    """
    runs = []
    for m in matches:
        if m["type"] == "transect" and m.get("transect"):
            if runs and runs[-1][0] == m["transect"]:
                runs[-1][1].append(m)
            else:
                runs.append((m["transect"], [m]))
    first_run = {}
    for i, (name, _) in enumerate(runs):
        first_run.setdefault(name, i)
    runs.sort(key=lambda run: (natural_key(run[0]), first_run[run[0]]))
    n_runs = Counter(name for name, _ in runs)
    seen = Counter()
    panels = []
    for name, points in runs:
        seen[name] += 1
        title = name if n_runs[name] == 1 else f"{name} ({seen[name]}/{n_runs[name]})"
        panels.append((title, along_transect(points)))
    return panels


def plot_transect_profiles(matches, label, filepath):
    panels = transect_panels(matches)
    if len(panels) == 0:
        print("  [WARN] No transect data for profiles")
        return

    n_trans = len(panels)

    ncols = min(4, n_trans)
    nrows = int(np.ceil(n_trans / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, (tname, pts) in enumerate(panels):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        eas = np.array([p["gt_easting"] for p in pts])
        nor = np.array([p["gt_northing"] for p in pts])
        seg_dist = np.sqrt(np.diff(eas) ** 2 + np.diff(nor) ** 2)
        cum_dist = np.concatenate([[0], np.cumsum(seg_dist)])

        gt_elev = np.array([p["gt_elevation"] for p in pts])
        ws_elev = np.array([p.get("ws_elevation", np.nan) for p in pts])
        pred_depth = np.array([p.get("pred_depth_m", np.nan) for p in pts])
        matched = np.array([p.get("matched", False) for p in pts])

        ax.plot(
            cum_dist, gt_elev, "k-o", markersize=3, linewidth=1.5, label="GT riverbed"
        )

        # Water surface
        ws_valid = np.isfinite(ws_elev)
        if np.any(ws_valid):
            ax.plot(
                cum_dist[ws_valid],
                ws_elev[ws_valid],
                "b--",
                linewidth=1.5,
                label="Water surface",
            )

        pred_valid = matched & np.isfinite(pred_depth) & np.isfinite(ws_elev)
        if np.any(pred_valid):
            pred_elev = ws_elev[pred_valid] - pred_depth[pred_valid]
            ax.plot(
                cum_dist[pred_valid],
                pred_elev,
                "r-s",
                markersize=3,
                linewidth=1.5,
                label="Predicted",
            )

        ax.set_title(tname, fontsize=11)
        ax.set_xlabel("Distance (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(n_trans, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Transect Profiles -- {label}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved transect profiles: {filepath}")


def plot_spatial_error_map(matches, grid, label, filepath):
    fig, ax = plt.subplots(figsize=(12, 10))

    sc = ax.scatter(
        grid["easting"],
        grid["northing"],
        c=grid["depth_m"],
        cmap="viridis",
        s=0.1,
        alpha=0.3,
        vmin=np.nanpercentile(grid["depth_m"], 1),
        vmax=np.nanpercentile(grid["depth_m"], 99),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Predicted Depth (m)", fontsize=11)

    valid = [
        m
        for m in matches
        if m["matched"]
        and np.isfinite(m.get("gt_depth_m", np.nan))
        and np.isfinite(m.get("pred_depth_m", np.nan))
    ]
    if valid:
        gt_e = np.array([m["gt_easting"] for m in valid])
        gt_n = np.array([m["gt_northing"] for m in valid])
        errors = np.array([m["pred_depth_m"] - m["gt_depth_m"] for m in valid])
        err_max = max(abs(errors.min()), abs(errors.max()), 0.5)
        sc2 = ax.scatter(
            gt_e,
            gt_n,
            c=errors,
            cmap="RdBu_r",
            s=30,
            edgecolors="black",
            linewidths=0.5,
            vmin=-err_max,
            vmax=err_max,
            zorder=5,
        )
        cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.5, pad=0.08)
        cbar2.set_label("Depth Error (pred - GT, m)", fontsize=10)

    unmatched = [m for m in matches if not m["matched"]]
    if unmatched:
        ax.scatter(
            [m["gt_easting"] for m in unmatched],
            [m["gt_northing"] for m in unmatched],
            c="gray",
            s=15,
            marker="x",
            alpha=0.5,
            label=f"Unmatched ({len(unmatched)})",
            zorder=4,
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("Easting (m, EPSG:25833)", fontsize=12)
    ax.set_ylabel("Northing (m, EPSG:25833)", fontsize=12)
    ax.set_title(f"Spatial Error Map -- {label}", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved spatial error map: {filepath}")


def plot_error_histogram(matches, stats, label, filepath):
    valid = [
        m
        for m in matches
        if m["matched"]
        and np.isfinite(m.get("gt_depth_m", np.nan))
        and np.isfinite(m.get("pred_depth_m", np.nan))
    ]
    if len(valid) < 3:
        print("  [WARN] Too few points for error histogram")
        return

    errors = np.array([m["pred_depth_m"] - m["gt_depth_m"] for m in valid])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(
        stats["bias"],
        color="red",
        linewidth=1.5,
        linestyle="-",
        label=f"Bias = {stats['bias']:+.3f} m",
    )

    ax.set_xlabel("Depth Error (pred - GT, m)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Error Distribution -- {label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    txt = (
        f"N = {stats['n_valid']}\n"
        f"RMSE = {stats['rmse']:.3f} m\n"
        f"MAE = {stats['mae']:.3f} m\n"
        f"Std = {stats['std_error']:.3f} m"
    )
    ax.text(
        0.95,
        0.95,
        txt,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved error histogram: {filepath}")


def main():
    args = parse_args()

    label = args.label or os.path.splitext(os.path.basename(args.pred_csv))[0]

    print("=" * 50)
    print(f"BATHYMETRY VALIDATION -- {label}")
    print("=" * 70)
    print(f"Predictions: {args.pred_csv}")
    print(f"WSM:         {args.wsm_path}")
    print(f"Max dist:    {args.max_dist} m")
    print(f"Output:      {args.output_dir}")
    print()

    for flag, path in (
        ("--pred_csv", args.pred_csv),
        ("--ref_csv", args.ref_csv),
        ("--transect_csv", args.transect_csv),
        ("--wsm_path", args.wsm_path),
    ):
        if not os.path.isfile(path):
            print(f"ERROR: {flag} not found: {path}")
            sys.exit(1)

    print("---Load predicted bathymetry ---")
    grid = load_predicted_grid(args.pred_csv)
    print()

    print("---Load ground truth ---")
    ref_points = load_gt_reference_targets(args.ref_csv)
    trans_points = load_gt_transect_points(args.transect_csv)
    all_gt = ref_points + trans_points
    print(f"Total GT points: {len(all_gt)}")

    print("\nComputing GT depths from WSM...")
    all_gt = compute_gt_depths(all_gt, args.wsm_path)

    gt_depths_valid = [
        p["gt_depth_m"] for p in all_gt if np.isfinite(p.get("gt_depth_m", np.nan))
    ]
    if gt_depths_valid:
        gt_arr = np.array(gt_depths_valid)
        print(f"\nGT depth summary ({len(gt_arr)} points):")
        print(f"  Mean:   {np.mean(gt_arr):.3f} m")
        print(f"  Median: {np.median(gt_arr):.3f} m")
        print(f"  Std:    {np.std(gt_arr):.3f} m")
        print(f"  Range:  [{np.min(gt_arr):.3f}, {np.max(gt_arr):.3f}] m")

    print("\nReference target depths:")
    for p in ref_points:
        ws = p.get("ws_elevation", np.nan)
        d = p.get("gt_depth_m", np.nan)
        print(
            f"  {p['name']:12s}  bed={p['elevation']:.3f}  ws={ws:.3f}  depth={d:.3f} m"
        )
    print()

    print("---Match GT to predictions ---")
    matches = match_gt_to_grid(all_gt, grid, max_dist_m=args.max_dist)
    print()

    print("---Compute error statistics ---")

    ref_matches = [m for m in matches if m["type"] == "reference_target"]
    trans_matches = [m for m in matches if m["type"] == "transect"]

    stats_all = compute_error_statistics(matches)
    stats_ref = compute_error_statistics(ref_matches)
    stats_trans = compute_error_statistics(trans_matches)

    n_matched = sum(1 for m in matches if m["matched"])

    summary_lines = format_summary(
        label,
        stats_all,
        stats_ref,
        stats_trans,
        len(matches),
        n_matched,
    )
    for line in summary_lines:
        print(line)

    print("---Save outputs ---")
    os.makedirs(args.output_dir, exist_ok=True)

    save_comparison_csv(os.path.join(args.output_dir, "comparison.csv"), matches)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Saved summary: {summary_path}")

    stats_json = {
        "model_label": label,
        "pred_csv": os.path.abspath(args.pred_csv),
        "n_gt_total": len(matches),
        "n_matched": n_matched,
        "n_unmatched": len(matches) - n_matched,
        "max_match_dist_m": args.max_dist,
        "stats_all": stats_all,
        "stats_reference_targets": stats_ref,
        "stats_transect_points": stats_trans,
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"  Saved stats JSON: {stats_path}")

    print("\n--- Step 6: Generate plots ---")

    plot_scatter(
        matches,
        stats_all,
        label,
        os.path.join(args.output_dir, "scatter.png"),
    )
    plot_transect_profiles(
        matches,
        label,
        os.path.join(args.output_dir, "transects.png"),
    )
    plot_spatial_error_map(
        matches,
        grid,
        label,
        os.path.join(args.output_dir, "spatial_error.png"),
    )
    plot_error_histogram(
        matches,
        stats_all,
        label,
        os.path.join(args.output_dir, "error_histogram.png"),
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
