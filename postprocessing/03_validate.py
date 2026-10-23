"""
Validate predicted depths against the underwater survey of the Pielach data: the
reference targets (U6-U13) and the transect points (QP*).

The GT depth of a point is the WSM elevation at the point minus its surveyed bed
elevation. Each GT point takes the depth of the nearest cell of the 02 grid within
--max_dist. Writes comparison.csv (every GT point with its cell, if matched),
summary.txt and stats.json (statistics for all points, the reference targets and the
transect points) and up to four plots to --output_dir. 10, 11 and 14 read
comparison.csv.

Usage examples:
    # NeRFrac 50m
    python 03_validate.py \\
        --pred_csv bathymetry/bathymetry_nerfrac_50m.csv \\
        --ref_csv ../data/gt/pielach/07_underwater_reference_targets/Pielach_20241024_underwater_reference_targets.csv \\
        --transect_csv ../data/gt/pielach/08_underwater_transect_points/Pielach_20241024_underwater_transect_points.csv \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --label "NeRFrac 50m" \\
        --output_dir validation/nerfrac_50m

    # VGGT 50m
    python 03_validate.py \\
        --pred_csv bathymetry/bathymetry_vggt_50m.csv \\
        --ref_csv ../data/gt/pielach/07_underwater_reference_targets/Pielach_20241024_underwater_reference_targets.csv \\
        --transect_csv ../data/gt/pielach/08_underwater_transect_points/Pielach_20241024_underwater_transect_points.csv \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --label "VGGT 50m" \\
        --output_dir validation/vggt_50m
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import sample_wsm_batch

# grid key, the suffix of a pred_* column of comparison.csv -> column of the 02 CSV
PRED_COLUMNS = {
    "depth_m": "depth_m",
    "easting": "utm_easting",
    "northing": "utm_northing",
    "elevation": "utm_elevation",
}
COMPARISON_COLUMNS = [
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
MIN_HISTOGRAM_POINTS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate predicted depths against the Pielach underwater survey."
    )

    # prediction
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="gridded bathymetry CSV from 02_extract_bathymetry.py",
    )

    # ground truth
    parser.add_argument(
        "--ref_csv",
        type=str,
        required=True,
        help="underwater reference targets: a header line, then name, easting, "
        "northing and height in the first four columns",
    )
    parser.add_argument(
        "--transect_csv",
        type=str,
        required=True,
        help="underwater transect points, same columns; the transect is the name up "
        "to the last underscore (QP15 for QP15_3)",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF; GT depth = WSM elevation - bed elevation",
    )

    # matching
    parser.add_argument(
        "--max_dist",
        type=float,
        default=2.0,
        help="largest distance in m from a GT point to its grid cell (default: 2.0)",
    )

    # output
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="model name for the plots, the summary and stats.json (default: the file "
        "name of --pred_csv without its extension)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation",
        help="output folder (default: validation)",
    )

    args = parser.parse_args()
    if not args.max_dist >= 0:
        parser.error("--max_dist must be 0 or more")
    return args


def natural_key(name):
    return [int(t) if t.isdecimal() else t for t in re.split(r"(\d+)", name)]


def cumulative_distance(easting, northing):
    return np.concatenate(
        [[0], np.cumsum(np.sqrt(np.diff(easting) ** 2 + np.diff(northing) ** 2))]
    )


def load_predicted_grid(csv_path):
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
    grid = {key: data[col] for key, col in PRED_COLUMNS.items()}

    depth = grid["depth_m"]
    print(
        f"Predicted grid: {len(data):,} cells, depth {np.nanmin(depth):.3f} to "
        f"{np.nanmax(depth):.3f} m (median {np.nanmedian(depth):.3f} m)"
    )
    return grid


def read_survey_csv(csv_path):
    """(name, easting, northing, elevation) per row of a Pielach survey CSV.

    The U12 height in the reference targets ends in a stray U+202C (pop directional
    formatting), so non-ASCII characters are dropped from the heights.
    """
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not any(field.strip() for field in row):
                continue
            if len(row) < 4:
                print(
                    f"ERROR: {csv_path} line {reader.line_num} has fewer than 4 columns"
                )
                sys.exit(1)
            name, easting, northing = (field.strip() for field in row[:3])
            height = row[3].strip().encode("ascii", "ignore").decode()
            rows.append((name, float(easting), float(northing), float(height)))
    return rows


def load_gt_points(ref_csv, transect_csv):
    points = []
    for path, point_type in ((ref_csv, "reference_target"), (transect_csv, "transect")):
        for name, easting, northing, elevation in read_survey_csv(path):
            # transect points are named <transect>_<number>, e.g. QP15_3
            transect = name.rsplit("_", 1)[0] if point_type == "transect" else ""
            points.append(
                {
                    "name": name,
                    "type": point_type,
                    "transect": transect,
                    "gt_easting": easting,
                    "gt_northing": northing,
                    "gt_elevation": elevation,
                }
            )

    n_ref = sum(p["type"] == "reference_target" for p in points)
    transects = sorted(
        {p["transect"] for p in points if p["transect"]}, key=natural_key
    )
    names = f" ({', '.join(transects)})" if transects else ""
    print(
        f"GT: {n_ref} reference targets, {len(points) - n_ref} transect points{names}"
    )
    return points


def add_gt_depths(gt_points, wsm_path):
    """GT depth = WSM elevation at the point - bed elevation (NaN off the WSM)."""
    ws_elevations = sample_wsm_batch(
        wsm_path,
        [p["gt_easting"] for p in gt_points],
        [p["gt_northing"] for p in gt_points],
    )
    for pt, ws in zip(gt_points, ws_elevations):
        pt["ws_elevation"] = float(ws)
        pt["gt_depth_m"] = float(ws - pt["gt_elevation"])

    depths = np.array([p["gt_depth_m"] for p in gt_points])
    depths = depths[np.isfinite(depths)]
    n_off = len(gt_points) - len(depths)
    note = f", {n_off} off the WSM" if n_off else ""
    print(f"GT depth for {len(depths)}/{len(gt_points)} points{note}")
    if len(depths) > 0:
        print(
            f"  mean {np.mean(depths):.3f} m, median {np.median(depths):.3f} m, "
            f"range {np.min(depths):.3f} to {np.max(depths):.3f} m"
        )
    for p in gt_points:
        if p["type"] == "reference_target":
            print(
                f"  {p['name']:12s}  bed={p['gt_elevation']:.3f}  "
                f"ws={p['ws_elevation']:.3f}  depth={p['gt_depth_m']:.3f} m"
            )


def match_gt_to_grid(gt_points, grid, max_dist_m):
    """Every GT point with its nearest grid cell, matched within max_dist_m."""
    matches = []
    for pt in gt_points:
        dx = grid["easting"] - pt["gt_easting"]
        dy = grid["northing"] - pt["gt_northing"]
        dists = np.sqrt(dx**2 + dy**2)
        i_min = np.argmin(dists)
        min_dist = float(dists[i_min])
        matched = min_dist <= max_dist_m
        match = {**pt, "match_dist_m": min_dist, "matched": matched}
        for key in PRED_COLUMNS:
            match[f"pred_{key}"] = float(grid[key][i_min]) if matched else np.nan
        matches.append(match)

    unmatched = [m["match_dist_m"] for m in matches if not m["matched"]]
    print(
        f"Matched {len(matches) - len(unmatched)}/{len(matches)} GT points within "
        f"{max_dist_m} m"
    )
    if unmatched:
        print(
            f"  nearest cell of the unmatched points: {min(unmatched):.2f} to "
            f"{max(unmatched):.2f} m"
        )
    return matches


def valid_matches(matches):
    """Matched points with a finite GT and predicted depth."""
    return [
        m
        for m in matches
        if m["matched"]
        and np.isfinite(m["gt_depth_m"])
        and np.isfinite(m["pred_depth_m"])
    ]


def compute_error_statistics(matches):
    valid = valid_matches(matches)
    if not valid:
        return {"n_valid": 0}

    gt = np.array([m["gt_depth_m"] for m in valid])
    pred = np.array([m["pred_depth_m"] for m in valid])
    errors = pred - gt
    abs_errors = np.abs(errors)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    if len(valid) >= 2 and np.std(gt) > 1e-12 and np.std(pred) > 1e-12:
        correlation = float(np.corrcoef(gt, pred)[0, 1])
    else:
        correlation = np.nan
    # mean(pred) / mean(GT), above 1 when the depths are overpredicted; 11 and 14
    # scale by the median GT / predicted depth at the reference targets instead
    if abs(np.mean(gt)) > 1e-12:
        scale_factor = float(np.mean(pred) / np.mean(gt))
    else:
        scale_factor = np.nan

    return {
        "n_valid": len(valid),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_errors)),
        "bias": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "r2": float(r2),
        "correlation": correlation,
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
        "scale_factor": scale_factor,
    }


def fmt(value):
    return f"{value:.4f}" if np.isfinite(value) else ""


def save_comparison_csv(path, matches):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COMPARISON_COLUMNS)
        for m in matches:
            writer.writerow(
                [
                    m["name"],
                    m["type"],
                    m["transect"],
                    fmt(m["gt_easting"]),
                    fmt(m["gt_northing"]),
                    fmt(m["gt_elevation"]),
                    fmt(m["ws_elevation"]),
                    fmt(m["gt_depth_m"]),
                    fmt(m["pred_easting"]),
                    fmt(m["pred_northing"]),
                    fmt(m["pred_elevation"]),
                    fmt(m["pred_depth_m"]),
                    fmt(m["pred_depth_m"] - m["gt_depth_m"]),
                    fmt(m["match_dist_m"]),
                    "yes" if m["matched"] else "no",
                ]
            )
    print(f"Saved: {path}")


def write_summary(
    path, label, stats_all, stats_ref, stats_transect, n_total, n_matched
):
    lines = [
        f"Model: {label}",
        f"GT points total: {n_total}",
        f"  Matched: {n_matched}",
        f"  Unmatched: {n_total - n_matched}",
        "",
    ]
    for section, stats in [
        ("ALL POINTS", stats_all),
        ("REFERENCE TARGETS ONLY", stats_ref),
        ("TRANSECT POINTS ONLY", stats_transect),
    ]:
        lines.append(f"--- {section} ---")
        if stats["n_valid"] == 0:
            lines += ["  No valid points", ""]
            continue
        lines += [
            f"  N valid:      {stats['n_valid']}",
            f"  RMSE:         {stats['rmse']:.4f} m",
            f"  MAE:          {stats['mae']:.4f} m",
            f"  Bias:         {stats['bias']:+.4f} m",
            f"  Std error:    {stats['std_error']:.4f} m",
            f"  R2:           {stats['r2']:.4f}",
            f"  Correlation:  {stats['correlation']:.4f}",
            f"  Scale factor: {stats['scale_factor']:.3f}x",
            f"  Max |error|:  {stats['max_abs_error']:.4f} m",
            f"  P50 |error|:  {stats['p50_abs_error']:.4f} m",
            f"  P90 |error|:  {stats['p90_abs_error']:.4f} m",
            f"  GT depth:     mean={stats['gt_mean']:.3f}, std={stats['gt_std']:.3f}, "
            f"range=[{stats['gt_min']:.3f}, {stats['gt_max']:.3f}] m",
            f"  Pred depth:   mean={stats['pred_mean']:.3f}, "
            f"std={stats['pred_std']:.3f}, "
            f"range=[{stats['pred_min']:.3f}, {stats['pred_max']:.3f}] m",
            "",
        ]

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    print()
    print(text)
    print(f"Saved: {path}")


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(matches, stats, label, path):
    valid = valid_matches(matches)
    if not valid:
        print("No valid points, no scatter plot")
        return

    gt = np.array([m["gt_depth_m"] for m in valid])
    pred = np.array([m["pred_depth_m"] for m in valid])
    ref_mask = np.array([m["type"] == "reference_target" for m in valid])
    transect_mask = ~ref_mask

    fig, ax = plt.subplots(figsize=(8, 8))
    if np.any(transect_mask):
        ax.scatter(
            gt[transect_mask],
            pred[transect_mask],
            c="steelblue",
            alpha=0.5,
            s=20,
            label=f"Transect ({np.sum(transect_mask)})",
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

    # bank points just above the WSM have negative GT depths
    lims = [min(gt.min(), pred.min()) - 0.1, max(gt.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="1:1 line")

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
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    save_figure(fig, path)


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
        if m["type"] == "transect" and m["transect"]:
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


def plot_transect_profiles(matches, label, path):
    panels = transect_panels(matches)
    if not panels:
        print("No transect points, no transect plot")
        return

    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, (title, pts) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        cum_dist = cumulative_distance(
            np.array([p["gt_easting"] for p in pts]),
            np.array([p["gt_northing"] for p in pts]),
        )
        gt_elev = np.array([p["gt_elevation"] for p in pts])
        ws_elev = np.array([p["ws_elevation"] for p in pts])
        pred_depth = np.array([p["pred_depth_m"] for p in pts])
        matched = np.array([p["matched"] for p in pts])

        ax.plot(
            cum_dist, gt_elev, "k-o", markersize=3, linewidth=1.5, label="GT riverbed"
        )

        ws_valid = np.isfinite(ws_elev)
        if np.any(ws_valid):
            ax.plot(
                cum_dist[ws_valid],
                ws_elev[ws_valid],
                "b--",
                linewidth=1.5,
                label="Water surface",
            )

        # the predicted bed at its depth below the GT water surface, the depth the
        # statistics compare; 02's elevation of plane_fit runs can be metres off
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

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Distance (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Transect Profiles -- {label}", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, path)


def plot_spatial_error_map(matches, grid, label, path):
    fig, ax = plt.subplots(figsize=(12, 10))

    cells = ax.scatter(
        grid["easting"],
        grid["northing"],
        c=grid["depth_m"],
        cmap="viridis",
        s=0.1,
        alpha=0.3,
        vmin=np.nanpercentile(grid["depth_m"], 1),
        vmax=np.nanpercentile(grid["depth_m"], 99),
    )
    depth_bar = fig.colorbar(cells, ax=ax, shrink=0.8, pad=0.02)
    depth_bar.set_label("Predicted Depth (m)", fontsize=11)

    valid = valid_matches(matches)
    if valid:
        errors = np.array([m["pred_depth_m"] - m["gt_depth_m"] for m in valid])
        err_max = max(np.abs(errors).max(), 0.5)
        error_points = ax.scatter(
            [m["gt_easting"] for m in valid],
            [m["gt_northing"] for m in valid],
            c=errors,
            cmap="RdBu_r",
            s=30,
            edgecolors="black",
            linewidths=0.5,
            vmin=-err_max,
            vmax=err_max,
            zorder=5,
        )
        error_bar = fig.colorbar(error_points, ax=ax, shrink=0.5, pad=0.08)
        error_bar.set_label("Depth Error (pred - GT, m)", fontsize=10)

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
    save_figure(fig, path)


def plot_error_histogram(matches, stats, label, path):
    valid = valid_matches(matches)
    if len(valid) < MIN_HISTOGRAM_POINTS:
        print(f"Fewer than {MIN_HISTOGRAM_POINTS} valid points, no error histogram")
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
    save_figure(fig, path)


def main():
    args = parse_args()
    for flag, path in (
        ("--pred_csv", args.pred_csv),
        ("--ref_csv", args.ref_csv),
        ("--transect_csv", args.transect_csv),
        ("--wsm_path", args.wsm_path),
    ):
        if not os.path.isfile(path):
            print(f"ERROR: {flag} not found: {path}")
            sys.exit(1)
    label = args.label or os.path.splitext(os.path.basename(args.pred_csv))[0]
    output_dir = args.output_dir
    print(f"Validation of {label} -> {output_dir}")

    grid = load_predicted_grid(args.pred_csv)
    gt_points = load_gt_points(args.ref_csv, args.transect_csv)
    add_gt_depths(gt_points, args.wsm_path)
    matches = match_gt_to_grid(gt_points, grid, args.max_dist)
    stats_all = compute_error_statistics(matches)
    stats_ref = compute_error_statistics(
        [m for m in matches if m["type"] == "reference_target"]
    )
    stats_transect = compute_error_statistics(
        [m for m in matches if m["type"] == "transect"]
    )
    n_total = len(matches)
    n_matched = sum(m["matched"] for m in matches)

    os.makedirs(output_dir, exist_ok=True)
    save_comparison_csv(os.path.join(output_dir, "comparison.csv"), matches)
    write_summary(
        os.path.join(output_dir, "summary.txt"),
        label,
        stats_all,
        stats_ref,
        stats_transect,
        n_total,
        n_matched,
    )
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "model_label": label,
                "pred_csv": os.path.abspath(args.pred_csv),
                "n_gt_total": n_total,
                "n_matched": n_matched,
                "n_unmatched": n_total - n_matched,
                "max_match_dist_m": args.max_dist,
                "stats_all": stats_all,
                "stats_reference_targets": stats_ref,
                "stats_transect_points": stats_transect,
            },
            f,
            indent=2,
        )
    print(f"Saved: {stats_path}")

    plot_scatter(matches, stats_all, label, os.path.join(output_dir, "scatter.png"))
    plot_transect_profiles(matches, label, os.path.join(output_dir, "transects.png"))
    plot_spatial_error_map(
        matches, grid, label, os.path.join(output_dir, "spatial_error.png")
    )
    plot_error_histogram(
        matches, stats_all, label, os.path.join(output_dir, "error_histogram.png")
    )
    print(f"Done, outputs in {output_dir}")


if __name__ == "__main__":
    main()
