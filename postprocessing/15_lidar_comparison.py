"""
Compare a model's bathymetry or terrain with the classified LiDAR point cloud.

The LiDAR points are gridded on the prediction's own grid, so every predicted cell is
compared with the median of the LiDAR points in the same cell. Writes
lidar_comparison.csv, lidar_stats.json, lidar_summary.txt, lidar_difference.ply and
three plots to --output_dir.

Usage examples:
    # underwater depth (LiDAR class 40) against the gridded bathymetry from 02
    python 15_lidar_comparison.py \\
        --pred_csv bathymetry/bathymetry_nerfrac_50m.csv \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --label "NeRFrac 50m" \\
        --output_dir validation_lidar/nerfrac_50m

    # full terrain (classes 2-5 and 40) against the point cloud from 07
    python 15_lidar_comparison.py \\
        --pred_npy pointclouds/nerfrac_50m/pointcloud_combined.npy \\
        --compare full \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --label "NeRFrac 50m" \\
        --output_dir validation_lidar/nerfrac_50m_full
"""

import argparse
import json
import os
import sys
import time

import laspy
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import sample_wsm_batch, write_ply_binary

LIDAR_CLASSES = {"underwater": [40], "land": [2, 3, 4, 5], "full": [2, 3, 4, 5, 40]}
CLASS_NAMES = {
    2: "ground",
    3: "low vegetation",
    4: "medium vegetation",
    5: "high vegetation",
    40: "river bottom",
}
MODE_TITLES = {
    "underwater": "Underwater Depth",
    "land": "Land Elevation",
    "full": "Full Terrain",
}
DEPTH_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MIN_MATCHED_CELLS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare predicted bathymetry or terrain with classified LiDAR."
    )

    # inputs
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="",
        help="gridded bathymetry CSV from 02_extract_bathymetry.py (underwater mode; "
        "land and full use it only without --pred_npy)",
    )
    parser.add_argument(
        "--pred_npy",
        type=str,
        default="",
        help="point cloud (N, 3) from 07_make_pointcloud.py (land and full modes)",
    )
    parser.add_argument(
        "--lidar_laz",
        type=str,
        required=True,
        help="classified LiDAR point cloud (LAZ or LAS)",
    )
    parser.add_argument(
        "--wsm_path", type=str, required=True, help="water surface model GeoTIFF"
    )

    # comparison
    parser.add_argument(
        "--compare",
        choices=["underwater", "land", "full"],
        default="underwater",
        help="underwater (default): river bottom (class 40) against the predicted "
        "depth; land: ground and vegetation (classes 2-5) against the predicted "
        "elevation; full: both. Land and full only use cells inside the WSM footprint",
    )
    parser.add_argument(
        "--grid_resolution",
        type=float,
        default=0.1,
        help="grid cell size in m, the one 02 used for --pred_csv (default: 0.1)",
    )

    # output
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="model name for the plots, the summary and the stats (default: the "
        "folder of --pred_npy or the file name of --pred_csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_lidar/lidar_comparison",
        help="output folder (default: validation_lidar/lidar_comparison)",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        help="fixed easting limits of the spatial plot (e.g. 527600 527715)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="fixed northing limits of the spatial plot (e.g. 5340240 5340320)",
    )
    return parser.parse_args()


def prediction_grid(e, n, resolution, centred):
    """Grid starting at the lowest prediction coordinates, half a cell lower when they
    are cell centres (the 02 CSV)."""
    e0, n0 = e.min(), n.min()
    if centred:
        e0, n0 = e0 - resolution / 2, n0 - resolution / 2
    n_cols = int((e.max() - e0) / resolution) + 1
    n_rows = int((n.max() - n0) / resolution) + 1
    return e0, n0, resolution, n_cols, n_rows


def cell_ids(e, n, grid):
    e0, n0, resolution, n_cols, n_rows = grid
    col = np.floor((e - e0) / resolution).astype(np.int64)
    row = np.floor((n - n0) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < n_cols) & (row >= 0) & (row < n_rows)
    return row * n_cols + col, inside


def cell_centres(cells, grid):
    e0, n0, resolution, n_cols, _ = grid
    return (
        e0 + (cells % n_cols + 0.5) * resolution,
        n0 + (cells // n_cols + 0.5) * resolution,
    )


def grid_median(e, n, z, grid):
    """Median z and point count per cell, for the points inside the grid."""
    cells, inside = cell_ids(e, n, grid)
    grouped = (
        pd.DataFrame({"cell": cells[inside], "z": z[inside]})
        .groupby("cell")["z"]
        .agg(["median", "count"])
    )
    return grouped.index.values, grouped["median"].values, grouped["count"].values


def load_prediction(pred_csv, pred_npy, mode, resolution):
    """The prediction's grid and its cells: the median elevation per cell of pred_npy
    if given, else the depth or elevation column of the 02 CSV."""
    if pred_npy:
        pc = np.load(pred_npy)
        print(f"Prediction: {len(pc):,} points in {pred_npy}")
        grid = prediction_grid(pc[:, 0], pc[:, 1], resolution, centred=False)
        cells, elevation, _ = grid_median(pc[:, 0], pc[:, 1], pc[:, 2], grid)
        e, n = cell_centres(cells, grid)
        print(f"  {len(cells):,} cells")
        return grid, {"cells": cells, "e": e, "n": n, "values": elevation}

    if not pred_csv:
        alternative = "" if mode == "underwater" else " or --pred_npy"
        print(f"ERROR: --compare {mode} needs --pred_csv{alternative}")
        sys.exit(1)
    if mode != "underwater":
        print("WARNING: --pred_csv only has the water cells, use --pred_npy for land")
    pred = pd.read_csv(pred_csv)
    e, n = pred["utm_easting"].values, pred["utm_northing"].values
    steps = np.concatenate([e - e.min(), n - n.min()]) / resolution
    index = np.rint(steps).astype(np.int64)
    # a coarser grid also lies on this lattice, but then all indices share a factor
    if np.abs(steps - index).max() > 0.01 or np.gcd.reduce(index) > 1:
        print(
            f"ERROR: The cells in {pred_csv} are not {resolution} m apart, "
            "set --grid_resolution to the one used in 02"
        )
        sys.exit(1)
    grid = prediction_grid(e, n, resolution, centred=True)
    cells, _ = cell_ids(e, n, grid)
    print(f"Prediction: {len(cells):,} cells in {pred_csv}")
    column = "depth_m" if mode == "underwater" else "utm_elevation"
    return grid, {"cells": cells, "e": e, "n": n, "values": pred[column].values}


def load_lidar(path, classes):
    las = laspy.read(path)
    cls = np.array(las.classification)
    keep = np.isin(cls, classes)
    for c in classes:
        print(f"LiDAR class {c} ({CLASS_NAMES[c]}): {(cls == c).sum():,} points")
    if not keep.any():
        print(f"ERROR: No LiDAR points of classes {classes} in {path}")
        sys.exit(1)
    return np.array(las.x[keep]), np.array(las.y[keep]), np.array(las.z[keep])


def lidar_reference(lidar_xyz, grid, wsm_path, mode):
    """Grid the LiDAR on the prediction grid and keep the cells on the WSM.

    Returns the cell ids, the reference values (depth below the WSM or elevation) and
    the WSM elevation of each cell.
    """
    cells, z, counts = grid_median(*lidar_xyz, grid)
    if len(cells) == 0:
        print("ERROR: No LiDAR point inside the prediction grid")
        sys.exit(1)
    median = int(np.median(counts))
    print(f"LiDAR: {len(cells):,} cells, median {median} points per cell")

    wsm = sample_wsm_batch(wsm_path, *cell_centres(cells, grid))
    on_wsm = ~np.isnan(wsm)
    if not on_wsm.any():
        print("ERROR: No LiDAR cell lies on the WSM")
        sys.exit(1)
    if mode != "underwater":
        print("Land and full modes only use cells inside the WSM footprint")
    ref = wsm - z if mode == "underwater" else z
    return {"cells": cells[on_wsm], "ref": ref[on_wsm], "wsm": wsm[on_wsm]}


def match_cells(pred, lidar, mode):
    """Pair every prediction cell with the LiDAR value of the same cell, sorted by
    cell. Underwater cells need a positive depth in both."""
    pos = np.searchsorted(lidar["cells"], pred["cells"])
    pos = np.minimum(pos, len(lidar["cells"]) - 1)
    pred_idx = np.flatnonzero(lidar["cells"][pos] == pred["cells"])
    pred_idx = pred_idx[np.argsort(pred["cells"][pred_idx], kind="stable")]
    lidar_idx = pos[pred_idx]

    values, ref = pred["values"][pred_idx], lidar["ref"][lidar_idx]
    if mode == "underwater":
        keep = (values > 0) & (ref > 0)
    else:
        keep = np.isfinite(values) & np.isfinite(ref)
    print(
        f"Matched: {keep.sum():,} of {len(pred['cells']):,} prediction cells "
        f"({len(pred_idx):,} with LiDAR on the WSM)"
    )
    return {
        "e": pred["e"][pred_idx][keep],
        "n": pred["n"][pred_idx][keep],
        "pred": values[keep],
        "ref": ref[keep],
        "wsm": lidar["wsm"][lidar_idx][keep],
    }


def compute_stats(pred, ref):
    error = pred - ref
    abs_error = np.abs(error)

    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(abs_error))
    bias = float(np.mean(error))
    std_err = float(np.std(error))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    scale = float(np.mean(pred) / np.mean(ref)) if np.mean(ref) != 0 else float("nan")

    return {
        "n": len(pred),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "std_error": std_err,
        "r2": r2,
        "pearson_r": float(sp_stats.pearsonr(pred, ref)[0]),
        "spearman_r": float(sp_stats.spearmanr(pred, ref)[0]),
        "scale_factor": scale,
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "ref_mean": float(ref.mean()),
        "ref_std": float(ref.std()),
    }


def print_stats(stats):
    print(f"  N:            {stats['n']:,}")
    print(f"  RMSE:         {stats['rmse']:.4f} m")
    print(f"  MAE:          {stats['mae']:.4f} m")
    print(f"  Bias:         {stats['bias']:+.4f} m")
    print(f"  Std error:    {stats['std_error']:.4f} m")
    print(f"  R²:           {stats['r2']:.4f}")
    print(f"  Pearson r:    {stats['pearson_r']:.4f}")
    print(f"  Spearman rho: {stats['spearman_r']:.4f}")
    print(f"  Scale:        {stats['scale_factor']:.3f}x")


def depth_stratified(pred, ref):
    """Stats of the cells deeper than each LiDAR depth threshold, if 5 or more."""
    rows = []
    print(
        f"  {'deeper than':>11s} {'N':>8s} {'RMSE':>8s} {'MAE':>8s} "
        f"{'Bias':>8s} {'Spearman':>8s}"
    )
    for thresh in DEPTH_THRESHOLDS:
        deeper = ref > thresh
        if deeper.sum() < 5:
            continue
        stats = compute_stats(pred[deeper], ref[deeper])
        rows.append({"threshold": thresh, **stats})
        print(
            f"  {thresh:9.1f} m {stats['n']:8,} {stats['rmse']:8.4f} "
            f"{stats['mae']:8.4f} {stats['bias']:+8.4f} {stats['spearman_r']:8.4f}"
        )
    return rows


def write_outputs(output_dir, label, mode, matched, stats, strat):
    value = "depth" if mode == "underwater" else "elevation"
    e, n, pred, ref = matched["e"], matched["n"], matched["pred"], matched["ref"]
    error = pred - ref
    pd.DataFrame(
        {
            "utm_easting": e,
            "utm_northing": n,
            f"pred_{value}": pred,
            f"lidar_{value}": ref,
            "error_m": error,
            "abs_error_m": np.abs(error),
        }
    ).to_csv(os.path.join(output_dir, "lidar_comparison.csv"), index=False)

    with open(os.path.join(output_dir, "lidar_stats.json"), "w") as f:
        json.dump(
            {"label": label, "mode": mode, **stats, "depth_stratified": strat},
            f,
            indent=2,
        )

    lines = [
        f"LiDAR {MODE_TITLES[mode]} Comparison: {label}",
        "=" * 60,
        f"Mode:        {mode}",
        f"Matched:     {stats['n']:,}",
        f"RMSE:        {stats['rmse']:.4f} m",
        f"MAE:         {stats['mae']:.4f} m",
        f"Bias:        {stats['bias']:+.4f} m",
        f"Spearman:    {stats['spearman_r']:.4f}",
        f"Scale:       {stats['scale_factor']:.3f}x",
    ]
    if mode != "underwater":
        lines.append("Area:        cells inside the WSM footprint")
    with open(os.path.join(output_dir, "lidar_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # underwater: z is the predicted riverbed, the WSM minus the predicted depth
    z = matched["wsm"] - pred if mode == "underwater" else pred
    write_ply_binary(
        os.path.join(output_dir, "lidar_difference.ply"),
        np.column_stack([e, n, z]),
        scalar_fields={f"pred_{value}": pred, f"lidar_{value}": ref, "error": error},
    )


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(output_dir, label, mode, matched, stats):
    value = "depth" if mode == "underwater" else "elevation"
    pred, ref = matched["pred"], matched["ref"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ref, pred, s=0.3, alpha=0.05, c="steelblue")
    hi = max(ref.max(), pred.max())
    if mode == "underwater":
        lo, hi = 0, hi * 1.05
    else:
        lo = min(ref.min(), pred.min())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(f"LiDAR {value} (m)")
    ax.set_ylabel(f"Predicted {value} (m)")
    ax.set_title(
        f"{label}\nRMSE={stats['rmse']:.3f}m, ρ={stats['spearman_r']:.3f}, "
        f"N={stats['n']:,}"
    )
    ax.set_aspect("equal")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "lidar_scatter.png"))


def plot_spatial(output_dir, label, mode, matched, stats, xlim, ylim):
    """Predicted, LiDAR and error maps. The first two share one colour range."""
    value = "depth" if mode == "underwater" else "elevation"
    e, n, pred, ref = matched["e"], matched["n"], matched["pred"], matched["ref"]
    error = pred - ref
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = np.percentile(ref, 99)
    vmin = 0 if mode == "underwater" else np.percentile(ref, 1)
    cmap = "viridis" if mode == "underwater" else "viridis_r"
    for ax, vals, title in [
        (axes[0], pred, f"Predicted {value} (m)"),
        (axes[1], ref, f"LiDAR {value} (m)"),
    ]:
        sc = ax.scatter(e, n, c=vals, s=0.2, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, shrink=0.7)

    err_lim = np.percentile(np.abs(error), 95)
    sc = axes[2].scatter(
        e, n, c=error, s=0.2, cmap="RdBu_r", vmin=-err_lim, vmax=err_lim
    )
    axes[2].set_title(f"Error (m)\nbias={stats['bias']:+.3f}, RMSE={stats['rmse']:.3f}")
    plt.colorbar(sc, ax=axes[2], shrink=0.7)

    for ax in axes:
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box" if xlim or ylim else "datalim")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
    fig.suptitle(f"LiDAR {MODE_TITLES[mode]} Comparison: {label}", fontsize=14)
    save_figure(fig, os.path.join(output_dir, "lidar_spatial_error.png"))


def plot_histogram(output_dir, label, mode, matched, stats):
    value = "depth" if mode == "underwater" else "elevation"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        matched["pred"] - matched["ref"],
        bins=100,
        alpha=0.7,
        color="steelblue",
        edgecolor="none",
    )
    ax.axvline(stats["bias"], color="red", lw=2, label=f"bias={stats['bias']:+.3f}m")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel(f"{value.capitalize()} error (pred - LiDAR) [m]")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: N={stats['n']:,}")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "lidar_error_histogram.png"))


def main():
    args = parse_args()
    t0 = time.time()
    mode = args.compare
    # the underwater depths always come from the 02 CSV
    pred_npy = args.pred_npy if mode != "underwater" else ""
    if args.label:
        label = args.label
    elif pred_npy:
        label = os.path.basename(os.path.dirname(os.path.abspath(pred_npy)))
    else:
        label = os.path.splitext(os.path.basename(args.pred_csv))[0]
    print(f"LiDAR comparison ({mode}): {label}")

    grid, pred = load_prediction(args.pred_csv, pred_npy, mode, args.grid_resolution)
    lidar_xyz = load_lidar(args.lidar_laz, LIDAR_CLASSES[mode])
    lidar = lidar_reference(lidar_xyz, grid, args.wsm_path, mode)
    matched = match_cells(pred, lidar, mode)
    n_matched = len(matched["pred"])
    if n_matched < MIN_MATCHED_CELLS:
        print(f"ERROR: Only {n_matched} matched cells, need {MIN_MATCHED_CELLS}")
        sys.exit(1)

    stats = compute_stats(matched["pred"], matched["ref"])
    print_stats(stats)
    strat = []
    if mode == "underwater":
        strat = depth_stratified(matched["pred"], matched["ref"])

    os.makedirs(args.output_dir, exist_ok=True)
    write_outputs(args.output_dir, label, mode, matched, stats, strat)
    plot_scatter(args.output_dir, label, mode, matched, stats)
    plot_spatial(args.output_dir, label, mode, matched, stats, args.xlim, args.ylim)
    plot_histogram(args.output_dir, label, mode, matched, stats)
    print(f"Done in {time.time() - t0:.1f} s, outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
