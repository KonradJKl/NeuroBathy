"""
Compare NeRFrac's learned water surface (Xs points) with the surveyed water surface
model (WSM).

Writes ws_analysis_stats.json, ws_analysis_summary.txt and the plots to --output_dir.

Usage example:
    python 08_water_surface_analysis.py \\
        --log_folder /path/to/experiment_log \\
        --testset renderonly_all_600000 \\
        --poses_bounds /path/to/pielach_data/poses_bounds.npy \\
        --factor 4 \\
        --mask_dir /path/to/pielach_data/masks_4 \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --umeyama_npz ../data/transforms/nerfrac_nadir_umeyama.npz \\
        --output_dir water_surface/f4 \\
        --label 'NeRFrac f4'
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from coord_utils import (
    NEAR,
    WATER_MASK_THRESHOLD,
    apply_umeyama,
    load_focal_from_poses_bounds,
    load_nerfrac_coord_metadata,
    load_umeyama,
    ndc_to_colmap,
    nerf_world_to_colmap,
    sample_wsm_batch,
)

# Land rays carry their origin on the NDC near plane (z = -1) as Xs
NDC_NEAR_PLANE_Z = -0.999
GRID_CELL_M = 0.1
PLANE_FIT_POINTS = 50000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare NeRFrac's water surface (Xs) with the surveyed WSM."
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="experiment folder with c2w.npy and llff_scale.npy",
    )
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="render folder inside --log_folder with the Xs_pts*.npy files, e.g. "
        "renderonly_all_600000",
    )
    parser.add_argument(
        "--poses_bounds",
        type=str,
        required=True,
        help="poses_bounds.npy of the scene, for the image size and focal",
    )
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="image downsample factor of the training",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF",
    )
    parser.add_argument(
        "--umeyama_npz",
        type=str,
        required=True,
        help="transform from 01_compute_umeyama.py",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="water masks (PNG), paired with the views by sorted file name. "
        "Without them, every pixel whose Xs is off the near plane counts as water",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="scene folder, to recompute the scene scale without llff_scale.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output folder (default: water_surface/<label>)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="model name for the plots and the summary (default: the log folder name)",
    )
    parser.add_argument(
        "--skip_multiview",
        action="store_true",
        help="skip the multiview_3d, multiview_analysis and cameras plots",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="gridded quality plot with two panels, without the metrics text",
    )
    return parser.parse_args()


def load_masks(mask_dir, n_views, H, W):
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)
    names = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if len(names) != n_views:
        print(f"ERROR: {len(names)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    masks = []
    for name in names:
        mask = np.array(Image.open(os.path.join(mask_dir, name)).convert("L"))
        if mask.shape != (H, W):
            mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
        masks.append(mask.flatten() > WATER_MASK_THRESHOLD)
    return masks


def load_water_surface(xs_files, H, W, focal, coord_meta, umeyama, mask_dir):
    """UTM Xs points of the water pixels of all views, and the view of each point."""
    masks = [None] * len(xs_files)
    if mask_dir:
        masks = load_masks(mask_dir, len(xs_files), H, W)
    else:
        print("No --mask_dir: every pixel with Xs off the near plane counts as water")

    s, R, t = umeyama["scale"], umeyama["rotation"], umeyama["translation"]
    points_utm, view_ids = [], []
    for i, (xs_file, mask) in enumerate(zip(xs_files, masks)):
        xs = np.load(xs_file).reshape(-1, 3)
        water = xs[:, 2] > NDC_NEAR_PLANE_Z
        if mask is not None:
            water &= mask
        xs_water = xs[water]
        print(f"  View {i}: {len(xs_water):,} water pixels of {len(xs):,}")
        if len(xs_water) == 0:
            continue
        colmap = ndc_to_colmap(
            xs_water, H, W, focal, coord_meta["c2w_recenter"], coord_meta["sc"], NEAR
        )
        points_utm.append(apply_umeyama(colmap, s, R, t))
        view_ids.append(np.full(len(xs_water), i, dtype=int))
    if not points_utm:
        print("ERROR: No water pixels in any view")
        sys.exit(1)

    points_utm = np.vstack(points_utm)
    print(f"{len(points_utm):,} water surface points from {len(xs_files)} views")
    return points_utm, np.concatenate(view_ids)


def error_stats(dz, label=""):
    valid = ~np.isnan(dz)
    n = valid.sum()
    if n == 0:
        return {"n": 0, "label": label}
    d = dz[valid]
    return {
        "label": label,
        "n": int(n),
        "mean_bias_m": float(np.mean(d)),
        "median_bias_m": float(np.median(d)),
        "rmse_m": float(np.sqrt(np.mean(d**2))),
        "mae_m": float(np.mean(np.abs(d))),
        "std_m": float(np.std(d)),
        "min_m": float(np.min(d)),
        "max_m": float(np.max(d)),
        "p05_m": float(np.percentile(d, 5)),
        "p95_m": float(np.percentile(d, 95)),
    }


def fit_plane(points):
    """Plane through a fixed random sample of PLANE_FIT_POINTS of the points.

    Returns the unit normal (pointing up), the offset and the centroid of the sample.
    """
    sample = points
    if len(points) > PLANE_FIT_POINTS:
        rng = np.random.default_rng(42)
        sample = points[rng.choice(len(points), size=PLANE_FIT_POINTS, replace=False)]
    centroid = np.mean(sample, axis=0)
    _, _, Vt = np.linalg.svd(sample - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -np.dot(normal, centroid), centroid


def plane_gradient(normal):
    """Slope of the plane: dZ/dE, dZ/dN, the tilt from horizontal and the uphill
    direction in degrees counter-clockwise from east."""
    a, b, c = normal
    dz_de = -a / c
    dz_dn = -b / c
    grad_mag = np.sqrt(dz_de**2 + dz_dn**2)
    return {
        "dz_de": dz_de,
        "dz_dn": dz_dn,
        "gradient_magnitude": grad_mag,
        "gradient_direction_deg": np.degrees(np.arctan2(dz_dn, dz_de)),
        "tilt_from_horizontal_deg": np.degrees(np.arctan(grad_mag)),
    }


def read_wsm(wsm_path):
    with rasterio.open(wsm_path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return data, src.transform


def wsm_gradient(wsm_path):
    """Mean slope of the WSM raster, from the gradient of every pixel."""
    data, transform = read_wsm(wsm_path)
    sx = transform.a
    sy = -transform.e
    grad_row, grad_col = np.gradient(data)
    dz_de = grad_col / sx
    dz_dn = -grad_row / sy
    valid = ~np.isnan(data)
    dz_de_mean = float(np.nanmean(dz_de[valid]))
    dz_dn_mean = float(np.nanmean(dz_dn[valid]))
    grad_mag = np.sqrt(dz_de_mean**2 + dz_dn_mean**2)
    slope_map = np.sqrt(dz_de**2 + dz_dn**2)
    return {
        "dz_de_mean": dz_de_mean,
        "dz_dn_mean": dz_dn_mean,
        "gradient_magnitude": grad_mag,
        "gradient_direction_deg": np.degrees(np.arctan2(dz_dn_mean, dz_de_mean)),
        "tilt_from_horizontal_deg": np.degrees(np.arctan(grad_mag)),
        "per_pixel_slope_mean": float(np.nanmean(slope_map[valid])),
        "per_pixel_slope_max": float(np.nanmax(slope_map[valid])),
    }


def grid_cells(e, n, cell_size_m):
    """Cell of each point, with the grid origin at the smallest easting and northing.

    Cells are numbered in the order of their first point. Returns the cell of each
    point, the number of points per cell and the cell centers.
    """
    e_min, n_min = e.min(), n.min()
    ei = ((e - e_min) / cell_size_m).astype(int)
    ni = ((n - n_min) / cell_size_m).astype(int)
    _, first, cell, counts = np.unique(
        ei * (ni.max() + 1) + ni,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    first = first[order]
    center_e = e_min + (ei[first] + 0.5) * cell_size_m
    center_n = n_min + (ni[first] + 0.5) * cell_size_m
    return rank[cell], counts[order], center_e, center_n


def cell_median(values, cell, counts):
    # np.median of every cell at once: sort by cell, then by value, and average the
    # two middle values (the same value twice for an odd count)
    starts = np.cumsum(counts) - counts
    v = values[np.lexsort((values, cell))]
    return (v[starts + (counts - 1) // 2] + v[starts + counts // 2]) / 2


def gridded_metrics(points_utm, wsm_elev, view_ids, cell_size_m):
    """Error of the median Xs elevation per cell against the median WSM elevation.

    Also the view consistency (Z std in cells seen by 2+ views) and the tilt of a
    plane through the cell medians. Returns the metrics and, for the plot, the
    error and the center of each cell.
    """
    valid = ~np.isnan(wsm_elev)
    pts = points_utm[valid]
    wsm_z = wsm_elev[valid]
    vids = view_ids[valid]

    cell, counts, grid_e, grid_n = grid_cells(pts[:, 0], pts[:, 1], cell_size_m)
    grid_z = cell_median(pts[:, 2], cell, counts)
    cell_dz = grid_z - cell_median(wsm_z, cell, counts)

    # np.std per cell: a vectorized sum adds in a different order, which changes the
    # last digits
    cell_std = np.zeros(len(counts), dtype=pts.dtype)
    starts = np.cumsum(counts) - counts
    z_by_cell = pts[np.argsort(cell, kind="stable"), 2]
    for c in np.flatnonzero(counts > 1):
        cell_std[c] = np.std(z_by_cell[starts[c] : starts[c] + counts[c]])
    # distinct views per cell, from the unique (cell, view) pairs
    n_ids = vids.max() + 1
    cell_nviews = np.bincount(np.unique(cell * n_ids + vids) // n_ids)

    multi = cell_nviews >= 2
    vc_mean = float(np.mean(cell_std[multi])) if multi.sum() > 0 else float("nan")
    vc_med = float(np.median(cell_std[multi])) if multi.sum() > 0 else float("nan")

    # tilt of a plane through the cell medians, without the highest and lowest 1%
    z1, z99 = np.percentile(grid_z, [1, 99])
    filt = (grid_z >= z1) & (grid_z <= z99)
    gef, gnf, gzf = grid_e[filt], grid_n[filt], grid_z[filt]
    if len(gef) > 10:
        A = np.column_stack([gef - gef.mean(), gnf - gnf.mean(), np.ones(len(gef))])
        params, _, _, _ = np.linalg.lstsq(A, gzf, rcond=None)
        nvec = np.array([-params[0], -params[1], 1.0])
        nvec /= np.linalg.norm(nvec)
        gridded_tilt = float(np.degrees(np.arccos(abs(nvec[2]))))
    else:
        gridded_tilt = float("nan")

    metrics = {
        "n_cells": int(len(cell_dz)),
        "gridded_bias": float(np.mean(cell_dz)),
        "gridded_rmse": float(np.sqrt(np.mean(cell_dz**2))),
        "gridded_mae": float(np.mean(np.abs(cell_dz))),
        "gridded_tilt": gridded_tilt,
        "within_05m": float((np.abs(cell_dz) < 0.5).mean() * 100),
        "within_1m": float((np.abs(cell_dz) < 1.0).mean() * 100),
        "within_2m": float((np.abs(cell_dz) < 2.0).mean() * 100),
        "view_consistency_mean": vc_mean,
        "view_consistency_median": vc_med,
        "n_multiview_cells": int(multi.sum()),
    }
    return metrics, (cell_dz, grid_e, grid_n)


def save_figure(output_path, dpi):
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gridded_quality(gm, cells, label, output_path, compact=False):
    cell_dz, grid_e, grid_n = cells

    ncols = 2 if compact else 3
    figw = 14 if compact else 20
    fig, axes = plt.subplots(1, ncols, figsize=(figw, 7))

    ax1 = axes[0]
    vmax = max(abs(np.percentile(cell_dz, 2)), abs(np.percentile(cell_dz, 98)), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax1.scatter(
        grid_e,
        grid_n,
        c=cell_dz,
        cmap="RdBu_r",
        norm=norm,
        s=1,
        alpha=0.7,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax1, shrink=0.8, label="dZ = Xs - WSM (m)")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.set_title(f"Gridded Water Surface Error\n{label}")
    ax1.set_aspect("equal")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    ax2.hist(cell_dz, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    for thresh, color, ls in [
        (0.5, "green", "--"),
        (1.0, "orange", "--"),
        (2.0, "red", "--"),
    ]:
        ax2.axvline(thresh, color=color, linestyle=ls, label=f"+{thresh}m")
        ax2.axvline(-thresh, color=color, linestyle=ls)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("dZ = Xs - WSM (m)")
    ax2.set_ylabel("Cell count")
    ax2.set_title(
        f"Planarity: {gm['within_05m']:.0f}% < 0.5m, {gm['within_1m']:.0f}% < 1m"
    )
    ax2.legend(fontsize=8)

    if not compact:
        ax3 = axes[2]
        ax3.axis("off")
        text = (
            f"Gridded Water Surface Metrics\n"
            f"{'=' * 35}\n\n"
            f"Cells:           {gm['n_cells']:,}\n"
            f"Bias:            {gm['gridded_bias']:+.3f} m\n"
            f"RMSE:            {gm['gridded_rmse']:.3f} m\n"
            f"MAE:             {gm['gridded_mae']:.3f} m\n"
            f"Tilt:            {gm['gridded_tilt']:.2f}°\n\n"
            f"Within 0.5m:     {gm['within_05m']:.1f}%\n"
            f"Within 1.0m:     {gm['within_1m']:.1f}%\n"
            f"Within 2.0m:     {gm['within_2m']:.1f}%\n\n"
            f"View consistency:\n"
            f"  Mean std:      {gm['view_consistency_mean']:.3f} m\n"
            f"  Median std:    {gm['view_consistency_median']:.3f} m\n"
            f"  Multi-view:    {gm['n_multiview_cells']:,} cells"
        )
        ax3.text(
            0.05,
            0.95,
            text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    save_figure(output_path, 200)


def plot_spatial_error_map(points_utm, dz, output_path):
    valid = ~np.isnan(dz)
    pts = points_utm[valid]
    vals = dz[valid]

    max_plot = 200000
    if len(pts) > max_plot:
        idx = np.random.default_rng(42).choice(len(pts), size=max_plot, replace=False)
        pts = pts[idx]
        vals = vals[idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = max(abs(np.percentile(vals, 2)), abs(np.percentile(vals, 98)))
    if vmax < 0.001:
        vmax = 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=vals,
        cmap="RdBu_r",
        norm=norm,
        s=0.1,
        alpha=0.5,
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("dZ = NeRFrac - WSM (m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Water Surface Elevation Error (NeRFrac - GT)")
    ax.set_aspect("equal")

    save_figure(output_path, 200)


def plot_histogram(dz, stats, output_path):
    dz_valid = dz[~np.isnan(dz)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(dz_valid, bins=200, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(
        stats["mean_bias_m"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean bias: {stats['mean_bias_m']:.3f} m",
    )
    ax.axvline(
        stats["median_bias_m"],
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median bias: {stats['median_bias_m']:.3f} m",
    )
    ax.set_xlabel("dZ = NeRFrac - WSM (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Elevation Difference Distribution (N={len(dz_valid):,})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    p5, p95 = np.percentile(dz_valid, [5, 95])
    clip = dz_valid[(dz_valid >= p5) & (dz_valid <= p95)]
    ax2.hist(clip, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
    ax2.axvline(0, color="black", linestyle="-", linewidth=1)
    ax2.axvline(stats["mean_bias_m"], color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("dZ = NeRFrac - WSM (m)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Zoomed [P5, P95] = [{p5:.3f}, {p95:.3f}] m")

    textbox = (
        f"RMSE: {stats['rmse_m']:.3f} m\n"
        f"MAE: {stats['mae_m']:.3f} m\n"
        f"Std: {stats['std_m']:.3f} m"
    )
    ax2.text(
        0.97,
        0.95,
        textbox,
        transform=ax2.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    save_figure(output_path, 200)


def plot_per_view(dz, view_ids, n_views, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views, biases, rmses = [], [], []
    for v in range(n_views):
        mask = (view_ids == v) & (~np.isnan(dz))
        d = dz[mask]
        views.append(v)
        biases.append(float(np.mean(d)) if mask.sum() > 0 else 0)
        rmses.append(float(np.sqrt(np.mean(d**2))) if mask.sum() > 0 else 0)

    ax = axes[0]
    colors = ["red" if b > 0 else "blue" for b in biases]
    ax.bar(views, biases, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("View")
    ax.set_ylabel("Mean Bias (m)")
    ax.set_title("Per-View Elevation Bias")
    ax.set_xticks(views)

    ax2 = axes[1]
    ax2.bar(views, rmses, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("View")
    ax2.set_ylabel("RMSE (m)")
    ax2.set_title("Per-View RMSE")
    ax2.set_xticks(views)

    ax3 = axes[2]
    boxdata = []
    boxlabels = []
    for v in range(n_views):
        mask = (view_ids == v) & (~np.isnan(dz))
        if mask.sum() > 0:
            d = dz[mask]
            if len(d) > 10000:
                d = np.random.default_rng(42).choice(d, 10000, replace=False)
            boxdata.append(d)
            boxlabels.append(str(v))
    bp = ax3.boxplot(
        boxdata, tick_labels=boxlabels, showfliers=False, patch_artist=True
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_xlabel("View")
    ax3.set_ylabel("dZ (m)")
    ax3.set_title("Per-View dZ Distribution")

    save_figure(output_path, 200)


def plot_unified_3d(points_utm, n_views, label, output_path):
    z_p01 = np.percentile(points_utm[:, 2], 1)
    z_p99 = np.percentile(points_utm[:, 2], 99)
    keep = (points_utm[:, 2] >= z_p01) & (points_utm[:, 2] <= z_p99)
    pts_kept = points_utm[keep]

    cell, counts, grid_e, grid_n = grid_cells(
        pts_kept[:, 0], pts_kept[:, 1], GRID_CELL_M
    )
    grid_z = cell_median(pts_kept[:, 2], cell, counts)

    max_3d = 50000
    if len(grid_e) > max_3d:
        idx3 = np.random.default_rng(42).choice(len(grid_e), size=max_3d, replace=False)
        grid_e, grid_n, grid_z = grid_e[idx3], grid_n[idx3], grid_z[idx3]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(grid_e, grid_n, grid_z, c=grid_z, cmap="viridis", s=2, alpha=0.7)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Elevation (m)")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(
        f"Unified Water Surface — {label}\n"
        f"({len(grid_e):,} gridded cells from {n_views} views, "
        f"{GRID_CELL_M}m resolution)"
    )
    ax.view_init(elev=30, azim=-60)

    save_figure(output_path, 150)


def stream_axes(wsm_path):
    """Centre of the WSM and its axes along and across the stream, from a PCA of the
    WSM pixel positions."""
    data, transform = read_wsm(wsm_path)
    rows, cols = np.where(~np.isnan(data))
    eastings = transform.c + (cols + 0.5) * transform.a
    northings = transform.f + (rows + 0.5) * transform.e
    center_e = np.mean(eastings)
    center_n = np.mean(northings)
    xy = np.column_stack([eastings - center_e, northings - center_n])
    _, _, Vt = np.linalg.svd(xy, full_matrices=False)
    return center_e, center_n, Vt[0], Vt[1]


def binned_median(x, values, n_bins):
    """Median of the values in equal-width bins of x (NaN for 10 points or fewer).

    Returns the bin centers and the medians.
    """
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    medians = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = (x >= edges[b]) & (x < edges[b + 1])
        if in_bin.sum() > 10:
            medians[b] = np.median(values[in_bin])
    return 0.5 * (edges[:-1] + edges[1:]), medians


def plot_profiles(points_utm, dz, wsm_elev, wsm_path, output_path):
    """Elevation and error along and across the stream, the main axis of the WSM."""
    valid = ~np.isnan(dz)
    pts = points_utm[valid]
    vals = dz[valid]
    elev_pred = pts[:, 2]
    wsm_elev = wsm_elev[valid]

    center_e, center_n, stream_dir, cross_dir = stream_axes(wsm_path)
    xy_centered = np.column_stack([pts[:, 0] - center_e, pts[:, 1] - center_n])
    along = xy_centered @ stream_dir
    cross = xy_centered @ cross_dir

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    max_plot = 50000
    idx = (
        np.random.default_rng(42).choice(len(pts), size=max_plot, replace=False)
        if len(pts) > max_plot
        else np.arange(len(pts))
    )

    n_bins = 30
    for row, (dist, name) in enumerate([(along, "Along"), (cross, "Cross")]):
        ax = axes[row, 0]
        ax.scatter(
            dist[idx],
            elev_pred[idx],
            c="red",
            s=0.3,
            alpha=0.2,
            label="NeRFrac Xs",
            rasterized=True,
        )
        ax.scatter(
            dist[idx],
            wsm_elev[idx],
            c="green",
            s=0.3,
            alpha=0.2,
            label="GT WSM",
            rasterized=True,
        )
        ax.set_xlabel(f"{name}-stream distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"{name}-Stream Elevation Profile")
        ax.legend(markerscale=10)

        ax2 = axes[row, 1]
        ax2.scatter(
            dist[idx], vals[idx], c="steelblue", s=0.3, alpha=0.2, rasterized=True
        )
        centers, medians = binned_median(dist, vals, n_bins)
        ax2.plot(
            centers,
            medians,
            "r-o",
            markersize=4,
            linewidth=2,
            label="Binned median",
        )
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel(f"{name}-stream distance (m)")
        ax2.set_ylabel("dZ (m)")
        ax2.set_title(f"{name}-Stream Elevation Error")
        ax2.legend()

    fig.suptitle(
        f"Stream axis: ({stream_dir[0]:.3f}, {stream_dir[1]:.3f}) | "
        f"Azimuth: {np.degrees(np.arctan2(stream_dir[0], stream_dir[1])):.1f} deg",
        fontsize=10,
    )

    save_figure(output_path, 200)


def plot_tilt_comparison(nerfrac_grad, wsm_grad, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.arrow(
        0,
        0,
        wsm_grad["dz_de_mean"] * 1000,
        wsm_grad["dz_dn_mean"] * 1000,
        head_width=0.1,
        head_length=0.05,
        fc="green",
        ec="green",
        linewidth=2,
        label="GT WSM",
    )
    ax.arrow(
        0,
        0,
        nerfrac_grad["dz_de"] * 1000,
        nerfrac_grad["dz_dn"] * 1000,
        head_width=0.1,
        head_length=0.05,
        fc="red",
        ec="red",
        linewidth=2,
        label="NeRFrac plane",
    )
    ax.set_xlabel("dZ/dE (mm/m)")
    ax.set_ylabel("dZ/dN (mm/m)")
    ax.set_title("Water Surface Gradient Vectors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    margin = (
        max(
            abs(wsm_grad["dz_de_mean"] * 1000),
            abs(wsm_grad["dz_dn_mean"] * 1000),
            abs(nerfrac_grad["dz_de"] * 1000),
            abs(nerfrac_grad["dz_dn"] * 1000),
        )
        * 1.5
    )
    if margin < 0.1:
        margin = 1.0
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        ["Property", "GT WSM", "NeRFrac Plane"],
        [
            "Tilt from horizontal",
            f"{wsm_grad['tilt_from_horizontal_deg']:.3f} deg",
            f"{nerfrac_grad['tilt_from_horizontal_deg']:.3f} deg",
        ],
        [
            "Gradient direction",
            f"{wsm_grad['gradient_direction_deg']:.1f} deg",
            f"{nerfrac_grad['gradient_direction_deg']:.1f} deg",
        ],
        [
            "dZ/dE",
            f"{wsm_grad['dz_de_mean'] * 1000:.2f} mm/m",
            f"{nerfrac_grad['dz_de'] * 1000:.2f} mm/m",
        ],
        [
            "dZ/dN",
            f"{wsm_grad['dz_dn_mean'] * 1000:.2f} mm/m",
            f"{nerfrac_grad['dz_dn'] * 1000:.2f} mm/m",
        ],
        [
            "Gradient magnitude",
            f"{wsm_grad['gradient_magnitude'] * 1000:.2f} mm/m",
            f"{nerfrac_grad['gradient_magnitude'] * 1000:.2f} mm/m",
        ],
    ]
    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title("Tilt/Slope Comparison", pad=20)

    save_figure(output_path, 200)


def plot_multiview_3d(points_utm, view_ids, n_views, output_path):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, len(points_utm) // 50000)
    subset = points_utm[::step]
    subset_views = view_ids[::step]

    sc = ax.scatter(
        subset[:, 0],
        subset[:, 1],
        subset[:, 2],
        c=subset_views,
        cmap="tab10",
        s=1,
        alpha=0.5,
    )
    fig.colorbar(sc, ax=ax, shrink=0.5, label="View Index")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(f"Water Surface from {n_views} Views (UTM33)")
    ax.view_init(elev=30, azim=-60)

    save_figure(output_path, 150)


def plot_multiview_analysis(points_utm, view_ids, n_views, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    views_with_water = [i for i in range(n_views) if np.any(view_ids == i)]
    if len(views_with_water) <= 20:
        for i in views_with_water:
            z_vals = points_utm[view_ids == i, 2]
            ax1.hist(z_vals, bins=50, alpha=0.5, label=f"View {i}", density=True)
        ax1.legend(fontsize=8)
        ax1.set_ylabel("Density")
        ax1.set_title("Z Distribution per View")
    else:
        ax1.hist(points_utm[:, 2], bins=100, color="steelblue", alpha=0.7)
        ax1.text(
            0.98,
            0.95,
            f"{len(views_with_water)} views",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax1.set_ylabel("Count")
        ax1.set_title("Z Distribution (all views)")
    ax1.set_xlabel("Elevation (m)")

    ax2 = axes[0, 1]
    z_all = points_utm[:, 2]
    ax2.hist(z_all, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    ax2.axvline(
        np.mean(z_all), color="red", linestyle="--", label=f"Mean: {np.mean(z_all):.3f}"
    )
    ax2.axvline(
        np.median(z_all),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(z_all):.3f}",
    )
    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Combined Z Distribution")
    ax2.legend()

    ax3 = axes[1, 0]
    normal, d, _ = fit_plane(points_utm)
    residuals = np.dot(points_utm, normal) + d
    ax3.hist(residuals, bins=100, color="coral", edgecolor="black", alpha=0.7)
    ax3.axvline(0, color="red", linestyle="--", linewidth=2)
    ax3.set_xlabel("Deviation from Best-Fit Plane (m)")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Plane Residuals (Std: {np.std(residuals):.4f} m)")

    ax4 = axes[1, 1]
    view_means = []
    view_stds = []
    view_list = []
    for i in range(n_views):
        z_vals = points_utm[view_ids == i, 2]
        if len(z_vals) == 0:
            continue
        view_list.append(i)
        view_means.append(np.mean(z_vals))
        view_stds.append(np.std(z_vals))
    ax4.errorbar(view_list, view_means, yerr=view_stds, fmt="o-", capsize=5, capthick=2)
    ax4.set_xlabel("View Index")
    ax4.set_ylabel("Elevation (m)")
    ax4.set_title("Mean Z +/- Std per View")
    ax4.grid(True, alpha=0.3)

    save_figure(output_path, 150)


def plot_camera_positions(testset_dir, points_utm, coord_meta, umeyama, output_path):
    pose_files = sorted(glob.glob(os.path.join(testset_dir, "pose_*.npy")))
    if not pose_files:
        print("No pose files, camera plot skipped")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, len(points_utm) // 10000)
    subset = points_utm[::step]
    ax.scatter(
        subset[:, 0],
        subset[:, 1],
        subset[:, 2],
        c="lightblue",
        s=1,
        alpha=0.2,
        label="Water Surface",
    )

    for i, pf in enumerate(pose_files):
        cam_colmap = nerf_world_to_colmap(
            np.load(pf)[:3, 3].copy(), coord_meta["c2w_recenter"], coord_meta["sc"]
        )
        cam_utm = apply_umeyama(
            cam_colmap, umeyama["scale"], umeyama["rotation"], umeyama["translation"]
        )
        ax.scatter(
            cam_utm[0],
            cam_utm[1],
            cam_utm[2],
            s=100,
            marker="^",
            c=f"C{i % 10}",
            edgecolors="black",
            linewidths=1,
        )
        ax.text(cam_utm[0], cam_utm[1], cam_utm[2], f" {i}", fontsize=8)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(f"Camera Positions & Water Surface — {len(pose_files)} cameras")
    ax.legend()

    save_figure(output_path, 150)


def write_summary(stats, path):
    g = stats["global_stats"]
    wsm = stats["wsm_gradient"]
    nerfrac = stats["nerfrac_gradient"]
    plane = stats["nerfrac_plane"]
    n_total, n_matched = stats["n_total_xs_points"], stats["n_matched_with_wsm"]
    lines = [
        f"Water Surface Analysis: {stats['model_label']}",
        "=" * 70,
        "",
        f"Model:   {stats['log_folder']}",
        f"Testset: {stats['testset']}",
        f"WSM:     {stats['wsm_path']}",
        "",
        "POINT MATCHING",
        f"  Total Xs points:      {n_total:,}",
        f"  Matched with WSM:     {n_matched:,} ({100 * n_matched / n_total:.1f}%)",
        "",
        "GLOBAL ELEVATION ERROR (dZ = NeRFrac - GT WSM)",
        f"  Mean bias:   {g['mean_bias_m']:+.4f} m",
        f"  Median bias: {g['median_bias_m']:+.4f} m",
        f"  RMSE:        {g['rmse_m']:.4f} m",
        f"  MAE:         {g['mae_m']:.4f} m",
        f"  Std:         {g['std_m']:.4f} m",
        f"  Range:       [{g['min_m']:.4f}, {g['max_m']:.4f}] m",
        f"  P5/P95:      [{g['p05_m']:.4f}, {g['p95_m']:.4f}] m",
        "",
        "PER-VIEW BREAKDOWN",
        f"  {'View':>5} {'N':>8} {'Bias (m)':>10} {'RMSE (m)':>10} {'Std (m)':>10}",
        f"  {'-' * 45}",
    ]
    for vs in stats["per_view_stats"]:
        if vs["n"] > 0:
            lines.append(
                f"  {vs['label']:>5} {vs['n']:>8,} {vs['mean_bias_m']:>+10.4f} "
                f"{vs['rmse_m']:>10.4f} {vs['std_m']:>10.4f}"
            )
    lines += [
        "",
        "TILT / SLOPE COMPARISON",
        "  NeRFrac: plane through a 50,000-point sample of the matched Xs points. "
        "WSM: mean gradient of the raster.",
        "  Directions point uphill, in degrees counter-clockwise from east.",
        f"  {'':>25} {'GT WSM':>15} {'NeRFrac':>15}",
        f"  {'-' * 55}",
        f"  {'Tilt from horizontal':>25} {wsm['tilt_from_horizontal_deg']:>14.4f}d "
        f"{nerfrac['tilt_from_horizontal_deg']:>14.4f}d",
        f"  {'Gradient direction':>25} {wsm['gradient_direction_deg']:>14.1f}d "
        f"{nerfrac['gradient_direction_deg']:>14.1f}d",
        "",
        f"  Tilt ratio (NeRFrac / WSM): {stats['tilt_ratio_nerfrac_over_wsm']:.1f}x",
        f"  Gradient direction diff: {stats['gradient_direction_diff_deg']:.1f} deg",
        "",
        "NERFRAC PLANE FIT (UTM33)",
        f"  Centroid: E={plane['centroid_utm33'][0]:.2f}, "
        f"N={plane['centroid_utm33'][1]:.2f}, Z={plane['centroid_utm33'][2]:.3f} m",
        f"  Residual std: {plane['residual_std_m']:.4f} m",
        f"  Normal: [{plane['normal'][0]:.6f}, {plane['normal'][1]:.6f}, "
        f"{plane['normal'][2]:.6f}]",
        "",
    ]
    gm = stats["gridded_metrics"]
    lines += [
        "GRIDDED WATER SURFACE (median Z per cell, dZ = cell Z - cell WSM)",
        f"  Cells:            {gm['n_cells']:,}",
        f"  Bias:             {gm['gridded_bias']:+.3f} m",
        f"  RMSE:             {gm['gridded_rmse']:.3f} m",
        f"  MAE:              {gm['gridded_mae']:.3f} m",
        f"  Gridded tilt:     {gm['gridded_tilt']:.2f} deg from horizontal "
        "(plane through the cell medians)",
        f"  Within 0.5/1/2 m: {gm['within_05m']:.1f}% / {gm['within_1m']:.1f}% / "
        f"{gm['within_2m']:.1f}%",
        f"  View consistency: {gm['view_consistency_mean']:.3f} m mean, "
        f"{gm['view_consistency_median']:.3f} m median Z std "
        f"({gm['n_multiview_cells']:,} cells seen by 2+ views)",
        "",
        "ELEVATION RANGES",
    ]
    z = stats["xs_elevation_utm33"]
    lines.append(
        f"  NeRFrac Xs: [{z['min']:.3f}, {z['max']:.3f}] m (mean={z['mean']:.3f})"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def main():
    args = parse_args()
    testset_dir = os.path.join(args.log_folder, args.testset)
    for path, exists, what in [
        (testset_dir, os.path.isdir, "Render folder"),
        (args.wsm_path, os.path.isfile, "WSM"),
        (args.umeyama_npz, os.path.isfile, "Umeyama transform"),
    ]:
        if not exists(path):
            print(f"ERROR: {what} not found: {path}")
            sys.exit(1)

    label = args.label or os.path.basename(os.path.normpath(args.log_folder))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "water_surface", label.replace("/", "_").replace(" ", "_")
        )
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "ws_analysis")
    print(f"Water surface analysis of {label}: {testset_dir} -> {output_dir}")

    coord_meta = load_nerfrac_coord_metadata(args.log_folder, data_dir=args.data_dir)
    if coord_meta["c2w_recenter"] is None or coord_meta["sc"] is None:
        print(
            f"ERROR: Could not load c2w.npy and llff_scale.npy from {args.log_folder}"
        )
        sys.exit(1)
    H, W, focal = load_focal_from_poses_bounds(args.poses_bounds, factor=args.factor)
    umeyama = load_umeyama(args.umeyama_npz)
    # poses_bounds gives H // factor (5460 // 8 = 682 for Pielach), but the rendered
    # images are as tall as images_<factor> (683), so take H from the rendered pixels
    xs_files = sorted(glob.glob(os.path.join(testset_dir, "Xs_pts*.npy")))
    if not xs_files:
        print(f"ERROR: No Xs_pts*.npy files in {testset_dir}")
        sys.exit(1)
    n_pixels = np.load(xs_files[0]).shape[0]
    if n_pixels % W != 0:
        print(f"ERROR: {n_pixels:,} rendered pixels do not fit W={W}, check --factor")
        sys.exit(1)
    H = n_pixels // W
    print(f"{W}x{H} px, focal {focal:.4f} px, Umeyama scale {umeyama['scale']:.4f}")

    points_utm, view_ids = load_water_surface(
        xs_files, H, W, focal, coord_meta, umeyama, args.mask_dir
    )
    n_views = len(xs_files)

    wsm_elev = sample_wsm_batch(args.wsm_path, points_utm[:, 0], points_utm[:, 1])
    matched = ~np.isnan(wsm_elev)
    n_matched, n_total = int(matched.sum()), len(wsm_elev)
    print(f"{n_matched:,} of {n_total:,} points on the WSM")
    if n_matched == 0:
        print("ERROR: No Xs point lies on the WSM")
        sys.exit(1)

    dz = points_utm[:, 2] - wsm_elev
    global_stats = error_stats(dz, "global")
    per_view_stats = [
        error_stats(dz[view_ids == v], f"view_{v}")
        for v in range(n_views)
        if np.any(view_ids == v)
    ]
    print(
        f"dZ = Xs - WSM: bias {global_stats['mean_bias_m']:+.4f} m, RMSE "
        f"{global_stats['rmse_m']:.4f} m, MAE {global_stats['mae_m']:.4f} m"
    )

    normal, d, centroid = fit_plane(points_utm[matched])
    nerfrac_grad = plane_gradient(normal)
    resid_std = float(np.std(np.dot(points_utm[matched], normal) + d))
    wsm_grad = wsm_gradient(args.wsm_path)
    print(
        f"Plane tilt {nerfrac_grad['tilt_from_horizontal_deg']:.4f} deg (plane "
        f"through a 50,000-point sample of the matched points, residual std "
        f"{resid_std:.4f} m), WSM tilt "
        f"{wsm_grad['tilt_from_horizontal_deg']:.4f} deg"
    )

    gm, cells = gridded_metrics(points_utm, wsm_elev, view_ids, GRID_CELL_M)
    print(
        f"Gridded ({GRID_CELL_M} m cells): {gm['n_cells']:,} cells, bias "
        f"{gm['gridded_bias']:+.3f} m, RMSE {gm['gridded_rmse']:.3f} m, tilt "
        f"{gm['gridded_tilt']:.2f} deg, {gm['within_05m']:.1f}% within 0.5 m, view "
        f"consistency {gm['view_consistency_mean']:.3f} m"
    )

    plot_unified_3d(points_utm, n_views, label, f"{prefix}_unified_3d.png")
    plot_gridded_quality(
        gm, cells, label, f"{prefix}_gridded_quality.png", args.compact
    )
    plot_spatial_error_map(points_utm, dz, f"{prefix}_spatial.png")
    plot_histogram(dz, global_stats, f"{prefix}_histogram.png")
    plot_per_view(dz, view_ids, n_views, f"{prefix}_per_view.png")
    plot_profiles(points_utm, dz, wsm_elev, args.wsm_path, f"{prefix}_profiles.png")
    plot_tilt_comparison(nerfrac_grad, wsm_grad, f"{prefix}_tilt.png")
    if not args.skip_multiview:
        plot_multiview_3d(points_utm, view_ids, n_views, f"{prefix}_multiview_3d.png")
        plot_multiview_analysis(
            points_utm, view_ids, n_views, f"{prefix}_multiview_analysis.png"
        )
        plot_camera_positions(
            testset_dir, points_utm, coord_meta, umeyama, f"{prefix}_cameras.png"
        )

    angle_diff_deg = abs(
        nerfrac_grad["gradient_direction_deg"] - wsm_grad["gradient_direction_deg"]
    )
    if angle_diff_deg > 180:
        angle_diff_deg = 360 - angle_diff_deg
    stats = {
        "model_label": label,
        "log_folder": args.log_folder,
        "testset": args.testset,
        "factor": args.factor,
        "wsm_path": args.wsm_path,
        "umeyama_npz": args.umeyama_npz,
        "n_total_xs_points": int(n_total),
        "n_matched_with_wsm": int(n_matched),
        "match_rate_pct": float(100 * n_matched / n_total),
        "global_stats": global_stats,
        "per_view_stats": per_view_stats,
        "nerfrac_plane": {
            "normal": normal.tolist(),
            "centroid_utm33": centroid.tolist(),
            "residual_std_m": float(resid_std),
        },
        "nerfrac_gradient": {k: float(v) for k, v in nerfrac_grad.items()},
        "wsm_gradient": {k: float(v) for k, v in wsm_grad.items()},
        "tilt_ratio_nerfrac_over_wsm": float(
            nerfrac_grad["tilt_from_horizontal_deg"]
            / max(wsm_grad["tilt_from_horizontal_deg"], 1e-6)
        ),
        "gradient_direction_diff_deg": float(angle_diff_deg),
        "xs_elevation_utm33": {
            "min": float(points_utm[:, 2].min()),
            "max": float(points_utm[:, 2].max()),
            "mean": float(points_utm[:, 2].mean()),
            "std": float(points_utm[:, 2].std()),
        },
        "gridded_metrics": gm,
    }
    with open(f"{prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {prefix}_stats.json")
    write_summary(stats, f"{prefix}_summary.txt")


if __name__ == "__main__":
    main()
