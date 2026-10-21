"""
Analyze NeRFrac's predicted water surface (Xs points) against a ground truth
Usage:
    python 08_water_surface_analysis.py \\
        --log_folder logs/pielach_sep_nerfrac_d-0.4_... \\
        --testset testset_all_600000 \\
        --poses_bounds data/nerf_llff_data/pielach_data/poses_bounds.npy \\
        --factor 4 \\
        --wsm_path '/path/to/Pielach_20241024_WSM.tif' \\
        --umeyama_npz ../data/transforms/nerfrac_nadir_umeyama.npz \\
        --output_dir ./water_surface/f4 \\
        --label 'NeRFrac f4'
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from coord_utils import (
    apply_umeyama,
    load_focal_from_poses_bounds,
    load_nerfrac_coord_metadata,
    load_umeyama,
    NEAR,
    ndc_to_colmap,
    sample_wsm_batch,
)



def _load_masks_sorted(mask_dir, H, W):
    from PIL import Image

    mask_files = sorted(
        [f for f in os.listdir(mask_dir)
         if f.endswith(('.png', '.jpg', '.npy'))]
    )

    masks = []
    for mf in mask_files:
        p = os.path.join(mask_dir, mf)
        if p.endswith(".npy"):
            mask = np.load(p)
        else:
            mask = np.array(Image.open(p).convert("L"))
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = np.array(
                Image.fromarray(mask).resize((W, H), Image.NEAREST)
            )
        masks.append(mask.flatten() > 127)

    return masks


def load_xs_all_views(testset_dir, H, W, focal, coord_meta, umeyama,
                      mask_dir=None):
    xs_files = sorted(glob.glob(os.path.join(testset_dir, "Xs_pts*.npy")))
    if not xs_files:
        print(f"[ERROR] No Xs_pts*.npy files found in {testset_dir}")
        sys.exit(1)

    masks = None
    if mask_dir:
        masks = _load_masks_sorted(mask_dir, H, W)
        print(f"  Using water masks from: {mask_dir} ({len(masks)} masks)")
        if len(masks) != len(xs_files):
            print(f"  [WARN] {len(masks)} masks but {len(xs_files)} Xs files — "
                  f"mask/view count mismatch!")
    else:
        print("  [WARN] No --mask_dir provided. Including ALL pixels with "
              "Z_ndc > -0.999. Land pixels may have meaningless Xs values!")

    c2w_recenter = coord_meta["c2w_recenter"]
    sc = coord_meta["sc"]
    s = umeyama["scale"]
    R = umeyama["rotation"]
    t = umeyama["translation"]

    all_colmap = []
    all_utm = []
    all_view_ids = []
    per_view_counts = []

    for i, xs_file in enumerate(xs_files):
        xs_ndc = np.load(xs_file).reshape(-1, 3)
        n_total = len(xs_ndc)
        ndc_mask = xs_ndc[:, 2] > -0.999

        if masks is not None and i < len(masks):
            water_mask = masks[i]
            combined_mask = ndc_mask & water_mask
        else:
            combined_mask = ndc_mask
            water_mask = None

        xs_water = xs_ndc[combined_mask]
        n_water = xs_water.shape[0]
        per_view_counts.append(n_water)

        n_ndc = ndc_mask.sum()
        if water_mask is not None:
            print(
                f"  View {i}: {n_water:,} water-masked pixels "
                f"(NDC: {n_ndc:,}, mask: {water_mask.sum():,}) "
                f"out of {n_total:,}"
            )
        else:
            print(
                f"  View {i}: {n_water:,} water pixels "
                f"({100 * n_water / n_total:.1f}%) out of {n_total:,}"
            )

        if n_water == 0:
            continue

        pts_colmap = ndc_to_colmap(xs_water, H, W, focal, c2w_recenter, sc, NEAR)
        pts_utm = apply_umeyama(pts_colmap, s, R, t)

        all_colmap.append(pts_colmap)
        all_utm.append(pts_utm)
        all_view_ids.append(np.full(n_water, i, dtype=int))

    if not all_utm:
        print("[ERROR] No water pixels found in any view!")
        sys.exit(1)

    points_colmap = np.vstack(all_colmap)
    points_utm = np.vstack(all_utm)
    view_ids = np.concatenate(all_view_ids)

    print(
        f"\nTotal: {len(points_utm):,} water surface points from {len(xs_files)} views"
    )

    return {
        "points_utm": points_utm,
        "points_colmap": points_colmap,
        "view_ids": view_ids,
        "n_views": len(xs_files),
        "per_view_counts": per_view_counts,
    }


def compute_error_stats(dz, label=""):
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


def fit_plane_svd(points, max_pts=50000):
    if len(points) > max_pts:
        idx = np.random.choice(len(points), size=max_pts, replace=False)
        sample = points[idx]
    else:
        sample = points

    centroid = np.mean(sample, axis=0)
    centered = sample - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]

    if normal[2] < 0:
        normal = -normal

    d = -np.dot(normal, centroid)
    return normal, d, centroid, S


def plane_gradient_info(normal):
    a, b, c = normal
    dz_de = -a / c
    dz_dn = -b / c
    grad_mag = np.sqrt(dz_de**2 + dz_dn**2)
    grad_dir_deg = np.degrees(np.arctan2(dz_dn, dz_de))
    tilt_deg = np.degrees(np.arctan(grad_mag))
    return {
        "dz_de": dz_de,
        "dz_dn": dz_dn,
        "gradient_magnitude": grad_mag,
        "gradient_direction_deg": grad_dir_deg,
        "tilt_from_horizontal_deg": tilt_deg,
    }


def compute_gridded_metrics(points_utm, wsm_elev, view_ids, cell_size=0.5):
    from collections import defaultdict

    valid = ~np.isnan(wsm_elev)
    pts = points_utm[valid]
    wsm_z = wsm_elev[valid]
    vids = view_ids[valid]

    if len(pts) == 0:
        return {k: 0.0 for k in [
            'n_cells', 'gridded_bias', 'gridded_rmse', 'gridded_mae',
            'gridded_tilt', 'within_05m', 'within_1m', 'within_2m',
            'view_consistency_mean', 'view_consistency_median',
            'n_multiview_cells',
        ]}

    e_min, n_min = pts[:, 0].min(), pts[:, 1].min()
    ei = ((pts[:, 0] - e_min) / cell_size).astype(int)
    ni = ((pts[:, 1] - n_min) / cell_size).astype(int)

    cells_z = defaultdict(list)
    cells_wsm = defaultdict(list)
    cells_views = defaultdict(set)

    for i in range(len(pts)):
        key = (ei[i], ni[i])
        cells_z[key].append(pts[i, 2])
        cells_wsm[key].append(wsm_z[i])
        cells_views[key].add(vids[i])

    cell_dz = []
    cell_std = []
    cell_nviews = []
    grid_e, grid_n, grid_z = [], [], []

    for key in cells_z:
        zvals = np.array(cells_z[key])
        wvals = np.array(cells_wsm[key])
        med_z = np.median(zvals)
        med_wsm = np.median(wvals)
        cell_dz.append(med_z - med_wsm)
        cell_std.append(np.std(zvals))
        cell_nviews.append(len(cells_views[key]))
        grid_e.append(e_min + (key[0] + 0.5) * cell_size)
        grid_n.append(n_min + (key[1] + 0.5) * cell_size)
        grid_z.append(med_z)

    cell_dz = np.array(cell_dz)
    cell_std = np.array(cell_std)
    cell_nviews = np.array(cell_nviews)
    grid_e, grid_n, grid_z = np.array(grid_e), np.array(grid_n), np.array(grid_z)

    within_05 = float((np.abs(cell_dz) < 0.5).mean() * 100)
    within_1 = float((np.abs(cell_dz) < 1.0).mean() * 100)
    within_2 = float((np.abs(cell_dz) < 2.0).mean() * 100)

    multi = cell_nviews >= 2
    vc_mean = float(np.mean(cell_std[multi])) if multi.sum() > 0 else float('nan')
    vc_med = float(np.median(cell_std[multi])) if multi.sum() > 0 else float('nan')

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
        gridded_tilt = float('nan')

    return {
        'n_cells': int(len(cell_dz)),
        'gridded_bias': float(np.mean(cell_dz)),
        'gridded_rmse': float(np.sqrt(np.mean(cell_dz ** 2))),
        'gridded_mae': float(np.mean(np.abs(cell_dz))),
        'gridded_tilt': gridded_tilt,
        'within_05m': within_05,
        'within_1m': within_1,
        'within_2m': within_2,
        'view_consistency_mean': vc_mean,
        'view_consistency_median': vc_med,
        'n_multiview_cells': int(multi.sum()),
        'cell_dz': cell_dz,
        'grid_e': grid_e,
        'grid_n': grid_n,
    }


def plot_gridded_quality(gridded_metrics, wsm_path, label, output_path,
                         compact=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    gm = gridded_metrics
    cell_dz = gm['cell_dz']
    grid_e = gm['grid_e']
    grid_n = gm['grid_n']

    ncols = 2 if compact else 3
    figw = 14 if compact else 20
    fig, axes = plt.subplots(1, ncols, figsize=(figw, 7))

    ax1 = axes[0]
    vmax = max(abs(np.percentile(cell_dz, 2)), abs(np.percentile(cell_dz, 98)), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax1.scatter(grid_e, grid_n, c=cell_dz, cmap='RdBu_r', norm=norm,
                     s=1, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax1, shrink=0.8, label='dZ = Xs - WSM (m)')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_title(f'Gridded Water Surface Error\n{label}')
    ax1.set_aspect('equal')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    ax2.hist(cell_dz, bins=80, color='steelblue', edgecolor='black', alpha=0.7)
    for thresh, color, ls in [(0.5, 'green', '--'), (1.0, 'orange', '--'), (2.0, 'red', '--')]:
        ax2.axvline(thresh, color=color, linestyle=ls, label=f'+{thresh}m')
        ax2.axvline(-thresh, color=color, linestyle=ls)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('dZ = Xs - WSM (m)')
    ax2.set_ylabel('Cell count')
    ax2.set_title(f'Planarity: {gm["within_05m"]:.0f}% < 0.5m, '
                  f'{gm["within_1m"]:.0f}% < 1m')
    ax2.legend(fontsize=8)

    if not compact:
        ax3 = axes[2]
        ax3.axis('off')
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
        ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_wsm_gradient(wsm_path):
    import rasterio

    with rasterio.open(wsm_path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        sx = src.transform.a
        sy = -src.transform.e

    if nodata is not None:
        data[data == nodata] = np.nan

    grad_row, grad_col = np.gradient(data)
    dz_de = grad_col / sx
    dz_dn = -grad_row / sy

    valid = ~np.isnan(data)
    dz_de_mean = float(np.nanmean(dz_de[valid]))
    dz_dn_mean = float(np.nanmean(dz_dn[valid]))

    grad_mag = np.sqrt(dz_de_mean**2 + dz_dn_mean**2)
    grad_dir_deg = np.degrees(np.arctan2(dz_dn_mean, dz_de_mean))
    tilt_deg = np.degrees(np.arctan(grad_mag))

    slope_map = np.sqrt(dz_de**2 + dz_dn**2)

    return {
        "dz_de_mean": dz_de_mean,
        "dz_dn_mean": dz_dn_mean,
        "gradient_magnitude": grad_mag,
        "gradient_direction_deg": grad_dir_deg,
        "tilt_from_horizontal_deg": tilt_deg,
        "per_pixel_slope_mean": float(np.nanmean(slope_map[valid])),
        "per_pixel_slope_max": float(np.nanmax(slope_map[valid])),
    }



def plot_spatial_error_map(points_utm, dz, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    valid = ~np.isnan(dz)
    pts = points_utm[valid]
    vals = dz[valid]

    max_plot = 200000
    if len(pts) > max_plot:
        idx = np.random.choice(len(pts), size=max_plot, replace=False)
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_histogram(dz, output_path):
    import matplotlib.pyplot as plt

    valid = dz[~np.isnan(dz)]
    stats = compute_error_stats(valid, "all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(valid, bins=200, color="steelblue", alpha=0.8, edgecolor="none")
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
    ax.set_title(f"Elevation Difference Distribution (N={len(valid):,})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    p5, p95 = np.percentile(valid, [5, 95])
    clip = valid[(valid >= p5) & (valid <= p95)]
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_view(dz, view_ids, n_views, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    view_stats = []
    for v in range(n_views):
        mask = (view_ids == v) & (~np.isnan(dz))
        if mask.sum() == 0:
            view_stats.append({"view": v, "n": 0, "bias": 0, "rmse": 0, "std": 0})
            continue
        d = dz[mask]
        view_stats.append(
            {
                "view": v,
                "n": int(mask.sum()),
                "bias": float(np.mean(d)),
                "rmse": float(np.sqrt(np.mean(d**2))),
                "std": float(np.std(d)),
                "median": float(np.median(d)),
            }
        )

    views = [s["view"] for s in view_stats]
    biases = [s["bias"] for s in view_stats]
    rmses = [s["rmse"] for s in view_stats]

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
                d = np.random.choice(d, 10000, replace=False)
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return view_stats


def plot_unified_3d(points_utm, n_views, label, output_path):
    import matplotlib.pyplot as plt
    from collections import defaultdict


    z_p01 = np.percentile(points_utm[:, 2], 1)
    z_p99 = np.percentile(points_utm[:, 2], 99)
    keep = (points_utm[:, 2] >= z_p01) & (points_utm[:, 2] <= z_p99)
    pts_valid = points_utm[keep]

    cell_size = 0.1
    e_min = pts_valid[:, 0].min()
    n_min = pts_valid[:, 1].min()
    ei = ((pts_valid[:, 0] - e_min) / cell_size).astype(int)
    ni = ((pts_valid[:, 1] - n_min) / cell_size).astype(int)

    cells = defaultdict(list)
    for k in range(len(pts_valid)):
        cells[(ei[k], ni[k])].append(pts_valid[k, 2])

    grid_e, grid_n, grid_z = [], [], []
    for (ci, cj), zvals in cells.items():
        grid_e.append(e_min + (ci + 0.5) * cell_size)
        grid_n.append(n_min + (cj + 0.5) * cell_size)
        grid_z.append(np.median(zvals))
    grid_e = np.array(grid_e)
    grid_n = np.array(grid_n)
    grid_z = np.array(grid_z)

    max_3d = 50000
    if len(grid_e) > max_3d:
        idx3 = np.random.choice(len(grid_e), size=max_3d, replace=False)
        grid_e, grid_n, grid_z = grid_e[idx3], grid_n[idx3], grid_z[idx3]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(grid_e, grid_n, grid_z, c=grid_z, cmap="viridis",
                    s=2, alpha=0.7)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Elevation (m)")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(f"Unified Water Surface — {label}\n"
                 f"({len(grid_e):,} gridded cells from {n_views} views, "
                 f"{cell_size}m resolution)")
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_profiles(points_utm, dz, wsm_path, output_path):
    import matplotlib.pyplot as plt
    import rasterio

    valid = ~np.isnan(dz)
    pts = points_utm[valid]
    vals = dz[valid]
    elev_pred = pts[:, 2]


    wsm_elev = sample_wsm_batch(wsm_path, pts[:, 0], pts[:, 1])


    with rasterio.open(wsm_path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    valid_wsm = ~np.isnan(data)
    rows_v, cols_v = np.where(valid_wsm)
    wsm_eastings = transform.c + (cols_v + 0.5) * transform.a
    wsm_northings = transform.f + (rows_v + 0.5) * transform.e
    wsm_center_e = np.mean(wsm_eastings)
    wsm_center_n = np.mean(wsm_northings)

    wsm_xy = np.column_stack(
        [wsm_eastings - wsm_center_e, wsm_northings - wsm_center_n]
    )
    _, _, Vt = np.linalg.svd(wsm_xy, full_matrices=False)
    stream_dir = Vt[0]
    cross_dir = Vt[1]

    xy_centered = np.column_stack([pts[:, 0] - wsm_center_e, pts[:, 1] - wsm_center_n])
    along = xy_centered @ stream_dir
    cross = xy_centered @ cross_dir

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    max_plot = 50000
    idx = (
        np.random.choice(len(pts), size=max_plot, replace=False)
        if len(pts) > max_plot
        else np.arange(len(pts))
    )

    ax = axes[0, 0]
    ax.scatter(
        along[idx],
        elev_pred[idx],
        c="red",
        s=0.3,
        alpha=0.2,
        label="NeRFrac Xs",
        rasterized=True,
    )
    wsm_ok = ~np.isnan(wsm_elev[idx])
    ax.scatter(
        along[idx][wsm_ok],
        wsm_elev[idx][wsm_ok],
        c="green",
        s=0.3,
        alpha=0.2,
        label="GT WSM",
        rasterized=True,
    )
    ax.set_xlabel("Along-stream distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Along-Stream Elevation Profile")
    ax.legend(markerscale=10)

    ax2 = axes[0, 1]
    ax2.scatter(along[idx], vals[idx], c="steelblue", s=0.3, alpha=0.2, rasterized=True)
    n_bins = 30
    bin_edges = np.linspace(along.min(), along.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_medians = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = (along >= bin_edges[b]) & (along < bin_edges[b + 1])
        if in_bin.sum() > 10:
            bin_medians[b] = np.median(vals[in_bin])
    ax2.plot(
        bin_centers,
        bin_medians,
        "r-o",
        markersize=4,
        linewidth=2,
        label="Binned median",
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Along-stream distance (m)")
    ax2.set_ylabel("dZ (m)")
    ax2.set_title("Along-Stream Elevation Error")
    ax2.legend()

    ax3 = axes[1, 0]
    ax3.scatter(
        cross[idx],
        elev_pred[idx],
        c="red",
        s=0.3,
        alpha=0.2,
        label="NeRFrac Xs",
        rasterized=True,
    )
    ax3.scatter(
        cross[idx][wsm_ok],
        wsm_elev[idx][wsm_ok],
        c="green",
        s=0.3,
        alpha=0.2,
        label="GT WSM",
        rasterized=True,
    )
    ax3.set_xlabel("Cross-stream distance (m)")
    ax3.set_ylabel("Elevation (m)")
    ax3.set_title("Cross-Stream Elevation Profile")
    ax3.legend(markerscale=10)

    ax4 = axes[1, 1]
    ax4.scatter(cross[idx], vals[idx], c="steelblue", s=0.3, alpha=0.2, rasterized=True)
    bin_edges_c = np.linspace(cross.min(), cross.max(), n_bins + 1)
    bin_centers_c = 0.5 * (bin_edges_c[:-1] + bin_edges_c[1:])
    bin_medians_c = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = (cross >= bin_edges_c[b]) & (cross < bin_edges_c[b + 1])
        if in_bin.sum() > 10:
            bin_medians_c[b] = np.median(vals[in_bin])
    ax4.plot(
        bin_centers_c,
        bin_medians_c,
        "r-o",
        markersize=4,
        linewidth=2,
        label="Binned median",
    )
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.set_xlabel("Cross-stream distance (m)")
    ax4.set_ylabel("dZ (m)")
    ax4.set_title("Cross-Stream Elevation Error")
    ax4.legend()

    fig.suptitle(
        f"Stream axis: ({stream_dir[0]:.3f}, {stream_dir[1]:.3f}) | "
        f"Azimuth: {np.degrees(np.arctan2(stream_dir[0], stream_dir[1])):.1f} deg",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_tilt_comparison(nerfrac_grad, wsm_grad, output_path):
    import matplotlib.pyplot as plt

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")




def plot_multiview_3d(points_utm, view_ids, n_views, dz, output_path):
    import matplotlib.pyplot as plt

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_multiview_analysis(points_utm, view_ids, n_views, output_path):

    import matplotlib.pyplot as plt

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
    normal, d, _, _ = fit_plane_svd(points_utm)
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
        mask = view_ids == i
        z_vals = points_utm[mask, 2]
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_camera_positions(
    testset_dir, points_utm, coord_meta, umeyama, n_views, output_path
):

    import matplotlib.pyplot as plt

    pose_files = sorted(glob.glob(os.path.join(testset_dir, "pose_*.npy")))
    if not pose_files:
        print("  No pose files found, skipping camera visualization.")
        return

    s_um = umeyama["scale"]
    R_um = umeyama["rotation"]
    t_um = umeyama["translation"]
    c2w_recenter = coord_meta["c2w_recenter"]
    sc = coord_meta["sc"]

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
        pose = np.load(pf)

        cam_nerf = pose[:3, 3].copy()

        cam_colmap = np.linalg.inv(c2w_recenter[:3, :3]) @ cam_nerf
        cam_colmap = cam_colmap / sc
        cam_utm = apply_umeyama(cam_colmap, s_um, R_um, t_um)

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")



def save_results(
    output_dir,
    label,
    args,
    global_stats,
    per_view_stats,
    points_utm,
    wsm_elev,
    nerfrac_grad,
    wsm_grad,
    nerfrac_normal,
    nerfrac_centroid,
    nerfrac_resid_std,
    n_matched,
    n_total,
    gridded_metrics=None,
):
    """Save JSON stats and human-readable text summary."""

    prefix = os.path.join(output_dir, "ws_analysis")

    tilt_ratio = nerfrac_grad["tilt_from_horizontal_deg"] / max(
        wsm_grad["tilt_from_horizontal_deg"], 1e-6
    )
    angle_diff = abs(
        nerfrac_grad["gradient_direction_deg"] - wsm_grad["gradient_direction_deg"]
    )
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    all_stats = {
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
            "normal": nerfrac_normal.tolist(),
            "centroid_utm33": nerfrac_centroid.tolist(),
            "residual_std_m": float(nerfrac_resid_std),
        },
        "nerfrac_gradient": {k: float(v) for k, v in nerfrac_grad.items()},
        "wsm_gradient": {k: float(v) for k, v in wsm_grad.items()},
        "tilt_ratio_nerfrac_over_wsm": float(tilt_ratio),
        "gradient_direction_diff_deg": float(angle_diff),
        "xs_elevation_utm33": {
            "min": float(points_utm[:, 2].min()),
            "max": float(points_utm[:, 2].max()),
            "mean": float(points_utm[:, 2].mean()),
            "std": float(points_utm[:, 2].std()),
        },
    }

    if gridded_metrics is not None:
        gm_save = {k: v for k, v in gridded_metrics.items()
                    if not isinstance(v, np.ndarray)}
        all_stats["gridded_metrics"] = gm_save

    stats_path = f"{prefix}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Text summary
    summary_path = f"{prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Water Surface Analysis: {label}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Model:   {args.log_folder}\n")
        f.write(f"Testset: {args.testset}\n")
        f.write(f"WSM:     {args.wsm_path}\n\n")

        f.write(f"POINT MATCHING\n")
        f.write(f"  Total Xs points:      {n_total:,}\n")
        f.write(
            f"  Matched with WSM:     {n_matched:,} "
            f"({100 * n_matched / n_total:.1f}%)\n\n"
        )

        f.write(f"GLOBAL ELEVATION ERROR (dZ = NeRFrac - GT WSM)\n")
        f.write(f"  Mean bias:   {global_stats['mean_bias_m']:+.4f} m\n")
        f.write(f"  Median bias: {global_stats['median_bias_m']:+.4f} m\n")
        f.write(f"  RMSE:        {global_stats['rmse_m']:.4f} m\n")
        f.write(f"  MAE:         {global_stats['mae_m']:.4f} m\n")
        f.write(f"  Std:         {global_stats['std_m']:.4f} m\n")
        f.write(
            f"  Range:       [{global_stats['min_m']:.4f}, "
            f"{global_stats['max_m']:.4f}] m\n"
        )
        f.write(
            f"  P5/P95:      [{global_stats['p05_m']:.4f}, "
            f"{global_stats['p95_m']:.4f}] m\n\n"
        )

        f.write(f"PER-VIEW BREAKDOWN\n")
        f.write(
            f"  {'View':>5} {'N':>8} {'Bias (m)':>10} "
            f"{'RMSE (m)':>10} {'Std (m)':>10}\n"
        )
        f.write(f"  {'-' * 45}\n")
        for vs in per_view_stats:
            if vs["n"] == 0:
                continue
            f.write(
                f"  {vs['label']:>5} {vs['n']:>8,} "
                f"{vs['mean_bias_m']:>+10.4f} "
                f"{vs['rmse_m']:>10.4f} "
                f"{vs['std_m']:>10.4f}\n"
            )

        f.write(f"\nTILT / SLOPE COMPARISON\n")
        f.write(f"  {'':>25} {'GT WSM':>15} {'NeRFrac':>15}\n")
        f.write(f"  {'-' * 55}\n")
        f.write(
            f"  {'Tilt from horizontal':>25} "
            f"{wsm_grad['tilt_from_horizontal_deg']:>14.4f}d "
            f"{nerfrac_grad['tilt_from_horizontal_deg']:>14.4f}d\n"
        )
        f.write(
            f"  {'Gradient direction':>25} "
            f"{wsm_grad['gradient_direction_deg']:>14.1f}d "
            f"{nerfrac_grad['gradient_direction_deg']:>14.1f}d\n"
        )
        f.write(f"\n  Tilt ratio (NeRFrac / WSM): {tilt_ratio:.1f}x\n")
        f.write(f"  Gradient direction diff: {angle_diff:.1f} deg\n\n")

        f.write(f"NERFRAC PLANE FIT (UTM33)\n")
        f.write(
            f"  Centroid: E={nerfrac_centroid[0]:.2f}, "
            f"N={nerfrac_centroid[1]:.2f}, "
            f"Z={nerfrac_centroid[2]:.3f} m\n"
        )
        f.write(f"  Residual std: {nerfrac_resid_std:.4f} m\n")
        f.write(
            f"  Normal: [{nerfrac_normal[0]:.6f}, "
            f"{nerfrac_normal[1]:.6f}, {nerfrac_normal[2]:.6f}]\n\n"
        )

        f.write(f"ELEVATION RANGES\n")
        f.write(
            f"  NeRFrac Xs: [{points_utm[:, 2].min():.3f}, "
            f"{points_utm[:, 2].max():.3f}] m "
            f"(mean={points_utm[:, 2].mean():.3f})\n"
        )

    print(f"Saved: {summary_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="NeRFrac water surface (Xs) vs GT WSM analysis"
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="Path to NeRFrac experiment log folder",
    )
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="Testset subdirectory name (e.g. testset_all_600000)",
    )
    parser.add_argument(
        "--poses_bounds",
        type=str,
        required=True,
        help="Path to poses_bounds.npy for this dataset",
    )
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="Image downscale factor used during training",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="Path to GT Water Surface Model GeoTIFF",
    )
    parser.add_argument(
        "--umeyama_npz",
        type=str,
        required=True,
        help="Path to Umeyama .npz (from 01_compute_umeyama.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-named in postprocessing/)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable model label for plots and summary",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Path to water mask directory. STRONGLY RECOMMENDED: without masks, "
             "land pixels with meaningless Xs values contaminate the analysis.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="LLFF data dir (for coord_meta fallback if log folder lacks it)",
    )
    parser.add_argument(
        "--skip_multiview",
        action="store_true",
        help="Skip multiview geometry visualizations (faster)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact gridded quality plot (2 panels, no metrics text)",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    testset_dir = os.path.join(args.log_folder, args.testset)
    if not os.path.isdir(testset_dir):
        print(f"[ERROR] Testset directory not found: {testset_dir}")
        sys.exit(1)
    if not os.path.isfile(args.wsm_path):
        print(f"[ERROR] WSM GeoTIFF not found: {args.wsm_path}")
        sys.exit(1)
    if not os.path.isfile(args.umeyama_npz):
        print(f"[ERROR] Umeyama .npz not found: {args.umeyama_npz}")
        sys.exit(1)

    label = args.label or os.path.basename(args.log_folder)
    output_dir = args.output_dir
    if output_dir is None:
        safe_label = label.replace("/", "_").replace(" ", "_")
        output_dir = os.path.join("postprocessing", "water_surface", safe_label)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Water Surface Analysis: {label}")
    print(f"  Log folder:  {args.log_folder}")
    print(f"  Testset:     {args.testset}")
    print(f"  WSM:         {args.wsm_path}")
    print(f"  Umeyama:     {args.umeyama_npz}")
    print(f"  Mask dir:    {args.mask_dir or '(none — land pixels included!)'}")
    print(f"  Output:      {output_dir}")
    print("=" * 80)

    print("\n--- Loading coordinate metadata ---")
    coord_meta = load_nerfrac_coord_metadata(args.log_folder, data_dir=args.data_dir)
    H, W, focal = load_focal_from_poses_bounds(args.poses_bounds, factor=args.factor)
    umeyama = load_umeyama(args.umeyama_npz)

    xs_files_check = sorted(glob.glob(os.path.join(testset_dir, "Xs_pts*.npy")))
    if xs_files_check:
        try:
            xs0 = np.load(xs_files_check[0])
            n_pixels = xs0.shape[0]
            H_actual = n_pixels // W
            if H_actual * W == n_pixels and H_actual != H:
                print(
                    f"  [INFO] Adjusting H from poses_bounds ({H}) to actual "
                    f"model output ({H_actual}) — {n_pixels} pixels = "
                    f"{H_actual}x{W}"
                )
                H = H_actual
            elif H_actual * W != n_pixels:
                print(
                    f"  [WARN] First Xs file has {n_pixels} pixels, "
                    f"not divisible by W={W}"
                )
        except Exception:
            pass

    print(f"  Image size: {H}x{W}, focal={focal:.4f}")
    print(f"  Umeyama scale: {umeyama['scale']:.4f}")

    print("\n--- Loading Xs points (NDC -> COLMAP -> UTM33) ---")
    xs_data = load_xs_all_views(testset_dir, H, W, focal, coord_meta, umeyama,
                                mask_dir=args.mask_dir)
    points_utm = xs_data["points_utm"]
    view_ids = xs_data["view_ids"]
    n_views = xs_data["n_views"]

    print(
        f"\nXs UTM33 elevation range: "
        f"[{points_utm[:, 2].min():.3f}, {points_utm[:, 2].max():.3f}] m"
    )

    print("\n--- Sampling WSM at Xs points ---")
    wsm_elev = sample_wsm_batch(args.wsm_path, points_utm[:, 0], points_utm[:, 1])
    n_matched = int(np.sum(~np.isnan(wsm_elev)))
    n_total = len(wsm_elev)
    print(f"  Matched: {n_matched:,} / {n_total:,} ({100 * n_matched / n_total:.1f}%)")

    if n_matched == 0:
        print("[ERROR] No Xs points overlap with WSM raster!")
        sys.exit(1)

    print("\n--- Computing elevation differences ---")
    dz = points_utm[:, 2] - wsm_elev

    global_stats = compute_error_stats(dz, "global")
    print(f"  Mean bias:  {global_stats['mean_bias_m']:+.4f} m")
    print(f"  RMSE:       {global_stats['rmse_m']:.4f} m")
    print(f"  MAE:        {global_stats['mae_m']:.4f} m")
    print(f"  Std:        {global_stats['std_m']:.4f} m")


    per_view_stats = []
    for v in range(n_views):
        mask = view_ids == v
        if mask.sum() == 0:
            continue
        per_view_stats.append(compute_error_stats(dz[mask], f"view_{v}"))

    print("\n--- Plane fit + tilt analysis ---")
    matched = ~np.isnan(wsm_elev)
    normal, d, centroid, S = fit_plane_svd(points_utm[matched])
    nerfrac_grad = plane_gradient_info(normal)
    resid = np.dot(points_utm[matched], normal) + d
    resid_std = float(np.std(resid))
    print(f"  Plane residual std: {resid_std:.4f} m")
    print(f"  Tilt: {nerfrac_grad['tilt_from_horizontal_deg']:.4f} deg")

    wsm_grad = compute_wsm_gradient(args.wsm_path)
    print(f"  WSM tilt: {wsm_grad['tilt_from_horizontal_deg']:.4f} deg")

    print("\n--- Gridded water surface metrics (0.1m cells, median Z) ---")
    gridded_metrics = compute_gridded_metrics(
        points_utm, wsm_elev, view_ids, cell_size=0.1
    )
    gm = gridded_metrics
    print(f"  Gridded cells:    {gm['n_cells']}")
    print(f"  Gridded bias:     {gm['gridded_bias']:+.3f} m")
    print(f"  Gridded RMSE:     {gm['gridded_rmse']:.3f} m")
    print(f"  Gridded MAE:      {gm['gridded_mae']:.3f} m")
    print(f"  Gridded tilt:     {gm['gridded_tilt']:.2f} deg")
    print(f"  Within 0.5m:      {gm['within_05m']:.1f}%")
    print(f"  Within 1.0m:      {gm['within_1m']:.1f}%")
    print(f"  Within 2.0m:      {gm['within_2m']:.1f}%")
    print(f"  View consistency: {gm['view_consistency_mean']:.3f} m "
          f"(median {gm['view_consistency_median']:.3f} m, "
          f"{gm['n_multiview_cells']} multi-view cells)")

    print("\n--- Generating visualizations ---")
    prefix = os.path.join(output_dir, "ws_analysis")

    plot_unified_3d(points_utm, n_views, label,
                    f"{prefix}_unified_3d.png")

    plot_gridded_quality(gridded_metrics, args.wsm_path, label,
                         f"{prefix}_gridded_quality.png",
                         compact=args.compact)

    plot_spatial_error_map(points_utm, dz, f"{prefix}_spatial.png")
    plot_histogram(dz, f"{prefix}_histogram.png")
    plot_per_view(dz, view_ids, n_views, f"{prefix}_per_view.png")
    plot_profiles(points_utm, dz, args.wsm_path, f"{prefix}_profiles.png")
    plot_tilt_comparison(nerfrac_grad, wsm_grad, f"{prefix}_tilt.png")

    if not args.skip_multiview:
        print("\n--- Generating multiview visualizations ---")
        plot_multiview_3d(points_utm, view_ids, n_views, dz, f"{prefix}_multiview_3d.png")
        plot_multiview_analysis(
            points_utm, view_ids, n_views, f"{prefix}_multiview_analysis.png"
        )
        plot_camera_positions(
            testset_dir,
            points_utm,
            coord_meta,
            umeyama,
            n_views,
            f"{prefix}_cameras.png",
        )

    print("\n--- Saving results ---")
    save_results(
        output_dir,
        label,
        args,
        global_stats,
        per_view_stats,
        points_utm,
        wsm_elev,
        nerfrac_grad,
        wsm_grad,
        normal,
        centroid,
        resid_std,
        n_matched,
        n_total,
        gridded_metrics=gridded_metrics,
    )

    print(f"\n{'=' * 80}")
    print(f"DONE. All outputs saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
