#!/usr/bin/env python
"""Compare NeRFrac/VGGT outputs against classified LiDAR point cloud.

Usage:
    # Underwater depth comparison (default)
    python 15_lidar_comparison.py \\
        --pred_csv bathymetry/bathymetry_nerfrac_50m.csv \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --compare underwater \\
        --label "NeRFrac 50m" \\
        --output_dir validation_lidar/nerfrac_50m

    # Land elevation comparison
    python 15_lidar_comparison.py \\
        --pred_npy pointclouds/nerfrac_50m/pointcloud_combined.npy \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --compare land \\
        --label "NeRFrac 50m" \\
        --output_dir validation_lidar/nerfrac_50m_land

    # Full terrain comparison (land + underwater)
    python 15_lidar_comparison.py \\
        --pred_npy pointclouds/nerfrac_50m/pointcloud_combined.npy \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --compare full \\
        --label "NeRFrac 50m" \\
        --output_dir validation_lidar/nerfrac_50m_full
"""

import argparse
import json
import os
import sys
import time

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import stats as sp_stats

from coord_utils import sample_wsm_batch, write_ply_binary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare predicted bathymetry/elevation against LiDAR reference."
    )
    parser.add_argument("--pred_csv", default="",
                        help="Predicted bathymetry grid CSV from 02_extract_bathymetry.py (for underwater mode)")
    parser.add_argument("--pred_npy", default="",
                        help="Predicted pointcloud NPY (N,3) from 07_make_pointcloud.py (for land/full mode)")
    parser.add_argument("--lidar_laz", required=True,
                        help="Path to classified LAZ/LAS file")
    parser.add_argument("--wsm_path", required=True,
                        help="Path to Water Surface Model GeoTIFF")
    parser.add_argument("--compare", choices=["underwater", "land", "full"],
                        default="underwater",
                        help="Comparison mode: underwater (class 40), land (classes 2-5: "
                             "ground and vegetation), or full (classes 2-5 and 40). Land and "
                             "full only use cells inside the WSM footprint. Default: underwater")
    parser.add_argument("--grid_resolution", type=float, default=0.1,
                        help="Grid cell size in meters, the same as for the --pred_csv "
                             "grid (default: 0.1)")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output_dir", type=str,
                        default="validation_lidar/lidar_comparison")
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                        help="Fixed easting limits for spatial plot (e.g. 527600 527715)")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Fixed northing limits for spatial plot (e.g. 5340240 5340320)")
    return parser.parse_args()


def sample_wsm_vectorized(wsm_path, eastings, northings):
    return sample_wsm_batch(wsm_path, eastings, northings)


def prediction_grid(e, n, resolution, centred):
    """The prediction's own grid. centred: e, n are cell centres (the 02 CSV)."""
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
    return e0 + (cells % n_cols + 0.5) * resolution, n0 + (cells // n_cols + 0.5) * resolution


def grid_points_fast(x, y, z, grid):
    cell_id, inside = cell_ids(x, y, grid)

    df = pd.DataFrame({"cell": cell_id[inside], "z": z[inside]})
    grouped = df.groupby("cell")["z"].agg(["median", "count"])

    cells = grouped.index.values
    grid_z = grouped["median"].values
    grid_count = grouped["count"].values.astype(int)

    return cells, grid_z, grid_count


def compute_stats(pred, ref, label=""):
    error = pred - ref
    abs_error = np.abs(error)
    n = len(pred)

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(abs_error))
    bias = float(np.mean(error))
    std_err = float(np.std(error))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson_r = float(sp_stats.pearsonr(pred, ref)[0]) if n > 2 else float("nan")
    spearman_r = float(sp_stats.spearmanr(pred, ref)[0]) if n > 2 else float("nan")
    scale = float(np.mean(pred) / np.mean(ref)) if np.mean(ref) != 0 else float("nan")

    return {
        "n": n, "rmse": rmse, "mae": mae, "bias": bias, "std_error": std_err,
        "r2": r2, "pearson_r": pearson_r, "spearman_r": spearman_r,
        "scale_factor": scale,
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        "ref_mean": float(ref.mean()), "ref_std": float(ref.std()),
    }


def print_stats(s, indent="  "):
    print(f"{indent}N:           {s['n']:,}")
    print(f"{indent}RMSE:        {s['rmse']:.4f} m")
    print(f"{indent}MAE:         {s['mae']:.4f} m")
    print(f"{indent}Bias:        {s['bias']:+.4f} m")
    print(f"{indent}Std error:   {s['std_error']:.4f} m")
    print(f"{indent}R²:          {s['r2']:.4f}")
    print(f"{indent}Pearson r:   {s['pearson_r']:.4f}")
    print(f"{indent}Spearman rho: {s['spearman_r']:.4f}")
    print(f"{indent}Scale:       {s['scale_factor']:.3f}x")


def save_ply(path, e, n, z, scalar_fields):
    pts = np.column_stack([e, n, z])
    write_ply_binary(path, pts, scalar_fields=scalar_fields)


def main():
    args = parse_args()
    t0 = time.time()

    res = args.grid_resolution
    mode = args.compare
    pred_col = "depth_m" if mode == "underwater" else "utm_elevation"
    use_npy = mode != "underwater" and bool(args.pred_npy)
    if args.label:
        label = args.label
    elif use_npy:
        label = os.path.basename(os.path.dirname(os.path.abspath(args.pred_npy)))
    else:
        label = os.path.splitext(os.path.basename(args.pred_csv))[0]

    class_map = {
        "underwater": [40],
        "land": [2, 3, 4, 5],
        "full": [2, 3, 4, 5, 40],
    }
    lidar_classes = class_map[mode]
    class_names = {2: "ground", 3: "low_veg", 4: "med_veg", 5: "high_veg",
                   40: "river_bottom", 41: "water_surface", 43: "water_column"}
    mode_label = {"underwater": "Underwater Depth", "land": "Land Elevation",
                  "full": "Full Terrain"}[mode]

    print("=" * 70)
    print(f"LIDAR COMPARISON ({mode_label}): {label}")
    print("=" * 70)

    print("\n--- Step 1: Load prediction ---")
    if not use_npy:
        if not args.pred_csv:
            if mode == "underwater":
                print("ERROR: --pred_csv required for underwater mode")
            else:
                print("ERROR: --pred_npy (or --pred_csv) required for land/full mode")
            sys.exit(1)
        if mode != "underwater":
            print("  WARNING: Using --pred_csv for land/full mode. This only has water cells.")
            print("  For full terrain, provide --pred_npy from 07_make_pointcloud.py")
        pred = pd.read_csv(args.pred_csv)
        print(f"  CSV: {len(pred):,} grid cells")
        pe = pred["utm_easting"].values
        pn = pred["utm_northing"].values
        pv = pred[pred_col].values
        steps = np.concatenate([pe - pe.min(), pn - pn.min()]) / res
        index = np.rint(steps).astype(np.int64)
        # a coarser grid also lies on this lattice, but then all indices share a factor
        if np.abs(steps - index).max() > 0.01 or np.gcd.reduce(index) > 1:
            print(f"ERROR: The cells in {args.pred_csv} are not {res} m apart, "
                  f"set --grid_resolution to the one used in 02")
            sys.exit(1)
        grid = prediction_grid(pe, pn, res, centred=True)
        pred_cells, _ = cell_ids(pe, pn, grid)
    else:
        print(f"  Loading pointcloud: {args.pred_npy}")
        pc = np.load(args.pred_npy)
        print(f"  {len(pc):,} points, Z=[{pc[:,2].min():.1f}, {pc[:,2].max():.1f}]")
        grid = prediction_grid(pc[:, 0], pc[:, 1], res, centred=False)
        pred_cells, pv, _ = grid_points_fast(pc[:, 0], pc[:, 1], pc[:, 2], grid)
        pe, pn = cell_centres(pred_cells, grid)
        print(f"  Gridded to {len(pred_cells):,} cells")
        print(f"  Elevation range: [{pv.min():.2f}, {pv.max():.2f}]")

    print("\n--- Step 2: Load LiDAR ---")
    las = laspy.read(args.lidar_laz)
    cls = np.array(las.classification)
    mask = np.isin(cls, lidar_classes)
    lx = np.array(las.x[mask])
    ly = np.array(las.y[mask])
    lz = np.array(las.z[mask])
    lc = cls[mask]
    print(f"  {mask.sum():,} points (classes {lidar_classes})")
    for c in lidar_classes:
        n_c = (lc == c).sum()
        print(f"    class {c} ({class_names.get(c, '?')}): {n_c:,}")
    if not mask.any():
        print(f"ERROR: No LiDAR points of classes {lidar_classes}")
        sys.exit(1)

    e0, n0, _, n_cols, n_rows = grid
    print(f"\n  Prediction grid: E=[{e0:.1f}, {e0 + n_cols * res:.1f}], "
          f"N=[{n0:.1f}, {n0 + n_rows * res:.1f}] ({n_cols} x {n_rows} cells)")

    print("\n--- Step 3: Grid LiDAR ---")
    lidar_cells, lg_z, lg_count = grid_points_fast(lx, ly, lz, grid)
    if len(lidar_cells) == 0:
        print("ERROR: No LiDAR point inside the prediction grid")
        sys.exit(1)
    print(f"  {len(lidar_cells):,} cells, median pts/cell={int(np.median(lg_count))}")

    print("\n--- Step 4: Compute reference values ---")
    lg_e, lg_n = cell_centres(lidar_cells, grid)
    wsm_elev = sample_wsm_vectorized(args.wsm_path, lg_e, lg_n)
    valid = ~np.isnan(wsm_elev)
    if not valid.any():
        print("ERROR: No LiDAR cell lies on the WSM")
        sys.exit(1)

    if mode == "underwater":
        lidar_ref = wsm_elev[valid] - lg_z[valid]
        value_label = "depth"
    else:
        lidar_ref = lg_z[valid]
        value_label = "elevation"
        print("  Land and full modes only use cells inside the WSM footprint")

    lidar_cells, wsm_elev = lidar_cells[valid], wsm_elev[valid]
    print(f"  {valid.sum():,} LiDAR cells with WSM, ref range="
          f"[{lidar_ref.min():.3f}, {lidar_ref.max():.3f}]")

    print("\n--- Step 5: Match cells ---")
    # every prediction cell is compared with the LiDAR median of the same cell
    pos = np.minimum(np.searchsorted(lidar_cells, pred_cells), len(lidar_cells) - 1)
    pred_idx = np.flatnonzero(lidar_cells[pos] == pred_cells)
    pred_idx = pred_idx[np.argsort(pred_cells[pred_idx], kind="stable")]
    lidar_idx = pos[pred_idx]
    print(f"  Prediction cells: {len(pred_cells):,}, with LiDAR: {len(pred_idx):,}")

    matched_pred_val = pv[pred_idx]
    matched_lidar_val = lidar_ref[lidar_idx]
    matched_wsm = wsm_elev[lidar_idx]
    matched_e, matched_n = pe[pred_idx], pn[pred_idx]

    if mode == "underwater":
        keep = (matched_pred_val > 0) & (matched_lidar_val > 0)
    else:
        keep = np.isfinite(matched_pred_val) & np.isfinite(matched_lidar_val)

    matched_pred_val = matched_pred_val[keep]
    matched_lidar_val = matched_lidar_val[keep]
    matched_wsm = matched_wsm[keep]
    matched_e = matched_e[keep]
    matched_n = matched_n[keep]
    print(f"  Final matched: {len(matched_pred_val):,}")

    if len(matched_pred_val) < 10:
        print("ERROR: Too few matched cells")
        sys.exit(1)

    print(f"\n--- Step 6: Error statistics ({value_label}) ---")
    s = compute_stats(matched_pred_val, matched_lidar_val)
    error = matched_pred_val - matched_lidar_val
    abs_error = np.abs(error)
    print_stats(s)

    strat_results = []
    if mode == "underwater":
        print(f"\n--- Step 7: Depth-stratified ---")
        fmt = "  {:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}"
        print(fmt.format("Thresh", "N", "RMSE", "MAE", "Bias", "Spearman"))
        print("  " + "-" * 55)
        for thresh in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            m = matched_lidar_val > thresh
            if m.sum() < 5:
                continue
            ss = compute_stats(matched_pred_val[m], matched_lidar_val[m])
            strat_results.append({"threshold": thresh, **ss})
            print(f"  >{thresh:4.1f}m  {ss['n']:8d}  {ss['rmse']:8.4f}  "
                  f"{ss['mae']:8.4f}  {ss['bias']:+8.4f}  {ss['spearman_r']:8.4f}")

    print(f"\n--- Step 8: Save outputs ---")
    os.makedirs(args.output_dir, exist_ok=True)

    comp_df = pd.DataFrame({
        "utm_easting": matched_e, "utm_northing": matched_n,
        f"pred_{value_label}": matched_pred_val,
        f"lidar_{value_label}": matched_lidar_val,
        "error_m": error, "abs_error_m": abs_error,
    })
    csv_path = os.path.join(args.output_dir, "lidar_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(comp_df):,} cells)")

    stats_dict = {"label": label, "mode": mode, **s,
                  "depth_stratified": strat_results}
    json_path = os.path.join(args.output_dir, "lidar_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"  Saved: {json_path}")

    txt_path = os.path.join(args.output_dir, "lidar_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"LiDAR {mode_label} Comparison: {label}\n{'=' * 60}\n")
        f.write(f"Mode:        {mode}\nMatched:     {s['n']:,}\n")
        f.write(f"RMSE:        {s['rmse']:.4f} m\nMAE:         {s['mae']:.4f} m\n")
        f.write(f"Bias:        {s['bias']:+.4f} m\nSpearman:    {s['spearman_r']:.4f}\n")
        f.write(f"Scale:       {s['scale_factor']:.3f}x\n")
        if mode != "underwater":
            f.write("Area:        cells inside the WSM footprint\n")
    print(f"  Saved: {txt_path}")

    ply_path = os.path.join(args.output_dir, "lidar_difference.ply")
    if mode == "underwater":
        ply_z = matched_wsm - matched_pred_val
    else:
        ply_z = matched_pred_val

    save_ply(ply_path, matched_e, matched_n, ply_z, {
        f"pred_{value_label}": matched_pred_val,
        f"lidar_{value_label}": matched_lidar_val,
        "error": error,
    })
    print(f"  Saved: {ply_path} ({len(matched_e):,} vertices)")

    print(f"\n--- Step 9: Plots ---")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(matched_lidar_val, matched_pred_val, s=0.3, alpha=0.05, c="steelblue")
    hi = max(matched_lidar_val.max(), matched_pred_val.max())
    if mode == "underwater":
        lo, hi = 0, hi * 1.05
    else:
        lo = min(matched_lidar_val.min(), matched_pred_val.min())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(f"LiDAR {value_label} (m)")
    ax.set_ylabel(f"Predicted {value_label} (m)")
    ax.set_title(f"{label}\nRMSE={s['rmse']:.3f}m, ρ={s['spearman_r']:.3f}, "
                 f"N={s['n']:,}")
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "lidar_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = np.percentile(matched_lidar_val, 99)
    vmin = 0 if mode == "underwater" else np.percentile(matched_lidar_val, 1)
    for ax, vals, title in [
        (axes[0], matched_pred_val, f"Predicted {value_label} (m)"),
        (axes[1], matched_lidar_val, f"LiDAR {value_label} (m)"),
    ]:
        cmap = "viridis" if mode == "underwater" else "viridis_r"
        sc = ax.scatter(matched_e, matched_n, c=vals, s=0.2, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, shrink=0.7)

    err_lim = np.percentile(abs_error, 95)
    sc = axes[2].scatter(matched_e, matched_n, c=error, s=0.2,
                         cmap="RdBu_r", vmin=-err_lim, vmax=err_lim)
    axes[2].set_title(f"Error (m)\nbias={s['bias']:+.3f}, RMSE={s['rmse']:.3f}")
    plt.colorbar(sc, ax=axes[2], shrink=0.7)

    if args.xlim:
        for ax in axes:
            ax.set_xlim(args.xlim[0], args.xlim[1])
    if args.ylim:
        for ax in axes:
            ax.set_ylim(args.ylim[0], args.ylim[1])
    adjustable = 'box' if (args.xlim or args.ylim) else 'datalim'
    for ax in axes:
        ax.set_aspect("equal", adjustable=adjustable)
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
    fig.suptitle(f"LiDAR {mode_label} Comparison: {label}", fontsize=14)
    fig.savefig(os.path.join(args.output_dir, "lidar_spatial_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(error, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(s["bias"], color="red", lw=2, label=f"bias={s['bias']:+.3f}m")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel(f"{value_label.capitalize()} error (pred - LiDAR) [m]")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: N={s['n']:,}")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "lidar_error_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'=' * 70}")
    print(f"DONE in {time.time()-t0:.1f}s. Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
