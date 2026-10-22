"""
Check a model's point cloud at the surveyed land control points, together with the
underwater points from 03_validate.py.

The underwater depths are also scaled by the median ratio of GT to predicted depth at
the reference targets. Writes control_point_comparison.csv, control_point_summary.txt,
four plots and pointcloud_scaled.ply (the point cloud with scaled underwater depths)
to --output_dir.

Usage examples:
    # NeRFrac 50m, water surface from the WSM
    python 14_control_point_validation.py \\
        --comparison_csv validation/nerfrac_50m/comparison.csv \\
        --control_csv ../data/gt/pielach/06_control_points/Pielach_2024102_control_points.csv \\
        --pointcloud_npy pointclouds/nerfrac_50m/pointcloud_combined.npy \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --bathymetry_csv bathymetry/bathymetry_nerfrac_50m.csv \\
        --output_dir validation/nerfrac_50m/control_point_analysis \\
        --label "NeRFrac 50m"

    # plane_fit model: the scaled point cloud takes the water surface from 08's plane
    python 14_control_point_validation.py \\
        --comparison_csv validation/nerfrac_50m_planefit/comparison.csv \\
        --control_csv ../data/gt/pielach/06_control_points/Pielach_2024102_control_points.csv \\
        --pointcloud_npy pointclouds/nerfrac_50m/pointcloud_combined.npy \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --fitted_wsm_json water_surface/nerfrac_50m/ws_analysis_stats.json \\
        --output_dir validation/nerfrac_50m_planefit/control_point_analysis \\
        --label "NeRFrac 50m plane fit"
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import sample_wsm_batch

PLY_VERTEX = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
)
PLY_CHUNK_POINTS = 5_000_000
BACKGROUND_POINTS = 50_000
UNDERWATER_TYPES = ["reference_target", "transect"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a point cloud at the land control points, together with "
        "the underwater points."
    )
    # inputs
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="comparison.csv from 03_validate.py (underwater points)",
    )
    parser.add_argument(
        "--control_csv",
        type=str,
        required=True,
        help="land control points, semicolon-separated",
    )
    parser.add_argument(
        "--pointcloud_npy",
        type=str,
        required=True,
        help="pointcloud_combined.npy from 07_make_pointcloud.py; the .ply next to "
        "it becomes pointcloud_scaled.ply",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF",
    )
    parser.add_argument(
        "--bathymetry_csv",
        type=str,
        default="",
        help="bathymetry CSV from 02_extract_bathymetry.py, the background of the "
        "spatial error map",
    )
    parser.add_argument(
        "--fitted_wsm_json",
        type=str,
        default="",
        help="ws_analysis_stats.json from 08_water_surface_analysis.py. "
        "pointcloud_scaled.ply then takes the water surface from NeRFrac's fitted "
        "plane instead of the WSM, for plane_fit models where the WSM covers too "
        "little of the river",
    )

    # matching and output
    parser.add_argument(
        "--max_dist",
        type=float,
        default=0.5,
        help="horizontal radius in m for the point cloud points of a control point",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output folder (default: control_point_analysis/ next to "
        "--comparison_csv)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="model name for the plots and the summary (default: the folder of "
        "--comparison_csv)",
    )
    return parser.parse_args()


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def sort_by_name(df):
    """Rows in natural order of the names: GCP_L2, GCP_L3, ..., GCP_L21."""
    return df.sort_values("name", key=lambda s: s.map(natural_key))


def control_group(name):
    """Group of a control point name: SR_A of SR_A1, NP of NP_3."""
    m = re.match(r"([A-Z]+_[A-Z]+|[A-Z]+)", name)
    return m.group(1) if m else name


def cumulative_distance(easting, northing):
    return np.concatenate(
        [[0], np.cumsum(np.sqrt(np.diff(easting) ** 2 + np.diff(northing) ** 2))]
    )


def rmse_mae_bias(errors):
    if len(errors) == 0:
        return np.nan, np.nan, np.nan
    return np.sqrt(np.mean(errors**2)), np.mean(np.abs(errors)), np.mean(errors)


def load_control_points(path):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "Point Id": "name",
            "Easting [m]": "easting",
            "Northing [m]": "northing",
            "WGS84 Ellip. Height [m]": "elevation",
        }
    )
    df["name"] = df["name"].astype(str)
    df["elevation"] = pd.to_numeric(
        df["elevation"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )
    return df


def match_control_to_pointcloud(control_df, npy_path, max_dist_m):
    """Median elevation of the point cloud within max_dist_m of each control point."""
    points = np.load(npy_path, mmap_mode="r")
    buffer_m = max(max_dist_m * 2, 5.0)
    easting = control_df["easting"].values
    northing = control_df["northing"].values
    in_box = (
        (points[:, 0] >= easting.min() - buffer_m)
        & (points[:, 0] <= easting.max() + buffer_m)
        & (points[:, 1] >= northing.min() - buffer_m)
        & (points[:, 1] <= northing.max() + buffer_m)
    )
    nearby = np.array(points[in_box])
    print(
        f"{len(nearby):,} of {len(points):,} point cloud points near the control points"
    )

    pred_elev = np.full(len(control_df), np.nan)
    n_matches = np.zeros(len(control_df), dtype=int)
    if len(nearby) > 0:
        tree = cKDTree(nearby[:, :2])
        for i, xy in enumerate(np.column_stack([easting, northing])):
            idxs = tree.query_ball_point(xy, r=max_dist_m)
            if idxs:
                pred_elev[i] = np.median(nearby[idxs, 2])
            n_matches[i] = len(idxs)
    return pd.DataFrame(
        {
            "name": control_df["name"].values,
            "gt_easting": easting,
            "gt_northing": northing,
            "gt_elevation": control_df["elevation"].values,
            "pred_elevation": pred_elev,
            "n_matches": n_matches,
        }
    )


def reference_scale(comp):
    """Median ratio of GT to predicted depth at the reference targets.

    1 with fewer than two targets or a predicted depth of 0 or less.
    """
    ref = comp["type"] == "reference_target"
    gt_depth = comp.loc[ref, "gt_depth_m"].values
    pred_depth = comp.loc[ref, "pred_depth_m"].values
    if len(gt_depth) >= 2 and np.all(pred_depth > 0):
        return np.median(gt_depth / pred_depth)
    print("Fewer than 2 reference targets or a predicted depth <= 0, scale factor 1")
    return 1.0


def build_unified(comp, ctrl, scale_factor):
    """One table of underwater and control points with raw and scaled elevations.

    The scale factor applies to the underwater depths; control points keep the raw
    point cloud elevation.
    """
    uw = pd.DataFrame(
        {
            "name": comp["name"].values,
            "type": comp["type"].values,
            "gt_easting": comp["gt_easting"].values,
            "gt_northing": comp["gt_northing"].values,
            "gt_elevation": comp["gt_elevation"].values,
            "ws_elevation": comp["ws_elevation"].values,
            "gt_depth_m": comp["gt_depth_m"].values,
            "pred_elevation": comp["pred_elevation"].values,
            "pred_depth_m": comp["pred_depth_m"].values,
            "transect": comp["transect"].values if "transect" in comp.columns else "",
            "n_matches": 1,
        }
    )
    ws_elev = ctrl["ws_elevation"].values
    gt_elev = ctrl["gt_elevation"].values
    pred_elev = ctrl["pred_elevation"].values
    land = pd.DataFrame(
        {
            "name": ctrl["name"].values,
            "type": "control_point",
            "gt_easting": ctrl["gt_easting"].values,
            "gt_northing": ctrl["gt_northing"].values,
            "gt_elevation": gt_elev,
            "ws_elevation": ws_elev,
            "gt_depth_m": np.where(
                np.isfinite(ws_elev) & np.isfinite(gt_elev), ws_elev - gt_elev, np.nan
            ),
            "pred_elevation": pred_elev,
            "pred_depth_m": np.where(
                np.isfinite(ws_elev) & np.isfinite(pred_elev),
                ws_elev - pred_elev,
                np.nan,
            ),
            "transect": "",
            "n_matches": ctrl["n_matches"].values,
        }
    )
    unified = pd.concat([uw, land], ignore_index=True)

    is_underwater = unified["type"].isin(UNDERWATER_TYPES)
    is_land = unified["type"] == "control_point"
    unified["scaled_pred_depth_m"] = unified["pred_depth_m"].copy()
    unified.loc[is_underwater, "scaled_pred_depth_m"] = (
        unified.loc[is_underwater, "pred_depth_m"] * scale_factor
    )
    unified["scaled_pred_elevation"] = (
        unified["ws_elevation"] - unified["scaled_pred_depth_m"]
    )
    unified.loc[is_land, "scaled_pred_elevation"] = unified.loc[
        is_land, "pred_elevation"
    ]
    unified["elev_error_raw"] = unified["pred_elevation"] - unified["gt_elevation"]
    unified["elev_error_scaled"] = (
        unified["scaled_pred_elevation"] - unified["gt_elevation"]
    )
    unified["matched"] = unified["pred_elevation"].notna()
    return unified


def fmt(v):
    return f"{v:8.3f}" if np.isfinite(v) else "     n/a"


def write_summary(
    unified, scale_factor, label, max_dist_m, n_ctrl_matched, n_ctrl, path
):
    matched = unified[unified["matched"]]
    n_uw = unified["type"].isin(UNDERWATER_TYPES).sum()
    n_land = (unified["type"] == "control_point").sum()
    lines = [
        f"CONTROL POINT VALIDATION: {label}",
        f"Scale factor: {scale_factor:.4f}",
        f"Max match distance: {max_dist_m}m",
        f"Control points matched: {n_ctrl_matched}/{n_ctrl}",
        "",
        "ELEVATION ERROR (pred_elevation - gt_elevation)",
        f"{'Type':<25s} {'N':>4s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s} | "
        f"{'RMSE_sc':>8s} {'MAE_sc':>8s} {'Bias_sc':>8s}",
        "-" * 90,
    ]
    groups = [
        ("reference_target", matched["type"] == "reference_target"),
        ("transect", matched["type"] == "transect"),
        ("control_point", matched["type"] == "control_point"),
        ("all_underwater", matched["type"].isin(UNDERWATER_TYPES)),
        ("all", np.ones(len(matched), dtype=bool)),
    ]
    for point_type, in_group in groups:
        rows = matched[in_group]
        if rows.empty:
            continue
        raw = " ".join(
            fmt(v) for v in rmse_mae_bias(rows["elev_error_raw"].dropna().values)
        )
        scaled = " ".join(
            fmt(v) for v in rmse_mae_bias(rows["elev_error_scaled"].dropna().values)
        )
        lines.append(f"{point_type:<25s} {len(rows):4d} {raw} | {scaled}")
    lines += [
        "",
        "NOTE: elevation_error = pred_elevation - gt_elevation.",
        "      Positive = prediction is too HIGH (too shallow for water).",
        f"      Scale correction applied to {n_uw} underwater points, "
        f"{n_land} land control points kept raw.",
    ]

    ctrl = matched[matched["type"] == "control_point"]
    if not ctrl.empty:
        lines += [
            "",
            "CONTROL POINT DETAILS",
            f"{'Name':<12s} {'GT Elev':>9s} {'Pred Elev':>10s} {'Error':>8s} "
            f"{'Sc Elev':>10s} {'Sc Err':>8s} {'N pts':>6s}",
            "-" * 72,
        ]
        for _, r in sort_by_name(ctrl).iterrows():
            lines.append(
                f"{r['name']:<12s} {fmt(r['gt_elevation'])} {fmt(r['pred_elevation'])} "
                f"{fmt(r['elev_error_raw'])} {fmt(r['scaled_pred_elevation'])} "
                f"{fmt(r['elev_error_scaled'])} {r['n_matches']:6.0f}"
            )

    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"Saved: {path}")


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_analysis(matched, scale_factor, label, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def scatter_panel(ax, elev_col, title):
        for point_type, color, marker, size, zorder in [
            ("transect", "tab:blue", "o", 15, 1),
            ("reference_target", "red", "*", 60, 5),
            ("control_point", "green", "D", 30, 3),
        ]:
            sub = matched[matched["type"] == point_type]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["gt_elevation"],
                sub[elev_col],
                s=size,
                alpha=0.7,
                c=color,
                marker=marker,
                label=point_type,
                zorder=zorder,
            )
        all_gt = matched["gt_elevation"]
        all_pred = matched[elev_col].dropna()
        lo = min(all_gt.min(), all_pred.min()) - 0.5
        hi = max(all_gt.max(), all_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlabel("GT Elevation (m)")
        ax.set_ylabel("Predicted Elevation (m)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    scatter_panel(axes[0], "pred_elevation", "Before Scaling")
    scatter_panel(
        axes[1], "scaled_pred_elevation", f"After Scaling (x{scale_factor:.3f})"
    )

    ax = axes[2]
    bar_types = ["reference_target", "transect", "control_point", "all"]
    bar_labels = ["Ref targets", "Transect", "Control pts", "All"]
    rmse_before = []
    rmse_after = []
    for point_type in bar_types:
        sub = matched if point_type == "all" else matched[matched["type"] == point_type]
        raw_err = sub["elev_error_raw"].dropna().values
        sc_err = sub["elev_error_scaled"].dropna().values
        rmse_before.append(np.sqrt(np.mean(raw_err**2)) if len(raw_err) > 0 else 0)
        rmse_after.append(np.sqrt(np.mean(sc_err**2)) if len(sc_err) > 0 else 0)

    x = np.arange(len(bar_labels))
    width = 0.35
    ax.bar(x - width / 2, rmse_before, width, label="Before scaling", color="salmon")
    ax.bar(x + width / 2, rmse_after, width, label="After scaling", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Elevation RMSE (m)")
    ax.set_title("RMSE by Point Type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Control Point Validation — {label}", fontsize=13, y=1.02)
    save_figure(path)


def transect_panels(matched):
    """(title, rows) of each transect, in natural order of the names.

    A transect name can label separate runs of rows (QP15 has two cross-sections),
    and each run gets its own panel.
    """
    rows = matched[matched["transect"].notna() & (matched["transect"] != "")].copy()
    rows["run"] = (rows["transect"] != rows["transect"].shift()).cumsum()
    panels = []
    for name in sorted(rows["transect"].unique(), key=natural_key):
        runs = rows.loc[rows["transect"] == name, "run"].unique()
        for k, run in enumerate(runs, 1):
            title = name if len(runs) == 1 else f"{name} ({k}/{len(runs)})"
            panels.append((title, rows[rows["run"] == run]))
    return panels


def plot_transects(panels, label, path):
    n_panels = len(panels)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, (title, sub) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        cum_dist = cumulative_distance(
            sub["gt_easting"].values, sub["gt_northing"].values
        )
        gt_elev = sub["gt_elevation"].values
        ws_elev = sub["ws_elevation"].values
        pred_elev = sub["pred_elevation"].values
        scaled_elev = sub["scaled_pred_elevation"].values

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
        pred_valid = np.isfinite(pred_elev)
        if np.any(pred_valid):
            ax.plot(
                cum_dist[pred_valid],
                pred_elev[pred_valid],
                "-^",
                color="orange",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
                label="Predicted (raw)",
            )
        sc_valid = np.isfinite(scaled_elev)
        if np.any(sc_valid):
            ax.plot(
                cum_dist[sc_valid],
                scaled_elev[sc_valid],
                "r-s",
                markersize=3,
                linewidth=1.5,
                label="Predicted (scaled)",
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Distance (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        f"Transect Profiles (before/after scaling) — {label}", fontsize=14, y=1.01
    )
    save_figure(path)


def plot_land_profiles(ctrl_pts, label, path):
    """Elevation profiles of the control point groups (SR_A, GCP_L, NP, ...)."""
    group = ctrl_pts["name"].map(control_group)
    groups = sorted(group.unique())
    n_groups = len(groups)
    ncols = min(4, n_groups)
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, gname in enumerate(groups):
        ax = axes[idx // ncols, idx % ncols]
        sub = sort_by_name(ctrl_pts[group == gname])
        cum_dist = cumulative_distance(
            sub["gt_easting"].values, sub["gt_northing"].values
        )
        gt_elev = sub["gt_elevation"].values
        pred_elev = sub["pred_elevation"].values

        ax.plot(
            cum_dist, gt_elev, "k-o", markersize=4, linewidth=1.5, label="GT elevation"
        )
        pred_valid = np.isfinite(pred_elev)
        if np.any(pred_valid):
            ax.plot(
                cum_dist[pred_valid],
                pred_elev[pred_valid],
                "-^",
                color="orange",
                markersize=4,
                linewidth=1.5,
                label="Predicted",
            )
        for i, name in enumerate(sub["name"]):
            ax.annotate(
                name,
                (cum_dist[i], gt_elev[i]),
                fontsize=5,
                alpha=0.6,
                rotation=30,
                ha="left",
                va="bottom",
            )

        rmse, _, bias = rmse_mae_bias(pred_elev[pred_valid] - gt_elev[pred_valid])
        ax.set_title(
            f"{gname} (n={len(sub)}, RMSE={rmse:.2f}m, bias={bias:+.2f}m)", fontsize=9
        )
        ax.set_xlabel("Distance (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Land Control Point Profiles — {label}", fontsize=14, y=1.01)
    save_figure(path)


def load_background(path):
    """Up to BACKGROUND_POINTS cells of the bathymetry CSV, or None."""
    if not path:
        return None
    try:
        bathy = pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        print(f"Could not read {path}, spatial error map without background")
        return None
    columns = ["utm_easting", "utm_northing", "depth_m"]
    if not set(columns) <= set(bathy.columns) or not all(
        pd.api.types.is_numeric_dtype(bathy[c]) for c in columns
    ):
        print(
            f"{path} has no numeric utm_easting/utm_northing/depth_m, spatial error "
            "map without background"
        )
        return None
    if len(bathy) > BACKGROUND_POINTS:
        bathy = bathy.sample(n=BACKGROUND_POINTS, random_state=42)
    return bathy


def plot_spatial_errors(matched, background, label, path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (err_col, title_suffix) in zip(
        axes,
        [("elev_error_raw", "Before Scaling"), ("elev_error_scaled", "After Scaling")],
    ):
        if background is not None:
            ax.scatter(
                background["utm_easting"],
                background["utm_northing"],
                c=background["depth_m"],
                cmap="Blues",
                s=1,
                alpha=0.3,
                zorder=0,
                rasterized=True,
            )

        valid_err = matched[err_col].dropna().abs()
        vmax = max(valid_err.quantile(0.95), 1.0) if len(valid_err) > 0 else 5.0

        sc_handle = None
        trn_pts = matched[matched["type"] == "transect"]
        if len(trn_pts) > 0:
            sc_handle = ax.scatter(
                trn_pts["gt_easting"],
                trn_pts["gt_northing"],
                c=trn_pts[err_col].abs(),
                cmap="RdYlGn_r",
                vmin=0,
                vmax=vmax,
                s=30,
                alpha=0.8,
                zorder=2,
                label="transect",
            )
        ref_pts = matched[matched["type"] == "reference_target"]
        if len(ref_pts) > 0:
            ax.scatter(
                ref_pts["gt_easting"],
                ref_pts["gt_northing"],
                c=ref_pts[err_col].abs(),
                cmap="RdYlGn_r",
                vmin=0,
                vmax=vmax,
                s=100,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="ref target",
            )
        ctrl_pts = matched[matched["type"] == "control_point"]
        if len(ctrl_pts) > 0:
            ax.scatter(
                ctrl_pts["gt_easting"],
                ctrl_pts["gt_northing"],
                c=ctrl_pts[err_col].abs(),
                cmap="RdYlGn_r",
                vmin=0,
                vmax=vmax,
                s=50,
                marker="D",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
                label="control pt",
            )
        if sc_handle is not None:
            plt.colorbar(sc_handle, ax=ax, label="|Elevation error| (m)")

        trn_labeled = matched[matched["transect"].notna() & (matched["transect"] != "")]
        for t in trn_labeled["transect"].unique():
            sub = trn_labeled[trn_labeled["transect"] == t]
            mid = sub.iloc[len(sub) // 2]
            ax.annotate(
                t, (mid["gt_easting"], mid["gt_northing"]), fontsize=6, alpha=0.7
            )

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Spatial Error Map — {title_suffix}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Spatial Error — {label}", fontsize=14, y=1.02)
    save_figure(path)


def read_ply_header(path):
    """Number of vertices and header size of a PLY from 07_make_pointcloud.py."""
    n_vertices = None
    with open(path, "rb") as f:
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError(f"No end_header in {path}")
            line = raw_line.decode("ascii", errors="replace").strip()
            if n_vertices is None and line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line == "end_header":
                return n_vertices, f.tell()


def water_surface(wsm_path, fitted_json):
    """Water surface elevation as a function of easting and northing: the WSM, or
    NeRFrac's fitted plane from fitted_json."""
    if fitted_json:
        with open(fitted_json) as f:
            plane = json.load(f)["nerfrac_plane"]
        pnx, pny, pnz = plane["normal"]
        pcx, pcy, pcz = plane["centroid_utm33"]
        print(f"Water surface of the scaled PLY: fitted plane from {fitted_json}")
        return lambda e, n: pcz - (pnx * (e - pcx) + pny * (n - pcy)) / pnz

    with rasterio.open(wsm_path) as wsm:
        transform, data, nodata = wsm.transform, wsm.read(1), wsm.nodata
    print("Water surface of the scaled PLY: WSM")

    def wsm_elevation(e, n):
        rows, cols = rasterio.transform.rowcol(transform, e, n)
        rows = np.array(rows)
        cols = np.array(cols)
        h, w = data.shape
        on_wsm = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        elev = np.full(len(e), np.nan)
        if np.any(on_wsm):
            values = data[rows[on_wsm], cols[on_wsm]]
            if nodata is not None:
                values = np.where(values == nodata, np.nan, values)
            elev[on_wsm] = values.astype(np.float64)
        return elev

    return wsm_elevation


def write_scaled_ply(src_path, out_path, scale_factor, wsm_path, fitted_json):
    """Write the source PLY with the depth of every point below the water surface
    multiplied by scale_factor."""
    n_vertices, header_bytes = read_ply_header(src_path)
    if n_vertices is None or n_vertices < 0:
        raise ValueError(f"No valid vertex count in the header of {src_path}")
    water_level = water_surface(wsm_path, fitted_json)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # written to a .part file first, so that a broken source leaves no partial PLY
    part_path = out_path + ".part"
    n_scaled = 0
    try:
        with open(src_path, "rb") as src, open(part_path, "wb") as out:
            src.seek(header_bytes)
            out.write(header.encode("ascii"))
            for start in range(0, n_vertices, PLY_CHUNK_POINTS):
                n = min(PLY_CHUNK_POINTS, n_vertices - start)
                raw = src.read(n * PLY_VERTEX.itemsize)
                if len(raw) < n * PLY_VERTEX.itemsize:
                    raise OSError(
                        f"{src_path} ends at vertex {start:,} of {n_vertices:,}"
                    )
                verts = np.frombuffer(raw, dtype=PLY_VERTEX)
                easting = verts["x"].astype(np.float64)
                northing = verts["y"].astype(np.float64)
                elevation = verts["z"].astype(np.float64)

                ws_elev = water_level(easting, northing)
                underwater = np.isfinite(ws_elev) & (elevation < ws_elev)
                scaled_elev = elevation.copy()
                if np.any(underwater):
                    depth = ws_elev[underwater] - elevation[underwater]
                    scaled_elev[underwater] = ws_elev[underwater] - depth * scale_factor
                    n_scaled += int(np.sum(underwater))

                out_verts = np.empty(n, dtype=PLY_VERTEX)
                out_verts["x"] = easting.astype(np.float32)
                out_verts["y"] = northing.astype(np.float32)
                out_verts["z"] = scaled_elev.astype(np.float32)
                for c in "rgb":
                    out_verts[c] = verts[c]
                out.write(out_verts.tobytes())
        os.replace(part_path, out_path)
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)

    print(
        f"Saved: {out_path} ({n_vertices:,} points, {n_scaled:,} underwater points "
        f"scaled by {scale_factor:.4f})"
    )
    if n_scaled == 0:
        print("  no point lies below the water surface, so the PLY is unscaled")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.comparison_csv), "control_point_analysis"
    )
    os.makedirs(output_dir, exist_ok=True)
    label = args.label or os.path.basename(
        os.path.dirname(os.path.abspath(args.comparison_csv))
    )
    print(f"Control point validation of {label}")

    comp = pd.read_csv(args.comparison_csv)
    comp = comp[comp["matched"] == "yes"].copy()
    comp["pred_elevation"] = comp["ws_elevation"] - comp["pred_depth_m"]
    ctrl = load_control_points(args.control_csv)
    print(f"{len(comp)} matched underwater points, {len(ctrl)} control points")

    ctrl = match_control_to_pointcloud(ctrl, args.pointcloud_npy, args.max_dist)
    n_ctrl_matched = (ctrl["n_matches"] > 0).sum()
    print(
        f"{n_ctrl_matched}/{len(ctrl)} control points matched within {args.max_dist} m"
    )

    # control points off the WSM get the mean water level of the underwater points
    ctrl["ws_elevation"] = sample_wsm_batch(
        args.wsm_path, ctrl["gt_easting"].values, ctrl["gt_northing"].values
    )
    n_off_wsm = ctrl["ws_elevation"].isna().sum()
    if n_off_wsm > 0:
        ws_fallback = comp["ws_elevation"].mean()
        print(
            f"{n_off_wsm} control points off the WSM get the mean water level of the "
            f"underwater points, {ws_fallback:.2f} m"
        )
        ctrl["ws_elevation"] = ctrl["ws_elevation"].fillna(ws_fallback)

    scale_factor = reference_scale(comp)
    print(f"Scale factor: {scale_factor:.4f}")
    unified = build_unified(comp, ctrl, scale_factor)
    csv_path = os.path.join(output_dir, "control_point_comparison.csv")
    unified.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")

    write_summary(
        unified,
        scale_factor,
        label,
        args.max_dist,
        n_ctrl_matched,
        len(ctrl),
        os.path.join(output_dir, "control_point_summary.txt"),
    )

    matched = unified[unified["matched"]].copy()
    plot_analysis(
        matched,
        scale_factor,
        label,
        os.path.join(output_dir, "control_point_analysis.png"),
    )
    panels = transect_panels(matched)
    if panels:
        plot_transects(panels, label, os.path.join(output_dir, "transects.png"))
    else:
        print("No transect points, no transect plot")
    ctrl_pts = matched[matched["type"] == "control_point"]
    if not ctrl_pts.empty:
        plot_land_profiles(
            ctrl_pts, label, os.path.join(output_dir, "transects_land.png")
        )
    else:
        print("No matched control points, no land profile plot")
    plot_spatial_errors(
        matched,
        load_background(args.bathymetry_csv),
        label,
        os.path.join(output_dir, "spatial_error_map.png"),
    )

    src_ply = os.path.splitext(args.pointcloud_npy)[0] + ".ply"
    if os.path.isfile(src_ply):
        write_scaled_ply(
            src_ply,
            os.path.join(output_dir, "pointcloud_scaled.ply"),
            scale_factor,
            args.wsm_path,
            args.fitted_wsm_json,
        )
    else:
        print(f"No {src_ply}, no pointcloud_scaled.ply")


if __name__ == "__main__":
    main()
