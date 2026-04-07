#!/usr/bin/env python
"""14_control_point_validation.py — Land control point + underwater combined analysis.

Usage:
    python 14_control_point_validation.py \\
        --comparison_csv validation/<model>/comparison.csv \\
        --control_csv /path/to/control_points.csv \\
        --pointcloud_npy pointclouds/<model>/pointcloud_combined.npy \\
        --wsm_path /path/to/WSM.tif \\
        --bathymetry_csv bathymetry/<model>.csv \\
        --output_dir validation/<model>/control_point_analysis/ \\
        --max_dist 0.5 \\
        --label "Model Name"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from coord_utils import compute_metrics, sample_wsm_batch



def log(msg):
    print(msg, flush=True)


def load_control_points(path):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Point Id": "name",
        "Easting [m]": "easting",
        "Northing [m]": "northing",
        "WGS84 Ellip. Height [m]": "elevation",
        "Code": "code",
    }
    df = df.rename(columns=rename)
    df["elevation"] = pd.to_numeric(
        df["elevation"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )
    return df


def sample_wsm(wsm_path, eastings, northings):
    return sample_wsm_batch(wsm_path, np.asarray(eastings), np.asarray(northings))


def match_control_to_pointcloud(control_df, npy_path, max_dist=0.5):
    log(f"Loading point cloud: {npy_path}")
    points = np.load(npy_path, mmap_mode="r")
    log(f"  Shape: {points.shape}, dtype: {points.dtype}")

    buffer = max(max_dist * 2, 5.0)
    ctrl_eas = control_df["easting"].values
    ctrl_nor = control_df["northing"].values
    bb = [
        ctrl_eas.min() - buffer,
        ctrl_eas.max() + buffer,
        ctrl_nor.min() - buffer,
        ctrl_nor.max() + buffer,
    ]

    log(f"  Filtering to bbox: E[{bb[0]:.0f}, {bb[1]:.0f}] N[{bb[2]:.0f}, {bb[3]:.0f}]")
    mask = (
        (points[:, 0] >= bb[0])
        & (points[:, 0] <= bb[1])
        & (points[:, 1] >= bb[2])
        & (points[:, 1] <= bb[3])
    )
    filtered = np.array(points[mask])
    del points
    log(f"  {len(filtered):,} points in control area")

    if len(filtered) == 0:
        log("[WARN] No point cloud points within control point bounding box")
        return pd.DataFrame(
            {
                "name": control_df["name"].values,
                "gt_easting": ctrl_eas,
                "gt_northing": ctrl_nor,
                "gt_elevation": control_df["elevation"].values,
                "pred_elevation": np.nan,
                "n_matches": 0,
                "match_dist_m": np.nan,
            }
        )

    if len(filtered) > 50_000_000:
        log(
            f"  [NOTE] Large filtered set ({len(filtered):,} pts). "
            f"Tree building may take ~1 min..."
        )
    log("  Building cKDTree...")
    tree = cKDTree(filtered[:, :2])

    results = []
    for _, row in control_df.iterrows():
        xy = np.array([row["easting"], row["northing"]])
        idxs = tree.query_ball_point(xy, r=max_dist)
        if len(idxs) > 0:
            pred_elev = np.median(filtered[idxs, 2])
            dist, _ = tree.query(xy, k=1)
            results.append(
                {
                    "name": row["name"],
                    "gt_easting": row["easting"],
                    "gt_northing": row["northing"],
                    "gt_elevation": row["elevation"],
                    "pred_elevation": pred_elev,
                    "n_matches": len(idxs),
                    "match_dist_m": dist,
                }
            )
        else:
            dist, _ = tree.query(xy, k=1)
            results.append(
                {
                    "name": row["name"],
                    "gt_easting": row["easting"],
                    "gt_northing": row["northing"],
                    "gt_elevation": row["elevation"],
                    "pred_elevation": np.nan,
                    "n_matches": 0,
                    "match_dist_m": dist,
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Land control point validation + combined land/water analysis."
    )
    parser.add_argument(
        "--comparison_csv",
        required=True,
        help="comparison.csv from step 03 (underwater points)",
    )
    parser.add_argument(
        "--control_csv",
        required=True,
        help="Land control points CSV (semicolon-separated)",
    )
    parser.add_argument(
        "--pointcloud_npy",
        required=True,
        help="pointcloud_combined.npy from step 07",
    )
    parser.add_argument(
        "--wsm_path",
        required=True,
        help="Water surface model GeoTIFF",
    )
    parser.add_argument(
        "--bathymetry_csv",
        default="",
        help="Bathymetry CSV for spatial map background (optional)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory (default: control_point_analysis/ alongside comparison_csv)",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=0.5,
        help="Max horizontal matching distance in meters (default: 0.5)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Model label for plots",
    )
    parser.add_argument(
        "--fitted_wsm_json",
        default="",
        help="ws_analysis_stats.json from step 08.  When provided, the scaled "
        "PLY uses the NeRFrac fitted water-surface plane to classify "
        "underwater points instead of the GT WSM raster.  Intended for "
        "planefit models where the GT WSM has limited spatial coverage.",
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(
            os.path.dirname(args.comparison_csv), "control_point_analysis"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.label:
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.comparison_csv)))
        args.label = parent

    log(f"=== 14_control_point_validation: {args.label} ===")

    log("Loading underwater comparison data...")
    comp = pd.read_csv(args.comparison_csv)
    comp = comp[comp["matched"] == "yes"].copy()
    log(f"  {len(comp)} matched underwater points")

    comp["pred_elevation"] = comp["ws_elevation"] - comp["pred_depth_m"]

    log("Loading control points...")
    ctrl = load_control_points(args.control_csv)
    log(f"  {len(ctrl)} control points loaded")

    log("Matching control points to point cloud...")
    ctrl_matched = match_control_to_pointcloud(ctrl, args.pointcloud_npy, args.max_dist)

    n_matched = (ctrl_matched["n_matches"] > 0).sum()
    n_total = len(ctrl_matched)
    log(f"  {n_matched}/{n_total} control points matched within {args.max_dist}m")

    log("Sampling WSM at control points...")
    ctrl_matched["ws_elevation"] = sample_wsm(
        args.wsm_path,
        ctrl_matched["gt_easting"].values,
        ctrl_matched["gt_northing"].values,
    )
    n_wsm = ctrl_matched["ws_elevation"].notna().sum()
    log(f"  {n_wsm}/{n_total} control points have WSM values")

    ws_fallback = comp["ws_elevation"].mean()
    n_fill = ctrl_matched["ws_elevation"].isna().sum()
    if n_fill > 0:
        log(
            f"  Filling {n_fill} missing WSM values with mean={ws_fallback:.2f}m "
            f"(from underwater points)"
        )
        ctrl_matched["ws_elevation"] = ctrl_matched["ws_elevation"].fillna(ws_fallback)


    ref_mask_uw = comp["type"] == "reference_target"
    ref_gt_depth = comp.loc[ref_mask_uw, "gt_depth_m"].values
    ref_pred_depth = comp.loc[ref_mask_uw, "pred_depth_m"].values

    if len(ref_gt_depth) >= 2 and np.all(ref_pred_depth > 0):
        scale_factor = np.median(ref_gt_depth / ref_pred_depth)
    else:
        scale_factor = 1.0
        log("[WARN] Could not compute scale factor, using 1.0")
    log(f"  Scale factor: {scale_factor:.4f}")

    log("Building unified comparison...")

    uw_rows = []
    for _, r in comp.iterrows():
        uw_rows.append(
            {
                "name": r["name"],
                "type": r["type"],
                "gt_easting": r["gt_easting"],
                "gt_northing": r["gt_northing"],
                "gt_elevation": r["gt_elevation"],
                "ws_elevation": r["ws_elevation"],
                "gt_depth_m": r["gt_depth_m"],
                "pred_elevation": r["pred_elevation"],
                "pred_depth_m": r["pred_depth_m"],
                "transect": r.get("transect", ""),
                "n_matches": 1,
            }
        )


    ctrl_rows = []
    for _, r in ctrl_matched.iterrows():
        ws = r["ws_elevation"]
        pred_e = r["pred_elevation"]
        gt_e = r["gt_elevation"]

        gt_depth = (ws - gt_e) if np.isfinite(ws) and np.isfinite(gt_e) else np.nan
        pred_depth = (
            (ws - pred_e) if np.isfinite(ws) and np.isfinite(pred_e) else np.nan
        )

        ctrl_rows.append(
            {
                "name": r["name"],
                "type": "control_point",
                "gt_easting": r["gt_easting"],
                "gt_northing": r["gt_northing"],
                "gt_elevation": gt_e,
                "ws_elevation": ws,
                "gt_depth_m": gt_depth,
                "pred_elevation": pred_e,
                "pred_depth_m": pred_depth,
                "transect": "",
                "n_matches": r["n_matches"],
            }
        )

    unified = pd.DataFrame(uw_rows + ctrl_rows)

    is_underwater = unified["type"].isin(["reference_target", "transect"])
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

    n_uw_scaled = is_underwater.sum()
    n_land_raw = is_land.sum()
    log(
        f"  Scale correction: {n_uw_scaled} underwater points scaled, "
        f"{n_land_raw} land control points raw"
    )

    unified["elev_error_raw"] = unified["pred_elevation"] - unified["gt_elevation"]
    unified["elev_error_scaled"] = (
        unified["scaled_pred_elevation"] - unified["gt_elevation"]
    )

    unified["matched"] = unified["pred_elevation"].notna()

    n_uw_matched = len(comp)
    n_ctrl_matched = n_matched
    n_unified = unified["matched"].sum()
    log(
        f"  Unified: {len(unified)} rows total, "
        f"{n_unified} with predictions "
        f"({n_uw_matched} underwater + {n_ctrl_matched} control)"
    )

    csv_path = os.path.join(args.output_dir, "control_point_comparison.csv")
    unified.to_csv(csv_path, index=False, float_format="%.4f")
    log(f"Saved: {csv_path}")

    matched = unified[unified["matched"]].copy()

    type_groups = [
        ("reference_target", matched[matched["type"] == "reference_target"]),
        ("transect", matched[matched["type"] == "transect"]),
        ("control_point", matched[matched["type"] == "control_point"]),
        (
            "all_underwater",
            matched[matched["type"].isin(["reference_target", "transect"])],
        ),
        ("all", matched),
    ]

    txt_lines = []
    txt_lines.append(f"CONTROL POINT VALIDATION — {args.label}")
    txt_lines.append(f"Scale factor: {scale_factor:.4f}")
    txt_lines.append(f"Max match distance: {args.max_dist}m")
    txt_lines.append(f"Control points matched: {n_matched}/{n_total}")
    txt_lines.append("")
    txt_lines.append("ELEVATION ERROR (pred_elevation - gt_elevation)")
    txt_lines.append(
        f"{'Type':<25s} {'N':>4s} "
        f"{'RMSE':>8s} {'MAE':>8s} {'Bias':>8s} | "
        f"{'RMSE_sc':>8s} {'MAE_sc':>8s} {'Bias_sc':>8s}"
    )
    txt_lines.append("-" * 90)

    def fmt(v):
        return f"{v:8.3f}" if np.isfinite(v) else "     n/a"

    for tname, tdf in type_groups:
        if len(tdf) < 1:
            continue
        raw_err = tdf["elev_error_raw"].dropna().values
        sc_err = tdf["elev_error_scaled"].dropna().values

        if len(raw_err) > 0:
            rmse_r = np.sqrt(np.mean(raw_err**2))
            mae_r = np.mean(np.abs(raw_err))
            bias_r = np.mean(raw_err)
        else:
            rmse_r = mae_r = bias_r = np.nan

        if len(sc_err) > 0:
            rmse_s = np.sqrt(np.mean(sc_err**2))
            mae_s = np.mean(np.abs(sc_err))
            bias_s = np.mean(sc_err)
        else:
            rmse_s = mae_s = bias_s = np.nan

        txt_lines.append(
            f"{tname:<25s} {len(tdf):4d} "
            f"{fmt(rmse_r)} {fmt(mae_r)} {fmt(bias_r)} | "
            f"{fmt(rmse_s)} {fmt(mae_s)} {fmt(bias_s)}"
        )

    txt_lines.append("")
    txt_lines.append("NOTE: elevation_error = pred_elevation - gt_elevation.")
    txt_lines.append("      Positive = prediction is too HIGH (too shallow for water).")
    txt_lines.append(
        f"      Scale correction applied to {n_uw_scaled} underwater points, "
        f"{n_land_raw} land control points kept raw."
    )

    ctrl_m = matched[matched["type"] == "control_point"].copy()
    if len(ctrl_m) > 0:
        txt_lines.append("")
        txt_lines.append("CONTROL POINT DETAILS")
        txt_lines.append(
            f"{'Name':<12s} {'GT Elev':>9s} {'Pred Elev':>10s} "
            f"{'Error':>8s} {'Sc Elev':>10s} {'Sc Err':>8s} "
            f"{'N pts':>6s}"
        )
        txt_lines.append("-" * 72)
        for _, r in ctrl_m.sort_values("name").iterrows():
            txt_lines.append(
                f"{r['name']:<12s} "
                f"{fmt(r['gt_elevation'])} {fmt(r['pred_elevation'])} "
                f"{fmt(r['elev_error_raw'])} "
                f"{fmt(r['scaled_pred_elevation'])} {fmt(r['elev_error_scaled'])} "
                f"{r['n_matches']:6.0f}"
            )

    txt_content = "\n".join(txt_lines) + "\n"
    txt_path = os.path.join(args.output_dir, "control_point_summary.txt")
    with open(txt_path, "w") as fout:
        fout.write(txt_content)
    log(f"Saved: {txt_path}")
    print(txt_content)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        def scatter_panel(ax, elev_col, title):
            for tname, color, marker, sz, zord in [
                ("transect", "tab:blue", "o", 15, 1),
                ("reference_target", "red", "*", 60, 5),
                ("control_point", "green", "D", 30, 3),
            ]:
                sub = matched[matched["type"] == tname]
                if len(sub) == 0:
                    continue
                ax.scatter(
                    sub["gt_elevation"],
                    sub[elev_col],
                    s=sz,
                    alpha=0.7,
                    c=color,
                    marker=marker,
                    label=tname,
                    zorder=zord,
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
            axes[1],
            "scaled_pred_elevation",
            f"After Scaling (x{scale_factor:.3f})",
        )

        ax = axes[2]
        bar_types = ["reference_target", "transect", "control_point", "all"]
        bar_labels = ["Ref targets", "Transect", "Control pts", "All"]
        rmse_before = []
        rmse_after = []
        for tname in bar_types:
            sub = matched if tname == "all" else matched[matched["type"] == tname]
            raw_err = sub["elev_error_raw"].dropna().values
            sc_err = sub["elev_error_scaled"].dropna().values
            rmse_before.append(np.sqrt(np.mean(raw_err**2)) if len(raw_err) > 0 else 0)
            rmse_after.append(np.sqrt(np.mean(sc_err**2)) if len(sc_err) > 0 else 0)

        x = np.arange(len(bar_labels))
        w = 0.35
        ax.bar(x - w / 2, rmse_before, w, label="Before scaling", color="salmon")
        ax.bar(x + w / 2, rmse_after, w, label="After scaling", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel("Elevation RMSE (m)")
        ax.set_title("RMSE by Point Type")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Control Point Validation — {args.label}", fontsize=13, y=1.02)
        plt.tight_layout()
        png_path = os.path.join(args.output_dir, "control_point_analysis.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved: {png_path}")
    except Exception as e:
        log(f"[WARN] Could not create analysis plot: {e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trn_df = matched[
            matched["transect"].notna() & (matched["transect"] != "")
        ].copy()
        transect_names = sorted(trn_df["transect"].unique())
        n_trans = len(transect_names)

        if n_trans > 0:
            ncols = min(4, n_trans)
            nrows = int(np.ceil(n_trans / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )

            for idx, tname in enumerate(transect_names):
                row, col = idx // ncols, idx % ncols
                ax = axes[row, col]

                sub = trn_df[trn_df["transect"] == tname].copy()
                eas = sub["gt_easting"].values
                nor = sub["gt_northing"].values
                seg_dist = np.sqrt(np.diff(eas) ** 2 + np.diff(nor) ** 2)
                cum_dist = np.concatenate([[0], np.cumsum(seg_dist)])

                gt_elev = sub["gt_elevation"].values
                ws_elev = sub["ws_elevation"].values
                pred_elev = sub["pred_elevation"].values
                scaled_elev = sub["scaled_pred_elevation"].values

                ax.plot(
                    cum_dist,
                    gt_elev,
                    "k-o",
                    markersize=3,
                    linewidth=1.5,
                    label="GT riverbed",
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

                ax.set_title(tname, fontsize=11)
                ax.set_xlabel("Distance (m)", fontsize=9)
                ax.set_ylabel("Elevation (m)", fontsize=9)
                ax.legend(fontsize=6, loc="best")
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

            for idx in range(n_trans, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.suptitle(
                f"Transect Profiles (before/after scaling) — {args.label}",
                fontsize=14,
                y=1.01,
            )
            plt.tight_layout()
            trans_path = os.path.join(args.output_dir, "transects.png")
            plt.savefig(trans_path, dpi=150, bbox_inches="tight")
            plt.close()
            log(f"Saved: {trans_path}")
        else:
            log("[WARN] No transect data found for profiles")
    except Exception as e:
        log(f"[WARN] Could not create transect plot: {e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import re

        ctrl_pts = matched[matched["type"] == "control_point"].copy()
        if len(ctrl_pts) > 0:
            def get_group(name):
                m = re.match(r"([A-Z]+_[A-Z]+|[A-Z]+)", name)
                return m.group(1) if m else name

            ctrl_pts["group"] = ctrl_pts["name"].apply(get_group)
            groups = sorted(ctrl_pts["group"].unique())
            n_groups = len(groups)

            if n_groups > 0:
                ncols = min(4, n_groups)
                nrows = int(np.ceil(n_groups / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
                )

                for idx, gname in enumerate(groups):
                    row, col = idx // ncols, idx % ncols
                    ax = axes[row, col]

                    sub = ctrl_pts[ctrl_pts["group"] == gname].copy()
                    sub = sub.sort_values("name")
                    eas = sub["gt_easting"].values
                    nor = sub["gt_northing"].values

                    if len(sub) > 1:
                        seg_dist = np.sqrt(np.diff(eas) ** 2 + np.diff(nor) ** 2)
                        cum_dist = np.concatenate([[0], np.cumsum(seg_dist)])
                    else:
                        cum_dist = np.array([0.0])

                    gt_elev = sub["gt_elevation"].values
                    pred_elev = sub["pred_elevation"].values

                    ax.plot(
                        cum_dist,
                        gt_elev,
                        "k-o",
                        markersize=4,
                        linewidth=1.5,
                        label="GT elevation",
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

                    for i, (_, pt) in enumerate(sub.iterrows()):
                        ax.annotate(
                            pt["name"],
                            (cum_dist[i], gt_elev[i]),
                            fontsize=5,
                            alpha=0.6,
                            rotation=30,
                            ha="left",
                            va="bottom",
                        )

                    rmse = (
                        np.sqrt(
                            np.mean((pred_elev[pred_valid] - gt_elev[pred_valid]) ** 2)
                        )
                        if np.sum(pred_valid) > 0
                        else np.nan
                    )
                    bias = (
                        np.mean(pred_elev[pred_valid] - gt_elev[pred_valid])
                        if np.sum(pred_valid) > 0
                        else np.nan
                    )

                    ax.set_title(
                        f"{gname} (n={len(sub)}, RMSE={rmse:.2f}m, bias={bias:+.2f}m)",
                        fontsize=9,
                    )
                    ax.set_xlabel("Distance (m)", fontsize=9)
                    ax.set_ylabel("Elevation (m)", fontsize=9)
                    ax.legend(fontsize=6, loc="best")
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)

                for idx in range(n_groups, nrows * ncols):
                    axes[idx // ncols, idx % ncols].set_visible(False)

                fig.suptitle(
                    f"Land Control Point Profiles — {args.label}",
                    fontsize=14,
                    y=1.01,
                )
                plt.tight_layout()
                land_path = os.path.join(args.output_dir, "transects_land.png")
                plt.savefig(land_path, dpi=150, bbox_inches="tight")
                plt.close()
                log(f"Saved: {land_path}")
        else:
            log("[INFO] No control points for land transect plot")
    except Exception as e:
        log(f"[WARN] Could not create land transect plot: {e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        bathy = None
        if args.bathymetry_csv and os.path.isfile(args.bathymetry_csv):
            try:
                bathy = pd.read_csv(args.bathymetry_csv)
                if "utm_easting" not in bathy.columns:
                    bathy = None
            except Exception:
                bathy = None

        for panel_idx, (err_col, title_suffix) in enumerate(
            [
                ("elev_error_raw", "Before Scaling"),
                ("elev_error_scaled", "After Scaling"),
            ]
        ):
            ax = axes[panel_idx]

            if bathy is not None and "depth_m" in bathy.columns:
                bathy_plot = bathy
                if len(bathy_plot) > 50000:
                    bathy_plot = bathy_plot.sample(n=50000, random_state=42)
                ax.scatter(
                    bathy_plot["utm_easting"],
                    bathy_plot["utm_northing"],
                    c=bathy_plot["depth_m"],
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

            trn_labeled = matched[
                matched["transect"].notna() & (matched["transect"] != "")
            ]
            for t in trn_labeled["transect"].unique():
                sub = trn_labeled[trn_labeled["transect"] == t]
                mid = sub.iloc[len(sub) // 2]
                ax.annotate(
                    t,
                    (mid["gt_easting"], mid["gt_northing"]),
                    fontsize=6,
                    alpha=0.7,
                )

            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            ax.set_title(f"Spatial Error Map — {title_suffix}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(f"Spatial Error — {args.label}", fontsize=14, y=1.02)
        plt.tight_layout()
        map_path = os.path.join(args.output_dir, "spatial_error_map.png")
        plt.savefig(map_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved: {map_path}")
    except Exception as e:
        log(f"[WARN] Could not create spatial error map: {e}")

    try:
        import rasterio

        log("Generating scaled pointcloud PLY...")

        src_ply_path = args.pointcloud_npy.replace(".npy", ".ply")
        if not os.path.isfile(src_ply_path):
            raise FileNotFoundError(f"Source PLY not found: {src_ply_path}")

        with open(src_ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="replace").strip()
                header_lines.append(line)
                if line == "end_header":
                    break
            ply_header_bytes = f.tell()

        N_pts = None
        for hl in header_lines:
            if hl.startswith("element vertex"):
                N_pts = int(hl.split()[-1])
                break
        if N_pts is None:
            raise ValueError("Could not parse vertex count from source PLY")

        VERTEX_BYTES = 15
        log(f"  Source PLY: {src_ply_path}")
        log(f"  {N_pts:,} vertices, header={ply_header_bytes} bytes")

        use_fitted_ply = bool(args.fitted_wsm_json)
        if use_fitted_ply:
            import json as _json

            with open(args.fitted_wsm_json) as _f:
                _plane = _json.load(_f)["nerfrac_plane"]
            _pnx, _pny, _pnz = _plane["normal"]
            _pcx, _pcy, _pcz = _plane["centroid_utm33"]
            log(f"  PLY water surface: fitted plane from {args.fitted_wsm_json}")
        else:
            with rasterio.open(args.wsm_path) as wsm_ds:
                wsm_transform = wsm_ds.transform
                wsm_data = wsm_ds.read(1)
                wsm_nodata = wsm_ds.nodata
            log(f"  PLY water surface: GT WSM raster")

        CHUNK = 5_000_000
        out_path = os.path.join(args.output_dir, "pointcloud_scaled.ply")

        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
        os.close(tmp_fd)

        n_written = 0
        n_scaled = 0

        try:
            src_ply_f = open(src_ply_path, "rb")
            src_ply_f.seek(ply_header_bytes)

            dtype_vertex = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("r", "u1"),
                    ("g", "u1"),
                    ("b", "u1"),
                ]
            )

            with open(tmp_path, "wb") as tmp_f:
                for start in range(0, N_pts, CHUNK):
                    end = min(start + CHUNK, N_pts)
                    chunk_size = end - start

                    raw = src_ply_f.read(chunk_size * VERTEX_BYTES)
                    if len(raw) < chunk_size * VERTEX_BYTES:
                        raise IOError(
                            f"Unexpected EOF at vertex {start}: "
                            f"got {len(raw)} bytes, expected {chunk_size * VERTEX_BYTES}"
                        )

                    verts = np.frombuffer(raw, dtype=dtype_vertex)

                    easting = verts["x"].astype(np.float64)
                    northing = verts["y"].astype(np.float64)
                    elevation = verts["z"].astype(np.float64)
                    rgb = np.column_stack([verts["r"], verts["g"], verts["b"]])

                    if use_fitted_ply:
                        ws_elev = (
                            _pcz
                            - (_pnx * (easting - _pcx) + _pny * (northing - _pcy))
                            / _pnz
                        )
                    else:
                        rows, cols = rasterio.transform.rowcol(
                            wsm_transform, easting, northing
                        )
                        rows = np.array(rows)
                        cols = np.array(cols)

                        h, w = wsm_data.shape
                        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

                        ws_elev = np.full(chunk_size, np.nan)
                        if np.any(valid):
                            ws_vals = wsm_data[rows[valid], cols[valid]]
                            if wsm_nodata is not None:
                                ws_vals = np.where(
                                    ws_vals == wsm_nodata, np.nan, ws_vals
                                )
                            ws_elev[valid] = ws_vals.astype(np.float64)

                    underwater = np.isfinite(ws_elev) & (elevation < ws_elev)

                    scaled_elev = elevation.copy()
                    if np.any(underwater):
                        depth = ws_elev[underwater] - elevation[underwater]
                        scaled_depth = depth * scale_factor
                        scaled_elev[underwater] = ws_elev[underwater] - scaled_depth
                        n_scaled += int(np.sum(underwater))


                    out_arr = np.empty(chunk_size, dtype=dtype_vertex)
                    out_arr["x"] = easting.astype(np.float32)
                    out_arr["y"] = northing.astype(np.float32)
                    out_arr["z"] = scaled_elev.astype(np.float32)
                    out_arr["r"] = rgb[:, 0]
                    out_arr["g"] = rgb[:, 1]
                    out_arr["b"] = rgb[:, 2]
                    tmp_f.write(out_arr.tobytes())
                    n_written += chunk_size

                    if start % (CHUNK * 5) == 0 and start > 0:
                        log(f"    Processed {start:,}/{N_pts:,} points...")

            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {n_written}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            with open(out_path, "wb") as ply_f:
                ply_f.write(header.encode("ascii"))
                with open(tmp_path, "rb") as tmp_f:
                    while True:
                        block = tmp_f.read(64 * 1024 * 1024)
                        if not block:
                            break
                        ply_f.write(block)
        finally:
            src_ply_f.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        log(
            f"Saved: {out_path} "
            f"({n_written:,} pts, {n_scaled:,} scaled underwater, "
            f"scale={scale_factor:.4f})"
        )
    except Exception as e:
        log(f"[WARN] Could not create scaled pointcloud: {e}")

    log(f"\n=== Done: {args.label} ===")


if __name__ == "__main__":
    main()
