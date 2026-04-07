#!/usr/bin/env python
"""Investigate scale and affine depth correction.

Usage:
    python 11_scale_correction.py \\
        --comparison_csv validation/nerfrac_50m_wsm/comparison.csv \\
        --output_dir     validation/nerfrac_50m_wsm/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from coord_utils import compute_metrics


def print_metrics_table(rows):
    fmt = "{:<35s} {:>5s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(
        fmt.format(
            "Variant", "N", "Pearson", "Spearman", "RMSE", "MAE", "Bias", "Scale"
        )
    )
    print("-" * 100)
    for r in rows:
        print(
            fmt.format(
                r["label"],
                str(int(r["N"])),
                f"{r['pearson_r']:.4f}" if not np.isnan(r["pearson_r"]) else "n/a",
                f"{r['spearman_rho']:.4f}"
                if not np.isnan(r["spearman_rho"])
                else "n/a",
                f"{r['rmse']:.3f}" if not np.isnan(r["rmse"]) else "n/a",
                f"{r['mae']:.3f}" if not np.isnan(r["mae"]) else "n/a",
                f"{r['bias']:.3f}" if not np.isnan(r["bias"]) else "n/a",
                f"{r['scale_ratio']:.2f}" if not np.isnan(r["scale_ratio"]) else "n/a",
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Investigate scale and affine depth correction."
    )
    parser.add_argument(
        "--comparison_csv",
        required=True,
        help="Path to comparison.csv from 03_validate.py",
    )
    parser.add_argument(
        "--bathymetry_csv",
        default=None,
        help="Path to bathymetry CSV (for scale-corrected PLY generation)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as comparison_csv dir)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.comparison_csv):
        print(f"[ERROR] File not found: {args.comparison_csv}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.comparison_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.comparison_csv)
    df = df[df["matched"] == "yes"].copy()
    print(f"Loaded {len(df)} matched points from {args.comparison_csv}")

    gt_depth = df["gt_depth_m"].values
    pred_depth = df["pred_depth_m"].values
    ref_mask = df["type"].values == "reference_target"
    trn_mask = ~ref_mask

    print("\n" + "=" * 100)
    print("SCALE / AFFINE CORRECTION ANALYSIS")
    print("=" * 100)

    results = []

    m_all = compute_metrics(gt_depth, pred_depth, "Uncorrected (all)")
    m_ref = compute_metrics(
        gt_depth[ref_mask], pred_depth[ref_mask], "Uncorrected (ref targets)"
    )
    m_trn = compute_metrics(
        gt_depth[trn_mask], pred_depth[trn_mask], "Uncorrected (transect)"
    )
    results.extend([m_all, m_ref, m_trn])

    # scale = median(gt / pred) for reference targets
    ref_gt = gt_depth[ref_mask]
    ref_pred = pred_depth[ref_mask]
    scale_factor = np.median(ref_gt / ref_pred)
    print(f"\nScale factor (median gt/pred on ref targets): {scale_factor:.4f}")
    print(f"  => pred_depth is {1 / scale_factor:.2f}x too large on average")

    scaled_pred = pred_depth * scale_factor

    m_s_all = compute_metrics(gt_depth, scaled_pred, "Scale-corrected (all)")
    m_s_ref = compute_metrics(
        gt_depth[ref_mask], scaled_pred[ref_mask], "Scale-corrected (ref targets)"
    )
    m_s_trn = compute_metrics(
        gt_depth[trn_mask], scaled_pred[trn_mask], "Scale-corrected (transect)"
    )
    results.extend([m_s_all, m_s_ref, m_s_trn])

    A = np.column_stack([ref_pred, np.ones_like(ref_pred)])
    coefs, _, _, _ = np.linalg.lstsq(A, ref_gt, rcond=None)
    a_coef, b_coef = coefs
    print(
        f"\nAffine fit on ref targets: depth_corrected = {a_coef:.4f} * pred + {b_coef:.4f}"
    )

    affine_pred = a_coef * pred_depth + b_coef

    m_a_all = compute_metrics(gt_depth, affine_pred, "Affine-corrected (all)")
    m_a_ref = compute_metrics(
        gt_depth[ref_mask], affine_pred[ref_mask], "Affine-corrected (ref targets)"
    )
    m_a_trn = compute_metrics(
        gt_depth[trn_mask], affine_pred[trn_mask], "Affine-corrected (transect)"
    )
    results.extend([m_a_all, m_a_ref, m_a_trn])


    fit_mask = gt_depth > 0.3
    if np.sum(fit_mask) >= 3:
        A2 = np.column_stack([pred_depth[fit_mask], np.ones(np.sum(fit_mask))])
        coefs2, _, _, _ = np.linalg.lstsq(A2, gt_depth[fit_mask], rcond=None)
        a2, b2 = coefs2
        print(
            f"\nAffine fit on all pts > 0.3m (N={np.sum(fit_mask)}): "
            f"depth_corrected = {a2:.4f} * pred + {b2:.4f}"
        )

        affine2_pred = a2 * pred_depth + b2

        m_a2_all = compute_metrics(gt_depth, affine2_pred, "Affine-all>0.3m (all)")
        m_a2_ref = compute_metrics(
            gt_depth[ref_mask], affine2_pred[ref_mask], "Affine-all>0.3m (ref targets)"
        )
        m_a2_trn = compute_metrics(
            gt_depth[trn_mask], affine2_pred[trn_mask], "Affine-all>0.3m (transect)"
        )
        results.extend([m_a2_all, m_a2_ref, m_a2_trn])

    print("\n" + "=" * 100)
    print_metrics_table(results)
    print("=" * 100)

    print("\n\nDEPTH-STRATIFIED: SCALE-CORRECTED")
    print("-" * 80)
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    strat_rows = []
    for thr in thresholds:
        mask = gt_depth > thr
        m = compute_metrics(
            gt_depth[mask], scaled_pred[mask], f"Scale-corrected >{thr:.1f}m"
        )
        strat_rows.append(m)
    print_metrics_table(strat_rows)

    df_out = df.copy()
    df_out["scale_factor"] = scale_factor
    df_out["scaled_pred_depth_m"] = scaled_pred
    df_out["scaled_error_m"] = scaled_pred - gt_depth
    df_out["affine_a"] = a_coef
    df_out["affine_b"] = b_coef
    df_out["affine_pred_depth_m"] = affine_pred
    df_out["affine_error_m"] = affine_pred - gt_depth

    csv_path = os.path.join(args.output_dir, "scale_correction_analysis.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved per-point CSV: {csv_path}")

    summary = pd.DataFrame(results)
    summary_csv = os.path.join(args.output_dir, "scale_correction_summary.csv")
    summary.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"Saved summary CSV:   {summary_csv}")

    txt_path = os.path.join(args.output_dir, "scale_correction_analysis.txt")
    with open(txt_path, "w") as f:
        f.write("SCALE / AFFINE CORRECTION ANALYSIS\n")
        f.write(f"Source: {args.comparison_csv}\n\n")

        f.write(
            f"Scale factor (median gt/pred on {int(ref_mask.sum())} ref targets): "
            f"{scale_factor:.4f}\n"
        )
        f.write(f"  => Predictions are {1 / scale_factor:.2f}x too large\n\n")

        f.write(
            f"Affine fit on ref targets: "
            f"corrected = {a_coef:.4f} * pred + {b_coef:.4f}\n\n"
        )

        f.write("KEY COMPARISON:\n")
        f.write(
            f"  Uncorrected  — RMSE={m_all['rmse']:.3f}m, "
            f"Pearson={m_all['pearson_r']:.4f}, "
            f"Spearman={m_all['spearman_rho']:.4f}\n"
        )
        f.write(
            f"  Scale-corr   — RMSE={m_s_all['rmse']:.3f}m, "
            f"Pearson={m_s_all['pearson_r']:.4f}, "
            f"Spearman={m_s_all['spearman_rho']:.4f}\n"
        )
        f.write(
            f"  Affine-corr  — RMSE={m_a_all['rmse']:.3f}m, "
            f"Pearson={m_a_all['pearson_r']:.4f}, "
            f"Spearman={m_a_all['spearman_rho']:.4f}\n\n"
        )

        f.write("NOTE: Correlation metrics (Pearson, Spearman) are unchanged by\n")
        f.write("scale/affine correction — they only measure rank/linear agreement.\n")
        f.write("The improvement shows in RMSE, MAE, and bias.\n")

    print(f"Saved text report:   {txt_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def scatter_panel(ax, gt, pred, title, color="tab:blue"):
            ax.scatter(
                gt[trn_mask],
                pred[trn_mask],
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

        scatter_panel(axes[0], gt_depth, pred_depth, "Uncorrected")
        scatter_panel(
            axes[1], gt_depth, scaled_pred, f"Scale-corrected (x{scale_factor:.3f})"
        )
        scatter_panel(
            axes[2], gt_depth, affine_pred, f"Affine ({a_coef:.3f}*pred + {b_coef:.3f})"
        )

        plt.tight_layout()
        png_path = os.path.join(args.output_dir, "scale_correction_analysis.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved plot: {png_path}")
    except Exception as e:
        print(f"[WARN] Could not create plot: {e}")

    if args.bathymetry_csv and os.path.isfile(args.bathymetry_csv):
        try:
            bathy = pd.read_csv(args.bathymetry_csv)
            required_cols = {"utm_easting", "utm_northing", "utm_elevation", "depth_m"}
            if not required_cols.issubset(set(bathy.columns)):
                print(
                    f"[WARN] Bathymetry CSV missing columns: "
                    f"{required_cols - set(bathy.columns)}"
                )
            else:
                bathy_sc = bathy.copy()
                bathy_sc["depth_m_orig"] = bathy_sc["depth_m"]
                bathy_sc["depth_m"] = bathy_sc["depth_m"] * scale_factor

                if "wsm_elevation" in bathy_sc.columns:
                    bathy_sc["utm_elevation"] = (
                        bathy_sc["wsm_elevation"] - bathy_sc["depth_m"]
                    )
                else:
                    surface = bathy_sc["utm_elevation"] + bathy_sc["depth_m_orig"]
                    bathy_sc["utm_elevation"] = surface - bathy_sc["depth_m"]

                points = np.column_stack(
                    [
                        bathy_sc["utm_easting"].values,
                        bathy_sc["utm_northing"].values,
                        bathy_sc["utm_elevation"].values,
                    ]
                ).astype(np.float32)

                depth_vals = bathy_sc["depth_m"].values
                d_min, d_max = np.nanmin(depth_vals), np.nanmax(depth_vals)
                if d_max > d_min:
                    norm = np.clip((depth_vals - d_min) / (d_max - d_min), 0, 1)
                else:
                    norm = np.zeros_like(depth_vals)
                colors = np.zeros((len(norm), 3), dtype=np.uint8)
                colors[:, 0] = (norm * 255).astype(np.uint8)
                colors[:, 2] = ((1 - norm) * 255).astype(
                    np.uint8
                )

                N = len(points)
                ply_header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {N}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                dtype = np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                    ]
                )
                vertices = np.empty(N, dtype=dtype)
                vertices["x"] = points[:, 0]
                vertices["y"] = points[:, 1]
                vertices["z"] = points[:, 2]
                vertices["red"] = colors[:, 0]
                vertices["green"] = colors[:, 1]
                vertices["blue"] = colors[:, 2]


                bathy_base = os.path.splitext(os.path.basename(args.bathymetry_csv))[0]
                ply_path = os.path.join(
                    os.path.dirname(args.bathymetry_csv),
                    f"{bathy_base}_scale_corrected.ply",
                )
                with open(ply_path, "wb") as f:
                    f.write(ply_header.encode("ascii"))
                    vertices.tofile(f)
                print(
                    f"Saved scale-corrected PLY ({N} vertices, "
                    f"scale={scale_factor:.4f}): {ply_path}"
                )
        except Exception as e:
            print(f"[WARN] Could not create scale-corrected PLY: {e}")
    elif args.bathymetry_csv:
        print(f"[WARN] Bathymetry CSV not found: {args.bathymetry_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
