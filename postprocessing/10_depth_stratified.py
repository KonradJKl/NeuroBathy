#!/usr/bin/env python
"""Depth-stratified validation analysis.

Computes Pearson, Spearman, RMSE, MAE, and bias at increasing GT depth
thresholds to reveal how model skill changes as near-waterline points are
excluded.

Usage:
    python 10_depth_stratified.py \\
        --comparison_csv validation/nerfrac_50m_wsm/comparison.csv \\
        --output_dir     validation/nerfrac_50m_wsm/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from coord_utils import compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Depth-stratified validation analysis."
    )
    parser.add_argument(
        "--comparison_csv",
        required=True,
        help="Path to comparison.csv from 03_validate.py",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as comparison_csv dir)",
    )
    parser.add_argument(
        "--thresholds",
        default="0.0,0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated GT depth thresholds (default: 0.0,...,1.0)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.comparison_csv):
        print(f"[ERROR] File not found: {args.comparison_csv}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.comparison_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds.split(",")]

    df = pd.read_csv(args.comparison_csv)
    df = df[df["matched"] == "yes"].copy()
    print(f"Loaded {len(df)} matched points from {args.comparison_csv}")

    gt_depth = df["gt_depth_m"].values
    pred_depth = df["pred_depth_m"].values

    ref_mask = df["type"].values == "reference_target"
    trn_mask = ~ref_mask

    rows = []
    for thr in thresholds:
        mask = gt_depth > thr
        m = compute_metrics(gt_depth[mask], pred_depth[mask])
        m["min_gt_depth"] = thr
        m["gt_range"] = f">{thr:.1f}m"

        m_ref = compute_metrics(gt_depth[mask & ref_mask], pred_depth[mask & ref_mask])
        m_trn = compute_metrics(gt_depth[mask & trn_mask], pred_depth[mask & trn_mask])
        m["N_ref"] = m_ref["N"]
        m["N_trn"] = m_trn["N"]
        m["pearson_r_ref"] = m_ref["pearson_r"]
        m["pearson_r_trn"] = m_trn["pearson_r"]
        m["spearman_rho_ref"] = m_ref["spearman_rho"]
        m["spearman_rho_trn"] = m_trn["spearman_rho"]

        rows.append(m)

    result = pd.DataFrame(rows)

    col_order = [
        "gt_range",
        "N",
        "N_ref",
        "N_trn",
        "pearson_r",
        "spearman_rho",
        "rmse",
        "mae",
        "bias",
        "pearson_r_ref",
        "spearman_rho_ref",
        "pearson_r_trn",
        "spearman_rho_trn",
        "min_gt_depth",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    csv_path = os.path.join(args.output_dir, "depth_stratified_analysis.csv")
    result.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    print("\n" + "=" * 100)
    print("DEPTH-STRATIFIED ANALYSIS")
    print("=" * 100)
    fmt = "{:>8s} {:>5s} {:>5s} {:>5s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}"
    print(
        fmt.format(
            "Subset",
            "N",
            "N_ref",
            "N_trn",
            "Pearson",
            "Spearman",
            "RMSE",
            "MAE",
            "Bias",
            "Pearson_R",
            "Spearman_R",
        )
    )
    print("-" * 100)
    for _, r in result.iterrows():
        print(
            fmt.format(
                r["gt_range"],
                str(int(r["N"])),
                str(int(r["N_ref"])),
                str(int(r["N_trn"])),
                f"{r['pearson_r']:.4f}" if not np.isnan(r["pearson_r"]) else "n/a",
                f"{r['spearman_rho']:.4f}"
                if not np.isnan(r["spearman_rho"])
                else "n/a",
                f"{r['rmse']:.3f}" if not np.isnan(r["rmse"]) else "n/a",
                f"{r['mae']:.3f}" if not np.isnan(r["mae"]) else "n/a",
                f"{r['bias']:.3f}" if not np.isnan(r["bias"]) else "n/a",
                f"{r['pearson_r_ref']:.4f}"
                if not np.isnan(r["pearson_r_ref"])
                else "n/a",
                f"{r['spearman_rho_ref']:.4f}"
                if not np.isnan(r["spearman_rho_ref"])
                else "n/a",
            )
        )
    print("=" * 100)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(
            result["min_gt_depth"],
            result["pearson_r"],
            "o-",
            color="tab:blue",
            label="Pearson r (all)",
        )
        ax.plot(
            result["min_gt_depth"],
            result["spearman_rho"],
            "s-",
            color="tab:orange",
            label="Spearman rho (all)",
        )
        ax.plot(
            result["min_gt_depth"],
            result["pearson_r_trn"],
            "^--",
            color="tab:blue",
            alpha=0.5,
            label="Pearson r (transect)",
        )
        ax.plot(
            result["min_gt_depth"],
            result["spearman_rho_trn"],
            "v--",
            color="tab:orange",
            alpha=0.5,
            label="Spearman rho (transect)",
        )
        ax.set_xlabel("Minimum GT depth threshold (m)")
        ax.set_ylabel("Correlation")
        ax.set_title("Correlation vs GT depth threshold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.05)

        ax2 = axes[1]
        ax2.plot(
            result["min_gt_depth"],
            result["rmse"],
            "o-",
            color="tab:red",
            label="RMSE (m)",
        )
        ax2.plot(
            result["min_gt_depth"],
            result["mae"],
            "s-",
            color="tab:green",
            label="MAE (m)",
        )
        ax2.set_xlabel("Minimum GT depth threshold (m)")
        ax2.set_ylabel("Error (m)")
        ax2.set_title("Error metrics vs GT depth threshold")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = ax2.twinx()
        ax3.bar(
            result["min_gt_depth"],
            result["N"],
            width=0.04,
            alpha=0.2,
            color="gray",
            label="N points",
        )
        ax3.set_ylabel("N points", color="gray")
        ax3.tick_params(axis="y", labelcolor="gray")

        plt.tight_layout()
        png_path = os.path.join(args.output_dir, "depth_stratified_analysis.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
    except Exception as e:
        print(f"[WARN] Could not create plot: {e}")

    txt_path = os.path.join(args.output_dir, "depth_stratified_analysis.txt")
    with open(txt_path, "w") as f:
        f.write("DEPTH-STRATIFIED ANALYSIS\n")
        f.write(f"Source: {args.comparison_csv}\n\n")

        best_sp = result.loc[result["spearman_rho"].idxmax()]
        f.write(
            f"Best Spearman rho: {best_sp['spearman_rho']:.4f} "
            f"at threshold {best_sp['gt_range']} "
            f"(N={int(best_sp['N'])})\n"
        )

        all_row = result.iloc[0]
        f.write(f"\nAll points (N={int(all_row['N'])}):\n")
        f.write(f"  Pearson r    = {all_row['pearson_r']:.4f}\n")
        f.write(f"  Spearman rho = {all_row['spearman_rho']:.4f}\n")
        f.write(f"  RMSE         = {all_row['rmse']:.3f} m\n")

        n_neg = int(np.sum(gt_depth < 0))
        n_shallow = int(np.sum(gt_depth < 0.3))
        f.write(f"\nShallow contamination:\n")
        f.write(f"  Points with GT depth < 0m:   {n_neg}\n")
        f.write(
            f"  Points with GT depth < 0.3m: {n_shallow} "
            f"({100 * n_shallow / len(gt_depth):.1f}%)\n"
        )

        f.write(f"\n{'=' * 100}\n")
        f.write(result.to_string(index=False, float_format="%.4f"))
        f.write(f"\n{'=' * 100}\n")

    print(f"Saved: {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
