"""
Depth-stratified validation of predicted depths against ground truth.

Usage examples:
    # Default thresholds (0-1 m), outputs next to comparison.csv
    python 10_depth_stratified.py \\
        --comparison_csv validation/nerfrac_50m/comparison.csv

    # Fewer thresholds
    python 10_depth_stratified.py \\
        --comparison_csv validation/nerfrac_50m/comparison.csv \\
        --thresholds 0.0,0.3,0.5,0.7
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import compute_metrics

COLUMNS = [
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
    "min_gt_depth_m",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-stratified validation of predicted depths against ground truth."
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="Path to comparison.csv from 03_validate.py",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.0,0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated GT depth thresholds in meters (keeps gt_depth_m > "
        "threshold, default: 0.0,0.1,0.2,0.3,0.5,0.7,1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as comparison_csv dir)",
    )
    return parser.parse_args()


def fmt_value(value, prec=4):
    return f"{value:.{prec}f}" if np.isfinite(value) else "n/a"


def stratify(gt_depth, pred_depth, ref_mask, thresholds):
    """Metrics of all, reference and transect points deeper than each threshold.

    Thresholds are (depth in m, label) pairs.
    """
    rows = []
    for thr, label in thresholds:
        deeper = gt_depth > thr
        row = compute_metrics(gt_depth[deeper], pred_depth[deeper])
        m_ref = compute_metrics(
            gt_depth[deeper & ref_mask], pred_depth[deeper & ref_mask]
        )
        m_trn = compute_metrics(
            gt_depth[deeper & ~ref_mask], pred_depth[deeper & ~ref_mask]
        )
        row.update(
            gt_range=f">{label}m",
            min_gt_depth_m=thr,
            N_ref=m_ref["N"],
            N_trn=m_trn["N"],
            pearson_r_ref=m_ref["pearson_r"],
            spearman_rho_ref=m_ref["spearman_rho"],
            pearson_r_trn=m_trn["pearson_r"],
            spearman_rho_trn=m_trn["spearman_rho"],
        )
        rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


def print_table(result):
    fmt = "{:>8s} {:>5s} {:>5s} {:>5s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s} {:>12s} {:>12s}"
    header = fmt.format(
        "Subset",
        "N",
        "N_ref",
        "N_trn",
        "Pearson",
        "Spearman",
        "RMSE",
        "MAE",
        "Bias",
        "Pearson_ref",
        "Spearman_ref",
    )
    print()
    print(header)
    print("-" * len(header))
    for _, r in result.iterrows():
        print(
            fmt.format(
                r["gt_range"],
                str(int(r["N"])),
                str(int(r["N_ref"])),
                str(int(r["N_trn"])),
                fmt_value(r["pearson_r"]),
                fmt_value(r["spearman_rho"]),
                fmt_value(r["rmse"], 3),
                fmt_value(r["mae"], 3),
                fmt_value(r["bias"], 3),
                fmt_value(r["pearson_r_ref"]),
                fmt_value(r["spearman_rho_ref"]),
            )
        )
    print()


def plot_stratified(result, filepath):
    x = result["min_gt_depth_m"]
    fig, (ax_corr, ax_err) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    ax_corr.plot(
        x, result["pearson_r"], "o-", color="tab:blue", label="Pearson r (all)"
    )
    ax_corr.plot(
        x,
        result["spearman_rho"],
        "s-",
        color="tab:orange",
        label="Spearman rho (all)",
    )
    ax_corr.plot(
        x,
        result["pearson_r_trn"],
        "^--",
        color="tab:blue",
        alpha=0.5,
        label="Pearson r (transect)",
    )
    ax_corr.plot(
        x,
        result["spearman_rho_trn"],
        "v--",
        color="tab:orange",
        alpha=0.5,
        label="Spearman rho (transect)",
    )
    ax_corr.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_xlabel("Minimum GT depth threshold (m)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_title("Correlation vs GT depth threshold")
    ax_corr.legend(fontsize=8)
    ax_corr.grid(True, alpha=0.3)

    ax_err.plot(x, result["rmse"], "o-", color="tab:red", label="RMSE (m)")
    ax_err.plot(x, result["mae"], "s-", color="tab:green", label="MAE (m)")
    ax_err.set_xlabel("Minimum GT depth threshold (m)")
    ax_err.set_ylabel("Error (m)")
    ax_err.set_title("Error metrics vs GT depth threshold")
    ax_err.legend(loc="upper left", fontsize=8)
    ax_err.grid(True, alpha=0.3)

    ax_n = ax_err.twinx()
    ax_n.bar(x, result["N"], width=0.04, alpha=0.2, color="gray")
    ax_n.set_ylabel("N points", color="gray")
    ax_n.tick_params(axis="y", labelcolor="gray")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def write_summary(result, gt_depth, source_csv, filepath):
    first = result.iloc[0]
    n_matched = len(gt_depth)
    n_above_ws = int(np.sum(gt_depth < 0))
    n_shallow = int(np.sum(gt_depth < 0.3))

    lines = [f"Depth-stratified analysis of {source_csv}"]
    if result["spearman_rho"].notna().any():
        best = result.loc[result["spearman_rho"].idxmax()]
        lines += [
            "",
            f"Best Spearman rho: {best['spearman_rho']:.4f} "
            f"at GT depth {best['gt_range']} (N={int(best['N'])})",
        ]
    lines += [
        "",
        f"GT depth {first['gt_range']} (N={int(first['N'])}):",
        f"  Pearson r    = {fmt_value(first['pearson_r'])}",
        f"  Spearman rho = {fmt_value(first['spearman_rho'])}",
        f"  RMSE (m)     = {fmt_value(first['rmse'], 3)}",
        "",
        f"Shallow contamination ({n_matched} matched points):",
        f"  GT depth < 0m:   {n_above_ws}",
        f"  GT depth < 0.3m: {n_shallow} ({100 * n_shallow / n_matched:.1f}%)",
        "",
        result.to_string(index=False, float_format="%.4f"),
    ]
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {filepath}")


def main():
    args = parse_args()

    if not os.path.isfile(args.comparison_csv):
        print(f"ERROR: Comparison CSV not found: {args.comparison_csv}")
        sys.exit(1)
    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.comparison_csv)
    )
    os.makedirs(output_dir, exist_ok=True)
    # sorted by depth, labelled as typed
    thresholds = sorted(
        {float(t): t.strip() for t in args.thresholds.split(",")}.items()
    )

    df = pd.read_csv(args.comparison_csv)
    df = df[df["matched"] == "yes"]
    print(f"Loaded {len(df)} matched points from {args.comparison_csv}")
    if df.empty:
        print("ERROR: No matched points")
        sys.exit(1)

    gt_depth = df["gt_depth_m"].values
    pred_depth = df["pred_depth_m"].values
    ref_mask = df["type"].values == "reference_target"

    result = stratify(gt_depth, pred_depth, ref_mask, thresholds)
    csv_path = os.path.join(output_dir, "depth_stratified_analysis.csv")
    result.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")

    print_table(result)
    plot_stratified(result, os.path.join(output_dir, "depth_stratified_analysis.png"))
    write_summary(
        result,
        gt_depth,
        args.comparison_csv,
        os.path.join(output_dir, "depth_stratified_analysis.txt"),
    )


if __name__ == "__main__":
    main()
