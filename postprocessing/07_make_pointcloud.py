"""
Generate colored scene point clouds from VGGT or NeRFrac output.

Usage (VGGT):
    python 07_make_pointcloud.py \\
        --mode vggt \\
        --vggt_dir '/path/to/vggt_output' \\
        --output_dir pointclouds/vggt_nadir \\
        --umeyama_npz ../data/transforms/vggt_nadir_umeyama.npz \\
        --conf_threshold 0.5

Usage (NeRFrac):
    python 07_make_pointcloud.py \\
        --mode nerfrac \\
        --log_folder /path/to/experiment_log \\
        --testset renderonly_all_600000 \\
        --poses_bounds /path/to/pielach_data/poses_bounds.npy \\
        --factor 4 \\
        --data_dir /path/to/pielach_data \\
        --umeyama_npz ../data/transforms/nerfrac_nadir_umeyama.npz \\
        --mask_dir /path/to/pielach_data/masks_4 \\
        --output_dir pointclouds/nerfrac_f4
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from coord_utils import (
    apply_umeyama,
    discover_nerfrac_views,
    discover_vggt_views,
    load_nerfrac_coord_metadata,
    load_focal_from_poses_bounds,
    load_umeyama,
    ndc_to_colmap,
    WATER_MASK_THRESHOLD,
    write_ply_binary,
)



def filter_outliers(points, colors, percentile=1.0):
    mask = np.ones(len(points), dtype=bool)
    for axis in range(3):
        vals = points[:, axis]
        lo = np.percentile(vals, percentile)
        hi = np.percentile(vals, 100 - percentile)
        mask &= (vals >= lo) & (vals <= hi)
    removed = np.sum(~mask)
    return points[mask], colors[mask], removed

def list_masks(mask_dir, n_views):
    if not os.path.isdir(mask_dir):
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        sys.exit(1)
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if len(mask_files) != n_views:
        print(f"[ERROR] {len(mask_files)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    return mask_files


def _vggt_view_indices(vggt_dir):
    return [v["index"] for v in discover_vggt_views(vggt_dir)]


def _load_water_mask_for_view(mask_dir, view_idx, n_views, H, W):
    import glob

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files or view_idx >= len(mask_files):
        return None

    m = imageio.imread(mask_files[view_idx])
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape[0] != H or m.shape[1] != W:
        import cv2

        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    return m > WATER_MASK_THRESHOLD


def generate_vggt_pointcloud(args):
    vggt_dir = Path(args.vggt_dir)
    view_indices = _vggt_view_indices(str(vggt_dir))
    if not view_indices:
        print(f"[ERROR] No world_points_*.npy files in {vggt_dir}")
        return None, None, {}
    print(f"Found {len(view_indices)} views in {vggt_dir}")

    umeyama = None
    if args.umeyama_npz:
        umeyama = load_umeyama(args.umeyama_npz)
        print(f"Loaded Umeyama (scale={umeyama['scale']:.4f}) from {args.umeyama_npz}")

    has_masks = bool(args.mask_dir)
    if has_masks:
        list_masks(args.mask_dir, len(view_indices))
    all_pts = []
    all_rgb = []
    all_water_pts = []
    all_water_rgb = []
    all_land_pts = []
    all_land_rgb = []

    for idx in view_indices:
        wp = np.load(vggt_dir / f"world_points_{idx:03d}.npy")  # (H, W, 3)
        H, W = wp.shape[:2]

        # Load RGB
        rgb_path = vggt_dir / f"rgb_{idx:03d}.png"
        if rgb_path.exists():
            img = imageio.imread(str(rgb_path))
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[:, :, :3]
            if img.shape[0] != H or img.shape[1] != W:
                import cv2

                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            img = np.full((H, W, 3), 128, dtype=np.uint8)


        valid = np.isfinite(wp).all(axis=2)

        if args.conf_threshold > 0:
            conf_path = vggt_dir / f"depth_conf_{idx:03d}.npy"
            if conf_path.exists():
                conf = np.load(str(conf_path))
                valid &= conf >= args.conf_threshold
            else:
                print(
                    f"  [WARN] No confidence file for view {idx}, skipping conf filter"
                )

        pts = wp[valid]
        rgb = img[valid]
        all_pts.append(pts)
        all_rgb.append(rgb)

        if has_masks:
            water_bool = _load_water_mask_for_view(
                args.mask_dir, idx, len(view_indices), H, W
            )
            if water_bool is not None:
                water_valid = valid & water_bool
                land_valid = valid & ~water_bool
                n_w = np.sum(water_valid)
                n_l = np.sum(land_valid)
                print(
                    f"  View {idx:03d}: {len(pts):,} valid "
                    f"({n_w:,} water, {n_l:,} land)"
                )
                if n_w > 0:
                    all_water_pts.append(wp[water_valid])
                    all_water_rgb.append(img[water_valid])
                if n_l > 0:
                    all_land_pts.append(wp[land_valid])
                    all_land_rgb.append(img[land_valid])
            else:
                print(f"  View {idx:03d}: {len(pts):,} valid (no mask)")
        else:
            print(f"  View {idx:03d}: {len(pts):,} valid points")

    if not all_pts:
        print("[ERROR] No valid points across all views")
        return None, None, {}

    all_pts = np.concatenate(all_pts, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    print(f"Total before filtering: {len(all_pts):,} points")

    if not args.no_filter:
        all_pts, all_rgb, removed = filter_outliers(
            all_pts, all_rgb, args.outlier_percentile
        )
        if removed > 0:
            print(f"Removed {removed:,} outlier points")

    if umeyama is not None:
        s = umeyama["scale"]
        R = umeyama["rotation"]
        t = umeyama["translation"]
        all_pts = apply_umeyama(all_pts, s, R, t)
        print(f"Applied Umeyama -> UTM33 ({len(all_pts):,} points)")

    extra = {}
    if has_masks:
        if all_water_pts:
            water_all = np.concatenate(all_water_pts, axis=0)
            water_clr = np.concatenate(all_water_rgb, axis=0)
            if not args.no_filter:
                water_all, water_clr, _ = filter_outliers(
                    water_all, water_clr, args.outlier_percentile
                )
            if umeyama is not None:
                water_all = apply_umeyama(
                    water_all,
                    umeyama["scale"],
                    umeyama["rotation"],
                    umeyama["translation"],
                )
            extra["water_points"] = water_all
            extra["water_colors"] = water_clr
        if all_land_pts:
            land_all = np.concatenate(all_land_pts, axis=0)
            land_clr = np.concatenate(all_land_rgb, axis=0)
            if not args.no_filter:
                land_all, land_clr, _ = filter_outliers(
                    land_all, land_clr, args.outlier_percentile
                )
            if umeyama is not None:
                land_all = apply_umeyama(
                    land_all,
                    umeyama["scale"],
                    umeyama["rotation"],
                    umeyama["translation"],
                )
            extra["land_points"] = land_all
            extra["land_colors"] = land_clr

    return all_pts, all_rgb, extra


NDC_NEAR_PLANE_Z = -0.999


def _discover_nerfrac_views_with_images(testset_dir, depth_type="mean"):
    views = discover_nerfrac_views(testset_dir, depth_type)
    for v in views:
        idx = v["index"]
        img_candidates = glob.glob(os.path.join(testset_dir, f"*{idx:03d}.png"))
        img_candidates = [
            f for f in img_candidates
            if not any(x in os.path.basename(f)
                       for x in ["masked", "rgba", "err", "bathymetry_view"])
        ]
        v["image"] = img_candidates[0] if img_candidates else None
    return views


def process_nerfrac_view(view, H, W, focal, c2w_recenter, sc, water_mask=None):
    idx = view["index"]

    if view["xs"] is None or view["rays"] is None or view["depth"] is None:
        print(f"  View {idx}: Missing required files, skipping")
        return None

    try:
        Xs = np.load(view["xs"])
        Rays = np.load(view["rays"])
        Depth = np.load(view["depth"])
    except Exception as e:
        print(f"  View {idx}: Error loading: {e}")
        return None

    Depth_flat = Depth.reshape(-1, 1)

    P_ndc = Xs + Depth_flat * Rays

    P_colmap = ndc_to_colmap(P_ndc, H, W, focal, c2w_recenter, sc)

    if view["image"] is not None:
        try:
            img = imageio.imread(view["image"])
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[:, :, :3]
            if img.shape[0] == H and img.shape[1] == W:
                colors = img.reshape(-1, 3)
            else:
                print(f"  View {idx}: Image size mismatch, using gray")
                colors = np.full((len(P_colmap), 3), 128, dtype=np.uint8)
        except Exception as e:
            print(f"  View {idx}: Error loading image: {e}")
            colors = np.full((len(P_colmap), 3), 128, dtype=np.uint8)
    else:
        colors = np.full((len(P_colmap), 3), 128, dtype=np.uint8)

    result = {
        "all_points": P_colmap,
        "all_colors": colors,
        "water_points": None,
        "water_colors": None,
        "water_surface": None,
        "land_points": None,
        "land_colors": None,
    }

    if water_mask is None:
        return result

    water_idx = np.where(water_mask)[0]
    land_idx = np.where(~water_mask)[0]

    model_water = Xs[:, 2] > NDC_NEAR_PLANE_Z
    agree = np.sum(water_mask == model_water)
    print(
        f"  View {idx}: Mask/model agreement: {agree:,}/{len(water_mask):,} "
        f"({100 * agree / len(water_mask):.1f}%)"
    )

    if len(water_idx) > 0:
        result["water_points"] = P_colmap[water_idx]
        result["water_colors"] = colors[water_idx]
        # Water surface = Xs points only (no depth), converted to COLMAP
        Xs_water_colmap = ndc_to_colmap(Xs[water_idx], H, W, focal, c2w_recenter, sc)
        result["water_surface"] = Xs_water_colmap
    else:
        result["water_points"] = np.empty((0, 3))
        result["water_colors"] = np.empty((0, 3), dtype=np.uint8)
        result["water_surface"] = np.empty((0, 3))

    if len(land_idx) > 0:
        result["land_points"] = P_colmap[land_idx]
        result["land_colors"] = colors[land_idx]
    else:
        result["land_points"] = np.empty((0, 3))
        result["land_colors"] = np.empty((0, 3), dtype=np.uint8)

    n_w, n_l = len(water_idx), len(land_idx)
    print(f"  View {idx}: {n_w:,} water, {n_l:,} land pixels")

    return result

def fit_water_surface_plane(all_surface_points, max_points=50000):
    N = len(all_surface_points)
    if N < 3:
        print("[WARN] Too few water surface points for plane fit")
        return None, None, None

    if N > max_points:
        idx = np.random.default_rng(42).choice(N, max_points, replace=False)
        pts = all_surface_points[idx]
    else:
        pts = all_surface_points

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)
    normal = Vh[2]

    if normal[2] > 0:
        normal = -normal

    d = -np.dot(normal, centroid)
    planarity = S[2] / S[1] if S[1] > 0 else 0

    residuals = np.dot(all_surface_points, normal) + d
    print(f"\nWater surface plane fit:")
    print(f"  Points used: {len(pts):,} / {N:,}")
    print(f"  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"  Normal:   [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  Planarity (S2/S1): {planarity:.6f} (smaller = more planar)")
    print(
        f"  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
        f"range=[{residuals.min():.4f}, {residuals.max():.4f}]"
    )

    return normal, d, centroid


def align_water_surfaces(per_view_data, normal, d):
    print("\nPer-view water surface alignment:")
    for i, vd in enumerate(per_view_data):
        if vd is None:
            continue
        ws = vd.get("water_surface")
        if ws is None or len(ws) == 0:
            print(f"  View {i}: no water surface points, skipping alignment")
            continue

        dists = np.dot(ws, normal) + d
        median_offset = np.median(dists)
        print(
            f"  View {i}: WS median offset = {median_offset:+.4f}, "
            f"std = {dists.std():.4f} ({len(ws):,} WS points)"
        )

        shift = -median_offset * normal
        if vd["water_points"] is not None and len(vd["water_points"]) > 0:
            vd["water_points"] = vd["water_points"] + shift
        if len(ws) > 0:
            vd["water_surface"] = ws + shift

    return per_view_data



def _load_nerfrac_masks(mask_dir, n_views, H, W):
    mask_files = list_masks(mask_dir, n_views)
    masks = []
    for i in range(min(len(mask_files), n_views)):
        m = imageio.imread(mask_files[i])
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape[0] != H or m.shape[1] != W:
            import cv2

            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        water_bool = (m > WATER_MASK_THRESHOLD).reshape(-1)
        n_water = np.sum(water_bool)
        print(
            f"  Mask {i}: {os.path.basename(mask_files[i])} — "
            f"{n_water:,} water ({100 * n_water / (H * W):.1f}%), "
            f"{H * W - n_water:,} land"
        )
        masks.append(water_bool)

    while len(masks) < n_views:
        masks.append(None)

    return masks


def generate_nerfrac_pointcloud(args):
    testset_dir = os.path.join(args.log_folder, args.testset)
    if not os.path.isdir(testset_dir):
        print(f"[ERROR] Testset dir not found: {testset_dir}")
        return None, None, {}

    data_dir = args.data_dir if args.data_dir else None
    coord_meta = load_nerfrac_coord_metadata(args.log_folder, data_dir=data_dir)
    if coord_meta["c2w_recenter"] is None or coord_meta["sc"] is None:
        print("[ERROR] Cannot load c2w_recenter or sc from log folder")
        return None, None, {}

    c2w_recenter = coord_meta["c2w_recenter"]
    sc = coord_meta["sc"]
    print(f"Loaded coord metadata: sc={sc:.8f}")

    if args.poses_bounds:
        H, W, focal = load_focal_from_poses_bounds(args.poses_bounds, args.factor)
    else:
        print("[ERROR] --poses_bounds is required for NeRFrac mode")
        return None, None, {}
    print(f"Resolution: {H}x{W}, focal={focal:.4f} (factor={args.factor})")

    depth_type = getattr(args, "depth_type", "mean")
    views = _discover_nerfrac_views_with_images(testset_dir, depth_type=depth_type)
    if not views:
        print(f"[ERROR] No NeRFrac views found in {testset_dir}")
        return None, None, {}
    n_views = len(views)
    print(f"Found {n_views} views in {testset_dir} (depth_type={depth_type})")

    for v in views:
        if v["xs"] is not None:
            try:
                xs0 = np.load(v["xs"])
                n_pixels = xs0.shape[0]
                H_actual = n_pixels // W
                if H_actual * W == n_pixels and H_actual != H:
                    print(
                        f"[INFO] Adjusting H from poses_bounds ({H}) to actual "
                        f"model output ({H_actual}) — {n_pixels} pixels = "
                        f"{H_actual}x{W}"
                    )
                    H = H_actual
                elif H_actual * W != n_pixels:
                    print(
                        f"[WARN] Model output has {n_pixels} pixels, "
                        f"not divisible by W={W}"
                    )
            except Exception:
                pass
            break

    umeyama = None
    if args.umeyama_npz:
        umeyama = load_umeyama(args.umeyama_npz)
        print(f"Loaded Umeyama (scale={umeyama['scale']:.4f}) from {args.umeyama_npz}")

    # Load masks
    has_masks = bool(args.mask_dir)
    masks = None
    if has_masks:
        print(f"\nLoading water masks from {args.mask_dir}...")
        masks = _load_nerfrac_masks(args.mask_dir, n_views, H, W)
        if masks is None:
            print("[WARN] No masks loaded — all pixels treated together")
            has_masks = False

    output_dir = args.output_dir if args.output_dir else testset_dir

    print("\nProcessing views...")
    per_view_data = []
    for view in views:
        idx = view["index"]
        mask_i = masks[idx] if masks is not None and idx < len(masks) else None
        vd = process_nerfrac_view(view, H, W, focal, c2w_recenter, sc, mask_i)
        per_view_data.append(vd)

    if has_masks and not args.no_align:
        all_ws = []
        for vd in per_view_data:
            if vd is not None and vd.get("water_surface") is not None:
                ws = vd["water_surface"]
                if len(ws) > 0:
                    all_ws.append(ws)
        if all_ws:
            all_ws_combined = np.vstack(all_ws)
            normal, d_plane, _ = fit_water_surface_plane(all_ws_combined)
            if normal is not None:
                per_view_data = align_water_surfaces(per_view_data, normal, d_plane)
        else:
            print("\n[WARN] No water surface points — skipping alignment")

    all_water_pts, all_water_clr = [], []
    all_land_pts, all_land_clr = [], []
    all_combined_pts, all_combined_clr = [], []

    for i, vd in enumerate(per_view_data):
        if vd is None:
            continue

        if has_masks and vd.get("water_points") is not None:
            wp = vd["water_points"]
            wc = vd["water_colors"]
            lp = vd["land_points"]
            lc = vd["land_colors"]

            if not args.no_filter:
                if len(wp) > 0:
                    wp, wc, _ = filter_outliers(wp, wc, args.outlier_percentile)
                if len(lp) > 0:
                    lp, lc, _ = filter_outliers(lp, lc, args.outlier_percentile)

            if len(wp) > 0:
                all_water_pts.append(wp)
                all_water_clr.append(wc)
            if len(lp) > 0:
                all_land_pts.append(lp)
                all_land_clr.append(lc)

            if args.per_view_ply:
                parts = [p for p in [wp, lp] if len(p) > 0]
                colors_parts = [c for c in [wc, lc] if len(c) > 0]
                if parts:
                    vp = np.vstack(parts)
                    vc = np.vstack(colors_parts)
                    view_ply = os.path.join(output_dir, f"pointcloud_view_{i:03d}.ply")
                    write_ply_binary(view_ply, vp, vc)

            parts = [p for p in [wp, lp] if len(p) > 0]
            colors_parts = [c for c in [wc, lc] if len(c) > 0]
            if parts:
                all_combined_pts.append(np.vstack(parts))
                all_combined_clr.append(np.vstack(colors_parts))
        else:
            pts = vd["all_points"]
            clr = vd["all_colors"]
            if not args.no_filter:
                pts, clr, _ = filter_outliers(pts, clr, args.outlier_percentile)
            if len(pts) > 0:
                all_combined_pts.append(pts)
                all_combined_clr.append(clr)
                if args.per_view_ply:
                    view_ply = os.path.join(output_dir, f"pointcloud_view_{i:03d}.ply")
                    write_ply_binary(view_ply, pts, clr)

    if not all_combined_pts:
        print("[ERROR] No valid points across all views")
        return None, None, {}

    combined_pts = np.vstack(all_combined_pts)
    combined_clr = np.vstack(all_combined_clr)
    print(f"\nTotal combined: {len(combined_pts):,} points")

    if not args.no_filter:
        combined_pts, combined_clr, removed = filter_outliers(
            combined_pts, combined_clr, percentile=0.5
        )
        if removed > 0:
            print(f"Removed {removed:,} global outliers")

    if umeyama is not None:
        s = umeyama["scale"]
        R = umeyama["rotation"]
        t = umeyama["translation"]
        combined_pts = apply_umeyama(combined_pts, s, R, t)
        print(f"Applied Umeyama -> UTM33 ({len(combined_pts):,} points)")

    extra = {}
    if has_masks:
        if all_water_pts:
            water_all = np.vstack(all_water_pts)
            water_clr = np.vstack(all_water_clr)
            if not args.no_filter:
                water_all, water_clr, _ = filter_outliers(water_all, water_clr, 0.5)
            if umeyama is not None:
                s_u, R_u, t_u = (
                    umeyama["scale"],
                    umeyama["rotation"],
                    umeyama["translation"],
                )
                water_all = apply_umeyama(water_all, s_u, R_u, t_u)
            extra["water_points"] = water_all
            extra["water_colors"] = water_clr
        if all_land_pts:
            land_all = np.vstack(all_land_pts)
            land_clr = np.vstack(all_land_clr)
            if not args.no_filter:
                land_all, land_clr, _ = filter_outliers(land_all, land_clr, 0.5)
            if umeyama is not None:
                s_u, R_u, t_u = (
                    umeyama["scale"],
                    umeyama["rotation"],
                    umeyama["translation"],
                )
                land_all = apply_umeyama(land_all, s_u, R_u, t_u)
            extra["land_points"] = land_all
            extra["land_colors"] = land_clr

    return combined_pts, combined_clr, extra




def save_preview(points, colors, output_dir, label=""):
    """Save a matplotlib 3D scatter preview."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, len(points) // 100000)
    pts = points[::step]
    clr = colors[::step].astype(float) / 255.0

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=clr, s=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Point Cloud ({len(points):,} pts total, {len(pts):,} shown) {label}")

    plt.tight_layout()
    path = os.path.join(output_dir, "pointcloud_preview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate colored scene point cloud (PLY) from model output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode",
        required=True,
        choices=["vggt", "nerfrac"],
        help="Model type",
    )

    # VGGT
    p.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="[vggt] VGGT output directory with world_points_NNN.npy + rgb_NNN.png",
    )

    # NeRFrac
    p.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="[nerfrac] Path to NeRFrac experiment log folder",
    )
    p.add_argument(
        "--testset",
        type=str,
        default="",
        help="[nerfrac] Testset subdirectory (e.g. testset_all_600000)",
    )
    p.add_argument(
        "--poses_bounds",
        type=str,
        default="",
        help="[nerfrac] Path to poses_bounds.npy (for H, W, focal)",
    )
    p.add_argument(
        "--factor",
        type=int,
        default=8,
        help="[nerfrac] Image downsample factor (must match training)",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="[nerfrac] Fallback dataset dir for sc recomputation",
    )
    p.add_argument(
        "--no_align",
        action="store_true",
        help="[nerfrac] Skip per-view water surface alignment",
    )
    p.add_argument(
        "--per_view_ply",
        action="store_true",
        help="[nerfrac] Also save per-view PLY files",
    )

    # Common
    p.add_argument(
        "--umeyama_npz",
        type=str,
        default="",
        help="Umeyama .npz file to georeference to UTM33 (from 01_compute_umeyama.py)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for PLY + preview (default: model output dir)",
    )
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="[vggt] Minimum depth_conf to include a point (0 = no filter)",
    )
    p.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Directory with water mask PNGs, paired with the views by sorted file "
        "name; adds separate water and land point clouds",
    )
    p.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable outlier filtering",
    )
    p.add_argument(
        "--outlier_percentile",
        type=float,
        default=1.0,
        help="Percentile for outlier removal (default 1.0)",
    )
    p.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip generating matplotlib preview PNG",
    )
    p.add_argument(
        "--depth_type",
        type=str,
        default="mode",
        choices=["mean", "mode"],
        help="[nerfrac] Which depth map to use. "
        "'mode': peak of weight distribution (*_depth_mode.npy). "
        "'mean': weighted average depth (*_depth.npy).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "vggt":
        if not args.vggt_dir:
            print("[ERROR] --vggt_dir is required for VGGT mode")
            sys.exit(1)
        if not os.path.isdir(args.vggt_dir):
            print(f"[ERROR] VGGT directory not found: {args.vggt_dir}")
            sys.exit(1)
    elif args.mode == "nerfrac":
        if not args.log_folder:
            print("[ERROR] --log_folder is required for NeRFrac mode")
            sys.exit(1)
        if not args.testset:
            print("[ERROR] --testset is required for NeRFrac mode")
            sys.exit(1)
        if not args.poses_bounds:
            print("[ERROR] --poses_bounds is required for NeRFrac mode")
            sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    elif args.mode == "vggt":
        output_dir = args.vggt_dir
    elif args.mode == "nerfrac":
        output_dir = os.path.join(args.log_folder, args.testset)
    else:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("POINT CLOUD GENERATION")
    print("=" * 60)
    print(f"Mode:             {args.mode}")
    if args.mode == "vggt":
        print(f"VGGT dir:         {args.vggt_dir}")
    elif args.mode == "nerfrac":
        print(f"Log folder:       {args.log_folder}")
        print(f"Testset:          {args.testset}")
        print(f"Factor:           {args.factor}")
    print(f"Output dir:       {output_dir}")
    print(f"Umeyama:          {args.umeyama_npz or '(none)'}")
    print(
        f"Outlier filter:   {'off' if args.no_filter else f'{args.outlier_percentile}%'}"
    )
    if args.mask_dir:
        if args.mode == "nerfrac":
            print(f"Mask dir:         {args.mask_dir} (water/land separation)")
            print(f"WS alignment:     {'off' if args.no_align else 'on'}")
        else:
            print(f"Mask dir:         {args.mask_dir} (water/land separation)")
    print()

    extra = {}
    if args.mode == "vggt":
        points, colors, extra = generate_vggt_pointcloud(args)
    elif args.mode == "nerfrac":
        points, colors, extra = generate_nerfrac_pointcloud(args)
    else:
        print(f"[ERROR] Mode '{args.mode}' not yet implemented")
        sys.exit(1)

    if points is None:
        print("[ERROR] Point cloud generation failed")
        sys.exit(1)

    ply_path = os.path.join(output_dir, "pointcloud_combined.ply")
    write_ply_binary(ply_path, points, colors)
    size_mb = os.path.getsize(ply_path) / (1024 * 1024)
    print(f"\nSaved: {ply_path} ({size_mb:.1f} MB, {len(points):,} points)")

    npy_path = os.path.join(output_dir, "pointcloud_combined.npy")
    np.save(npy_path, points)
    print(f"Saved: {npy_path}")

    if "water_points" in extra:
        wp = extra["water_points"]
        wc = extra["water_colors"]
        water_ply = os.path.join(output_dir, "pointcloud_water.ply")
        write_ply_binary(water_ply, wp, wc)
        np.save(os.path.join(output_dir, "pointcloud_water.npy"), wp)
        print(f"Saved: {water_ply} ({len(wp):,} water points)")
    if "land_points" in extra:
        lp = extra["land_points"]
        lc = extra["land_colors"]
        land_ply = os.path.join(output_dir, "pointcloud_land.ply")
        write_ply_binary(land_ply, lp, lc)
        np.save(os.path.join(output_dir, "pointcloud_land.npy"), lp)
        print(f"Saved: {land_ply} ({len(lp):,} land points)")

    print(f"\nPoint cloud bounds:")
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"  {axis}: [{points[:, i].min():.4f}, {points[:, i].max():.4f}]")

    if not args.no_preview:
        label = (
            "(UTM33)"
            if args.umeyama_npz
            else "(COLMAP)"
            if args.mode == "nerfrac"
            else ""
        )
        save_preview(points, colors, output_dir, label=label)

    print(f"\nDone. {len(points):,} points saved to {output_dir}")


if __name__ == "__main__":
    main()
