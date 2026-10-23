"""
Analytical Snell's law refraction correction for VGGT depth predictions.

    # Multi-view correction
    python 04_correct_refraction.py \\
        --vggt_dir  '/path/to/vggt_output' \\
        --output_dir '/path/to/corrected_output' \\
        --mask_dir  '/path/to/masks' \\
        --mode multi_view

    # Optimized correction
    python 04_correct_refraction.py \\
        --vggt_dir  '/path/to/vggt_output' \\
        --output_dir '/path/to/corrected_output' \\
        --mask_dir  '/path/to/masks' \\
        --mode optimized

    # Afterwards extract bathymetry from the corrected output:
    python 02_extract_bathymetry.py --mode vggt \\
        --vggt_dir '/path/to/corrected_output' \\
        --umeyama_npz ../data/transforms/<umeyama.npz> \\
        --wsm_path <WSM.tif> \\
        --mask_dir '/path/to/masks' \\
        --output_csv bathymetry/bathymetry_corrected.csv

Pipeline position: runs AFTER 01_compute_umeyama.py, BEFORE 02_extract_bathymetry.py.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize


from coord_utils import WATER_MASK_THRESHOLD, discover_vggt_views

N_WATER_DEFAULT = 1.333


def parse_args():
    p = argparse.ArgumentParser(
        description="Snell's law refraction correction for VGGT outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vggt_dir", type=str, required=True, help="Input VGGT output directory"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for corrected data",
    )
    p.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory with water-mask PNGs (sorted order must match views)",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="multi_view",
        choices=["single_view", "multi_view", "optimized"],
        help="Correction mode",
    )
    p.add_argument(
        "--n_water",
        type=float,
        default=N_WATER_DEFAULT,
        help="Refractive index of water",
    )
    p.add_argument(
        "--shoreline_buffer",
        type=int,
        default=1,
        help="Mask dilation radius (pixels) for shoreline extraction",
    )
    p.add_argument(
        "--mask_threshold",
        type=int,
        default=WATER_MASK_THRESHOLD,
        help="Threshold for binary mask (>threshold = water)",
    )
    p.add_argument(
        "--ransac_iterations",
        type=int,
        default=2000,
        help="RANSAC iterations for plane fitting",
    )
    p.add_argument(
        "--ransac_threshold",
        type=float,
        default=None,
        help="RANSAC inlier distance threshold (auto if None)",
    )
    p.add_argument(
        "--opt_subsample_views",
        type=int,
        default=0,
        help="Use about this many evenly spaced views for optimisation (0 = all)",
    )
    p.add_argument(
        "--opt_subsample_pixels",
        type=int,
        default=4,
        help="Use every Nth water pixel for optimisation",
    )
    p.add_argument(
        "--opt_grid_res",
        type=float,
        default=0.01,
        help="2-D grid cell size (VGGT units) for optimisation loss",
    )
    args = p.parse_args()
    if args.n_water < 1:
        p.error("--n_water must be at least 1")
    if args.ransac_threshold is not None and args.ransac_threshold <= 0:
        p.error("--ransac_threshold must be positive")
    if args.opt_grid_res <= 0:
        p.error("--opt_grid_res must be positive")
    return args


def load_mask(mask_path, target_H, target_W, threshold):
    import cv2

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    # VGGT scales the width to its frame and rounds the height to its 14 px patches;
    # a larger mismatch means it cropped or padded the image, and a resize misaligns
    if abs(m.shape[0] * target_W / m.shape[1] - target_H) > 7:
        raise ValueError(
            f"Mask {Path(mask_path).name} ({m.shape[1]} x {m.shape[0]}) does not fit "
            f"the VGGT frame ({target_W} x {target_H}): VGGT cropped or padded the images"
        )
    if m.shape[0] != target_H or m.shape[1] != target_W:
        m = cv2.resize(m, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
    return m > threshold


def load_all_masks(mask_dir, n_views, target_H, target_W, threshold):
    mask_files = sorted(Path(mask_dir).glob("*.png"))
    if len(mask_files) == 0:
        raise FileNotFoundError(f"No PNG masks in {mask_dir}")
    if len(mask_files) != n_views:
        raise ValueError(f"{len(mask_files)} masks in {mask_dir} for {n_views} views")
    masks = []
    for i in range(n_views):
        masks.append(load_mask(mask_files[i], target_H, target_W, threshold))
    print(
        f"  Loaded {n_views} masks from {mask_dir}  (resized to {target_H}x{target_W})"
    )
    return masks


def camera_centre(extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return -R.T @ t


def extract_shoreline_points(world_points, water_mask, buffer_px=1):
    struct = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), dtype=bool)
    dilated = binary_dilation(water_mask, structure=struct)
    ring = dilated & (~water_mask)
    valid = ring & np.isfinite(world_points).all(axis=2)
    return world_points[valid], valid


def fit_plane_ransac(points, n_iter=2000, threshold=None, seed=42):
    rng = np.random.default_rng(seed)
    N = len(points)
    if N < 3:
        raise ValueError(f"Need >= 3 points for plane fitting, got {N}")

    if threshold is None:
        centroid = points.mean(axis=0)
        _, _, Vh = np.linalg.svd(points - centroid, full_matrices=False)
        n0 = Vh[-1]
        d0 = -n0 @ centroid
        dists = np.abs(points @ n0 + d0)
        threshold = max(2.0 * np.median(dists), np.percentile(dists, 75), 1e-8)

    best_count, best_n, best_d = 0, None, None
    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        v1 = points[idx[1]] - points[idx[0]]
        v2 = points[idx[2]] - points[idx[0]]
        n = np.cross(v1, v2)
        nrm = np.linalg.norm(n)
        if nrm < 1e-12:
            continue
        n /= nrm
        d = -n @ points[idx[0]]
        cnt = np.sum(np.abs(points @ n + d) < threshold)
        if cnt > best_count:
            best_count, best_n, best_d = cnt, n, d

    if best_n is None:
        centroid = points.mean(axis=0)
        _, _, Vh = np.linalg.svd(points - centroid, full_matrices=False)
        best_n = Vh[-1]
        best_d = -best_n @ centroid

    mask = np.abs(points @ best_n + best_d) < threshold
    if mask.sum() >= 3:
        inlier_pts = points[mask]
        c = inlier_pts.mean(axis=0)
        _, _, Vh = np.linalg.svd(inlier_pts - c, full_matrices=False)
        normal = Vh[-1]
        d = -normal @ c
    else:
        normal, d = best_n, best_d

    residuals = points @ normal + d
    mask = np.abs(residuals) < threshold
    res_std = float(np.std(residuals[mask])) if mask.any() else float("inf")
    return normal, d, mask, res_std, threshold


def orient_normal_toward_cameras(normal, d, camera_positions):
    """Flip normal so it points toward the camera centroid (= air side)."""
    cam_centroid = camera_positions.mean(axis=0)
    if normal @ cam_centroid + d < 0:
        normal, d = -normal, -d
    return normal, d


def snell_correct_view(
    world_points,
    water_mask,
    C,
    plane_normal,
    plane_d,
    n_water=N_WATER_DEFAULT,
):
    H, W = water_mask.shape
    corrected = world_points.copy()

    wy, wx = np.where(water_mask & np.isfinite(world_points).all(axis=2))
    N = len(wy)
    if N == 0:
        return corrected, {"n_water": 0, "n_corrected": 0, "n_skipped": 0}

    # the apparent points come from the same arrays as the shoreline points, so a
    # point-map run is corrected on its own geometry
    P_app = world_points[wy, wx].astype(np.float64)
    dirs_unnorm = P_app - C[None, :]  # (N, 3)

    n_dot_dir = dirs_unnorm @ plane_normal  # (N,)
    n_dot_C_d = plane_normal @ C + plane_d

    safe = np.abs(n_dot_dir) > 1e-12
    t_surface = np.full(N, np.inf)
    t_surface[safe] = -n_dot_C_d / n_dot_dir[safe]

    valid = safe & (t_surface > 0)

    P_surf = C[None, :] + t_surface[:, None] * dirs_unnorm  # (N, 3)

    dir_norms = np.linalg.norm(dirs_unnorm, axis=1, keepdims=True)
    dirs_norm = dirs_unnorm / np.maximum(dir_norms, 1e-12)
    cos_theta_i = np.abs(dirs_norm @ plane_normal)
    cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)
    sin_theta_i = np.sqrt(1.0 - cos_theta_i**2)

    sin_theta_r = sin_theta_i / n_water
    valid &= sin_theta_r <= 1.0
    cos_theta_r = np.sqrt(np.maximum(1.0 - sin_theta_r**2, 0.0))

    # vector Snell's law
    sign = -np.sign(dirs_norm @ plane_normal)  # (N,)
    n_hat = sign[:, None] * plane_normal[None, :]  # (N, 3)
    eta = 1.0 / n_water
    dir_refr = eta * dirs_norm + (eta * cos_theta_i - cos_theta_r)[:, None] * n_hat
    ref_len = np.linalg.norm(dir_refr, axis=1, keepdims=True)
    valid &= ref_len.squeeze() > 1e-12
    dir_refr /= np.maximum(ref_len, 1e-12)

    # correct point 3-D position; apparent points above the surface stay uncorrected
    signed_dist_to_plane = -(P_app @ plane_normal + plane_d)  # >0 = below surface
    d_uw = np.abs(signed_dist_to_plane) / np.maximum(cos_theta_i, 1e-12)
    valid &= signed_dist_to_plane > 1e-12
    depth_perp_app = np.abs(signed_dist_to_plane)
    depth_perp_true = depth_perp_app * n_water
    valid &= cos_theta_r > 1e-12
    t_true = np.zeros(N)
    t_true[valid] = depth_perp_true[valid] / cos_theta_r[valid]

    P_true = P_surf + t_true[:, None] * dir_refr
    corrected[wy[valid], wx[valid]] = P_true[valid]

    n_above_plane = int((signed_dist_to_plane < 0).sum())
    if valid.any():
        angles = np.degrees(np.arccos(cos_theta_i[valid]))
        disp = np.linalg.norm(P_true[valid] - P_app[valid], axis=1)
        stats = {
            "n_water": int(N),
            "n_corrected": int(valid.sum()),
            "n_skipped": int((~valid).sum()),
            "n_above_plane": n_above_plane,
            "mean_incidence_deg": round(float(angles.mean()), 2),
            "max_incidence_deg": round(float(angles.max()), 2),
            "mean_displacement": round(float(disp.mean()), 6),
        }
    else:
        stats = {"n_water": int(N), "n_corrected": 0, "n_skipped": int(N)}
    return corrected, stats


def _build_opt_data(views_data, sub_views=0, sub_px=4):
    """Pre-compute lightweight arrays for the optimiser."""
    n = len(views_data)
    use = list(range(n))
    if 0 < sub_views < n:
        use = list(range(0, n, max(1, n // sub_views)))

    out = []
    for vi in use:
        vd = views_data[vi]
        mask = vd["water_mask"]
        wy, wx = np.where(mask & np.isfinite(vd["world_points"]).all(axis=2))
        if len(wy) == 0:
            continue
        if sub_px > 1:
            sel = np.arange(0, len(wy), sub_px)
            wy, wx = wy[sel], wx[sel]
        C = vd["C"]
        P_app = vd["world_points"][wy, wx].astype(np.float64)
        dirs_unnorm = P_app - C[None, :]
        out.append(
            {
                "C": C,
                "dirs_unnorm": dirs_unnorm,
                "P_app": P_app,
                "vi": vi,
            }
        )
    return out


def _opt_loss(params, opt_data, base_normal, base_d, n_water, grid_res):
    dh, dtx, dty = params
    cx, sx = np.cos(dtx), np.sin(dtx)
    cy, sy = np.cos(dty), np.sin(dty)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    normal = Ry @ Rx @ base_normal
    normal /= np.linalg.norm(normal)
    d = base_d - dh

    all_keys = []
    all_depths = []
    all_views = []

    for vd in opt_data:
        C = vd["C"]
        dirs = vd["dirs_unnorm"]
        P_app = vd["P_app"]
        N = len(P_app)

        n_dot_dir = dirs @ normal
        safe = np.abs(n_dot_dir) > 1e-12
        t_surf = np.full(N, np.inf)
        t_surf[safe] = -(normal @ C + d) / n_dot_dir[safe]
        valid = safe & (t_surf > 0)
        if valid.sum() < 10:
            continue

        P_surf = C[None, :] + t_surf[:, None] * dirs
        dir_norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_n = dirs / np.maximum(dir_norms, 1e-12)
        cos_ti = np.clip(np.abs(dirs_n @ normal), 0, 1)
        sin_ti = np.sqrt(1 - cos_ti**2)
        sin_tr = sin_ti / n_water
        valid &= sin_tr <= 1.0
        cos_tr = np.sqrt(np.maximum(1 - sin_tr**2, 0))

        signed_dist = -(P_app @ normal + d)
        # unlike the correction, the loss keeps the points above the plane: without
        # them the plane could lower the loss by moving away from all points
        valid &= (np.abs(signed_dist) > 1e-12) & (cos_tr > 1e-12)
        depth_perp = np.abs(signed_dist) * n_water

        uv = np.floor(P_surf[valid, :2] / grid_res).astype(np.int64)
        all_keys.append(uv)
        all_depths.append(depth_perp[valid])
        all_views.append(np.full(valid.sum(), vd["vi"], dtype=np.int32))

    if not all_keys:
        return np.inf

    all_keys = np.concatenate(all_keys, axis=0)
    all_depths = np.concatenate(all_depths)
    all_views = np.concatenate(all_views)

    cells = defaultdict(lambda: defaultdict(list))
    for i in range(len(all_keys)):
        cells[(all_keys[i, 0], all_keys[i, 1])][all_views[i]].append(all_depths[i])

    total_var = 0.0
    n_cells = 0
    for view_depths in cells.values():
        if len(view_depths) < 2:
            continue
        means = [np.mean(dl) for dl in view_depths.values()]
        total_var += np.var(means)
        n_cells += 1

    # without a cell seen twice there is nothing to compare, not a perfect fit
    if n_cells == 0:
        return np.inf
    return total_var / n_cells


def optimize_water_surface(
    views_data, base_normal, base_d, n_water, sub_views, sub_px, grid_res
):
    print("\n --- Optimising water surface plane --")
    opt_data = _build_opt_data(views_data, sub_views, sub_px)
    total_pts = sum(len(vd["P_app"]) for vd in opt_data)
    print(f"    Optimisation data: {len(opt_data)} views, {total_pts:,} points")

    loss0 = _opt_loss([0, 0, 0], opt_data, base_normal, base_d, n_water, grid_res)
    if not np.isfinite(loss0):
        print("    No grid cell is seen from two views; keeping the RANSAC plane")
        return base_normal, base_d, None
    print(f"    Initial cross-view depth variance: {loss0:.8f}")

    result = minimize(
        _opt_loss,
        x0=[0.0, 0.0, 0.0],
        args=(opt_data, base_normal, base_d, n_water, grid_res),
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-10},
    )

    dh, dtx, dty = result.x
    cx, sx = np.cos(dtx), np.sin(dtx)
    cy, sy = np.cos(dty), np.sin(dty)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    opt_normal = Ry @ Rx @ base_normal
    opt_normal /= np.linalg.norm(opt_normal)
    opt_d = base_d - dh

    improv = (1 - result.fun / max(loss0, 1e-15)) * 100
    print(
        f"    Optimised: dh={dh:.6f},  "
        f"dtx={np.degrees(dtx):.4f} deg,  dty={np.degrees(dty):.4f} deg"
    )
    print(
        f"    Final variance: {result.fun:.8f}  "
        f"(was {loss0:.8f},  improvement {improv:.1f}%)"
    )
    return opt_normal, opt_d, result

def setup_output_dir(output_dir, vggt_dir, view_indices):
    output_dir = Path(output_dir)
    vggt_dir = Path(vggt_dir)
    # 02 would read the views of an earlier, larger run along with this one
    stale = sorted(
        f.name
        for ext in ("npy", "png")
        for f in output_dir.glob(f"*_[0-9][0-9][0-9].{ext}")
        if int(f.stem[-3:]) not in view_indices
    )
    if stale:
        raise ValueError(
            f"{output_dir} holds views of an earlier run ({stale[0]}), use a new folder"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "depth_{idx:03d}.npy",
        "depth_conf_{idx:03d}.npy",
        "extrinsic_{idx:03d}.npy",
        "intrinsic_{idx:03d}.npy",
        "rgb_{idx:03d}.png",
        "world_points_conf_{idx:03d}.npy",
    ]
    n_linked = 0
    for idx in view_indices:
        for pat in patterns:
            fname = pat.format(idx=idx)
            src = vggt_dir / fname
            dst = output_dir / fname
            # a link or copy left by an earlier run may belong to another input
            dst.unlink(missing_ok=True)
            if src.exists():
                os.symlink(src.resolve(), dst)
                n_linked += 1
    print(f"  Symlinked {n_linked} unchanged files to {output_dir}")


def save_metadata(output_dir, args, plane_normal, plane_d, plane_stats, per_view_stats):
    output_dir = Path(output_dir)
    refr_meta = {
        "correction_script": "04_correct_refraction.py",
        "mode": args.mode,
        "n_water": args.n_water,
        "shoreline_buffer_px": args.shoreline_buffer,
        "mask_threshold": args.mask_threshold,
        "ransac_iterations": args.ransac_iterations,
        "ransac_threshold": args.ransac_threshold,
        "opt_subsample_views": args.opt_subsample_views,
        "opt_subsample_pixels": args.opt_subsample_pixels,
        "opt_grid_res": args.opt_grid_res,
        "source_vggt_dir": str(args.vggt_dir),
        "mask_dir": str(args.mask_dir),
        # single_view stores each view's plane in its stats instead
        "plane_normal": None if plane_normal is None else plane_normal.tolist(),
        "plane_d": None if plane_d is None else float(plane_d),
        "plane_stats": plane_stats,
        "per_view_stats": per_view_stats,
    }
    # unlink first, so that a link in place of an output is not written through
    (output_dir / "refraction_metadata.json").unlink(missing_ok=True)
    with open(output_dir / "refraction_metadata.json", "w") as f:
        json.dump(refr_meta, f, indent=2)

    orig_path = Path(args.vggt_dir) / "metadata.json"
    if orig_path.exists():
        with open(orig_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["refraction_corrected"] = True
    meta["refraction_mode"] = args.mode
    meta["refraction_n_water"] = args.n_water
    (output_dir / "metadata.json").unlink(missing_ok=True)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()
    vggt_dir = Path(args.vggt_dir)

    print("=" * 70)
    print(f"REFRACTION CORRECTION  (mode={args.mode}, n_water={args.n_water})")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    if (
        vggt_dir.is_dir()
        and output_dir.is_dir()
        and os.path.samefile(output_dir, vggt_dir)
    ):
        raise ValueError("--output_dir must differ from --vggt_dir")
    meta_path = vggt_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            if json.load(f).get("refraction_corrected"):
                raise ValueError(f"{vggt_dir} is already refraction-corrected")

    view_indices = [v["index"] for v in discover_vggt_views(vggt_dir)]
    n_views = len(view_indices)
    if n_views == 0:
        raise FileNotFoundError(f"No world_points_NNN.npy files in {vggt_dir}")
    print(f"\nFound {n_views} views in {vggt_dir}")

    H, W = np.load(vggt_dir / f"world_points_{view_indices[0]:03d}.npy").shape[:2]
    print(f"Resolution: {W} x {H}")

    masks = load_all_masks(args.mask_dir, n_views, H, W, args.mask_threshold)

    print("  Loading per-view data...")
    views_data = []
    camera_positions = []
    for i, idx in enumerate(view_indices):
        C = camera_centre(np.load(vggt_dir / f"extrinsic_{idx:03d}.npy"))
        camera_positions.append(C)
        views_data.append(
            {
                "index": idx,
                "world_points": np.load(vggt_dir / f"world_points_{idx:03d}.npy"),
                "water_mask": masks[i],
                "C": C,
            }
        )
    camera_positions = np.array(camera_positions)
    print(f"  Camera span: {np.ptp(camera_positions, axis=0)}")

    setup_output_dir(args.output_dir, args.vggt_dir, view_indices)

    if args.mode == "single_view":
        print(f"\n--- Single-view mode: per-view surface estimation ---")
        per_view_planes = []
        all_stats = []

        for i, vd in enumerate(views_data):
            shore_pts, _ = extract_shoreline_points(
                vd["world_points"], vd["water_mask"], args.shoreline_buffer
            )

            if len(shore_pts) < 20:
                print(
                    f"  View {vd['index']}: only {len(shore_pts)} shoreline "
                    f"pts -- skipping correction"
                )
                per_view_planes.append(None)
                n_water = int(
                    (vd["water_mask"] & np.isfinite(vd["world_points"]).all(axis=2)).sum()
                )
                all_stats.append(
                    {
                        "n_water": n_water,
                        "n_corrected": 0,
                        "n_skipped": n_water,
                        "skipped_reason": "too few shoreline pts",
                    }
                )
                continue

            normal, d_p, inlier_mask, res_std, _ = fit_plane_ransac(
                shore_pts, args.ransac_iterations, args.ransac_threshold
            )
            normal, d_p = orient_normal_toward_cameras(normal, d_p, camera_positions)
            per_view_planes.append((normal, d_p))
            all_stats.append(None)  # filled in by the correction below

            n_inl = int(inlier_mask.sum())
            if i < 5 or i == n_views - 1:
                print(
                    f"  View {vd['index']}: {len(shore_pts)} shore pts, "
                    f"{n_inl} inliers, res_std={res_std:.6f}"
                )

        for i, vd in enumerate(views_data):
            out_path = Path(args.output_dir) / f"world_points_{vd['index']:03d}.npy"
            out_path.unlink(missing_ok=True)
            if per_view_planes[i] is None:
                np.save(out_path, vd["world_points"])
                continue
            normal, d_p = per_view_planes[i]
            corrected, stats = snell_correct_view(
                vd["world_points"],
                vd["water_mask"],
                vd["C"],
                normal,
                d_p,
                args.n_water,
            )
            np.save(out_path, corrected)
            stats["plane_normal"] = normal.tolist()
            stats["plane_d"] = float(d_p)
            all_stats[i] = stats
            if i < 5 or i == n_views - 1:
                print(
                    f"  View {vd['index']}: corrected "
                    f"{stats['n_corrected']}/{stats['n_water']} water px"
                )

        save_metadata(
            args.output_dir,
            args,
            None,
            None,
            {"method": "single_view_ransac"},
            all_stats,
        )

    else:
        print(f"\n--- Collecting shoreline points from all {n_views} views ---")
        all_shore = []
        for i, vd in enumerate(views_data):
            pts, _ = extract_shoreline_points(
                vd["world_points"], vd["water_mask"], args.shoreline_buffer
            )
            all_shore.append(pts)
            if i < 3 or i == n_views - 1:
                print(f"  View {vd['index']}: {len(pts):,} shoreline points")

        all_shore = np.concatenate(all_shore, axis=0)
        print(f"  Total shoreline points: {len(all_shore):,}")

        normal, d_plane, inlier_mask, res_std, threshold = fit_plane_ransac(
            all_shore, args.ransac_iterations, args.ransac_threshold
        )
        normal, d_plane = orient_normal_toward_cameras(
            normal, d_plane, camera_positions
        )
        n_inl = int(inlier_mask.sum())
        print(
            f"  RANSAC plane: {n_inl}/{len(all_shore)} inliers "
            f"({100 * n_inl / len(all_shore):.1f}%)"
        )
        print(f"  Normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        print(f"  d = {d_plane:.6f}")
        print(f"  Residual std: {res_std:.6f}")

        plane_stats = {
            "method": f"{'optimized' if args.mode == 'optimized' else 'multi_view'}_ransac",
            "n_shoreline_total": len(all_shore),
            "n_inliers": n_inl,
            "residual_std": float(res_std),
            "inlier_threshold": float(threshold),
        }

        if args.mode == "optimized":
            normal, d_plane, opt_result = optimize_water_surface(
                views_data,
                normal,
                d_plane,
                args.n_water,
                args.opt_subsample_views,
                args.opt_subsample_pixels,
                args.opt_grid_res,
            )
            plane_stats["optimized"] = opt_result is not None
            if opt_result is not None:
                plane_stats["opt_dh"] = float(opt_result.x[0])
                plane_stats["opt_dtx_deg"] = float(np.degrees(opt_result.x[1]))
                plane_stats["opt_dty_deg"] = float(np.degrees(opt_result.x[2]))
                plane_stats["opt_final_loss"] = float(opt_result.fun)

        print(f"\n--- Applying Snell's law correction (n_water={args.n_water}) ---")
        all_stats = []
        for i, vd in enumerate(views_data):
            corrected, stats = snell_correct_view(
                vd["world_points"],
                vd["water_mask"],
                vd["C"],
                normal,
                d_plane,
                args.n_water,
            )
            out_path = Path(args.output_dir) / f"world_points_{vd['index']:03d}.npy"
            out_path.unlink(missing_ok=True)
            np.save(out_path, corrected)
            all_stats.append(stats)
            if i < 5 or i == n_views - 1:
                print(
                    f"  View {vd['index']}: "
                    f"{stats['n_corrected']}/{stats['n_water']} corrected  "
                    f"angle={stats.get('mean_incidence_deg', 0):.1f} deg"
                )
            elif i == 5:
                print(f"  ... (showing first 5 and last)")

        save_metadata(args.output_dir, args, normal, d_plane, plane_stats, all_stats)

    elapsed = time.time() - t0
    total_water = sum(s.get("n_water", 0) for s in all_stats)
    total_corr = sum(s.get("n_corrected", 0) for s in all_stats)

    print(f"\n{'=' * 70}")
    print("REFRACTION CORRECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Refractive index: {args.n_water}")
    print(f"  Total water pixels: {total_water:,}")
    print(f"  Total corrected:    {total_corr:,}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
