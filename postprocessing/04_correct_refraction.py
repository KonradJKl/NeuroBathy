"""
Snell's law refraction correction of the VGGT water points.

The water surface is a plane fitted with RANSAC to the shoreline, the world points of
the land pixels in a ring around the water mask. The ray from the camera to the
apparent point of each water pixel is refracted where it crosses the plane, and the
point is moved along the refracted ray to n_water times its apparent depth below the
plane, the apparent-depth relation of a vertical view. Apparent points above the plane
are left uncorrected.

Modes:
    single_view  one plane per view, from that view's shoreline
    multi_view   one plane for all views, from the shoreline of all views
    optimized    the multi_view plane, refined in height and tilt (Nelder-Mead) so
                 that the corrected depths of the views agree on a grid

The output folder holds the corrected world_points_NNN.npy, links to the unchanged
per-view files, metadata.json and refraction_metadata.json, so that
02_extract_bathymetry.py reads it like any VGGT folder.

Usage example:
    # VGGT nadir, one plane for all views
    python 04_correct_refraction.py \\
        --vggt_dir /path/to/vggt_nadir \\
        --mask_dir /path/to/pielach_data/masks \\
        --mode multi_view \\
        --output_dir refraction/vggt_nadir_multi_view

    # then the bathymetry of the corrected folder
    python 02_extract_bathymetry.py \\
        --mode vggt \\
        --vggt_dir refraction/vggt_nadir_multi_view \\
        --umeyama_npz ../data/transforms/vggt_nadir_umeyama.npz \\
        --mask_dir /path/to/pielach_data/masks \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --output_csv bathymetry/bathymetry_vggt_nadir_snell_mv.csv
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize

from coord_utils import WATER_MASK_THRESHOLD, discover_vggt_views

N_WATER = 1.333
VGGT_PATCH_PX = 14
MIN_SHORELINE_POINTS = 20
# views with fewer rays that cross the plane are left out of the optimized-mode loss
MIN_LOSS_POINTS = 10
# per-view files the correction leaves unchanged; the output links them
LINKED_FILES = [
    "depth_{:03d}.npy",
    "depth_conf_{:03d}.npy",
    "extrinsic_{:03d}.npy",
    "intrinsic_{:03d}.npy",
    "rgb_{:03d}.png",
    "world_points_conf_{:03d}.npy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Snell's law refraction correction of the VGGT water points."
    )

    # inputs and output
    parser.add_argument(
        "--vggt_dir",
        type=str,
        required=True,
        help="output folder of run_vggt_inference.py",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="water mask PNGs, one per view, in view order when sorted by name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="new folder for the corrected views, the --vggt_dir of 02",
    )

    # correction
    parser.add_argument(
        "--mode",
        choices=["single_view", "multi_view", "optimized"],
        default="multi_view",
        help="single_view: one plane per view; multi_view (default): one plane for "
        "all views; optimized: the multi_view plane, refined for cross-view "
        "consistency",
    )
    parser.add_argument(
        "--n_water",
        type=float,
        default=N_WATER,
        help=f"refractive index of the water, at least 1 (default: {N_WATER})",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=WATER_MASK_THRESHOLD,
        help=f"mask values above this are water (default: {WATER_MASK_THRESHOLD})",
    )
    parser.add_argument(
        "--shoreline_buffer",
        type=int,
        default=1,
        help="width of the shoreline ring around the water mask, in px (default: 1)",
    )
    parser.add_argument(
        "--ransac_iterations",
        type=int,
        default=2000,
        help="RANSAC iterations of the plane fit (default: 2000)",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=None,
        help="RANSAC inlier distance in VGGT units (default: the larger of twice the "
        "median and the 75th percentile of the distances to a least-squares plane)",
    )

    # optimized mode
    parser.add_argument(
        "--opt_subsample_views",
        type=int,
        default=0,
        help="use every (number of views // N)-th view, so at least N views "
        "(default: 0, all views)",
    )
    parser.add_argument(
        "--opt_subsample_pixels",
        type=int,
        default=4,
        help="use every Nth water pixel of a view (default: 4)",
    )
    parser.add_argument(
        "--opt_grid_res",
        type=float,
        default=0.01,
        help="cell size of the grid that pairs the views, in VGGT units "
        "(default: 0.01)",
    )
    args = parser.parse_args()
    if args.n_water < 1:
        parser.error("--n_water must be at least 1")
    if args.ransac_threshold is not None and args.ransac_threshold <= 0:
        parser.error("--ransac_threshold must be positive")
    if args.opt_grid_res <= 0:
        parser.error("--opt_grid_res must be positive")
    return args


def load_views(vggt_dir, indices):
    views = []
    for idx in indices:
        extrinsic = np.load(vggt_dir / f"extrinsic_{idx:03d}.npy")
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        views.append(
            {
                "index": idx,
                "world_points": np.load(vggt_dir / f"world_points_{idx:03d}.npy"),
                "C": -R.T @ t,  # camera centre, extrinsics are camera-from-world
            }
        )
    return views


def load_mask(path, H, W, threshold):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERROR: Cannot read mask {path}")
        sys.exit(1)
    # VGGT scales the width to its frame and rounds the height to its patches; a
    # larger mismatch means it cropped or padded the image, and a resize misaligns
    if abs(mask.shape[0] * W / mask.shape[1] - H) > VGGT_PATCH_PX / 2:
        print(
            f"ERROR: Mask {path.name} ({mask.shape[1]} x {mask.shape[0]}) does not "
            f"have the aspect ratio of the VGGT frame ({W} x {H})"
        )
        sys.exit(1)
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask > threshold


def load_masks(mask_dir, n_views, H, W, threshold):
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)
    paths = sorted(Path(mask_dir).glob("*.png"))
    if len(paths) != n_views:
        print(f"ERROR: {len(paths)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    print(f"Masks: {n_views} from {mask_dir}")
    return [load_mask(p, H, W, threshold) for p in paths]


def water_pixels(view):
    return np.where(view["water_mask"] & np.isfinite(view["world_points"]).all(axis=2))


def shoreline_points(view, buffer_px):
    size = 2 * buffer_px + 1
    water = view["water_mask"]
    ring = binary_dilation(water, structure=np.ones((size, size), dtype=bool)) & ~water
    points = view["world_points"]
    return points[ring & np.isfinite(points).all(axis=2)]


def svd_plane(points):
    centroid = points.mean(axis=0)
    _, _, Vh = np.linalg.svd(points - centroid, full_matrices=False)
    return Vh[-1], -Vh[-1] @ centroid


def fit_plane_ransac(points, n_iter, threshold=None, seed=42):
    """RANSAC plane normal . P + d = 0, refitted by SVD on its inliers.

    Returns the plane, the number of inliers, the std of their residuals and the
    inlier threshold used.
    """
    rng = np.random.default_rng(seed)
    if threshold is None:
        normal, d = svd_plane(points)
        dists = np.abs(points @ normal + d)
        threshold = max(2.0 * np.median(dists), np.percentile(dists, 75), 1e-8)

    best_count, best = 0, None
    for _ in range(n_iter):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(normal)
        if length < 1e-12:  # collinear sample
            continue
        normal /= length
        d = -normal @ p0
        count = np.sum(np.abs(points @ normal + d) < threshold)
        if count > best_count:
            best_count, best = count, (normal, d)
    normal, d = best if best is not None else svd_plane(points)

    close = np.abs(points @ normal + d) < threshold
    if close.sum() >= 3:
        normal, d = svd_plane(points[close])
    residuals = points @ normal + d
    inliers = np.abs(residuals) < threshold
    res_std = float(np.std(residuals[inliers])) if inliers.any() else float("inf")
    return normal, d, int(inliers.sum()), res_std, threshold


def orient_toward_cameras(normal, d, camera_positions):
    """Flip the plane so that the camera centroid, the air side, is in front of it."""
    if normal @ camera_positions.mean(axis=0) + d < 0:
        return -normal, -d
    return normal, d


def trace_to_plane(C, rays, normal, d):
    """Where the rays from C cross the plane; valid marks those that cross it ahead."""
    n_dot_ray = rays @ normal
    valid = np.abs(n_dot_ray) > 1e-12
    t_surface = np.full(len(rays), np.inf)
    t_surface[valid] = -(normal @ C + d) / n_dot_ray[valid]
    valid &= t_surface > 0
    return C + t_surface[:, None] * rays, valid


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def snell_correct(view, normal, d, n_water):
    """Correct the water points of a view at the plane normal . P + d = 0.

    The normal points to the air side. Returns the corrected world points and the
    stats of the view.
    """
    corrected = view["world_points"].copy()
    wy, wx = water_pixels(view)
    N = len(wy)
    if N == 0:
        return corrected, {"n_water": 0, "n_corrected": 0, "n_skipped": 0}

    C = view["C"]
    P_app = view["world_points"][wy, wx].astype(np.float64)
    rays = P_app - C
    P_surf, valid = trace_to_plane(C, rays, normal, d)

    incident = unit(rays)
    signed_cos_i = incident @ normal
    cos_i = np.clip(np.abs(signed_cos_i), 0.0, 1.0)
    sin_r = np.sqrt(1.0 - cos_i**2) / n_water
    cos_r = np.sqrt(np.maximum(1.0 - sin_r**2, 0.0))
    valid &= (sin_r <= 1.0) & (cos_r > 1e-12)

    # vector form of Snell's law, with the normal on the side of the incoming ray
    eta = 1.0 / n_water
    n_hat = -np.sign(signed_cos_i)[:, None] * normal
    refracted = eta * incident + (eta * cos_i - cos_r)[:, None] * n_hat
    length = np.linalg.norm(refracted, axis=1, keepdims=True)
    valid &= length[:, 0] > 1e-12
    refracted /= np.maximum(length, 1e-12)

    # n_water times the apparent depth below the plane: exact for a vertical ray,
    # an underestimate for oblique ones
    depth_app = -(P_app @ normal + d)
    valid &= depth_app > 1e-12
    t_true = np.zeros(N)
    t_true[valid] = depth_app[valid] * n_water / cos_r[valid]
    P_true = P_surf + t_true[:, None] * refracted
    corrected[wy[valid], wx[valid]] = P_true[valid]

    if not valid.any():
        return corrected, {"n_water": N, "n_corrected": 0, "n_skipped": N}
    incidence_deg = np.degrees(np.arccos(cos_i[valid]))
    displacement = np.linalg.norm(P_true[valid] - P_app[valid], axis=1)
    return corrected, {
        "n_water": N,
        "n_corrected": int(valid.sum()),
        "n_skipped": int((~valid).sum()),
        "n_above_plane": int((depth_app < 0).sum()),
        "mean_incidence_deg": round(float(incidence_deg.mean()), 2),
        "max_incidence_deg": round(float(incidence_deg.max()), 2),
        "mean_displacement": round(float(displacement.mean()), 6),
    }


def tilted_plane(normal, d, params):
    """The plane raised by dh along its normal, which is turned by dtx about x and
    then by dty about y (radians)."""
    dh, dtx, dty = params
    cx, sx = np.cos(dtx), np.sin(dtx)
    cy, sy = np.cos(dty), np.sin(dty)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    tilted = Ry @ Rx @ normal
    return tilted / np.linalg.norm(tilted), d - dh


def optimization_samples(views, n_sample_views, pixel_step):
    """Every pixel_step-th water point of the views the optimisation uses."""
    view_ids = range(len(views))
    if 0 < n_sample_views < len(views):
        view_ids = range(0, len(views), len(views) // n_sample_views)
    samples = []
    for vi in view_ids:
        wy, wx = water_pixels(views[vi])
        if len(wy) == 0:
            continue
        if pixel_step > 1:
            wy, wx = wy[::pixel_step], wx[::pixel_step]
        P_app = views[vi]["world_points"][wy, wx].astype(np.float64)
        C = views[vi]["C"]
        samples.append({"C": C, "rays": P_app - C, "P_app": P_app, "view": vi})
    return samples


def cross_view_variance(params, samples, normal, d, n_water, grid_res):
    """Loss of the optimized mode: in each grid cell seen from two or more views, the
    variance of the views' mean corrected depth, averaged over these cells; inf when
    no cell is seen from two views."""
    normal, d = tilted_plane(normal, d, params)
    cells = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        P_surf, valid = trace_to_plane(sample["C"], sample["rays"], normal, d)
        if valid.sum() < MIN_LOSS_POINTS:
            continue
        cos_i = np.clip(np.abs(unit(sample["rays"]) @ normal), 0, 1)
        sin_r = np.sqrt(1 - cos_i**2) / n_water
        cos_r = np.sqrt(np.maximum(1 - sin_r**2, 0))
        depth_app = -(sample["P_app"] @ normal + d)
        # unlike the correction, the loss keeps the points above the plane: without
        # them the plane could lower the loss by moving away from all points
        valid &= (sin_r <= 1.0) & (np.abs(depth_app) > 1e-12) & (cos_r > 1e-12)
        keys = np.floor(P_surf[valid, :2] / grid_res).astype(np.int64)
        depths = np.abs(depth_app[valid]) * n_water
        for key, depth in zip(map(tuple, keys.tolist()), depths.tolist()):
            cells[key][sample["view"]].append(depth)

    variances = [
        np.var([np.mean(view_depths) for view_depths in per_view.values()])
        for per_view in cells.values()
        if len(per_view) > 1
    ]
    if not variances:
        return np.inf
    return sum(variances) / len(variances)


def optimize_plane(views, normal, d, args):
    """Refine the plane by Nelder-Mead on cross_view_variance; the plane stays as it
    is when no grid cell is seen from two views. Returns the plane and its stats."""
    samples = optimization_samples(
        views, args.opt_subsample_views, args.opt_subsample_pixels
    )
    n_points = sum(len(s["P_app"]) for s in samples)
    print(f"Optimising the plane on {len(samples)} view(s), {n_points:,} points")
    loss_args = (samples, normal, d, args.n_water, args.opt_grid_res)
    initial_loss = cross_view_variance([0, 0, 0], *loss_args)
    if not np.isfinite(initial_loss):
        print("No grid cell is seen from two views, keeping the RANSAC plane")
        return normal, d, {"optimized": False}
    result = minimize(
        cross_view_variance,
        x0=[0.0, 0.0, 0.0],
        args=loss_args,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-10},
    )
    dh, dtx, dty = result.x
    print(
        f"  plane raised by {dh:+.6f}, tilted {np.degrees(dtx):+.4f} deg about x and "
        f"{np.degrees(dty):+.4f} deg about y; variance "
        f"{initial_loss:.3e} -> {result.fun:.3e}"
    )
    stats = {
        "optimized": True,
        "opt_dh": float(dh),
        "opt_dtx_deg": float(np.degrees(dtx)),
        "opt_dty_deg": float(np.degrees(dty)),
        "opt_final_loss": float(result.fun),
    }
    return (*tilted_plane(normal, d, result.x), stats)


def new_file(path):
    """path, unlinked first so that writing it never follows a link into the input."""
    path.unlink(missing_ok=True)
    return path


def save_world_points(output_dir, view, points):
    np.save(new_file(output_dir / f"world_points_{view['index']:03d}.npy"), points)


def print_view(i, n_views, view, stats):
    if i < 5 or i == n_views - 1:
        line = (
            f"  View {view['index']}: corrected {stats['n_corrected']:,} of "
            f"{stats['n_water']:,} water px"
        )
        if "mean_incidence_deg" in stats:
            line += f", mean incidence {stats['mean_incidence_deg']:.1f} deg"
        print(line)
    elif i == 5:
        print("  ...")


def correct_single_view(views, output_dir, camera_positions, args):
    planes = []
    for view in views:
        shore = shoreline_points(view, args.shoreline_buffer)
        if len(shore) < MIN_SHORELINE_POINTS:
            print(f"  View {view['index']}: {len(shore)} shoreline points, uncorrected")
            planes.append(None)
            continue
        normal, d, _, _, _ = fit_plane_ransac(
            shore, args.ransac_iterations, args.ransac_threshold
        )
        planes.append(orient_toward_cameras(normal, d, camera_positions))

    view_stats = []
    for i, (view, plane) in enumerate(zip(views, planes)):
        if plane is None:
            n_water_px = len(water_pixels(view)[0])
            save_world_points(output_dir, view, view["world_points"])
            view_stats.append(
                {
                    "n_water": n_water_px,
                    "n_corrected": 0,
                    "n_skipped": n_water_px,
                    "skipped_reason": "too few shoreline pts",
                }
            )
            continue
        normal, d = plane
        corrected, stats = snell_correct(view, normal, d, args.n_water)
        save_world_points(output_dir, view, corrected)
        stats["plane_normal"] = normal.tolist()
        stats["plane_d"] = float(d)
        view_stats.append(stats)
        print_view(i, len(views), view, stats)
    # each view's plane is in its stats, so the metadata has no single plane
    return None, None, {"method": "single_view_ransac"}, view_stats


def correct_shared_plane(views, output_dir, camera_positions, args):
    shore = np.concatenate([shoreline_points(v, args.shoreline_buffer) for v in views])
    if len(shore) < 3:
        print(f"ERROR: {len(shore)} shoreline points, the plane fit needs 3")
        sys.exit(1)
    normal, d, n_inliers, res_std, threshold = fit_plane_ransac(
        shore, args.ransac_iterations, args.ransac_threshold
    )
    normal, d = orient_toward_cameras(normal, d, camera_positions)
    print(
        f"Plane: {n_inliers:,} of {len(shore):,} shoreline points are inliers, "
        f"residual std {res_std:.6f}"
    )
    plane_stats = {
        "method": f"{args.mode}_ransac",
        "n_shoreline_total": len(shore),
        "n_inliers": n_inliers,
        "residual_std": res_std,
        "inlier_threshold": float(threshold),
    }
    if args.mode == "optimized":
        normal, d, opt_stats = optimize_plane(views, normal, d, args)
        plane_stats.update(opt_stats)

    view_stats = []
    for i, view in enumerate(views):
        corrected, stats = snell_correct(view, normal, d, args.n_water)
        save_world_points(output_dir, view, corrected)
        view_stats.append(stats)
        print_view(i, len(views), view, stats)
    return normal, d, plane_stats, view_stats


def setup_output_dir(output_dir, vggt_dir, indices):
    """Create output_dir with links to the unchanged per-view files of vggt_dir."""
    # 02 would read the views of an earlier, larger run along with this one
    stale = sorted(
        f.name
        for ext in ("npy", "png")
        for f in output_dir.glob(f"*_[0-9][0-9][0-9].{ext}")
        if int(f.stem[-3:]) not in indices
    )
    if stale:
        print(
            f"ERROR: {output_dir} holds views of an earlier run ({stale[0]}), "
            "use a new folder"
        )
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        for pattern in LINKED_FILES:
            name = pattern.format(idx)
            src = vggt_dir / name
            # a link or copy left by an earlier run may belong to another input
            dst = new_file(output_dir / name)
            if src.exists():
                dst.symlink_to(src.resolve())


def read_vggt_metadata(vggt_dir):
    if not (vggt_dir / "metadata.json").exists():
        return {}
    with open(vggt_dir / "metadata.json") as f:
        return json.load(f)


def save_metadata(output_dir, args, normal, d, plane_stats, view_stats):
    refraction_metadata = {
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
        "source_vggt_dir": args.vggt_dir,
        "mask_dir": args.mask_dir,
        "plane_normal": None if normal is None else normal.tolist(),
        "plane_d": None if d is None else float(d),
        "plane_stats": plane_stats,
        "per_view_stats": view_stats,
    }
    with open(new_file(output_dir / "refraction_metadata.json"), "w") as f:
        json.dump(refraction_metadata, f, indent=2)

    # VGGT's own metadata, marked as corrected
    vggt_metadata = read_vggt_metadata(Path(args.vggt_dir))
    vggt_metadata["refraction_corrected"] = True
    vggt_metadata["refraction_mode"] = args.mode
    vggt_metadata["refraction_n_water"] = args.n_water
    with open(new_file(output_dir / "metadata.json"), "w") as f:
        json.dump(vggt_metadata, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()
    vggt_dir, output_dir = Path(args.vggt_dir), Path(args.output_dir)
    print(f"Refraction correction ({args.mode}, n_water {args.n_water})")
    if (
        vggt_dir.is_dir()
        and output_dir.is_dir()
        and os.path.samefile(output_dir, vggt_dir)
    ):
        print("ERROR: --output_dir must differ from --vggt_dir")
        sys.exit(1)
    if read_vggt_metadata(vggt_dir).get("refraction_corrected"):
        print(f"ERROR: {vggt_dir} is already refraction-corrected")
        sys.exit(1)

    indices = [v["index"] for v in discover_vggt_views(vggt_dir)]
    if not indices:
        print(f"ERROR: No world_points_NNN.npy files in {vggt_dir}")
        sys.exit(1)
    views = load_views(vggt_dir, indices)
    H, W = views[0]["world_points"].shape[:2]
    print(f"VGGT: {len(views)} view(s) at {W} x {H} px in {vggt_dir}")
    masks = load_masks(args.mask_dir, len(views), H, W, args.mask_threshold)
    for view, mask in zip(views, masks):
        view["water_mask"] = mask
    camera_positions = np.array([v["C"] for v in views])

    setup_output_dir(output_dir, vggt_dir, indices)
    if args.mode == "single_view":
        normal, d, plane_stats, view_stats = correct_single_view(
            views, output_dir, camera_positions, args
        )
    else:
        normal, d, plane_stats, view_stats = correct_shared_plane(
            views, output_dir, camera_positions, args
        )
    save_metadata(output_dir, args, normal, d, plane_stats, view_stats)

    n_water_px = sum(s["n_water"] for s in view_stats)
    n_corrected = sum(s["n_corrected"] for s in view_stats)
    print(
        f"Corrected {n_corrected:,} of {n_water_px:,} water pixels in "
        f"{time.time() - t0:.1f} s, outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
