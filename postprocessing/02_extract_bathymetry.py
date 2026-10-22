"""
Extract bathymetric depth from NeRFrac or VGGT output and grid it.

Writes the grid to --output_csv and every point to <csv stem>_raw.npz.

Usage examples:
    # NeRFrac 50m
    python 02_extract_bathymetry.py \\
        --mode nerfrac \\
        --testset_dir /path/to/experiment_log/renderonly_all_950000 \\
        --log_folder /path/to/experiment_log \\
        --data_dir /path/to/pielach_50m \\
        --umeyama_npz ../data/transforms/nerfrac_50m_umeyama.npz \\
        --mask_dir /path/to/pielach_50m/masks_8 \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --output_csv bathymetry/bathymetry_nerfrac_50m.csv

    # VGGT 50m
    python 02_extract_bathymetry.py \\
        --mode vggt \\
        --vggt_dir /path/to/vggt_output \\
        --umeyama_npz ../data/transforms/vggt_50m_v1_umeyama.npz \\
        --mask_dir /path/to/pielach_50m/masks_8 \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --output_csv bathymetry/bathymetry_vggt_50m.csv
"""

import argparse
import csv
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import (
    NEAR,
    WATER_MASK_THRESHOLD,
    apply_umeyama,
    discover_nerfrac_views,
    discover_vggt_views,
    load_nerfrac_coord_metadata,
    load_umeyama,
    ndc_to_colmap,
    sample_wsm_batch,
    write_ply_binary,
)

CSV_FIELDS = [
    "utm_easting",
    "utm_northing",
    "utm_elevation",
    "depth_m",
    "depth_std",
    "wsm_elevation",
    "n_observations",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bathymetric depth from NeRFrac or VGGT output."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["nerfrac", "vggt"],
        help="which model's output to read",
    )

    # NeRFrac input
    parser.add_argument(
        "--testset_dir",
        type=str,
        default="",
        help="[nerfrac] render folder with Xs_pts, rays and depth .npy files, "
        "e.g. renderonly_all_950000",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="[nerfrac] experiment folder with c2w.npy and llff_scale.npy",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="[nerfrac] scene folder; its poses_bounds.npy gives the focal length",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=0,
        help="[nerfrac] render height in px (default: from the depth maps)",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=0,
        help="[nerfrac] render width in px (default: from the depth maps)",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=0.0,
        help="[nerfrac] focal length in px at render resolution (default: from "
        "--data_dir)",
    )
    parser.add_argument(
        "--depth_type",
        type=str,
        default="mode",
        choices=["mean", "mode"],
        help="[nerfrac] 'mode' (default, used for the headline results): depth of "
        "the sample with the highest weight, *_depth_mode.npy; 'mean': expected "
        "depth, *_depth.npy",
    )
    parser.add_argument(
        "--depth_method",
        type=str,
        default="wsm",
        choices=["wsm", "plane_fit"],
        help="[nerfrac] 'wsm': WSM elevation minus point elevation; 'plane_fit': "
        "distance below a plane fitted to the learned water surface, times the "
        "Umeyama scale, so neither the WSM height nor the transform's Z offset "
        "enters the depth",
    )

    # VGGT input
    parser.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="[vggt] output folder of run_vggt_inference.py with world_points_NNN.npy",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="[vggt] minimum depth confidence",
    )
    parser.add_argument(
        "--no_z_auto_correct",
        action="store_true",
        help="[vggt] keep the heights instead of shifting them by the median offset "
        "between the highest water points and the WSM",
    )
    parser.add_argument(
        "--z_calib_views",
        type=int,
        default=20,
        help="[vggt] number of evenly spaced views used for the height shift",
    )
    parser.add_argument(
        "--z_surface_percentile",
        type=float,
        default=90.0,
        help="[vggt] water points above this height percentile stand in for the "
        "water surface (default 90: the highest 10%%)",
    )

    # georeferencing and filtering
    parser.add_argument(
        "--umeyama_npz",
        type=str,
        required=True,
        help="transform from 01_compute_umeyama.py",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="water masks, paired with the views by sorted file name (without "
        "masks, every pixel counts as water)",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.0,
        help="points shallower than this are dropped, in m (the default 0 drops "
        "points above the water surface)",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=5.0,
        help="points deeper than this are dropped, in m. Raise it, e.g. to 50, for "
        "models that overpredict depth and need 11_scale_correction.py",
    )
    parser.add_argument(
        "--grid_resolution",
        type=float,
        default=0.1,
        help="grid cell size in m",
    )

    # output
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="gridded bathymetry, one row per cell",
    )
    parser.add_argument(
        "--output_ply",
        action="store_true",
        help="also write every point to <csv stem>_combined.ply, colored by depth",
    )
    parser.add_argument(
        "--per_view_png",
        action="store_true",
        help="also write depth maps to <csv stem>_views/, every 10th view when "
        "there are more than 20",
    )
    return parser.parse_args()


def list_masks(mask_dir, n_views):
    """Mask file of each view, or None for every view without --mask_dir."""
    if not mask_dir:
        return [None] * n_views
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)
    names = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if len(names) != n_views:
        print(f"ERROR: {len(names)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    return [os.path.join(mask_dir, name) for name in names]


def load_mask(path, H, W):
    if path is None:
        return None
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
    return mask > WATER_MASK_THRESHOLD


def water_pixels(mask, n_pixels):
    return np.arange(n_pixels) if mask is None else np.flatnonzero(mask)


def depth_below_wsm(P_utm, wsm_path, depth_min, depth_max):
    wsm_elevs = sample_wsm_batch(wsm_path, P_utm[:, 0], P_utm[:, 1])
    depths = wsm_elevs - P_utm[:, 2]
    valid = (
        np.isfinite(wsm_elevs)
        & np.isfinite(depths)
        & (depths >= depth_min)
        & (depths <= depth_max)
    )
    return depths, wsm_elevs, valid


def view_result(pixels, P_utm, depths, wsm_elevs, shape, with_map):
    result = {"utm_points": P_utm, "depths": depths, "wsm_elevs": wsm_elevs}
    if with_map:
        depth_map = np.full(shape, np.nan, dtype=np.float32)
        depth_map.flat[pixels] = depths
        result["depth_map"] = depth_map
    return result


def print_view(vi, n_views, result):
    if result is None:
        if vi % 50 == 0:
            print(f"  View {vi:3d}: no valid points")
    elif vi % 50 == 0 or vi == n_views - 1:
        print(
            f"  View {vi:3d}: {len(result['depths']):9,} points, "
            f"median depth {np.median(result['depths']):.3f} m"
        )


def read_nerfrac_view(view, H, W):
    """Xs, refracted ray and depth of every pixel, or None if a file is missing."""
    if view["xs"] is None or view["rays"] is None or view["depth"] is None:
        return None
    xs = np.load(view["xs"]).reshape(-1, 3)
    if len(xs) != H * W:
        print(f"  View {view['index']}: {len(xs):,} Xs points for {H}x{W} px, skipped")
        return None
    rays = np.load(view["rays"]).reshape(-1, 3)
    depth = np.load(view["depth"]).reshape(-1, 1)
    return xs, rays, depth


def underwater_points(arrays, pixels, ndc):
    """P = Xs + depth * ray of the given pixels, in COLMAP coordinates."""
    xs, rays, depth = arrays
    return ndc_to_colmap(xs[pixels] + depth[pixels] * rays[pixels], **ndc)


def render_resolution(args, views):
    """H, W and focal length of the renders.

    Values not given as arguments come from the first depth map and poses_bounds.npy.
    """
    H, W, focal = args.H, args.W, args.focal
    if H and W and focal:
        return H, W, focal
    if views[0]["depth"] is None:
        print(f"ERROR: No {args.depth_type} depth map for view 0 in {args.testset_dir}")
        sys.exit(1)
    H_map, W_map = np.load(views[0]["depth"]).shape
    H, W = H or H_map, W or W_map
    if not focal:
        pb_path = os.path.join(args.data_dir, "poses_bounds.npy")
        if not args.data_dir or not os.path.exists(pb_path):
            print("ERROR: Cannot read the focal length. Provide --focal or --data_dir")
            sys.exit(1)
        poses = np.load(pb_path)[:, :15].reshape(-1, 3, 5)
        H_full, W_full, focal_full = poses[0, :, 4]
        focal = focal_full * W / W_full
    return H, W, focal


def fit_water_plane(views, mask_paths, ndc, s, R, t):
    """Plane through the learned water surface points of all views.

    Returns the unit normal and the offset in COLMAP coordinates, or None without
    water surface points.
    """
    H, W = ndc["H"], ndc["W"]
    surface = []
    for view, mask_path in zip(views, mask_paths):
        if view["xs"] is None:
            continue
        xs = np.load(view["xs"]).reshape(-1, 3)
        if len(xs) != H * W:
            continue
        pixels = water_pixels(load_mask(mask_path, H, W), H * W)
        if len(pixels) > 0:
            surface.append(ndc_to_colmap(xs[pixels], **ndc))
    if not surface:
        return None
    surface = np.concatenate(surface)

    centroid = surface.mean(axis=0)
    _, _, Vt = np.linalg.svd(surface - centroid, full_matrices=False)
    normal = Vt[-1]  # direction of least variance
    offset = -np.dot(normal, centroid)

    residuals = np.dot(surface, normal) + offset
    centroid_utm = apply_umeyama(centroid, s, R, t)
    print(f"Plane through {len(surface):,} water surface points")
    print(f"  normal (COLMAP) [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(
        f"  centroid (UTM) E {centroid_utm[0]:.2f}, N {centroid_utm[1]:.2f}, "
        f"Z {centroid_utm[2]:.2f}"
    )
    print(
        f"  residuals std {residuals.std() * s:.4f} m, "
        f"max {np.abs(residuals).max() * s:.4f} m"
    )
    return normal, offset


def orient_plane(views, mask_paths, ndc, normal, offset):
    """Orient the plane normal so that depths below the surface are positive.

    Decided by up to 5000 underwater points of the first usable view.
    """
    H, W = ndc["H"], ndc["W"]
    for vi, view in enumerate(views):
        arrays = read_nerfrac_view(view, H, W)
        if arrays is None:
            continue
        pixels = water_pixels(load_mask(mask_paths[vi], H, W), H * W)
        if len(pixels) == 0:
            continue
        if len(pixels) > 5000:
            pixels = np.random.default_rng(42).choice(pixels, 5000, replace=False)
        distances = np.dot(underwater_points(arrays, pixels, ndc), normal) + offset
        if np.median(distances) > 0:
            print(f"  normal flipped (median distance in view {vi} is positive)")
            return -normal, -offset
        return normal, offset
    print("  no usable view to orient the normal, depths may have the wrong sign")
    return normal, offset


def nerfrac_view(view, mask_path, ndc, plane, s, R, t, args):
    H, W = ndc["H"], ndc["W"]
    arrays = read_nerfrac_view(view, H, W)
    if arrays is None:
        return None
    pixels = water_pixels(load_mask(mask_path, H, W), H * W)
    if len(pixels) == 0:
        return None
    P_colmap = underwater_points(arrays, pixels, ndc)

    if plane is None:
        P_utm = apply_umeyama(P_colmap, s, R, t)
        depths, wsm_elevs, valid = depth_below_wsm(
            P_utm, args.wsm_path, args.depth_min, args.depth_max
        )
        P_utm, wsm_elevs = P_utm[valid], wsm_elevs[valid]
    else:
        normal, offset = plane
        depths = -(np.dot(P_colmap, normal) + offset) * s
        valid = (
            np.isfinite(depths)
            & (depths >= args.depth_min)
            & (depths <= args.depth_max)
        )
        P_utm = apply_umeyama(P_colmap[valid], s, R, t)
        # the water surface height that the plane implies
        wsm_elevs = P_utm[:, 2] + depths[valid]

    if not valid.any():
        return None
    return view_result(
        pixels[valid], P_utm, depths[valid], wsm_elevs, (H, W), args.per_view_png
    )


def run_nerfrac(args, s, R, t):
    if not args.testset_dir or not args.log_folder:
        print("ERROR: --mode nerfrac needs --testset_dir and --log_folder")
        sys.exit(1)
    meta = load_nerfrac_coord_metadata(args.log_folder, data_dir=args.data_dir or None)
    if meta["c2w_recenter"] is None or meta["sc"] is None:
        print(
            f"ERROR: Could not load c2w.npy and llff_scale.npy from {args.log_folder}"
        )
        sys.exit(1)
    views = discover_nerfrac_views(args.testset_dir, args.depth_type)
    if not views:
        print(f"ERROR: No NeRFrac views found in {args.testset_dir}")
        sys.exit(1)
    mask_paths = list_masks(args.mask_dir, len(views))

    H, W, focal = render_resolution(args, views)
    ndc = {
        "H": H,
        "W": W,
        "focal": focal,
        "c2w_recenter": meta["c2w_recenter"],
        "sc": meta["sc"],
        "near": NEAR,
    }
    print(
        f"{len(views)} views at {W}x{H} px, focal {focal:.4f} px, LLFF scale "
        f"{meta['sc']:.8f}, {args.depth_type} depth, {args.depth_method}"
    )

    plane = None
    if args.depth_method == "plane_fit":
        plane = fit_water_plane(views, mask_paths, ndc, s, R, t)
        if plane is None:
            print("ERROR: No water surface points to fit the plane")
            sys.exit(1)
        plane = orient_plane(views, mask_paths, ndc, *plane)

    results = []
    for vi, view in enumerate(views):
        results.append(nerfrac_view(view, mask_paths[vi], ndc, plane, s, R, t, args))
        print_view(vi, len(views), results[-1])
    return results


def vggt_height_shift(views, mask_paths, s, R, t, wsm_path, n_views, percentile):
    """Median height offset of the highest water points above the WSM.

    Uses n_views evenly spaced views; the points above the percentile stand in for
    the water surface.
    """
    points = []
    for vi in np.linspace(0, len(views) - 1, min(n_views, len(views)), dtype=int):
        world_points = np.load(views[vi]["world_points"])
        pts = world_points.reshape(-1, 3)
        valid = np.all(np.isfinite(pts), axis=1) & np.any(pts != 0.0, axis=1)
        mask = load_mask(mask_paths[vi], *world_points.shape[:2])
        if mask is not None:
            valid &= mask.reshape(-1)
        if valid.any():
            points.append(apply_umeyama(pts[valid], s, R, t))
    if not points:
        print("No water points in the calibration views, heights not shifted")
        return 0.0
    points = np.concatenate(points)

    top = points[points[:, 2] >= np.percentile(points[:, 2], percentile)]
    wsm_elevs = sample_wsm_batch(wsm_path, top[:, 0], top[:, 1])
    on_wsm = np.isfinite(wsm_elevs)
    if on_wsm.sum() < 10:
        print(
            f"Only {on_wsm.sum()} surface points on the WSM, the shift may be "
            "unreliable"
        )
    if not on_wsm.any():
        return 0.0
    offsets = top[on_wsm, 2] - wsm_elevs[on_wsm]
    z_offset = float(np.median(offsets))
    print(
        f"Height shift {-z_offset:+.3f} m (std {np.std(offsets):.3f} m) from "
        f"{on_wsm.sum():,} of the highest {100 - percentile:.0f}% of "
        f"{len(points):,} water points in {min(n_views, len(views))} views"
    )
    return z_offset


def vggt_view(view, mask_path, s, R, t, z_offset, args):
    world_points = np.load(view["world_points"])
    shape = world_points.shape[:2]
    pts = world_points.reshape(-1, 3)

    valid = np.all(np.isfinite(pts), axis=1) & np.any(pts != 0.0, axis=1)
    mask = load_mask(mask_path, *shape)
    if mask is not None:
        valid &= mask.reshape(-1)
    if view["depth_conf"] is not None:
        valid &= np.load(view["depth_conf"]).reshape(-1) >= args.conf_threshold
    pixels = np.flatnonzero(valid)
    if len(pixels) == 0:
        return None

    P_utm = apply_umeyama(pts[pixels], s, R, t)
    P_utm[:, 2] -= z_offset
    depths, wsm_elevs, keep = depth_below_wsm(
        P_utm, args.wsm_path, args.depth_min, args.depth_max
    )
    if not keep.any():
        return None
    return view_result(
        pixels[keep],
        P_utm[keep],
        depths[keep],
        wsm_elevs[keep],
        shape,
        args.per_view_png,
    )


def run_vggt(args, s, R, t):
    if not args.vggt_dir:
        print("ERROR: --mode vggt needs --vggt_dir")
        sys.exit(1)
    views = discover_vggt_views(args.vggt_dir)
    if not views:
        print(f"ERROR: No VGGT views found in {args.vggt_dir}")
        sys.exit(1)
    mask_paths = list_masks(args.mask_dir, len(views))
    print(f"{len(views)} views")

    z_offset = 0.0
    if not args.no_z_auto_correct:
        z_offset = vggt_height_shift(
            views,
            mask_paths,
            s,
            R,
            t,
            args.wsm_path,
            args.z_calib_views,
            args.z_surface_percentile,
        )

    results = []
    for vi, view in enumerate(views):
        results.append(vggt_view(view, mask_paths[vi], s, R, t, z_offset, args))
        print_view(vi, len(views), results[-1])
    return results


def cell_median(values, cell, starts, counts):
    # np.median of every cell at once: sort by cell, then by value, and average the
    # two middle values (the same value twice for an odd count)
    v = values[np.lexsort((values, cell))]
    return (v[starts + (counts - 1) // 2] + v[starts + counts // 2]) / 2


def grid_points(points, depths, wsm_elevs, resolution):
    """Median of each grid cell.

    The grid origin is the smallest easting and northing. Returns the CSV columns,
    with the cells in the order of their first point.
    """
    e_min = points[:, 0].min()
    n_min = points[:, 1].min()
    gi = ((points[:, 0] - e_min) / resolution).astype(int)
    gj = ((points[:, 1] - n_min) / resolution).astype(int)
    cell_ids = gi * (gj.max() + 1) + gj
    _, first, cell, counts = np.unique(
        cell_ids,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    starts = np.cumsum(counts) - counts

    # np.std per cell: a vectorized sum adds in a different order, which changes the
    # last digits
    depth_std = np.zeros(len(counts))
    by_cell = depths[np.argsort(cell, kind="stable")]
    for c in np.flatnonzero(counts > 1):
        depth_std[c] = np.std(by_cell[starts[c] : starts[c] + counts[c]])

    order = np.argsort(first)
    return {
        "utm_easting": e_min + (gi[first[order]] + 0.5) * resolution,
        "utm_northing": n_min + (gj[first[order]] + 0.5) * resolution,
        "utm_elevation": cell_median(points[:, 2], cell, starts, counts)[order],
        "depth_m": cell_median(depths, cell, starts, counts)[order],
        "depth_std": depth_std[order],
        "wsm_elevation": cell_median(wsm_elevs, cell, starts, counts)[order],
        "n_observations": counts[order],
    }


def save_csv(grid, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for row in zip(*(grid[name] for name in CSV_FIELDS)):
            writer.writerow([f"{value:.4f}" for value in row[:-1]] + [row[-1]])
    print(f"Saved: {path} ({len(grid['depth_m']):,} cells)")


def save_ply(path, points, depths):
    norm = plt.Normalize(vmin=np.nanmin(depths), vmax=np.nanmax(depths))
    colors = (plt.cm.viridis(norm(depths))[:, :3] * 255).astype(np.uint8)
    write_ply_binary(path, points, colors=colors, scalar_fields={"depth_m": depths})
    print(f"Saved: {path} ({len(points):,} points)")


def save_depth_png(path, depth_map, vmin, vmax):
    H, W = depth_map.shape
    fig, ax = plt.subplots(figsize=(max(6, W / 150), max(4, H / 150)), dpi=150)
    im = ax.imshow(
        np.ma.masked_invalid(depth_map),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=9)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("depth (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_view_pngs(depth_maps, views_dir):
    """Save the (view index, depth map) pairs as PNGs on one color scale.

    The scale runs from the 1st to the 99th percentile of all depths.
    """
    os.makedirs(views_dir, exist_ok=True)
    finite = np.concatenate([m[np.isfinite(m)] for _, m in depth_maps])
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))

    every = 10 if len(depth_maps) > 20 else 1
    n_saved = 0
    for k, (vi, depth_map) in enumerate(depth_maps):
        if k % every == 0 or k == len(depth_maps) - 1:
            save_depth_png(
                os.path.join(views_dir, f"depth_view_{vi:03d}.png"),
                depth_map,
                vmin,
                vmax,
            )
            n_saved += 1
    print(f"Saved: {n_saved} depth maps to {views_dir} ({vmin:.3f} to {vmax:.3f} m)")


def save_outputs(args, results):
    found = [r for r in results if r is not None]
    if not found:
        print("ERROR: No valid points in any view")
        sys.exit(1)
    points = np.concatenate([r["utm_points"] for r in found])
    depths = np.concatenate([r["depths"] for r in found])
    wsm_elevs = np.concatenate([r["wsm_elevs"] for r in found])

    print(f"\n{len(points):,} points from {len(found)} of {len(results)} views")
    print(
        f"  depth median {np.median(depths):.3f} m, mean {np.mean(depths):.3f} m, "
        f"std {np.std(depths):.3f} m, range {depths.min():.3f} to {depths.max():.3f} m"
    )
    for axis, name in enumerate(["E", "N", "Z"]):
        print(
            f"  UTM {name} {points[:, axis].min():.2f} to {points[:, axis].max():.2f}"
        )

    grid = grid_points(points, depths, wsm_elevs, args.grid_resolution)
    cell_depths = grid["depth_m"]
    n_obs = grid["n_observations"]
    print(
        f"{len(cell_depths):,} cells of {args.grid_resolution} m: median depth "
        f"{np.median(cell_depths):.3f} m, range {cell_depths.min():.3f} to "
        f"{cell_depths.max():.3f} m, {np.median(n_obs):.0f} points per cell "
        f"(max {n_obs.max()})"
    )

    stem = os.path.splitext(args.output_csv)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    save_csv(grid, args.output_csv)
    np.savez_compressed(
        f"{stem}_raw.npz", utm_points=points, depths=depths, wsm_elevations=wsm_elevs
    )
    print(f"Saved: {stem}_raw.npz")
    if args.output_ply:
        save_ply(f"{stem}_combined.ply", points, depths)
    if args.per_view_png:
        depth_maps = [
            (vi, r["depth_map"]) for vi, r in enumerate(results) if r is not None
        ]
        save_view_pngs(depth_maps, f"{stem}_views")


def main():
    args = parse_args()

    umeyama = load_umeyama(args.umeyama_npz)
    s, R, t = umeyama["scale"], umeyama["rotation"], umeyama["translation"]
    print(
        f"Umeyama scale {s:.6f}, XY RMS {umeyama.get('xy_rms', float('nan')):.3f} m, "
        f"Z correction {umeyama.get('z_correction', 0.0):+.4f} m"
    )

    if args.mode == "nerfrac":
        results = run_nerfrac(args, s, R, t)
    else:
        results = run_vggt(args, s, R, t)
    save_outputs(args, results)


if __name__ == "__main__":
    main()
