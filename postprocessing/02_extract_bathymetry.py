"""
Extract bathymetric depth from NeRFrac or VGGT model output.

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
import glob
import os
import sys

import numpy as np

from coord_utils import (
    apply_umeyama,
    discover_nerfrac_views,
    discover_vggt_views,
    load_nerfrac_coord_metadata,
    load_umeyama,
    NEAR,
    ndc_to_colmap,
    sample_wsm_batch,
    WATER_MASK_THRESHOLD,
    write_ply_binary,
)


DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_GRID_RESOLUTION = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bathymetric depth from model output."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["nerfrac", "vggt"],
        help="Model type: nerfrac or vggt",
    )

    # NeRFrac-specific
    parser.add_argument(
        "--testset_dir",
        type=str,
        default="",
        help="[nerfrac] Path to testset folder with Xs_pts/rays/depth .npy files",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="[nerfrac] Path to experiment log folder (contains c2w.npy, llff_scale.npy)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="[nerfrac] Dataset folder (fallback for poses_bounds.npy)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=0,
        help="[nerfrac] Image height at training resolution (auto-detected if 0)",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=0,
        help="[nerfrac] Image width at training resolution (auto-detected if 0)",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=0.0,
        help="[nerfrac] Focal length at training resolution (auto-detected if 0)",
    )

    # VGGT-specific
    parser.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="[vggt] Path to VGGT output directory with world_points_NNN.npy",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="[vggt] Minimum depth confidence (0-1, default 0.5)",
    )

    parser.add_argument(
        "--umeyama_npz",
        type=str,
        required=True,
        help="Path to Umeyama .npz from 01_compute_umeyama.py",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="Path to Water Surface Model GeoTIFF",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Path to binary water masks directory (optional but recommended). "
        "The masks are paired with the views by sorted file name.",
    )
    parser.add_argument(
        "--grid_resolution",
        type=float,
        default=DEFAULT_GRID_RESOLUTION,
        help="Grid cell size in meters (default 0.1)",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.0,
        help="Minimum valid depth in meters (default 0.0 — reject points above WSM)",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=5.0,
        help="Maximum valid depth in meters (default 5.0). Raise it, e.g. to 50, for "
        "models that overpredict depth and need 11_scale_correction.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--depth_method",
        type=str,
        default="wsm",
        choices=["wsm", "plane_fit"],
        help="[nerfrac] Depth computation method. "
        "'wsm' (default): depth = WSM_elev - point_Z in UTM33. "
        "'plane_fit': fit water surface plane to Xs_pts in COLMAP space, "
        "compute signed distance below plane, multiply by Umeyama scale "
        "for metric depth. Uses NeRFrac's own water surface estimate — "
        "no GT data dependency and immune to Umeyama Z-offset errors.",
    )
    parser.add_argument(
        "--no_z_auto_correct",
        action="store_true",
        default=False,
        help="[vggt] Disable automatic Z-offset correction. "
        "By default, the extraction auto-detects and corrects the "
        "WGS84-to-geoid elevation offset for VGGT mode.",
    )
    parser.add_argument(
        "--z_calib_views",
        type=int,
        default=20,
        help="[vggt] Number of views to use for Z-offset calibration (default 20)",
    )
    parser.add_argument(
        "--z_surface_percentile",
        type=float,
        default=90.0,
        help="[vggt] Percentile for selecting near-surface points "
        "during Z-offset calibration (default 90 = top 10%%)",
    )

    parser.add_argument(
        "--depth_type",
        type=str,
        default="mode",
        choices=["mean", "mode"],
        help="[nerfrac] Which depth map to use. "
        "'mode' (default, used for the thesis results): peak of weight distribution "
        "(*_depth_mode.npy). 'mean': weighted average depth (*_depth.npy).",
    )

    # PLY point cloud -> for debugging
    parser.add_argument(
        "--output_ply",
        action="store_true",
        default=False,
        help="Also save a combined PLY point cloud (xyz + depth_m field). "
        "Output path: <output_csv stem>_combined.ply",
    )
    parser.add_argument(
        "--per_view_png",
        action="store_true",
        default=False,
        help="Save per-view colorized depth map PNGs. "
        "Output directory: <output_csv stem>_views/",
    )

    return parser.parse_args()


def list_masks(mask_dir, n_views):
    if not mask_dir:
        return None
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if len(mask_files) != n_views:
        print(f"ERROR: {len(mask_files)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    return [os.path.join(mask_dir, f) for f in mask_files]


def load_mask(mask_files, view_idx, H, W):
    if mask_files is None:
        return None

    import imageio

    mask = imageio.imread(mask_files[view_idx])
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.shape != (H, W):
        from PIL import Image as PILImage

        mask_pil = PILImage.fromarray(mask)
        mask_pil = mask_pil.resize((W, H), PILImage.NEAREST)
        mask = np.array(mask_pil)
    return mask > WATER_MASK_THRESHOLD


def extract_nerfrac_view(
    view,
    H,
    W,
    focal,
    c2w_recenter,
    sc,
    s,
    R,
    t,
    water_mask,
    wsm_path,
    depth_min,
    depth_max,
    return_depth_map=False,
):
    if view["xs"] is None or view["rays"] is None or view["depth"] is None:
        return None

    xs_ndc = np.load(view["xs"]).reshape(-1, 3)
    rays_ndc = np.load(view["rays"]).reshape(-1, 3)
    depth_map = np.load(view["depth"])  # (H, W)...

    expected_n = H * W
    if xs_ndc.shape[0] != expected_n:
        print(f"  [WARN] Xs shape {xs_ndc.shape[0]} != H*W={expected_n}")
        return None

    depth_flat = depth_map.reshape(-1, 1)

    if water_mask is not None:
        mask_flat = water_mask.reshape(-1)
    else:
        mask_flat = np.ones(expected_n, dtype=bool)

    water_idx = np.where(mask_flat)[0]
    if len(water_idx) == 0:
        return None

    # nerFRac: P = Xs + depth * rays
    xs_w = xs_ndc[water_idx]
    rays_w = rays_ndc[water_idx]
    depth_w = depth_flat[water_idx]

    P_ndc = xs_w + depth_w * rays_w

    P_colmap = ndc_to_colmap(P_ndc, H, W, focal, c2w_recenter, sc, NEAR)

    P_utm = apply_umeyama(P_colmap, s, R, t)

    wsm_elevs = sample_wsm_batch(wsm_path, P_utm[:, 0], P_utm[:, 1])

    depths = wsm_elevs - P_utm[:, 2]

    valid = (
        np.isfinite(wsm_elevs)
        & np.isfinite(depths)
        & (depths >= depth_min)
        & (depths <= depth_max)
    )

    if valid.sum() == 0:
        return None

    result = {
        "utm_points": P_utm[valid],
        "depths": depths[valid],
        "wsm_elevs": wsm_elevs[valid],
    }

    if return_depth_map:
        depth_map_2d = np.full((H, W), np.nan, dtype=np.float32)
        valid_water_idx = water_idx[valid]
        rows = valid_water_idx // W
        cols = valid_water_idx % W
        depth_map_2d[rows, cols] = depths[valid].astype(np.float32)
        result["depth_map_2d"] = depth_map_2d

    return result


# plane-fit extraction
def fit_water_surface_plane(Xs_colmap):
    centroid = Xs_colmap.mean(axis=0)
    centered = Xs_colmap - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # last singular vector = smallest variance = plane normal

    d = -np.dot(normal, centroid)
    return normal, d, centroid


def extract_nerfrac_views_plane_fit(
    views,
    H,
    W,
    focal,
    c2w_recenter,
    sc,
    umeyama_scale,
    R,
    t,
    mask_files,
    wsm_path,
    depth_min,
    depth_max,
    return_depth_maps=False,
):
    from coord_utils import apply_umeyama, ndc_to_colmap, sample_wsm_batch

    print("\n  [plane_fit] Pass 1: Collecting water surface points...")
    all_Xs_colmap = []
    for vi, view in enumerate(views):
        if view["xs"] is None:
            continue

        xs_ndc = np.load(view["xs"]).reshape(-1, 3)
        expected_n = H * W
        if xs_ndc.shape[0] != expected_n:
            continue

        mask = load_mask(mask_files, vi, H, W)
        if mask is not None:
            mask_flat = mask.reshape(-1)
        else:
            mask_flat = np.ones(expected_n, dtype=bool)

        water_idx = np.where(mask_flat)[0]
        if len(water_idx) == 0:
            continue

        xs_water = xs_ndc[water_idx]
        Xs_colmap = ndc_to_colmap(xs_water, H, W, focal, c2w_recenter, sc, NEAR)
        all_Xs_colmap.append(Xs_colmap)

    if not all_Xs_colmap:
        print("  [plane_fit] ERROR: No water surface points found")
        return None, None, None, len(views), len(views)

    all_Xs_colmap = np.concatenate(all_Xs_colmap, axis=0)
    print(f"  [plane_fit] Total water surface points: {len(all_Xs_colmap):,}")

    normal, d_plane, centroid = fit_water_surface_plane(all_Xs_colmap)
    print(
        f"  [plane_fit] Plane normal (COLMAP):  [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]"
    )
    print(
        f"  [plane_fit] Plane centroid (COLMAP): [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]"
    )
    # Need UTM reported:
    centroid_utm = apply_umeyama(centroid, umeyama_scale, R, t)
    print(
        f"  [plane_fit] Plane centroid (UTM33):  E={centroid_utm[0]:.2f}, N={centroid_utm[1]:.2f}, Z={centroid_utm[2]:.2f}"
    )

    surface_dists = np.dot(all_Xs_colmap, normal) + d_plane
    print(
        f"  [plane_fit] Water surface plane residuals: "
        f"mean={surface_dists.mean():.6f}, std={surface_dists.std():.6f}, "
        f"max_abs={np.abs(surface_dists).max():.6f} (COLMAP units)"
    )
    print(
        f"  [plane_fit] Water surface plane residuals (metric): "
        f"std={surface_dists.std() * umeyama_scale:.4f} m, "
        f"max_abs={np.abs(surface_dists).max() * umeyama_scale:.4f} m"
    )
    del all_Xs_colmap

    print("\n  [plane_fit] Orienting plane normal...")
    normal_oriented = False
    for vi, view in enumerate(views):
        if view["xs"] is None or view["rays"] is None or view["depth"] is None:
            continue

        xs_ndc_probe = np.load(view["xs"]).reshape(-1, 3)
        rays_ndc_probe = np.load(view["rays"]).reshape(-1, 3)
        depth_probe = np.load(view["depth"]).reshape(-1, 1)

        expected_n = H * W
        if xs_ndc_probe.shape[0] != expected_n:
            continue

        mask_probe = load_mask(mask_files, vi, H, W)
        if mask_probe is not None:
            water_idx_probe = np.where(mask_probe.reshape(-1))[0]
        else:
            water_idx_probe = np.arange(expected_n)

        if len(water_idx_probe) == 0:
            continue

        if len(water_idx_probe) > 5000:
            rng = np.random.default_rng(42)
            water_idx_probe = rng.choice(water_idx_probe, 5000, replace=False)

        xs_sub = xs_ndc_probe[water_idx_probe]
        rays_sub = rays_ndc_probe[water_idx_probe]
        depth_sub = depth_probe[water_idx_probe]

        P_ndc_probe = xs_sub + depth_sub * rays_sub
        P_colmap_probe = ndc_to_colmap(P_ndc_probe, H, W, focal, c2w_recenter, sc, NEAR)

        signed_dist_probe = np.dot(P_colmap_probe, normal) + d_plane
        median_sd = np.median(signed_dist_probe)

        print(
            f"  [plane_fit] Normal orientation probe (view {vi}): "
            f"median signed_dist = {median_sd:.6f}"
        )

        if median_sd > 0:
            normal = -normal
            d_plane = -d_plane
            print(
                f"  [plane_fit] Flipped normal → "
                f"[{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]"
            )
        else:
            print(f"  [plane_fit] Normal already correctly oriented (no flip needed)")

        normal_oriented = True
        break

    if not normal_oriented:
        print(
            "  [plane_fit] WARNING: Could not orient normal — no valid views "
            "found for probing. Depth signs may be wrong."
        )


    # Extract bathymetric depths with new plane
    print(f"\n  [plane_fit] Pass 2: Extracting depths from {len(views)} views...")

    all_points = []
    all_depths = []
    all_wsm_elevs = []
    depth_maps = []
    n_skipped = 0

    for vi, view in enumerate(views):
        if view["xs"] is None or view["rays"] is None or view["depth"] is None:
            n_skipped += 1
            continue

        xs_ndc = np.load(view["xs"]).reshape(-1, 3)
        rays_ndc = np.load(view["rays"]).reshape(-1, 3)
        depth_map = np.load(view["depth"])

        expected_n = H * W
        if xs_ndc.shape[0] != expected_n:
            n_skipped += 1
            continue

        depth_flat = depth_map.reshape(-1, 1)

        mask = load_mask(mask_files, vi, H, W)
        if mask is not None:
            mask_flat = mask.reshape(-1)
        else:
            mask_flat = np.ones(expected_n, dtype=bool)

        water_idx = np.where(mask_flat)[0]
        if len(water_idx) == 0:
            n_skipped += 1
            continue

        xs_w = xs_ndc[water_idx]
        rays_w = rays_ndc[water_idx]
        depth_w = depth_flat[water_idx]

        P_ndc = xs_w + depth_w * rays_w
        P_colmap = ndc_to_colmap(P_ndc, H, W, focal, c2w_recenter, sc, NEAR)

        signed_dist = np.dot(P_colmap, normal) + d_plane
        depth_colmap = -signed_dist  # positive = below plane

        depth_metric = depth_colmap * umeyama_scale

        valid = (
            np.isfinite(depth_metric)
            & (depth_metric >= depth_min)
            & (depth_metric <= depth_max)
        )

        if valid.sum() == 0:
            n_skipped += 1
            if vi % 50 == 0:
                print(f"  View {vi:3d}: skipped (no valid points after filtering)")
            continue

        P_colmap_valid = P_colmap[valid]
        depth_metric_valid = depth_metric[valid]

        P_utm = apply_umeyama(P_colmap_valid, umeyama_scale, R, t)

        ws_elevs = P_utm[:, 2] + depth_metric_valid

        n_pts = valid.sum()
        med_depth = np.median(depth_metric_valid)

        all_points.append(P_utm)
        all_depths.append(depth_metric_valid)
        all_wsm_elevs.append(ws_elevs)

        if return_depth_maps:
            depth_map_2d = np.full((H, W), np.nan, dtype=np.float32)
            valid_water_idx = water_idx[valid]
            rows = valid_water_idx // W
            cols = valid_water_idx % W
            depth_map_2d[rows, cols] = depth_metric[valid].astype(np.float32)
            depth_maps.append((vi, depth_map_2d))

        if vi % 50 == 0 or vi == len(views) - 1:
            print(f"  View {vi:3d}: {n_pts:6d} pts, median depth {med_depth:.3f} m")

    if return_depth_maps:
        return all_points, all_depths, all_wsm_elevs, len(views), n_skipped, depth_maps
    return all_points, all_depths, all_wsm_elevs, len(views), n_skipped


def extract_vggt_view(
    view,
    s,
    R,
    t,
    water_mask,
    conf_threshold,
    wsm_path,
    depth_min,
    depth_max,
    z_offset=0.0,
    return_depth_map=False,
):
    world_points = np.load(view["world_points"])  # (H, W, 3)
    H_v, W_v = world_points.shape[:2]

    pts_flat = world_points.reshape(-1, 3)

    valid = np.ones(pts_flat.shape[0], dtype=bool)

    if water_mask is not None:
        mask_resized = water_mask
        if water_mask.shape != (H_v, W_v):
            from PIL import Image as PILImage

            mask_pil = PILImage.fromarray(water_mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((W_v, H_v), PILImage.NEAREST)
            mask_resized = np.array(mask_pil) > WATER_MASK_THRESHOLD
        valid &= mask_resized.reshape(-1)

    if view["depth_conf"] is not None:
        conf = np.load(view["depth_conf"]).reshape(-1)
        valid &= conf >= conf_threshold

    valid &= np.all(np.isfinite(pts_flat), axis=1)
    valid &= np.any(pts_flat != 0.0, axis=1)

    water_idx = np.where(valid)[0]
    if len(water_idx) == 0:
        return None

    pts_water = pts_flat[water_idx]

    P_utm = apply_umeyama(pts_water, s, R, t)

    #  Z-offset is needed
    if z_offset != 0.0:
        P_utm[:, 2] -= z_offset

    wsm_elevs = sample_wsm_batch(wsm_path, P_utm[:, 0], P_utm[:, 1])

    depths = wsm_elevs - P_utm[:, 2]

    keep = (
        np.isfinite(wsm_elevs)
        & np.isfinite(depths)
        & (depths >= depth_min)
        & (depths <= depth_max)
    )

    if keep.sum() == 0:
        return None

    result = {
        "utm_points": P_utm[keep],
        "depths": depths[keep],
        "wsm_elevs": wsm_elevs[keep],
    }

    if return_depth_map:
        depth_map_2d = np.full((H_v, W_v), np.nan, dtype=np.float32)
        valid_water_idx = water_idx[keep]
        rows = valid_water_idx // W_v
        cols = valid_water_idx % W_v
        depth_map_2d[rows, cols] = depths[keep].astype(np.float32)
        result["depth_map_2d"] = depth_map_2d

    return result


def grid_points(all_points, all_depths, all_wsm_elevs, resolution):
    if len(all_points) == 0:
        return []

    e_min = all_points[:, 0].min()
    n_min = all_points[:, 1].min()

    gi = ((all_points[:, 0] - e_min) / resolution).astype(int)
    gj = ((all_points[:, 1] - n_min) / resolution).astype(int)

    from collections import defaultdict

    cells = defaultdict(list)
    for k in range(len(all_points)):
        cells[(gi[k], gj[k])].append(k)

    rows = []
    for (i, j), indices in cells.items():
        indices = np.array(indices)
        e_center = e_min + (i + 0.5) * resolution
        n_center = n_min + (j + 0.5) * resolution

        pts_cell = all_points[indices]
        depths_cell = all_depths[indices]
        wsm_cell = all_wsm_elevs[indices]

        rows.append(
            {
                "utm_easting": e_center,
                "utm_northing": n_center,
                "utm_elevation": float(np.median(pts_cell[:, 2])),
                "depth_m": float(np.median(depths_cell)),
                "depth_std": float(np.std(depths_cell))
                if len(depths_cell) > 1
                else 0.0,
                "wsm_elevation": float(np.median(wsm_cell)),
                "n_observations": len(indices),
            }
        )

    return rows


def save_csv(rows, output_path):
    """Save gridded bathymetry to CSV."""
    if not rows:
        print("[WARN] No data to save")
        return

    fieldnames = [
        "utm_easting",
        "utm_northing",
        "utm_elevation",
        "depth_m",
        "depth_std",
        "wsm_elevation",
        "n_observations",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utm_easting": f"{row['utm_easting']:.4f}",
                    "utm_northing": f"{row['utm_northing']:.4f}",
                    "utm_elevation": f"{row['utm_elevation']:.4f}",
                    "depth_m": f"{row['depth_m']:.4f}",
                    "depth_std": f"{row['depth_std']:.4f}",
                    "wsm_elevation": f"{row['wsm_elevation']:.4f}",
                    "n_observations": row["n_observations"],
                }
            )

    print(f"Saved {len(rows)} grid cells to {output_path}")



def write_ply_with_depth(filepath, points, depths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(points)
    if N == 0:
        print(f"  [WARN] No points for PLY: {filepath}")
        return

    norm = plt.Normalize(vmin=np.nanmin(depths), vmax=np.nanmax(depths))
    cmap = plt.cm.viridis
    rgba = cmap(norm(depths))
    colors = (rgba[:, :3] * 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float depth_m\n"
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
            ("depth_m", "<f4"),
        ]
    )
    vertices = np.empty(N, dtype=dtype)
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["depth_m"] = depths.astype(np.float32)

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)

    print(f"  Wrote PLY: {filepath} ({N:,} vertices)")



def save_depth_png(filepath, depth_2d, vmin=None, vmax=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = depth_2d.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(6, W / 150), max(4, H / 150)), dpi=150)

    masked = np.ma.masked_invalid(depth_2d)
    if vmin is None:
        vmin = float(np.nanmin(depth_2d)) if np.any(np.isfinite(depth_2d)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(depth_2d)) if np.any(np.isfinite(depth_2d)) else 1.0

    im = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(os.path.basename(filepath).replace(".png", ""), fontsize=9)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("depth (m)")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    args = parse_args()

    print("=" * 60)
    print(f"BATHYMETRY EXTRACTION  ({args.mode.upper()})")
    print("=" * 60)

    umeyama = load_umeyama(args.umeyama_npz)
    s = umeyama["scale"]
    R = umeyama["rotation"]
    t = umeyama["translation"]
    print(f"Scale: {s:.6f}")
    print(f"XY RMS: {umeyama.get('xy_rms', '?')} m")
    print(f"Z correction: {umeyama.get('z_correction', 0.0):+.4f} m")

    if args.mode == "nerfrac":
        _run_nerfrac(args, s, R, t)
    else:
        _run_vggt(args, s, R, t)


def _run_nerfrac(args, s, R, t):
    """NeRFrac extraction path."""
    if not args.testset_dir:
        print("ERROR: --testset_dir required for nerfrac mode")
        sys.exit(1)
    if not args.log_folder:
        print("ERROR: --log_folder required for nerfrac mode")
        sys.exit(1)

    coord_meta = load_nerfrac_coord_metadata(
        args.log_folder, data_dir=args.data_dir or None
    )
    c2w_recenter = coord_meta["c2w_recenter"]
    sc = coord_meta["sc"]
    if c2w_recenter is None or sc is None:
        print("ERROR: Could not load c2w.npy and/or llff_scale.npy")
        sys.exit(1)
    print(f"LLFF scale: sc={sc:.8f}  (1/sc={1.0 / sc:.4f})")

    views = discover_nerfrac_views(args.testset_dir, args.depth_type)
    if not views:
        print(f"ERROR: No NeRFrac views found in {args.testset_dir}")
        sys.exit(1)
    mask_files = list_masks(args.mask_dir, len(views))

    # Get focal
    H, W, focal = args.H, args.W, args.focal
    if H == 0 or W == 0 or focal == 0.0:
        if views[0]["depth"] is not None:
            d0 = np.load(views[0]["depth"])
            H_auto, W_auto = d0.shape
            if H == 0:
                H = H_auto
            if W == 0:
                W = W_auto
            if focal == 0.0:
                if args.data_dir:
                    pb_path = os.path.join(args.data_dir, "poses_bounds.npy")
                    if os.path.exists(pb_path):
                        pb = np.load(pb_path)
                        poses = pb[:, :15].reshape(-1, 3, 5)
                        H_full, W_full, focal_full = poses[0, :, 4]
                        focal = focal_full * W / W_full
                        print(f"Auto-detected focal={focal:.4f} from poses_bounds.npy")
                if focal == 0.0:
                    print(
                        "ERROR: Cannot auto-detect focal. Provide --focal or --data_dir"
                    )
                    sys.exit(1)
            print(f"Resolution: H={H}, W={W}, focal={focal:.4f}")

    print(
        f"\n--- Step 2: Process {len(views)} views (depth_method={args.depth_method}, depth_type={args.depth_type}) ---"
    )

    if args.depth_method == "plane_fit":
        want_maps = args.per_view_png
        pf_result = extract_nerfrac_views_plane_fit(
            views,
            H,
            W,
            focal,
            c2w_recenter,
            sc,
            s,
            R,
            t,
            mask_files,
            args.wsm_path,
            args.depth_min,
            args.depth_max,
            return_depth_maps=want_maps,
        )
        if pf_result[0] is None:
            print("\nERROR: No valid points extracted from any view")
            sys.exit(1)
        if want_maps:
            all_points, all_depths, all_wsm_elevs, n_views, n_skipped, depth_maps = (
                pf_result
            )
        else:
            all_points, all_depths, all_wsm_elevs, n_views, n_skipped = pf_result
            depth_maps = []

        if args.per_view_png and depth_maps:
            _save_per_view_pngs(args, depth_maps)

        _finalize(args, all_points, all_depths, all_wsm_elevs, n_views, n_skipped)
    else:
        all_points = []
        all_depths = []
        all_wsm_elevs = []
        depth_maps = []
        n_skipped = 0

        for vi, view in enumerate(views):
            mask = load_mask(mask_files, vi, H, W)

            result = extract_nerfrac_view(
                view,
                H,
                W,
                focal,
                c2w_recenter,
                sc,
                s,
                R,
                t,
                mask,
                args.wsm_path,
                args.depth_min,
                args.depth_max,
                return_depth_map=args.per_view_png,
            )

            if result is None:
                n_skipped += 1
                if vi % 50 == 0:
                    print(f"  View {vi:3d}: skipped (no valid points)")
                continue

            n_pts = len(result["depths"])
            med_depth = np.median(result["depths"])
            all_points.append(result["utm_points"])
            all_depths.append(result["depths"])
            all_wsm_elevs.append(result["wsm_elevs"])

            if args.per_view_png and "depth_map_2d" in result:
                depth_maps.append((vi, result["depth_map_2d"]))

            if vi % 50 == 0 or vi == len(views) - 1:
                print(f"  View {vi:3d}: {n_pts:6d} pts, median depth {med_depth:.3f} m")

        if args.per_view_png and depth_maps:
            _save_per_view_pngs(args, depth_maps)

        _finalize(args, all_points, all_depths, all_wsm_elevs, len(views), n_skipped)


def _calibrate_z_offset_vggt(
    views, s, R, t, mask_files, wsm_path, n_calib_views, surface_percentile
):

    n_use = min(n_calib_views, len(views))
    indices = np.linspace(0, len(views) - 1, n_use, dtype=int)

    all_pts_utm = []
    for vi in indices:
        wp = np.load(views[vi]["world_points"])  # (H, W, 3)
        H_v, W_v = wp.shape[:2]
        pts = wp.reshape(-1, 3)

        valid = np.all(np.isfinite(pts), axis=1) & np.any(pts != 0.0, axis=1)

        mask = load_mask(mask_files, vi, H_v, W_v)
        if mask is not None:
            if mask.shape != (H_v, W_v):
                from PIL import Image as PILImage

                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((W_v, H_v), PILImage.NEAREST)
                mask = np.array(mask_pil) > WATER_MASK_THRESHOLD
            valid &= mask.reshape(-1)

        pts_valid = pts[valid]
        if len(pts_valid) == 0:
            continue

        utm = apply_umeyama(pts_valid, s, R, t)
        all_pts_utm.append(utm)

    if not all_pts_utm:
        print("  [WARN] Z-calibration: no valid points from calibration views")
        return 0.0

    all_pts_utm = np.concatenate(all_pts_utm, axis=0)

    z_threshold = np.percentile(all_pts_utm[:, 2], surface_percentile)
    surface_mask = all_pts_utm[:, 2] >= z_threshold
    surface_pts = all_pts_utm[surface_mask]

    wsm_elevs = sample_wsm_batch(wsm_path, surface_pts[:, 0], surface_pts[:, 1])
    wsm_valid = np.isfinite(wsm_elevs)

    if wsm_valid.sum() < 10:
        print(
            f"  [WARN] Z-calibration: only {wsm_valid.sum()} surface points "
            f"within WSM extent — offset may be unreliable"
        )
        if wsm_valid.sum() == 0:
            return 0.0

    offsets = surface_pts[:, 2][wsm_valid] - wsm_elevs[wsm_valid]
    z_offset = float(np.median(offsets))

    print(f"  Z-offset calibration:")
    print(f"    Calibration views:     {n_use}")
    print(f"    Total pts sampled:     {len(all_pts_utm):,}")
    print(
        f"    Surface proxy pts:     {len(surface_pts):,} "
        f"(top {100 - surface_percentile:.0f}% by Z)"
    )
    print(f"    WSM-matched:           {wsm_valid.sum():,}")
    print(f"    Median Z offset:       {z_offset:+.3f} m")
    print(
        f"    Offset std:            {np.std(offsets[wsm_valid] if wsm_valid.all() else offsets):.3f} m"
    )
    print(f"    -> Will subtract {z_offset:.3f} m from all Z values")

    return z_offset


def _run_vggt(args, s, R, t):
    """VGGT extraction path."""
    if not args.vggt_dir:
        print("ERROR: --vggt_dir required for vggt mode")
        sys.exit(1)

    views = discover_vggt_views(args.vggt_dir)
    print(f"\n--- Step 2: Process {len(views)} views ---")

    if len(views) == 0:
        print("ERROR: No VGGT views found")
        sys.exit(1)

    mask_files = list_masks(args.mask_dir, len(views))
    wp0 = np.load(views[0]["world_points"])
    H_v, W_v = wp0.shape[:2]
    del wp0

    z_offset = 0.0
    if not args.no_z_auto_correct:
        print("\n--- Step 2a: Z-offset calibration ---")
        z_offset = _calibrate_z_offset_vggt(
            views,
            s,
            R,
            t,
            mask_files,
            args.wsm_path,
            args.z_calib_views,
            args.z_surface_percentile,
        )
    else:
        print("  Z auto-correction disabled (--no_z_auto_correct)")

    all_points = []
    all_depths = []
    all_wsm_elevs = []
    depth_maps = []  # list of (view_index, depth_map_2d)
    n_skipped = 0

    for vi, view in enumerate(views):
        mask = load_mask(mask_files, vi, H_v, W_v)

        result = extract_vggt_view(
            view,
            s,
            R,
            t,
            mask,
            args.conf_threshold,
            args.wsm_path,
            args.depth_min,
            args.depth_max,
            z_offset=z_offset,
            return_depth_map=args.per_view_png,
        )

        if result is None:
            n_skipped += 1
            if vi % 50 == 0:
                print(f"  View {vi:3d}: skipped")
            continue

        n_pts = len(result["depths"])
        med_depth = np.median(result["depths"])
        all_points.append(result["utm_points"])
        all_depths.append(result["depths"])
        all_wsm_elevs.append(result["wsm_elevs"])

        if args.per_view_png and "depth_map_2d" in result:
            depth_maps.append((vi, result["depth_map_2d"]))

        if vi % 50 == 0 or vi == len(views) - 1:
            print(f"  View {vi:3d}: {n_pts:6d} pts, median depth {med_depth:.3f} m")

    if args.per_view_png and depth_maps:
        _save_per_view_pngs(args, depth_maps)

    _finalize(args, all_points, all_depths, all_wsm_elevs, len(views), n_skipped)


def _save_per_view_pngs(args, depth_maps):

    csv_stem = os.path.splitext(args.output_csv)[0]
    views_dir = csv_stem + "_views"
    os.makedirs(views_dir, exist_ok=True)

    all_finite_depths = []
    for _, dm in depth_maps:
        finite = dm[np.isfinite(dm)]
        if len(finite) > 0:
            all_finite_depths.append(finite)

    if not all_finite_depths:
        print("  [per_view_png] No finite depth values — skipping PNGs")
        return

    combined = np.concatenate(all_finite_depths)
    global_vmin = float(np.percentile(combined, 1))
    global_vmax = float(np.percentile(combined, 99))
    del combined, all_finite_depths
    print(
        f"\n--- Per-view depth PNGs ---"
        f"\n  Output dir:  {views_dir}"
        f"\n  Views:       {len(depth_maps)}"
        f"\n  Colorbar:    [{global_vmin:.3f}, {global_vmax:.3f}] m (p1/p99)"
    )

    # >20 views, only save every 10th
    save_every = 1
    if len(depth_maps) > 20:
        save_every = 10
        print(
            f"  Throttling:  saving every {save_every}th view ({len(depth_maps)} total)"
        )

    n_saved = 0
    for i, (vi, dm) in enumerate(depth_maps):
        if i % save_every != 0 and i != len(depth_maps) - 1:
            continue
        png_path = os.path.join(views_dir, f"depth_view_{vi:03d}.png")
        save_depth_png(png_path, dm, vmin=global_vmin, vmax=global_vmax)
        n_saved += 1

    print(f"  Saved {n_saved} depth PNGs to {views_dir}")


def _finalize(args, all_points, all_depths, all_wsm_elevs, n_views, n_skipped):
    if not all_points:
        print("\nERROR: No valid points extracted from any view")
        sys.exit(1)

    all_points = np.concatenate(all_points, axis=0)
    all_depths = np.concatenate(all_depths, axis=0)
    all_wsm_elevs = np.concatenate(all_wsm_elevs, axis=0)

    print(f"\n--- Step 3: Summary ---")
    print(f"  Views processed: {n_views - n_skipped} / {n_views}")
    print(f"  Total points:    {len(all_points)}")
    print(f"  Depth range:     [{all_depths.min():.3f}, {all_depths.max():.3f}] m")
    print(f"  Depth median:    {np.median(all_depths):.3f} m")
    print(f"  Depth mean:      {np.mean(all_depths):.3f} m")
    print(f"  Depth std:       {np.std(all_depths):.3f} m")
    print(
        f"  UTM E range:     [{all_points[:, 0].min():.2f}, {all_points[:, 0].max():.2f}]"
    )
    print(
        f"  UTM N range:     [{all_points[:, 1].min():.2f}, {all_points[:, 1].max():.2f}]"
    )
    print(
        f"  UTM Z range:     [{all_points[:, 2].min():.2f}, {all_points[:, 2].max():.2f}]"
    )
    print(f"\n--- Step 4: Grid (resolution={args.grid_resolution}m) ---")
    rows = grid_points(all_points, all_depths, all_wsm_elevs, args.grid_resolution)
    print(f"  Grid cells: {len(rows)}")

    if rows:
        grid_depths = np.array([r["depth_m"] for r in rows])
        grid_obs = np.array([r["n_observations"] for r in rows])
        print(
            f"  Gridded depth range:  [{grid_depths.min():.3f}, {grid_depths.max():.3f}] m"
        )
        print(f"  Gridded depth median: {np.median(grid_depths):.3f} m")
        print(
            f"  Observations/cell:    median={np.median(grid_obs):.0f}, "
            f"max={grid_obs.max()}"
        )

    print(f"\n--- Step 5: Save ---")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    save_csv(rows, args.output_csv)

    raw_path = os.path.splitext(args.output_csv)[0] + "_raw.npz"
    np.savez_compressed(
        raw_path,
        utm_points=all_points,
        depths=all_depths,
        wsm_elevations=all_wsm_elevs,
    )
    print(f"Saved {len(all_points)} raw points to {raw_path}")

    if args.output_ply:
        ply_path = os.path.splitext(args.output_csv)[0] + "_combined.ply"
        write_ply_with_depth(ply_path, all_points, all_depths)


if __name__ == "__main__":
    main()
