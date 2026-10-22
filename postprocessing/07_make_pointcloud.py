"""
Generate colored scene point clouds from VGGT or NeRFrac output.

Writes pointcloud_combined.ply/.npy and, with --mask_dir, the water and land points
to pointcloud_water.ply/.npy and pointcloud_land.ply/.npy.

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

import cv2
import imageio.v2 as imageio
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_utils import (
    WATER_MASK_THRESHOLD,
    apply_umeyama,
    discover_nerfrac_views,
    discover_vggt_views,
    load_focal_from_poses_bounds,
    load_nerfrac_coord_metadata,
    load_umeyama,
    ndc_to_colmap,
    write_ply_binary,
)

# NeRFrac returns the ray origin on the NDC near plane (z = -1) as Xs of land rays.
# Water rays have Xs on the learned surface, deeper in NDC.
NDC_NEAR_PLANE_Z = -0.999
PLANE_FIT_POINTS = 50000


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a colored scene point cloud (PLY) from model output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["vggt", "nerfrac"])

    # VGGT
    p.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="[vggt] output folder with world_points_NNN.npy and rgb_NNN.png",
    )
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="[vggt] minimum depth confidence of a point (0: no filter)",
    )

    # NeRFrac
    p.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="[nerfrac] experiment folder with c2w.npy and llff_scale.npy",
    )
    p.add_argument(
        "--testset",
        type=str,
        default="",
        help="[nerfrac] render folder inside --log_folder, e.g. renderonly_all_600000",
    )
    p.add_argument(
        "--poses_bounds",
        type=str,
        default="",
        help="[nerfrac] poses_bounds.npy of the scene, for the image size and focal",
    )
    p.add_argument(
        "--factor",
        type=int,
        default=8,
        help="[nerfrac] image downsample factor of the training",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="[nerfrac] scene folder, to recompute the scene scale without "
        "llff_scale.npy",
    )
    p.add_argument(
        "--depth_type",
        type=str,
        default="mode",
        choices=["mean", "mode"],
        help="[nerfrac] 'mode': depth of the sample with the highest weight, "
        "*_depth_mode.npy; 'mean': expected depth, *_depth.npy",
    )
    p.add_argument(
        "--no_align",
        action="store_true",
        help="[nerfrac] don't move each view's water points onto one water surface "
        "plane",
    )
    p.add_argument(
        "--per_view_ply",
        action="store_true",
        help="[nerfrac] also write each view to pointcloud_view_NNN.ply, in COLMAP "
        "coordinates",
    )

    # common
    p.add_argument(
        "--umeyama_npz",
        type=str,
        default="",
        help="transform from 01_compute_umeyama.py; without it the points stay in "
        "model coordinates",
    )
    p.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="water masks, paired with the views by sorted file name; adds the "
        "water and land point clouds",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output folder (default: the model output folder)",
    )
    p.add_argument(
        "--no_filter",
        action="store_true",
        help="keep the outliers",
    )
    p.add_argument(
        "--outlier_percentile",
        type=float,
        default=1.0,
        help="points below this percentile or above 100 minus it, on any axis, are "
        "dropped from each view (from the combined clouds at 0.5)",
    )
    p.add_argument(
        "--no_preview",
        action="store_true",
        help="skip the 3D scatter preview PNG",
    )
    return p.parse_args()


def list_masks(mask_dir, n_views):
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if len(mask_files) != n_views:
        print(f"ERROR: {len(mask_files)} masks in {mask_dir} for {n_views} views")
        sys.exit(1)
    return mask_files


def load_mask(path, H, W):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask > WATER_MASK_THRESHOLD


def load_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def georeference(points, umeyama):
    if umeyama is None:
        return points
    return apply_umeyama(
        points, umeyama["scale"], umeyama["rotation"], umeyama["translation"]
    )


def filter_outliers(points, colors, percentile):
    keep = np.ones(len(points), dtype=bool)
    for axis in range(3):
        values = points[:, axis]
        lo = np.percentile(values, percentile)
        hi = np.percentile(values, 100 - percentile)
        keep &= (values >= lo) & (values <= hi)
    return points[keep], colors[keep], np.sum(~keep)


def finish_cloud(points, colors, percentile, args, umeyama):
    """Stack the parts, drop the outliers and georeference."""
    points, colors = np.concatenate(points), np.concatenate(colors)
    if not args.no_filter:
        points, colors, removed = filter_outliers(points, colors, percentile)
        if removed > 0:
            print(f"  removed {removed:,} outliers")
    return georeference(points, umeyama), colors


def vggt_pointcloud(args, umeyama):
    indices = [v["index"] for v in discover_vggt_views(args.vggt_dir)]
    if not indices:
        print(f"ERROR: No world_points_*.npy files in {args.vggt_dir}")
        sys.exit(1)
    mask_files = list_masks(args.mask_dir, len(indices)) if args.mask_dir else None
    print(f"{len(indices)} views in {args.vggt_dir}")

    parts = {"combined": ([], []), "water": ([], []), "land": ([], [])}
    for k, idx in enumerate(indices):
        world_points = np.load(
            os.path.join(args.vggt_dir, f"world_points_{idx:03d}.npy")
        )
        H, W = world_points.shape[:2]

        rgb_path = os.path.join(args.vggt_dir, f"rgb_{idx:03d}.png")
        if os.path.exists(rgb_path):
            img = load_rgb(rgb_path)
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            img = np.full((H, W, 3), 128, dtype=np.uint8)

        valid = np.isfinite(world_points).all(axis=2)
        if args.conf_threshold > 0:
            conf_path = os.path.join(args.vggt_dir, f"depth_conf_{idx:03d}.npy")
            if os.path.exists(conf_path):
                valid &= np.load(conf_path) >= args.conf_threshold
            else:
                print(f"  View {idx:03d}: no depth_conf file, confidence not filtered")
        parts["combined"][0].append(world_points[valid])
        parts["combined"][1].append(img[valid])

        if mask_files is None:
            print(f"  View {idx:03d}: {valid.sum():,} points")
            continue
        water = load_mask(mask_files[k], H, W)
        for name, region in (("water", valid & water), ("land", valid & ~water)):
            if region.any():
                parts[name][0].append(world_points[region])
                parts[name][1].append(img[region])
        print(
            f"  View {idx:03d}: {valid.sum():,} points "
            f"({(valid & water).sum():,} water, {(valid & ~water).sum():,} land)"
        )

    clouds = {}
    for name, (points, colors) in parts.items():
        if points:
            clouds[name] = finish_cloud(
                points, colors, args.outlier_percentile, args, umeyama
            )
    return clouds


def nerfrac_view(view, ndc, image_shape, mask):
    """Points and colors of one view in COLMAP coordinates.

    With a mask they are split into water and land, and the water points come with
    the learned surface points (Xs) of the same pixels.
    """
    xs = np.load(view["xs"])
    rays = np.load(view["rays"])
    depth = np.load(view["depth"]).reshape(-1, 1)
    points = ndc_to_colmap(xs + depth * rays, **ndc)

    colors = None
    if view["image"] is not None:
        img = load_rgb(view["image"])
        if img.shape[:2] == image_shape:
            colors = img.reshape(-1, 3)
        else:
            print(f"  View {view['index']}: image size mismatch, using gray")
    if colors is None:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

    if mask is None:
        return {"points": points, "colors": colors}

    agree = np.sum(mask == (xs[:, 2] > NDC_NEAR_PLANE_Z))
    water, land = np.flatnonzero(mask), np.flatnonzero(~mask)
    print(
        f"  View {view['index']}: {len(water):,} water, {len(land):,} land pixels, "
        f"{100 * agree / len(mask):.1f}% agree with the model's water rays"
    )
    return {
        "water_points": points[water],
        "water_colors": colors[water],
        "water_surface": ndc_to_colmap(xs[water], **ndc),
        "land_points": points[land],
        "land_colors": colors[land],
    }


def fit_water_plane(surface):
    """Plane through the water surface points, normal pointing down (-z).

    Fitted to a fixed random sample of PLANE_FIT_POINTS points.
    """
    if len(surface) < 3:
        print("Too few water surface points for the plane fit")
        return None
    sample = surface
    if len(surface) > PLANE_FIT_POINTS:
        rng = np.random.default_rng(42)
        sample = surface[rng.choice(len(surface), PLANE_FIT_POINTS, replace=False)]

    centroid = sample.mean(axis=0)
    _, S, Vh = np.linalg.svd(sample - centroid, full_matrices=False)
    normal = Vh[2]
    if normal[2] > 0:
        normal = -normal
    offset = -np.dot(normal, centroid)

    residuals = np.dot(surface, normal) + offset
    print(f"\nWater surface plane from {len(sample):,} of {len(surface):,} points")
    print(f"  normal [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  planarity S2/S1 {S[2] / S[1] if S[1] > 0 else 0:.6f}")
    print(
        f"  residuals std {residuals.std():.4f}, range {residuals.min():.4f} to "
        f"{residuals.max():.4f}"
    )
    return normal, offset


def align_water_surfaces(views_data, normal, offset):
    """Shift each view's water points along the plane normal.

    Afterwards the median of the view's water surface points lies on the plane.
    """
    for i, data in enumerate(views_data):
        if data is None or "water_surface" not in data:
            continue
        surface = data["water_surface"]
        if len(surface) == 0:
            print(f"  View {i}: no water surface points, not aligned")
            continue
        median = np.median(np.dot(surface, normal) + offset)
        shift = -median * normal
        print(f"  View {i}: shifted by {-median:+.4f} along the normal")
        if len(data["water_points"]) > 0:
            data["water_points"] = data["water_points"] + shift
        data["water_surface"] = surface + shift


def nerfrac_pointcloud(args, umeyama, output_dir):
    testset_dir = os.path.join(args.log_folder, args.testset)
    if not os.path.isdir(testset_dir):
        print(f"ERROR: Render folder not found: {testset_dir}")
        sys.exit(1)
    meta = load_nerfrac_coord_metadata(args.log_folder, data_dir=args.data_dir or None)
    if meta["c2w_recenter"] is None or meta["sc"] is None:
        print(
            f"ERROR: Could not load c2w.npy and llff_scale.npy from {args.log_folder}"
        )
        sys.exit(1)

    views = discover_nerfrac_views(testset_dir, args.depth_type)
    if not views:
        print(f"ERROR: No NeRFrac views found in {testset_dir}")
        sys.exit(1)
    for view in views:
        images = [
            f
            for f in glob.glob(os.path.join(testset_dir, f"*{view['index']:03d}.png"))
            if not any(
                x in os.path.basename(f)
                for x in ["masked", "rgba", "err", "bathymetry_view"]
            )
        ]
        view["image"] = images[0] if images else None

    H, W, focal = load_focal_from_poses_bounds(args.poses_bounds, args.factor)
    # poses_bounds gives H // factor (5460 // 8 = 682 for Pielach), but the rendered
    # images are as tall as images_<factor> (683), so take H from the rendered pixels
    first = next((v for v in views if v["xs"] is not None), None)
    if first is not None:
        n_pixels = np.load(first["xs"]).shape[0]
        if n_pixels % W == 0:
            H = n_pixels // W
        else:
            print(f"WARNING: {n_pixels:,} rendered pixels are not a multiple of W={W}")
    print(
        f"{len(views)} views at {W}x{H} px, focal {focal:.4f} px, factor "
        f"{args.factor}, {args.depth_type} depth"
    )

    masks = [None] * len(views)
    if args.mask_dir:
        masks = [
            load_mask(path, H, W).reshape(-1)
            for path in list_masks(args.mask_dir, len(views))
        ]

    ndc = {
        "H": H,
        "W": W,
        "focal": focal,
        "c2w_recenter": meta["c2w_recenter"],
        "sc": meta["sc"],
    }
    views_data = []
    for view, mask in zip(views, masks):
        if view["xs"] is None or view["rays"] is None or view["depth"] is None:
            print(f"  View {view['index']}: missing files, skipped")
            views_data.append(None)
        else:
            views_data.append(nerfrac_view(view, ndc, (H, W), mask))

    if args.mask_dir and not args.no_align:
        surfaces = [
            d["water_surface"]
            for d in views_data
            if d is not None and len(d["water_surface"]) > 0
        ]
        plane = fit_water_plane(np.vstack(surfaces)) if surfaces else None
        if plane is not None:
            align_water_surfaces(views_data, *plane)

    parts = {"combined": ([], []), "water": ([], []), "land": ([], [])}
    for i, data in enumerate(views_data):
        if data is None:
            continue
        if args.mask_dir:
            view_parts = []
            for name in ("water", "land"):
                points, colors = data[f"{name}_points"], data[f"{name}_colors"]
                if not args.no_filter and len(points) > 0:
                    points, colors, _ = filter_outliers(
                        points, colors, args.outlier_percentile
                    )
                if len(points) > 0:
                    parts[name][0].append(points)
                    parts[name][1].append(colors)
                    view_parts.append((points, colors))
            if not view_parts:
                continue
            points = np.vstack([p for p, _ in view_parts])
            colors = np.vstack([c for _, c in view_parts])
        else:
            points, colors = data["points"], data["colors"]
            if not args.no_filter:
                points, colors, _ = filter_outliers(
                    points, colors, args.outlier_percentile
                )
            if len(points) == 0:
                continue
        parts["combined"][0].append(points)
        parts["combined"][1].append(colors)
        if args.per_view_ply:
            write_ply_binary(
                os.path.join(output_dir, f"pointcloud_view_{i:03d}.ply"), points, colors
            )

    if not parts["combined"][0]:
        print("ERROR: No valid points in any view")
        sys.exit(1)
    clouds = {}
    for name, (points, colors) in parts.items():
        if points:
            clouds[name] = finish_cloud(points, colors, 0.5, args, umeyama)
    return clouds


def save_preview(points, colors, path, label):
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
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    args = parse_args()
    if args.mode == "vggt" and not os.path.isdir(args.vggt_dir):
        print(f"ERROR: --vggt_dir not found: {args.vggt_dir!r}")
        sys.exit(1)
    if args.mode == "nerfrac" and not (
        args.log_folder and args.testset and args.poses_bounds
    ):
        print("ERROR: --mode nerfrac needs --log_folder, --testset and --poses_bounds")
        sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    elif args.mode == "vggt":
        output_dir = args.vggt_dir
    else:
        output_dir = os.path.join(args.log_folder, args.testset)
    os.makedirs(output_dir, exist_ok=True)

    umeyama = None
    if args.umeyama_npz:
        umeyama = load_umeyama(args.umeyama_npz)
        print(f"Umeyama scale {umeyama['scale']:.4f} from {args.umeyama_npz}")

    if args.mode == "vggt":
        clouds = vggt_pointcloud(args, umeyama)
    else:
        clouds = nerfrac_pointcloud(args, umeyama, output_dir)

    for name, (points, colors) in clouds.items():
        ply_path = os.path.join(output_dir, f"pointcloud_{name}.ply")
        write_ply_binary(ply_path, points, colors)
        np.save(os.path.join(output_dir, f"pointcloud_{name}.npy"), points)
        print(f"Saved: {ply_path} and .npy ({len(points):,} points)")

    points, colors = clouds["combined"]
    for axis, name in enumerate(["X", "Y", "Z"]):
        print(f"  {name} {points[:, axis].min():.4f} to {points[:, axis].max():.4f}")
    if not args.no_preview:
        if args.umeyama_npz:
            label = "(UTM33)"
        else:
            label = "(COLMAP)" if args.mode == "nerfrac" else ""
        save_preview(
            points, colors, os.path.join(output_dir, "pointcloud_preview.png"), label
        )


if __name__ == "__main__":
    main()
