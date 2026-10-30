"""
Build the scene cloud of a NeRFrac render folder one view at a time and thin it as
make_viewer_data.py does. 07_make_pointcloud.py holds all views in memory at once, which
does not fit for large models.

The cloud is the one of 07 run with --mask_dir --no_align and its default mode depth,
made with the functions of 07 in three passes over the views, each holding one view at
a time:
    1. 07's filter of each view, and the lowest and highest values of each axis of the
       water and of the land points, for the thresholds of 07's final filter
    2. both filters and the georeferencing, for the minimum and maximum of the water
       and land points together
    3. the thinning of make_viewer_data.py: one point per voxel, the point nearest the
       voxel centre, on a lattice from that minimum, in chunks of views

Writes to --output_dir, in the format of 07:
    pointcloud_water.npy, .ply   the thinned water points (float64) and their colors
    pointcloud_land.npy, .ply    the thinned land points
    stream_cloud_counts.txt      the point counts after each step

The points of each cloud are in voxel order. They equal the points that the thinning of
make_viewer_data.py keeps of the clouds of 07, and make_viewer_data.py reads them with
--thinned.

Usage, from the postprocessing folder of NeuroBathy:
    python /path/to/gh-pages/build/stream_cloud.py \\
        --log_folder $log --testset testset_950000 \\
        --poses_bounds $scene/poses_bounds.npy --factor 8 \\
        --umeyama_npz ../data/transforms/nerfrac_50m_umeyama.npz \\
        --mask_dir $scene/masks_8 --postprocessing_dir . \\
        --voxel_size 0.4 --output_dir pointclouds/pielach_all280
"""

import argparse
import contextlib
import glob
import importlib.util
import io
import os
import sys
import time

import numpy as np

PARTS = ("water", "land")
DEPTH_TYPE = "mode"  # the default of 07
VIEW_PERCENTILE = 1.0  # the default --outlier_percentile of 07, for each view
FINAL_PERCENTILE = 0.5  # 07's filter of the combined clouds
CHUNK_POINTS = 5_000_000  # points thinned at a time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the thinned water and land clouds of a NeRFrac render "
        "folder view by view."
    )

    # input
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="experiment folder with c2w.npy and llff_scale.npy",
    )
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="render folder inside --log_folder, e.g. testset_950000",
    )
    parser.add_argument(
        "--poses_bounds",
        type=str,
        required=True,
        help="poses_bounds.npy of the scene, for the image size and focal",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="image downsample factor of the training",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="water masks, paired with the views by sorted file name",
    )
    parser.add_argument(
        "--umeyama_npz",
        type=str,
        required=True,
        help="transform from 01_compute_umeyama.py",
    )
    parser.add_argument(
        "--postprocessing_dir",
        type=str,
        required=True,
        help="postprocessing folder of NeuroBathy, for 07_make_pointcloud.py and "
        "coord_utils",
    )

    # thinning
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.23,
        help="voxel size in m; the point nearest the center of each voxel is kept",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output folder",
    )
    return parser.parse_args()


def import_make_pointcloud(postprocessing_dir):
    """07_make_pointcloud.py as a module, loaded from its path since its name starts
    with a digit."""
    path = os.path.join(postprocessing_dir, "07_make_pointcloud.py")
    spec = importlib.util.spec_from_file_location("make_pointcloud", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_renders(args, make_pointcloud):
    """The views, masks and NDC parameters of the render folder, as 07 finds them."""
    from coord_utils import (
        discover_nerfrac_views,
        load_focal_from_poses_bounds,
        load_nerfrac_coord_metadata,
    )

    testset_dir = os.path.join(args.log_folder, args.testset)
    if not os.path.isdir(testset_dir):
        print(f"ERROR: Render folder not found: {testset_dir}")
        sys.exit(1)
    meta = load_nerfrac_coord_metadata(args.log_folder)
    if meta["c2w_recenter"] is None or meta["sc"] is None:
        print(
            f"ERROR: Could not load c2w.npy and llff_scale.npy from {args.log_folder}"
        )
        sys.exit(1)

    views = discover_nerfrac_views(testset_dir, DEPTH_TYPE)
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

    # H from the rendered pixels, as 07 takes it
    H, W, focal = load_focal_from_poses_bounds(args.poses_bounds, args.factor)
    first = next((v for v in views if v["xs"] is not None), None)
    if first is not None:
        n_pixels = np.load(first["xs"], mmap_mode="r").shape[0]
        if n_pixels % W == 0:
            H = n_pixels // W
        else:
            print(f"WARNING: {n_pixels:,} rendered pixels are not a multiple of W={W}")
    print(
        f"{len(views)} views at {W}x{H} px, focal {focal:.4f} px, factor "
        f"{args.factor}, {DEPTH_TYPE} depth"
    )

    mask_files = make_pointcloud.list_masks(args.mask_dir, len(views))
    ndc = {
        "H": H,
        "W": W,
        "focal": focal,
        "c2w_recenter": meta["c2w_recenter"],
        "sc": meta["sc"],
    }
    return views, mask_files, ndc


def filtered_views(make_pointcloud, views, mask_files, ndc, quiet):
    """Water and land points and colors of each view after 07's filter of the view, in
    COLMAP coordinates. quiet drops the lines that 07 prints for each view."""
    shape = (ndc["H"], ndc["W"])
    for view, mask_path in zip(views, mask_files):
        if view["xs"] is None or view["rays"] is None or view["depth"] is None:
            if not quiet:
                print(f"  View {view['index']}: missing files, skipped")
            continue
        mask = make_pointcloud.load_mask(mask_path, *shape).reshape(-1)
        silence = contextlib.redirect_stdout(io.StringIO())
        with silence if quiet else contextlib.nullcontext():
            data = make_pointcloud.nerfrac_view(view, ndc, shape, mask)
        parts = {}
        for name in PARTS:
            points, colors = data[f"{name}_points"], data[f"{name}_colors"]
            if len(points) > 0:
                points, colors, _ = make_pointcloud.filter_outliers(
                    points, colors, VIEW_PERCENTILE
                )
            parts[name] = (points, colors)
        yield parts


def update_tails(tails, points, n_tail):
    """Keep the n_tail lowest and highest values of each axis of the points so far."""
    for axis in range(3):
        low = np.concatenate([tails["low"][axis], points[:, axis]])
        high = np.concatenate([tails["high"][axis], points[:, axis]])
        if len(low) > n_tail:
            low = np.partition(low, n_tail - 1)[:n_tail]
            high = np.partition(high, len(high) - n_tail)[-n_tail:]
        tails["low"][axis], tails["high"][axis] = low, high
    tails["count"] += len(points)


def percentile_of_run(run, first_rank, n, percentile):
    """np.percentile of n values, from a sorted run of them that starts at rank
    first_rank.

    The arithmetic of the linear method of numpy, so the result is the threshold of 07.
    """
    index = (n - 1) * np.true_divide(percentile, 100)
    below = int(np.floor(index))
    if below < first_rank or below + 1 >= first_rank + len(run):
        print(f"ERROR: rank {below} of {n:,} values is not in the run")
        sys.exit(1)
    a, b = run[below - first_rank], run[below + 1 - first_rank]
    gamma = index - below
    diff = b - a
    if gamma >= 0.5:
        return b - diff * (1 - gamma)
    return a + diff * gamma


def final_thresholds(tails):
    """Lower and upper bounds of each axis of 07's final filter of one cloud."""
    n = tails["count"]
    lo, hi = np.empty(3), np.empty(3)
    for axis in range(3):
        low, high = np.sort(tails["low"][axis]), np.sort(tails["high"][axis])
        lo[axis] = percentile_of_run(low, 0, n, FINAL_PERCENTILE)
        hi[axis] = percentile_of_run(high, n - len(high), n, 100 - FINAL_PERCENTILE)
    return lo, hi


def scan_thresholds(scan, n_pixels):
    """Pass 1: the thresholds of 07's final filter of the water and the land cloud."""
    # the needed ranks lie within the lowest and highest n_tail of any count up to
    # n_pixels
    n_tail = int(np.ceil(n_pixels * FINAL_PERCENTILE / 100)) + 2
    tails = {
        name: {"low": [np.empty(0)] * 3, "high": [np.empty(0)] * 3, "count": 0}
        for name in PARTS
    }
    for parts in scan(quiet=False):
        for name in PARTS:
            update_tails(tails[name], parts[name][0], n_tail)
    thresholds = {name: final_thresholds(tails[name]) for name in PARTS}
    counts = {name: tails[name]["count"] for name in PARTS}
    return thresholds, counts


def final_views(scan, make_pointcloud, thresholds, umeyama):
    """Water and land points and colors of each view after both filters of 07,
    georeferenced."""
    for parts in scan(quiet=True):
        result = {}
        for name in PARTS:
            points, colors = parts[name]
            lo, hi = thresholds[name]
            keep = np.all((points >= lo) & (points <= hi), axis=1)
            result[name] = (
                make_pointcloud.georeference(points[keep], umeyama),
                colors[keep],
            )
        yield result


def scan_bounds(views):
    """Pass 2: minimum and maximum of the water and land points together, and the
    count of each cloud."""
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    counts = dict.fromkeys(PARTS, 0)
    for parts in views:
        for name in PARTS:
            points = parts[name][0]
            if len(points) > 0:
                lower = np.minimum(lower, points.min(axis=0))
                upper = np.maximum(upper, points.max(axis=0))
            counts[name] += len(points)
    return lower, upper, counts


def voxel_keys(points, origin, dims, voxel_size):
    """Voxel ids and squared distances to the voxel centres, as voxel_pick of
    make_viewer_data.py computes them."""
    scaled = (points - origin) / voxel_size
    cells = np.floor(scaled).astype(np.int64)
    voxel_ids = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    dist2 = np.sum((scaled - cells - 0.5) ** 2, axis=1)
    return voxel_ids, dist2


def thin(kept, chunk):
    """One point per voxel of kept and chunk together, in voxel order.

    Ties in the distance go to the lower rank, the position in the water and land
    clouds stacked, as with voxel_pick of make_viewer_data.py.
    """
    merged = {k: np.concatenate([kept[k], chunk[k]]) for k in chunk}
    order = np.lexsort((merged["rank"], merged["dist2"], merged["voxel_id"]))
    sorted_ids = merged["voxel_id"][order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    return {k: v[order[first]] for k, v in merged.items()}


def thin_views(views, origin, upper, n_water, voxel_size):
    """Pass 3: thin the water and land points in chunks of views."""
    dims = np.floor((upper - origin) / voxel_size).astype(np.int64) + 1
    kept = {
        "voxel_id": np.empty(0, np.int64),
        "dist2": np.empty(0),
        "rank": np.empty(0, np.int64),
        "points": np.empty((0, 3)),
        "colors": np.empty((0, 3), np.uint8),
    }
    next_rank = {"water": 0, "land": n_water}
    chunk, n_chunk = [], 0
    for parts in views:
        for name in PARTS:
            points, colors = parts[name]
            voxel_ids, dist2 = voxel_keys(points, origin, dims, voxel_size)
            rank = next_rank[name] + np.arange(len(points))
            next_rank[name] += len(points)
            chunk.append(
                {
                    "voxel_id": voxel_ids,
                    "dist2": dist2,
                    "rank": rank,
                    "points": points,
                    "colors": colors,
                }
            )
            n_chunk += len(points)
        if n_chunk >= CHUNK_POINTS:
            kept = thin(kept, {k: np.concatenate([c[k] for c in chunk]) for k in kept})
            chunk, n_chunk = [], 0
    if chunk:
        kept = thin(kept, {k: np.concatenate([c[k] for c in chunk]) for k in kept})
    is_water = kept["rank"] < n_water
    return {
        "water": (kept["points"][is_water], kept["colors"][is_water]),
        "land": (kept["points"][~is_water], kept["colors"][~is_water]),
    }


def main():
    args = parse_args()
    sys.path.insert(0, args.postprocessing_dir)
    from coord_utils import load_umeyama, write_ply_binary

    make_pointcloud = import_make_pointcloud(args.postprocessing_dir)
    umeyama = load_umeyama(args.umeyama_npz)
    print(f"Umeyama scale {umeyama['scale']:.4f} from {args.umeyama_npz}")
    views, mask_files, ndc = open_renders(args, make_pointcloud)
    n_pixels = len(views) * ndc["H"] * ndc["W"]

    def scan(quiet):
        return filtered_views(make_pointcloud, views, mask_files, ndc, quiet)

    t0 = time.time()
    print("\nPass 1: filter of each view, thresholds of the final filter")
    thresholds, view_counts = scan_thresholds(scan, n_pixels)
    for name in PARTS:
        lo, hi = thresholds[name]
        print(
            f"  {name}: {view_counts[name]:,} points after the filter of each view, "
            f"final filter from {np.array2string(lo, precision=6)} to "
            f"{np.array2string(hi, precision=6)} ({time.time() - t0:.0f} s)"
        )

    print("\nPass 2: final filter and georeferencing, bounds of the cloud")
    origin, upper, final_counts = scan_bounds(
        final_views(scan, make_pointcloud, thresholds, umeyama)
    )
    for name in PARTS:
        removed = view_counts[name] - final_counts[name]
        print(
            f"  {name}: removed {removed:,} outliers, {final_counts[name]:,} points "
            f"({time.time() - t0:.0f} s)"
        )
    for axis, name in enumerate(["X", "Y", "Z"]):
        print(f"  {name} {origin[axis]:.4f} to {upper[axis]:.4f}")

    print(f"\nPass 3: thinning with {args.voxel_size} m voxels")
    clouds = thin_views(
        final_views(scan, make_pointcloud, thresholds, umeyama),
        origin,
        upper,
        final_counts["water"],
        args.voxel_size,
    )
    print(
        f"  {len(clouds['water'][0]):,} water and {len(clouds['land'][0]):,} land "
        f"points ({time.time() - t0:.0f} s)"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for name, (points, colors) in clouds.items():
        ply_path = os.path.join(args.output_dir, f"pointcloud_{name}.ply")
        write_ply_binary(ply_path, points, colors)
        np.save(os.path.join(args.output_dir, f"pointcloud_{name}.npy"), points)
        print(f"Saved: {ply_path} and .npy ({len(points):,} points)")

    path = os.path.join(args.output_dir, "stream_cloud_counts.txt")
    with open(path, "w") as f:
        f.write(f"render folder {os.path.join(args.log_folder, args.testset)}\n")
        f.write(f"views {len(views)}, {DEPTH_TYPE} depth\n")
        f.write(
            "cloud, after the filter of each view, after the final filter, thinned\n"
        )
        for name in PARTS:
            f.write(
                f"{name}, {view_counts[name]}, {final_counts[name]}, "
                f"{len(clouds[name][0])}\n"
            )
        f.write(f"voxel size {args.voxel_size} m\n")
        f.write(f"minimum {' '.join(f'{v:.4f}' for v in origin)}\n")
        f.write(f"maximum {' '.join(f'{v:.4f}' for v in upper)}\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
