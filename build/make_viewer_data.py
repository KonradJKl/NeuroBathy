"""
Build the data of the 3D viewer from the scene point cloud of the best model and the
validation of 03_validate.py.

Writes to --output_dir:
    scene_0.bin, scene_1.bin   the cloud, thinned to one point per voxel. Level 0 is
                               an even preview, level 1 holds the other points.
    scene.json                 origin, step and depth scale of the .bin files
    survey_points.json         the survey points in the frame of scene.json, with
                               their surveyed and predicted depth

Each .bin file is little-endian: count * 3 uint16 positions (x, y, z of each point,
position = origin + q * step), then count * 3 uint8 colors, then count uint8 depth
codes. Code 0 is land or water off the WSM; codes 1-255 map linearly to 0 to
depth_max m below the WSM, deeper and shallower points clipped.

Usage, from the postprocessing folder of NeuroBathy after 02 and 03 as in
"Reproducing the best model" of its README. 07 runs with --no_align: by default it
shifts the water points of each view onto a plane fitted to the learned water
surface, away from the points that 02 measures.
    python 07_make_pointcloud.py --mode nerfrac \\
        --log_folder $log --testset renderonly_all_950000 \\
        --poses_bounds $scene/poses_bounds.npy --factor 8 \\
        --umeyama_npz ../data/transforms/nerfrac_oblique_umeyama.npz \\
        --mask_dir $scene/masks_8 --no_align \\
        --output_dir pointclouds/pielach_oblique

    python /path/to/gh-pages/build/make_viewer_data.py \\
        --pointcloud_dir pointclouds/pielach_oblique \\
        --comparison_csv validation/pielach_oblique/comparison.csv \\
        --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --postprocessing_dir . \\
        --output_dir /path/to/gh-pages/data
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

CRS = "EPSG:25833"
ATTRIBUTION = (
    "Derived from the Pielach River research dataset October 2024 by G. Mandlburger, "
    "TU Wien, licensed CC BY 4.0, https://doi.org/10.48436/taz19-r6618"
)
LICENSE = "CC BY 4.0"
SURVEY_FILE = "survey_points.json"
MAX_POSITION_CODE = 65535  # uint16
N_DEPTH_STEPS = 254  # codes 1-255
DEPTH_MAX = 2.0  # m, the depth colour range of the figures
PLY_TYPES = {"double": "<f8", "float": "<f4", "uchar": "u1"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the point cloud and survey data of the 3D viewer."
    )

    # input
    parser.add_argument(
        "--pointcloud_dir",
        type=str,
        required=True,
        help="output folder of 07_make_pointcloud.py run with --mask_dir, with the "
        "water and land clouds (pointcloud_water and pointcloud_land .npy and .ply)",
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="comparison.csv of 03_validate.py",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF; depth = WSM elevation - point elevation",
    )
    parser.add_argument(
        "--postprocessing_dir",
        type=str,
        required=True,
        help="postprocessing folder of NeuroBathy, for sample_wsm_batch of coord_utils",
    )

    # thinning
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.23,
        help="voxel size in m; the point nearest the center of each voxel is kept",
    )
    parser.add_argument(
        "--preview_voxel_size",
        type=float,
        default=0.6,
        help="voxel size in m of level 0, the preview",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output folder, the data folder of the page",
    )
    return parser.parse_args()


def read_ply(path):
    """Vertices of a binary PLY from write_ply_binary of coord_utils."""
    fields, n_vertices = [], 0
    with open(path, "rb") as f:
        line = f.readline()
        while line and line != b"end_header\n":
            words = line.decode("ascii").split()
            if words[0] == "element":
                n_vertices = int(words[2])
            elif words[0] == "property":
                fields.append((words[2], PLY_TYPES[words[1]]))
            line = f.readline()
        return np.fromfile(f, dtype=fields, count=n_vertices)


def load_cloud(pointcloud_dir, name):
    """Points of a 07 cloud from its .npy (float64) and colors from its .ply.

    The .ply stores z as float32, so only x and y are compared with the .npy.
    """
    points = np.load(os.path.join(pointcloud_dir, f"pointcloud_{name}.npy"))
    vertices = read_ply(os.path.join(pointcloud_dir, f"pointcloud_{name}.ply"))
    if not (
        np.array_equal(vertices["x"], points[:, 0])
        and np.array_equal(vertices["y"], points[:, 1])
    ):
        print(f"ERROR: pointcloud_{name}.ply and .npy hold different points")
        sys.exit(1)
    colors = np.stack([vertices[c] for c in ("red", "green", "blue")], axis=1)
    print(f"{name}: {len(points):,} points")
    return points, colors


def report_depths(depths):
    """Print how the depths of the water points on the WSM fit the colour range."""
    print(
        f"Depth of {len(depths):,} water points on the WSM: median "
        f"{np.median(depths):.3f} m, {np.mean(depths > DEPTH_MAX):.1%} deeper than "
        f"{DEPTH_MAX} m, {np.mean(depths < 0):.1%} above the WSM"
    )


def voxel_pick(points, voxel_size):
    """Index of the point nearest the center of each occupied voxel, in voxel order."""
    scaled = (points - points.min(axis=0)) / voxel_size
    cells = np.floor(scaled).astype(np.int64)
    dims = cells.max(axis=0) + 1
    voxel_ids = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    dist2 = np.sum((scaled - cells - 0.5) ** 2, axis=1)
    order = np.lexsort((dist2, voxel_ids))
    sorted_ids = voxel_ids[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    return order[first]


def position_step(points):
    """Smallest multiple of 1 mm that fits the largest extent into 0..65535."""
    extent = np.ptp(points, axis=0).max()
    return int(np.ceil(extent * 1000 / MAX_POSITION_CODE)) / 1000


def depth_codes(depths, depth_max):
    codes = np.zeros(len(depths), dtype=np.uint8)
    on_wsm = np.isfinite(depths)
    clipped = np.clip(depths[on_wsm], 0, depth_max)
    codes[on_wsm] = 1 + np.round(clipped / depth_max * N_DEPTH_STEPS).astype(np.uint8)
    return codes


def write_level(path, positions, colors, codes):
    with open(path, "wb") as f:
        f.write(positions.astype("<u2").tobytes())
        f.write(colors.astype(np.uint8).tobytes())
        f.write(codes.tobytes())
    print(f"Saved: {path} ({len(codes):,} points, {os.path.getsize(path):,} bytes)")


def check_level(path, points, colors, codes, origin, step):
    """Decode a written level and compare it with the float64 points."""
    n = len(points)
    data = np.fromfile(path, dtype=np.uint8)
    if len(data) != 10 * n:
        print(f"ERROR: {path} has {len(data):,} bytes for {n:,} points")
        sys.exit(1)
    decoded = origin + data[: 6 * n].view("<u2").reshape(n, 3) * step
    error = np.abs(decoded - points).max()
    same = np.array_equal(data[6 * n : 9 * n].reshape(n, 3), colors) and (
        np.array_equal(data[9 * n :], codes)
    )
    # 1e-6 m of slack for the float64 rounding of UTM coordinates
    if error > step / 2 + 1e-6 or not same:
        print(f"ERROR: {path} does not decode to the points")
        sys.exit(1)
    print(
        f"  decoded: max position error {error * 1000:.3f} mm (step / 2: "
        f"{step * 500:.1f} mm), colors and depth codes equal"
    )


def float_or_none(field):
    return float(field) if field else None


def survey_points(comparison_csv, origin):
    """GT points of comparison.csv, moved to the origin of the cloud.

    z is the surveyed bed elevation; depth_pred is None for unmatched points.
    """
    points = []
    with open(comparison_csv, newline="") as f:
        for row in csv.DictReader(f):
            kind = "target" if row["type"] == "reference_target" else "transect"
            points.append(
                {
                    "name": row["name"],
                    "kind": kind,
                    "x": round(float(row["gt_easting"]) - origin[0], 4),
                    "y": round(float(row["gt_northing"]) - origin[1], 4),
                    "z": round(float(row["gt_elevation"]) - origin[2], 4),
                    "depth_gt": float_or_none(row["gt_depth_m"]),
                    "depth_pred": float_or_none(row["pred_depth_m"]),
                }
            )
    n_targets = sum(p["kind"] == "target" for p in points)
    n_matched = sum(p["depth_pred"] is not None for p in points)
    print(
        f"Survey: {n_targets} targets, {len(points) - n_targets} transect points, "
        f"{n_matched} with a predicted depth"
    )
    return points


def main():
    args = parse_args()
    sys.path.insert(0, args.postprocessing_dir)
    from coord_utils import sample_wsm_batch

    water, water_colors = load_cloud(args.pointcloud_dir, "water")
    land, land_colors = load_cloud(args.pointcloud_dir, "land")
    wsm_elevations = sample_wsm_batch(args.wsm_path, water[:, 0], water[:, 1])
    water_depths = wsm_elevations - water[:, 2]
    report_depths(water_depths[np.isfinite(water_depths)])

    # land points have no depth
    points = np.vstack([water, land])
    colors = np.vstack([water_colors, land_colors])
    depths = np.concatenate([water_depths, np.full(len(land), np.nan)])

    keep = voxel_pick(points, args.voxel_size)
    points, colors, depths = points[keep], colors[keep], depths[keep]
    in_preview = np.zeros(len(points), dtype=bool)
    in_preview[voxel_pick(points, args.preview_voxel_size)] = True
    print(
        f"Thinned with {args.voxel_size} m voxels to {len(points):,} points "
        f"({np.isfinite(depths).sum():,} water points on the WSM), "
        f"{in_preview.sum():,} in the preview ({args.preview_voxel_size} m voxels)"
    )

    origin = points.min(axis=0)
    step = position_step(points)
    positions = np.round((points - origin) / step).astype(np.uint16)
    codes = depth_codes(depths, DEPTH_MAX)
    extent = np.ptp(points, axis=0)
    print(
        f"Origin E {origin[0]:.2f}, N {origin[1]:.2f}, Z {origin[2]:.2f}, extent "
        f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m, step "
        f"{step * 1000:.0f} mm"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    levels = []
    for i, part in enumerate((in_preview, ~in_preview)):
        name = f"scene_{i}.bin"
        path = os.path.join(args.output_dir, name)
        write_level(path, positions[part], colors[part], codes[part])
        check_level(path, points[part], colors[part], codes[part], origin, step)
        levels.append({"file": name, "count": int(part.sum())})

    survey = survey_points(args.comparison_csv, origin)
    path = os.path.join(args.output_dir, SURVEY_FILE)
    with open(path, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(p) for p in survey) + "\n]\n")
    print(f"Saved: {path}")

    scene = {
        "crs": CRS,
        "origin": [float(v) for v in origin],
        "step": step,
        "levels": levels,
        "depth_max": DEPTH_MAX,
        "survey": SURVEY_FILE,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
    }
    path = os.path.join(args.output_dir, "scene.json")
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
        f.write("\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
