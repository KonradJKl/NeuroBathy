"""
Build the data of the 3D viewer for one of its models: the point cloud, the depth grid
with its errors against the LiDAR and the survey points of the model, and the riverbed
of the classified LiDAR of the dataset, which the models share. MODELS lists the
models; run the script once for each.

Writes to --output_dir, with the file names that MODELS gives the model:
    two levels              the cloud, thinned to one point per voxel. Level 0 is an
                            even preview, level 1 holds the other points.
    <model>_grid.bin        the depth grid of 02, one point per 0.1 m cell
    <model>_error.bin       the depth of each cell of the grid minus the LiDAR depth,
                            from 15_lidar_comparison.py
    lidar_bed.bin           the riverbed of the LiDAR, one point per 0.1 m cell
    <model>_replaced.bin    the water points of the levels that the cells of the grid
                            and of the LiDAR replace in the viewer
    a survey file           the survey points in the frame of the levels, with their
                            surveyed and predicted depth
    scene.json              the CRS, the depth scale, the models with their files,
                            counts, origins and steps, the LiDAR, the attribution and
                            the license. The entry of the model is replaced, those of
                            the other models are kept.

Each level is little-endian: count * 3 uint16 positions (x, y, z of each point,
position = origin + q * step), then count * 3 uint8 colors, then count uint8 depth
codes. Code 0 is land or water off the WSM; codes 1-255 map linearly to 0 to
depth_max m below the WSM, deeper and shallower points clipped.

The cell files (grid and LiDAR) are little-endian as well: count * 3 uint16 positions
with their own origin and step, then count uint16 depths below the WSM in mm, not
clipped. The grid holds every cell of the 02 CSV at its utm_easting, utm_northing and
utm_elevation, with its depth_m. The LiDAR holds the 0.1 m cells, counted from the
lower-left corner of the LAS header, that contain river bottom points (class 40): each
at the cell centre and the median elevation of its points (grid_median of
15_lidar_comparison.py), with the depth below the WSM at the cell centre. As in 15,
cells off the WSM or with a depth of 0 or less are left out.

The error file holds one little-endian int16 per cell of the grid, in its order: the
predicted minus the LiDAR depth in mm (error_m of the lidar_comparison.csv of 15) for
the cells that 15 compares, NO_LIDAR for the others, which have no river bottom points
or a LiDAR or predicted depth of 0 or less.

The replaced file holds one uint8 per point of the levels, in their order. Bit 0 is set
for the water points in a 0.1 m cell of the 02 lattice (as 15 builds it for the CSV)
that holds a cell of the grid, bit 1 for those in a cell of the LiDAR. Each riverbed
view of the viewer hides the points that its cells replace and keeps the other water
points, so no holes open where a layer has no cell.

Usage, from the postprocessing folder of NeuroBathy. The best model needs 02 and 03 as
in "Reproducing the best model" of its README, and 15 as in make_figures.py. 07 runs
with --no_align: by default it shifts the water points of each view onto a plane
fitted to the learned water surface, away from the points that 02 measures.
    python 07_make_pointcloud.py --mode nerfrac \\
        --log_folder $log --testset renderonly_all_950000 \\
        --poses_bounds $scene/poses_bounds.npy --factor 8 \\
        --umeyama_npz ../data/transforms/nerfrac_oblique_umeyama.npz \\
        --mask_dir $scene/masks_8 --no_align \\
        --output_dir pointclouds/pielach_oblique

    python /path/to/gh-pages/build/make_viewer_data.py --model oblique9 \\
        --pointcloud_dir pointclouds/pielach_oblique \\
        --grid_csv bathymetry/pielach_oblique.csv \\
        --comparison_csv validation/pielach_oblique/comparison.csv \\
        --lidar_dir validation_lidar/pielach_oblique \\
        --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --postprocessing_dir . \\
        --output_dir /path/to/gh-pages/data

The model on all 280 images needs its 02, 03 and 15, called as in the README of the
page. Its views do not fit in memory at once, so stream_cloud.py builds its cloud,
thinned already, in place of 07:
    python /path/to/gh-pages/build/stream_cloud.py \\
        --log_folder $log --testset testset_950000 \\
        --poses_bounds $scene/poses_bounds.npy --factor 8 \\
        --umeyama_npz ../data/transforms/nerfrac_50m_umeyama.npz \\
        --mask_dir $scene/masks_8 --postprocessing_dir . \\
        --voxel_size 0.4 --output_dir pointclouds/pielach_all280

    python /path/to/gh-pages/build/make_viewer_data.py --model all280 \\
        --pointcloud_dir pointclouds/pielach_all280 --thinned \\
        --preview_voxel_size 0.8 \\
        --grid_csv bathymetry/pielach_all280.csv \\
        --comparison_csv validation/pielach_all280/comparison.csv \\
        --lidar_dir validation_lidar/pielach_all280 \\
        --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --postprocessing_dir . \\
        --output_dir /path/to/gh-pages/data
"""

import argparse
import csv
import importlib.util
import json
import os
import sys

import laspy
import numpy as np

CRS = "EPSG:25833"
ATTRIBUTION = (
    "Derived from the Pielach River research dataset October 2024 by G. Mandlburger, "
    "TU Wien, licensed CC BY 4.0, https://doi.org/10.48436/taz19-r6618"
)
LICENSE = "CC BY 4.0"
# The models of the viewer, in the order of its switch. The first one opens the page,
# and its origin is the frame of the view. home is what the home view of a model
# frames: the river (its water points on the WSM) or its whole cloud.
MODELS = {
    "oblique9": {
        "label": "9 oblique images",
        "home": "river",
        "levels": ("scene_0.bin", "scene_1.bin"),
        "survey": "survey_points.json",
    },
    "all280": {
        "label": "280 nadir and oblique images",
        "home": "cloud",
        "levels": ("all280_scene_0.bin", "all280_scene_1.bin"),
        "survey": "all280_survey_points.json",
    },
}
LIDAR_FILE = "lidar_bed.bin"
MAX_POSITION_CODE = 65535  # uint16
N_DEPTH_STEPS = 254  # codes 1-255
DEPTH_MAX = 2.0  # m, the depth colour range of the figures
RESOLUTION = 0.1  # cell size of the 02 grid and of the LiDAR riverbed, m
RIVER_BOTTOM = 40  # LiDAR class
NO_LIDAR = -32768  # int16 error of the cells that 15 does not compare
LAS_CHUNK = 2_000_000  # points read from the LiDAR at a time
PLY_TYPES = {"double": "<f8", "float": "<f4", "uchar": "u1"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the point cloud, riverbed and survey data of a model of the "
        "3D viewer."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        required=True,
        help="the model to build; MODELS gives its label, its home view and the names "
        "of its files",
    )

    # input
    parser.add_argument(
        "--pointcloud_dir",
        type=str,
        required=True,
        help="output folder of 07_make_pointcloud.py run with --mask_dir, with the "
        "water and land clouds (pointcloud_water and pointcloud_land .npy and .ply), "
        "or of stream_cloud.py",
    )
    parser.add_argument(
        "--grid_csv",
        type=str,
        required=True,
        help="gridded bathymetry CSV of 02_extract_bathymetry.py, the depth grid",
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="comparison.csv of 03_validate.py",
    )
    parser.add_argument(
        "--lidar_dir",
        type=str,
        required=True,
        help="output folder of 15_lidar_comparison.py (underwater mode) for "
        "--grid_csv, with lidar_comparison.csv and lidar_stats.json",
    )
    parser.add_argument(
        "--wsm_path",
        type=str,
        required=True,
        help="water surface model GeoTIFF; depth = WSM elevation - point elevation",
    )
    parser.add_argument(
        "--lidar_laz",
        type=str,
        required=True,
        help="classified LiDAR point cloud (folder 03 of the dataset); its river "
        "bottom (class 40) is the LiDAR riverbed",
    )
    parser.add_argument(
        "--postprocessing_dir",
        type=str,
        required=True,
        help="postprocessing folder of NeuroBathy, for sample_wsm_batch of coord_utils "
        "and the grid functions of 15_lidar_comparison.py",
    )

    # thinning
    parser.add_argument(
        "--thinned",
        action="store_true",
        help="the clouds of --pointcloud_dir are thinned already (stream_cloud.py), "
        "so all their points are kept",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.23,
        help="voxel size in m; the point nearest the center of each voxel is kept "
        "(not with --thinned)",
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


def import_lidar_comparison(postprocessing_dir):
    """15_lidar_comparison.py as a module, loaded from its path since its name starts
    with a digit."""
    path = os.path.join(postprocessing_dir, "15_lidar_comparison.py")
    spec = importlib.util.spec_from_file_location("lidar_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def read_grid(grid_csv):
    """Cells of a 02 CSV: utm_easting, utm_northing and utm_elevation, and depth_m."""
    table = np.genfromtxt(grid_csv, delimiter=",", names=True)
    grid_points = np.column_stack(
        [table["utm_easting"], table["utm_northing"], table["utm_elevation"]]
    )
    depths = table["depth_m"]
    print(
        f"Grid: {len(grid_points):,} cells of {grid_csv}, depth {depths.min():.3f} to "
        f"{depths.max():.3f} m, {np.sum(depths > DEPTH_MAX):,} deeper than "
        f"{DEPTH_MAX} m"
    )
    return grid_points, depths


def replaced_points(points, is_water, grid, cells, lidar_comparison):
    """The water points whose cell of grid (a grid tuple of 15) is one of cells."""
    point_cells, inside = lidar_comparison.cell_ids(points[:, 0], points[:, 1], grid)
    return is_water & inside & np.isin(point_cells, cells)


def write_flags(path, flags):
    flags.tofile(path)
    print(f"Saved: {path} ({len(flags):,} points, {os.path.getsize(path):,} bytes)")


def check_flags(path, flags):
    """Read a written flag file back and compare it with the flags."""
    if not np.array_equal(np.fromfile(path, dtype=np.uint8), flags):
        print(f"ERROR: {path} does not hold the flags")
        sys.exit(1)
    print(f"  read back: {len(flags):,} flags equal")


def read_river_bottom(lidar_laz):
    """x, y, z of the river bottom points of a LAS or LAZ file, read in chunks, and
    the header of the file."""
    parts = []
    with laspy.open(lidar_laz) as reader:
        header = reader.header
        for chunk in reader.chunk_iterator(LAS_CHUNK):
            bottom = np.asarray(chunk.classification) == RIVER_BOTTOM
            parts.append(
                np.column_stack(
                    [np.asarray(chunk[axis])[bottom] for axis in ("x", "y", "z")]
                )
            )
    points = np.vstack(parts)
    print(
        f"LiDAR: {len(points):,} river bottom points (class {RIVER_BOTTOM}) of "
        f"{header.point_count:,}"
    )
    return points, header


def header_grid(header):
    """Grid tuple of 15 over the extent of a LAS header, from its lower-left corner."""
    e0, n0 = header.mins[:2]
    n_cols, n_rows = ((header.maxs[:2] - header.mins[:2]) / RESOLUTION).astype(int) + 1
    return e0, n0, RESOLUTION, n_cols, n_rows


def lidar_riverbed(lidar_laz, wsm_path, lidar_comparison, sample_wsm_batch):
    """The river bottom cells on the WSM with a positive depth, as 15 compares them:
    the grid tuple, the cell ids, the points (cell centre and median elevation) and
    their depths."""
    bottom, header = read_river_bottom(lidar_laz)
    grid = header_grid(header)
    ids, elevations, _ = lidar_comparison.grid_median(*bottom.T, grid)
    e, n = lidar_comparison.cell_centres(ids, grid)
    depths = sample_wsm_batch(wsm_path, e, n) - elevations
    on_wsm = np.isfinite(depths)
    riverbed = on_wsm & (depths > 0)
    print(
        f"LiDAR cells of {RESOLUTION} m from E {grid[0]:.2f}, N {grid[1]:.2f}: "
        f"{len(ids):,}, {on_wsm.sum():,} on the WSM, {riverbed.sum():,} with a depth "
        f"> 0 (up to {depths[riverbed].max():.3f} m, "
        f"{np.sum(depths[riverbed] > DEPTH_MAX):,} deeper than {DEPTH_MAX} m)"
    )
    points = np.column_stack([e, n, elevations])[riverbed]
    return grid, ids[riverbed], points, depths[riverbed]


def write_cells(path, positions, depths_mm):
    with open(path, "wb") as f:
        f.write(positions.astype("<u2").tobytes())
        f.write(depths_mm.astype("<u2").tobytes())
    print(f"Saved: {path} ({len(depths_mm):,} cells, {os.path.getsize(path):,} bytes)")


def check_cells(path, points, depths, origin, step):
    """Decode a written cell file and compare it with the float64 cells and depths."""
    n = len(points)
    data = np.fromfile(path, dtype="<u2")
    if len(data) != 4 * n:
        print(f"ERROR: {path} has {2 * len(data):,} bytes for {n:,} cells")
        sys.exit(1)
    decoded = origin + data[: 3 * n].reshape(n, 3) * step
    error = np.abs(decoded - points).max()
    depth_error = np.abs(data[3 * n :] / 1000 - depths).max()
    # 1e-6 m of slack for the float64 rounding of UTM coordinates and depths
    if error > step / 2 + 1e-6 or depth_error > 0.0005 + 1e-6:
        print(f"ERROR: {path} does not decode to the cells")
        sys.exit(1)
    print(
        f"  decoded: max position error {error * 1000:.3f} mm (step / 2: "
        f"{step * 500:.1f} mm), max depth error {depth_error * 1000:.3f} mm"
    )


def cell_file(output_dir, name, points, depths):
    """Write and check a cell file; returns its entry of scene.json."""
    origin = points.min(axis=0)
    step = position_step(points)
    positions = np.round((points - origin) / step).astype(np.uint16)
    depths_mm = np.round(depths * 1000).astype(np.uint16)
    path = os.path.join(output_dir, name)
    write_cells(path, positions, depths_mm)
    check_cells(path, points, depths, origin, step)
    return {
        "file": name,
        "count": len(points),
        "origin": [float(v) for v in origin],
        "step": step,
        "cell_size": RESOLUTION,
    }


def grid_errors(lidar_dir, grid, cells, lidar_comparison):
    """Predicted minus LiDAR depth in mm of each cell of the 02 grid (cells, the ids of
    its rows on the grid tuple of 15), NO_LIDAR where 15 compared none, and the stats of
    15."""
    table = np.genfromtxt(
        os.path.join(lidar_dir, "lidar_comparison.csv"), delimiter=",", names=True
    )
    with open(os.path.join(lidar_dir, "lidar_stats.json")) as f:
        stats = json.load(f)
    compared, inside = lidar_comparison.cell_ids(
        table["utm_easting"], table["utm_northing"], grid
    )
    order = np.argsort(cells)
    found = np.minimum(np.searchsorted(cells, compared, sorter=order), len(cells) - 1)
    rows = order[found]
    if not (
        inside.all()
        and np.array_equal(cells[rows], compared)
        and len(np.unique(rows)) == len(rows)
    ):
        print(
            f"ERROR: The cells of lidar_comparison.csv in {lidar_dir} are not cells "
            "of the grid"
        )
        sys.exit(1)
    errors = np.full(len(cells), NO_LIDAR, dtype=np.int16)
    errors[rows] = np.round(table["error_m"] * 1000).astype(np.int16)
    print(
        f"Errors against the LiDAR: {len(rows):,} of {len(cells):,} cells, RMSE "
        f"{stats['rmse']:.4f} m, bias {stats['bias']:+.4f} m"
    )
    return errors, stats


def check_errors(path, errors, stats):
    """Read a written error file back and compare it with the errors and with the count,
    RMSE and bias of 15."""
    data = np.fromfile(path, dtype="<i2")
    compared = data[data != NO_LIDAR] / 1000
    rmse = np.sqrt(np.mean(compared**2))
    bias = np.mean(compared)
    # 0.5 mm of slack for the rounding to whole millimetres
    if not (
        np.array_equal(data, errors)
        and len(compared) == stats["n"]
        and abs(rmse - stats["rmse"]) <= 0.0005
        and abs(bias - stats["bias"]) <= 0.0005
    ):
        print(f"ERROR: {path} does not hold the errors of 15")
        sys.exit(1)
    print(
        f"  read back: {len(compared):,} errors, RMSE {rmse:.4f} m, bias {bias:+.4f} m"
    )


def error_file(output_dir, name, errors, stats):
    """Write and check an error file; returns its entry of scene.json."""
    path = os.path.join(output_dir, name)
    errors.astype("<i2").tofile(path)
    print(f"Saved: {path} ({len(errors):,} cells, {os.path.getsize(path):,} bytes)")
    check_errors(path, errors, stats)
    return {"file": name, "rmse": stats["rmse"]}


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


def write_scene(path, model, lidar_layer):
    """Write scene.json with the model in place of its old entry, the other models of
    the file and the LiDAR; the models in the order of MODELS."""
    models = {}
    if os.path.exists(path):
        with open(path) as f:
            models = {entry["id"]: entry for entry in json.load(f)["models"]}
    models[model["id"]] = model
    scene = {
        "crs": CRS,
        "depth_max": DEPTH_MAX,
        "models": [models[name] for name in MODELS if name in models],
        "lidar": lidar_layer,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
    }
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
        f.write("\n")
    names = ", ".join(entry["id"] for entry in scene["models"])
    print(f"Saved: {path} (models {names})")


def main():
    args = parse_args()
    sys.path.insert(0, args.postprocessing_dir)
    from coord_utils import sample_wsm_batch

    lidar_comparison = import_lidar_comparison(args.postprocessing_dir)
    spec = MODELS[args.model]
    print(f"Model {args.model}: {spec['label']}")

    water, water_colors = load_cloud(args.pointcloud_dir, "water")
    land, land_colors = load_cloud(args.pointcloud_dir, "land")
    wsm_elevations = sample_wsm_batch(args.wsm_path, water[:, 0], water[:, 1])
    water_depths = wsm_elevations - water[:, 2]
    report_depths(water_depths[np.isfinite(water_depths)])

    # land points have no depth
    points = np.vstack([water, land])
    colors = np.vstack([water_colors, land_colors])
    depths = np.concatenate([water_depths, np.full(len(land), np.nan)])

    if args.thinned:
        keep = np.arange(len(points))
    else:
        keep = voxel_pick(points, args.voxel_size)
        print(f"Thinned with {args.voxel_size} m voxels to {len(keep):,} points")
    points, colors, depths = points[keep], colors[keep], depths[keep]
    is_water = keep < len(water)
    in_preview = np.zeros(len(points), dtype=bool)
    in_preview[voxel_pick(points, args.preview_voxel_size)] = True
    print(
        f"{len(points):,} points ({np.isfinite(depths).sum():,} water points on the "
        f"WSM), {in_preview.sum():,} in the preview ({args.preview_voxel_size} m voxels)"
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
    parts = (in_preview, ~in_preview)
    levels = []
    for name, part in zip(spec["levels"], parts):
        path = os.path.join(args.output_dir, name)
        write_level(path, positions[part], colors[part], codes[part])
        check_level(path, points[part], colors[part], codes[part], origin, step)
        levels.append({"file": name, "count": int(part.sum())})

    # the depth grid, its cells on the grid that 15 builds for the 02 CSV, and their
    # errors against the LiDAR
    grid_points, grid_depths = read_grid(args.grid_csv)
    grid_file = f"{args.model}_grid.bin"
    grid_layer = cell_file(args.output_dir, grid_file, grid_points, grid_depths)
    pred_grid, pred = lidar_comparison.load_prediction(
        args.grid_csv, "", "underwater", RESOLUTION
    )
    errors, stats = grid_errors(
        args.lidar_dir, pred_grid, pred["cells"], lidar_comparison
    )
    error_layer = error_file(args.output_dir, f"{args.model}_error.bin", errors, stats)

    bed_grid, bed_ids, bed, bed_depths = lidar_riverbed(
        args.lidar_laz, args.wsm_path, lidar_comparison, sample_wsm_batch
    )
    lidar_layer = cell_file(args.output_dir, LIDAR_FILE, bed, bed_depths)

    # bit 0: a cell of the grid replaces the water point, bit 1: a cell of the LiDAR
    by_grid = replaced_points(
        points, is_water, pred_grid, pred["cells"], lidar_comparison
    )
    by_lidar = replaced_points(points, is_water, bed_grid, bed_ids, lidar_comparison)
    print(
        f"Water points replaced by the grid: {by_grid.sum():,}, by the LiDAR: "
        f"{by_lidar.sum():,}, of {is_water.sum():,}"
    )
    flags = (by_grid + 2 * by_lidar).astype(np.uint8)
    replaced_file = f"{args.model}_replaced.bin"
    path = os.path.join(args.output_dir, replaced_file)
    in_levels = np.concatenate([flags[part] for part in parts])
    write_flags(path, in_levels)
    check_flags(path, in_levels)

    survey = survey_points(args.comparison_csv, origin)
    path = os.path.join(args.output_dir, spec["survey"])
    with open(path, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(p) for p in survey) + "\n]\n")
    print(f"Saved: {path}")

    model = {
        "id": args.model,
        "label": spec["label"],
        "home": spec["home"],
        "origin": [float(v) for v in origin],
        "step": step,
        "levels": levels,
        "replaced": replaced_file,
        "grid": grid_layer,
        "error": error_layer,
        "survey": spec["survey"],
    }
    write_scene(os.path.join(args.output_dir, "scene.json"), model, lidar_layer)


if __name__ == "__main__":
    main()
