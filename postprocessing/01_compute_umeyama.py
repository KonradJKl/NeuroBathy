"""
Similarity transform (Umeyama) from a model frame to the projected CRS of the ground
truth, fitted to the camera centres and the GPS positions of the same images.

The camera centres come from COLMAP's images.bin (NeRFrac) or from VGGT's extrinsics,
the GPS positions from the EXIF data of the source images or from the position priors
of a COLMAP database. Images are matched by file name without extension, so PNG copies
of the JPGs match. The GPS heights are used as recorded: they must be in the height
system of the ground truth (ellipsoidal for the RTK-tagged Pielach images and ground
truth).

Writes scale, rotation and translation, and the RMS of the horizontal, vertical and 3D
camera residuals, to --output. 02, 07 and 08 read it with --umeyama_npz.

Usage examples:
    # NeRFrac, oblique Pielach images
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/pielach_oblique/sparse/0 \\
        --image_dir /path/to/pielach_oblique/images \\
        --output transforms/nerfrac_oblique_umeyama.npz

    # VGGT, nadir Pielach images
    python 01_compute_umeyama.py \\
        --mode vggt \\
        --vggt_dir /path/to/vggt_nadir \\
        --image_dir /path/to/pielach_data/images \\
        --output transforms/vggt_nadir_umeyama.npz

    # NeRFrac, GPS from the position priors of the COLMAP database
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/pielach_oblique/sparse/0 \\
        --gps_source colmap_db \\
        --database_path /path/to/pielach_oblique/database.db \\
        --output transforms/nerfrac_oblique_umeyama.npz
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError

from coord_utils import (
    compute_umeyama_residuals,
    extract_colmap_camera_centers,
    extract_gps_from_colmap_db,
    extract_vggt_camera_centers_from_extrinsics,
    load_vggt_image_names,
    read_all_exif_gps,
    save_umeyama,
    umeyama_alignment,
    wgs84_dict_to_projected,
)

MIN_MATCHED_IMAGES = 3
# below this ratio of the second to the first spread the cameras lie on a line
MIN_SPREAD_RATIO = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Similarity transform from a model frame to the CRS of the "
        "ground truth."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["nerfrac", "vggt"],
        help="nerfrac: COLMAP camera centres; vggt: VGGT extrinsics",
    )

    # camera centres
    parser.add_argument(
        "--colmap_sparse_dir",
        type=str,
        default="",
        help="COLMAP model folder with images.bin (nerfrac mode)",
    )
    parser.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="output folder of run_vggt_inference.py (vggt mode)",
    )

    # GPS positions
    parser.add_argument(
        "--gps_source",
        choices=["exif", "colmap_db"],
        default="exif",
        help="exif (default): the EXIF GPS of the images in --image_dir; colmap_db: "
        "the position priors in --database_path",
    )
    parser.add_argument(
        "--image_dir", type=str, default="", help="source images with EXIF GPS"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default="",
        help="COLMAP database.db with position priors (opened read-only)",
    )
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=25833,
        help="EPSG code of the projected CRS of the ground truth, in metres "
        "(default: 25833, ETRS89 / UTM 33N, the Pielach data)",
    )

    # output
    parser.add_argument("--output", type=str, required=True, help="output .npz")
    return parser.parse_args()


def check_target_crs(epsg):
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError:
        print(f"ERROR: Unknown EPSG code {epsg}")
        sys.exit(1)
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        print(f"ERROR: EPSG:{epsg} is not a projected CRS in metres")
        sys.exit(1)


def camera_centres(args):
    """Camera centre per image name."""
    if args.mode == "nerfrac":
        if not os.path.isdir(args.colmap_sparse_dir):
            print(f"ERROR: --colmap_sparse_dir not found: {args.colmap_sparse_dir!r}")
            sys.exit(1)
        centres = extract_colmap_camera_centers(args.colmap_sparse_dir)
        print(f"COLMAP: {len(centres)} camera centres")
        return centres

    if not os.path.isdir(args.vggt_dir):
        print(f"ERROR: --vggt_dir not found: {args.vggt_dir!r}")
        sys.exit(1)
    centres = extract_vggt_camera_centers_from_extrinsics(args.vggt_dir)
    names = load_vggt_image_names(args.vggt_dir)
    if not names:
        print(f"ERROR: No image_names in {args.vggt_dir}/metadata.json")
        sys.exit(1)
    if len(centres) < len(names):
        print(
            f"ERROR: {len(centres)} extrinsic files for {len(names)} images in "
            "metadata.json"
        )
        sys.exit(1)
    # extrinsics beyond the named views are left from an earlier, larger run
    n_left_out = len(centres) - len(names)
    note = f", {n_left_out} more extrinsic file(s) left out" if n_left_out else ""
    print(f"VGGT: {len(names)} camera centres{note}")
    return dict(zip(names, centres))


def gps_positions(args):
    """(lat, lon, height) per image name."""
    if args.gps_source == "exif":
        if not os.path.isdir(args.image_dir):
            print(f"ERROR: --image_dir not found: {args.image_dir!r}")
            sys.exit(1)
        gps = read_all_exif_gps(args.image_dir)
        print(f"GPS: {len(gps)} images with EXIF GPS")
        return gps

    # sqlite would create an empty database at a wrong path
    if not os.path.isfile(args.database_path):
        print(f"ERROR: --database_path not found: {args.database_path!r}")
        sys.exit(1)
    gps = extract_gps_from_colmap_db(args.database_path)
    print(f"GPS: {len(gps)} position priors in the database")
    return gps


def match_images(centres, gps):
    """(model name, GPS name) pairs with the same file name without extension."""
    model_by_stem = {os.path.splitext(n)[0]: n for n in centres}
    gps_by_stem = {os.path.splitext(n)[0]: n for n in gps}
    stems = sorted(set(model_by_stem) & set(gps_by_stem))
    if len(stems) < MIN_MATCHED_IMAGES:
        print(
            f"ERROR: {len(stems)} images match between the model and the GPS, "
            f"need {MIN_MATCHED_IMAGES}"
        )
        print(f"  model: {sorted(centres)[:5]}")
        print(f"  GPS:   {sorted(gps)[:5]}")
        sys.exit(1)
    for side, names in (("model", centres), ("GPS", gps)):
        stem_counts = Counter(os.path.splitext(n)[0] for n in names)
        doubled = sorted(
            n
            for n in names
            if stem_counts[os.path.splitext(n)[0]] > 1
            and os.path.splitext(n)[0] in stems
        )
        if doubled:
            print(f"ERROR: {side} images that differ only in the extension: {doubled}")
            sys.exit(1)

    n_unmatched = len(centres) - len(stems)
    note = f", {n_unmatched} model images have no GPS" if n_unmatched else ""
    print(f"Matched {len(stems)} images{note}")
    return [(model_by_stem[s], gps_by_stem[s]) for s in stems]


def main():
    args = parse_args()
    check_target_crs(args.target_epsg)
    print(f"Umeyama transform ({args.mode} -> EPSG:{args.target_epsg})")
    centres = camera_centres(args)
    gps = gps_positions(args)
    pairs = match_images(centres, gps)

    projected = wgs84_dict_to_projected(
        {name: gps[name] for _, name in pairs}, args.target_epsg
    )
    model_points = np.array([centres[model] for model, _ in pairs])
    gps_points = np.array([projected[name] for _, name in pairs])
    # cameras along one line leave the rotation about that line undetermined
    spread = np.linalg.svd(gps_points - gps_points.mean(axis=0), compute_uv=False)
    if spread[1] < MIN_SPREAD_RATIO * spread[0]:
        print("ERROR: The matched cameras lie almost on one line")
        sys.exit(1)

    s, R, t = umeyama_alignment(model_points, gps_points)
    stats = compute_umeyama_residuals(model_points, gps_points, s, R, t)
    print(f"Scale {s:.6f}, translation [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    print(
        f"Camera residuals: horizontal RMS {stats['xy_rms']:.3f} m (max "
        f"{stats['max_xy']:.3f} m), vertical RMS {stats['z_rms']:.3f} m (max "
        f"{stats['max_z']:.3f} m)"
    )

    # np.savez adds the extension
    path = args.output if args.output.endswith(".npz") else args.output + ".npz"
    print(f"Saving: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_umeyama(
        args.output,
        s=s,
        R=R,
        t=t,
        # translation_raw and z_correction keep the format of the shipped transforms;
        # there is no Z correction, the transformed cameras keep their GPS heights
        translation_raw=t,
        z_correction=0.0,
        xy_rms=stats["xy_rms"],
        z_rms=stats["z_rms"],
        total_rms=stats["total_rms"],
        n_points=len(pairs),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
