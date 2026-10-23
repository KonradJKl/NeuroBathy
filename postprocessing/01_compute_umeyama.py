"""
Compute Umeyama similarity transform for georeferencing model outputs to UTM.

Usage examples:
    # NeRFrac 50m (EXIF GPS)
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/sparse/0 \\
        --image_dir /path/to/source/images \\
        --output transforms/nerfrac_50m_umeyama.npz

    # VGGT 50m (EXIF GPS)
    python 01_compute_umeyama.py \\
        --mode vggt \\
        --vggt_dir /path/to/vggt_output \\
        --image_dir /path/to/source/images \\
        --output transforms/vggt_50m_umeyama.npz

    # NeRFrac with COLMAP DB GPS fallback
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/sparse/0 \\
        --gps_source colmap_db \\
        --database_path /path/to/database.db \\
        --output transforms/nerfrac_50m_umeyama.npz
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError

from coord_utils import (
    apply_umeyama,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Umeyama similarity transform for georeferencing."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["nerfrac", "vggt"],
        help="Model type: nerfrac or vggt",
    )

    # Source of model camera positions
    parser.add_argument(
        "--colmap_sparse_dir",
        type=str,
        default="",
        help="[nerfrac] Path to COLMAP sparse/0/ directory with images.bin",
    )
    parser.add_argument(
        "--vggt_dir",
        type=str,
        default="",
        help="[vggt] Path to VGGT output directory with extrinsic_NNN.npy",
    )

    parser.add_argument(
        "--gps_source",
        type=str,
        default="exif",
        choices=["exif", "colmap_db"],
        help="Where to read GPS from: 'exif' (source images) or 'colmap_db' (database.db)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="[gps_source=exif] Directory with source JPEG images for GPS EXIF reading",
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default="",
        help="[gps_source=colmap_db] Path to COLMAP database.db",
    )

    # Target CRS
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=25833,
        help="EPSG code of the projected CRS of the ground truth. Default: 25833 "
        "(ETRS89 / UTM 33N, the Pielach data).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz file path",
    )

    return parser.parse_args()


def _stem(name):
    """Strip extension from filename."""
    return os.path.splitext(name)[0]


def check_target_crs(epsg):
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError:
        print(f"ERROR: Unknown EPSG code {epsg}")
        sys.exit(1)
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        print(f"ERROR: EPSG:{epsg} is not a projected CRS in metres")
        sys.exit(1)


def main():
    args = parse_args()
    check_target_crs(args.target_epsg)

    print("=" * 60)
    print(f"UMEYAMA ALIGNMENT  ({args.mode.upper()} -> EPSG:{args.target_epsg})")
    print("=" * 60)

    # Load model camera positions
    print("\n ---Load model camera positions ---")

    if args.mode == "nerfrac":
        if not os.path.isdir(args.colmap_sparse_dir):
            print(f"ERROR: --colmap_sparse_dir not found: {args.colmap_sparse_dir!r}")
            sys.exit(1)
        model_positions = extract_colmap_camera_centers(args.colmap_sparse_dir)
        print(f"Loaded {len(model_positions)} COLMAP camera centers")

    elif args.mode == "vggt":
        if not os.path.isdir(args.vggt_dir):
            print(f"ERROR: --vggt_dir not found: {args.vggt_dir!r}")
            sys.exit(1)

        centers = extract_vggt_camera_centers_from_extrinsics(args.vggt_dir)
        print(f"Loaded {len(centers)} VGGT camera positions from extrinsics")

        # Map indices to image names
        image_names = load_vggt_image_names(args.vggt_dir)
        if not image_names:
            print(f"ERROR: No image_names in {args.vggt_dir}/metadata.json")
            sys.exit(1)
        if len(centers) < len(image_names):
            print(
                f"ERROR: {len(centers)} extrinsic files for {len(image_names)} "
                f"images in metadata.json"
            )
            sys.exit(1)
        # extrinsics beyond the named views are left from an earlier, larger run
        model_positions = dict(zip(image_names, centers))
        print(f"Mapped {len(model_positions)} positions via metadata.json")

    # Load GPS positions
    print("\n ---Load GPS positions ---")

    if args.gps_source == "exif":
        if not os.path.isdir(args.image_dir):
            print(f"ERROR: --image_dir not found: {args.image_dir!r}")
            sys.exit(1)
        gps_wgs84 = read_all_exif_gps(args.image_dir)
        print(f"Read {len(gps_wgs84)} GPS positions from EXIF")
    else:
        # sqlite would create an empty database at a wrong path
        if not os.path.isfile(args.database_path):
            print(f"ERROR: --database_path not found: {args.database_path!r}")
            sys.exit(1)
        gps_wgs84 = extract_gps_from_colmap_db(args.database_path)
        print(f"Read {len(gps_wgs84)} GPS positions from COLMAP database")

    # Match model positions to GPS by image name
    print("\n ---Match model <-> GPS ---")

    # by file name without extension: the model may use PNG copies of the JPGs
    model_stem_map = {_stem(n): n for n in model_positions}
    gps_stem_map = {_stem(n): n for n in gps_wgs84}
    common_stems = sorted(set(model_stem_map) & set(gps_stem_map))
    for side, names in (("model", model_positions), ("GPS", gps_wgs84)):
        counts = Counter(_stem(n) for n in names)
        doubled = sorted(
            n for n in names if counts[_stem(n)] > 1 and _stem(n) in common_stems
        )
        if doubled:
            print(f"ERROR: {side} images that differ only in the extension: {doubled}")
            sys.exit(1)
    if len(common_stems) < 3:
        print(f"ERROR: {len(common_stems)} images match between model and GPS, need 3")
        print(f"  Model names sample: {sorted(model_positions)[:5]}")
        print(f"  GPS names sample:   {sorted(gps_wgs84)[:5]}")
        sys.exit(1)
    common = [model_stem_map[stem] for stem in common_stems]
    gps_wgs84_matched = {
        model_stem_map[stem]: gps_wgs84[gps_stem_map[stem]] for stem in common_stems
    }
    print(f"Matched {len(common)} images by file name")

    if len(common) < len(model_positions):
        n_missing = len(model_positions) - len(common)
        print(f"  ({n_missing} model images have no GPS match)")

    print(f"  GPS to EPSG:{args.target_epsg}")
    gps_utm33 = wgs84_dict_to_projected(gps_wgs84_matched, args.target_epsg)

    # This build aligned arrays
    model_pts = np.array([model_positions[n] for n in common])
    utm_pts = np.array([gps_utm33[n] for n in common])
    # cameras along one line leave the rotation about that line undetermined
    spread = np.linalg.svd(utm_pts - utm_pts.mean(axis=0), compute_uv=False)
    if spread[1] < 0.05 * spread[0]:
        print("ERROR: The matched cameras lie almost on one line")
        sys.exit(1)

    print(f"\nModel positions range:")
    for ax, label in enumerate(["X", "Y", "Z"]):
        print(
            f"  {label}: [{model_pts[:, ax].min():.4f}, {model_pts[:, ax].max():.4f}]"
        )
    print(f"UTM33 GPS range:")
    for ax, label in enumerate(["E", "N", "Z"]):
        print(f"  {label}: [{utm_pts[:, ax].min():.2f}, {utm_pts[:, ax].max():.2f}]")

    # Umeyama alignment
    print("\n ---Umeyama alignment ---")

    s, R, t = umeyama_alignment(model_pts, utm_pts)

    print(f"Scale:       {s:.10f}")
    print(f"Rotation:\n{R}")
    print(f"Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")

    # Stats for thesis
    stats = compute_umeyama_residuals(model_pts, utm_pts, s, R, t)
    print(f"\nResiduals:")
    print(f"  XY RMS:    {stats['xy_rms']:.4f} m")
    print(f"  Z RMS:     {stats['z_rms']:.4f} m")
    print(f"  Total RMS: {stats['total_rms']:.4f} m")
    print(f"  Max XY:    {stats['max_xy']:.4f} m")
    print(f"  Max Z:     {stats['max_z']:.4f} m")


    print(f"\n ---Save to {args.output} ---")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    save_umeyama(
        args.output,
        s=s,
        R=R,
        t=t,
        # no Z correction: the transformed cameras keep their flight height
        translation_raw=t,
        z_correction=0.0,
        xy_rms=stats["xy_rms"],
        z_rms=stats["z_rms"],
        total_rms=stats["total_rms"],
        n_points=len(common),
        mode=args.mode,
    )
    print(f"Saved Umeyama transform ({len(common)} points, scale={s:.4f})")

    # Results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Mode:          {args.mode}")
    print(f"  N points:      {len(common)}")
    print(f"  Scale:         {s:.6f}")
    print(f"  XY RMS:        {stats['xy_rms']:.4f} m")
    print(f"  Z RMS:         {stats['z_rms']:.4f} m")
    print(f"  Output:        {args.output}")


if __name__ == "__main__":
    main()
