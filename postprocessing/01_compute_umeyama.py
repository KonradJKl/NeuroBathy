"""
Compute Umeyama similarity transform for georeferencing model outputs to UTM.

Usage examples:
    # NeRFrac 50m (EXIF GPS)
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/sparse/0 \\
        --image_dir /path/to/source/images \\
        --wsm_path /path/to/WSM.tif \\
        --output transforms/nerfrac_50m_umeyama.npz

    # VGGT 50m (EXIF GPS)
    python 01_compute_umeyama.py \\
        --mode vggt \\
        --vggt_dir /path/to/vggt_output \\
        --image_dir /path/to/source/images \\
        --wsm_path /path/to/WSM.tif \\
        --output transforms/vggt_50m_umeyama.npz

    # NeRFrac with COLMAP DB GPS fallback
    python 01_compute_umeyama.py \\
        --mode nerfrac \\
        --colmap_sparse_dir /path/to/sparse/0 \\
        --gps_source colmap_db \\
        --database_path /path/to/database.db \\
        --wsm_path /path/to/WSM.tif \\
        --output transforms/nerfrac_50m_umeyama.npz
"""

import argparse
import os
import sys

import numpy as np

from coord_utils import (
    apply_umeyama,
    compute_umeyama_residuals,
    extract_colmap_camera_centers,
    extract_gps_from_colmap_db,
    extract_vggt_camera_centers_from_extrinsics,
    load_vggt_image_names,
    read_all_exif_gps,
    sample_wsm,
    save_umeyama,
    umeyama_alignment,
    wgs84_dict_to_utm33,
    wgs84_dict_to_utm34,
    wgs84_dict_to_utm36,
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

    # WSM Z-correction
    parser.add_argument(
        "--wsm_path",
        type=str,
        default="",
        help="Path to Water Surface Model GeoTIFF for Z-correction (optional)",
    )

    # UTM zone
    parser.add_argument(
        "--utm_zone",
        type=int,
        default=33,
        choices=[33, 34, 36],
        help="UTM zone for GPS conversion: 33 (Pielach), 34 (Santa Maria), 36 (Agia Napa/Cyprus). Default: 33.",
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


def main():
    args = parse_args()

    print("=" * 60)
    print(f"UMEYAMA ALIGNMENT  ({args.mode.upper()} -> UTM33)")
    print("=" * 60)

    # Load model camera positions
    print("\n ---Load model camera positions ---")

    if args.mode == "nerfrac":
        if not args.colmap_sparse_dir:
            print("ERROR: --colmap_sparse_dir required for nerfrac mode")
            sys.exit(1)
        model_positions = extract_colmap_camera_centers(args.colmap_sparse_dir)
        print(f"Loaded {len(model_positions)} COLMAP camera centers")

    elif args.mode == "vggt":
        if not args.vggt_dir:
            print("ERROR: --vggt_dir required for vggt mode")
            sys.exit(1)

        centers = extract_vggt_camera_centers_from_extrinsics(args.vggt_dir)
        print(f"Loaded {len(centers)} VGGT camera positions from extrinsics")

        # Map indices to image names
        image_names = load_vggt_image_names(args.vggt_dir)
        if image_names is not None:
            if len(image_names) != len(centers):
                print(
                    f"[WARN] metadata.json has {len(image_names)} names "
                    f"but found {len(centers)} extrinsic files"
                )
            model_positions = {}
            for i, center in enumerate(centers):
                if i < len(image_names):
                    model_positions[image_names[i]] = center
                else:
                    model_positions[f"frame_{i:03d}"] = center
            print(f"Mapped {len(model_positions)} positions via metadata.json")
        else:
            if args.image_dir and os.path.isdir(args.image_dir):
                all_imgs = sorted(
                    [
                        f
                        for f in os.listdir(args.image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                model_positions = {}
                for i, center in enumerate(centers):
                    if i < len(all_imgs):
                        model_positions[all_imgs[i]] = center
                    else:
                        model_positions[f"frame_{i:03d}"] = center
                print(
                    f"Mapped {len(model_positions)} positions by "
                    f"sorted image directory listing (no metadata.json)"
                )
            else:
                print(
                    "ERROR: No metadata.json found and no --image_dir for "
                    "index-to-name mapping"
                )
                sys.exit(1)

    # Load GPS positions
    print("\n ---Load GPS positions ---")

    if args.gps_source == "exif":
        if not args.image_dir:
            print("ERROR: --image_dir required for gps_source=exif")
            sys.exit(1)
        gps_wgs84 = read_all_exif_gps(args.image_dir)
        print(f"Read {len(gps_wgs84)} GPS positions from EXIF")
    else:
        if not args.database_path:
            print("ERROR: --database_path required for gps_source=colmap_db")
            sys.exit(1)
        gps_wgs84 = extract_gps_from_colmap_db(args.database_path)
        print(f"Read {len(gps_wgs84)} GPS positions from COLMAP database")

    # Match model positions to GPS by image name
    print("\n ---Match model <-> GPS ---")

    model_names = set(model_positions.keys())
    gps_names = set(gps_wgs84.keys())

    # Try full-name match
    common = sorted(model_names & gps_names)

    if len(common) == 0:
        # Maybe stem?
        model_stem_map = {_stem(n): n for n in model_names}
        gps_stem_map = {_stem(n): n for n in gps_names}
        common_stems = sorted(set(model_stem_map.keys()) & set(gps_stem_map.keys()))

        if len(common_stems) > 0:
            common = []
            matched_gps = {}
            for stem in common_stems:
                mname = model_stem_map[stem]
                gname = gps_stem_map[stem]
                common.append(mname)
                matched_gps[mname] = gps_wgs84[gname]
            gps_wgs84_matched = matched_gps
            print(f"Matched {len(common)} images by stem (extension-agnostic)")
        else:
            print("ERROR: No matching images found between model and GPS")
            print(f"  Model names sample: {sorted(model_names)[:5]}")
            print(f"  GPS names sample:   {sorted(gps_names)[:5]}")
            sys.exit(1)
    else:
        gps_wgs84_matched = {n: gps_wgs84[n] for n in common}
        print(f"Matched {len(common)} images by filename")

    if len(common) < 3:
        print(f"ERROR: Need at least 3 matched points, got {len(common)}")
        sys.exit(1)

    if len(common) < len(model_positions):
        n_missing = len(model_positions) - len(common)
        print(f"  ({n_missing} model images have no GPS match)")

    # Convert GPS to UTM
    if args.utm_zone == 36:
        print(f"  Using UTM Zone 36N (EPSG:32636)")
        gps_utm33 = wgs84_dict_to_utm36(gps_wgs84_matched)
    elif args.utm_zone == 34:
        print(f"  Using UTM Zone 34N (EPSG:32634)")
        gps_utm33 = wgs84_dict_to_utm34(gps_wgs84_matched)
    else:
        print(f"  Using UTM Zone 33N (EPSG:25833)")
        gps_utm33 = wgs84_dict_to_utm33(gps_wgs84_matched)

    # This build aligned arrays
    model_pts = np.array([model_positions[n] for n in common])
    utm_pts = np.array([gps_utm33[n] for n in common])

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
    print(f"\nResiduals (before Z-correction):")
    print(f"  XY RMS:    {stats['xy_rms']:.4f} m")
    print(f"  Z RMS:     {stats['z_rms']:.4f} m")
    print(f"  Total RMS: {stats['total_rms']:.4f} m")
    print(f"  Max XY:    {stats['max_xy']:.4f} m")
    print(f"  Max Z:     {stats['max_z']:.4f} m")

    # Z-correction for VGGT
    z_correction = 0.0
    t_corrected = t.copy()

    if args.wsm_path and os.path.exists(args.wsm_path):
        print("\n ---Z-correction using WSM ---")
        centroid_utm = stats["transformed"].mean(axis=0)
        print(
            f"Transformed centroid: "
            f"E={centroid_utm[0]:.2f}, N={centroid_utm[1]:.2f}, Z={centroid_utm[2]:.3f}"
        )

        wsm_elev = sample_wsm(args.wsm_path, centroid_utm[0], centroid_utm[1])
        if wsm_elev is not None:
            z_correction = wsm_elev - centroid_utm[2]
            t_corrected[2] += z_correction
            print(f"WSM elevation at centroid: {wsm_elev:.3f} m")
            print(f"Z correction: {z_correction:+.4f} m")

            #  Important: Recompute Z residuals with correction!!!
            stats_corrected = compute_umeyama_residuals(
                model_pts, utm_pts, s, R, t_corrected
            )
            print(f"Z RMS after correction: {stats_corrected['z_rms']:.4f} m")
        else:
            print("[WARN] Could not sample WSM at centroid — no Z-correction applied")
    else:
        if args.wsm_path:
            print(f"\n[WARN] WSM not found at {args.wsm_path} — skipping Z-correction")
        else:
            print("\n[INFO] No --wsm_path provided — skipping Z-correction")


    print(f"\n ---Save to {args.output} ---")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    save_umeyama(
        args.output,
        s=s,
        R=R,
        t=t_corrected,
        translation_raw=t,
        z_correction=z_correction,
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
    print(f"  Z correction:  {z_correction:+.4f} m")
    print(f"  Output:        {args.output}")


if __name__ == "__main__":
    main()
