"""
Coordinate transforms, georeferencing and I/O shared by the postprocessing scripts,
and the depth metrics of 10 and 11.

NeRFrac's surface points, refracted rays and depths live in NDC. ndc_to_colmap undoes
the NDC warp, the recentering and the scaling of load_llff_data, and the Umeyama
transform of 01 maps COLMAP or VGGT coordinates to the projected CRS of the ground
truth.
"""

import glob
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD
from pyproj import Transformer
from scipy.stats import spearmanr

from colmap_io import qvec2rotmat, read_colmap_images_binary

WATER_MASK_THRESHOLD = 127  # uint8 mask: pixel > threshold = water
NEAR = 1.0  # NDC near plane distance
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# pred / gt explodes for GT depths near 0, so scale_ratio leaves those points out
SCALE_RATIO_MIN_DEPTH_M = 0.1


def _dms_to_degrees(dms, ref):
    degrees = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
    return -degrees if ref in ("S", "W") else degrees


def read_exif_gps(image_path):
    """(lat, lon, height) from the EXIF of an image, or None."""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
    except Exception:
        return None
    gps_raw = exif.get(IFD.GPSInfo) if exif is not None else None
    if gps_raw is None:
        return None
    gps = {GPSTAGS.get(tag, tag): gps_raw[tag] for tag in gps_raw}

    try:
        lat = _dms_to_degrees(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _dms_to_degrees(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        # without a height the image cannot enter the 3D fit
        height = float(gps["GPSAltitude"])
        height_ref = gps.get("GPSAltitudeRef", b"\x00")
        if (isinstance(height_ref, bytes) and height_ref == b"\x01") or (
            isinstance(height_ref, int) and height_ref == 1
        ):
            height = -height  # below the reference level
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    if lat == 0 and lon == 0:  # the placeholder of a camera without a fix
        return None
    return (lat, lon, height)


def read_all_exif_gps(image_dir, extensions=IMAGE_EXTENSIONS):
    gps_dict = {}
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in extensions:
            continue
        gps = read_exif_gps(os.path.join(image_dir, fname))
        if gps is not None:
            gps_dict[fname] = np.array(gps)
    return gps_dict


def extract_gps_from_colmap_db(db_path):
    """(lat, lon, height) per image name from the position priors of a COLMAP
    database; the feature extractor stores the EXIF GPS as prior_tx/ty/tz."""
    # read-only, so that the WAL of a COLMAP project is not written into its database
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT name, prior_tx, prior_ty, prior_tz FROM images "
        "WHERE prior_tx IS NOT NULL AND prior_ty IS NOT NULL AND prior_tz IS NOT NULL"
    ).fetchall()
    conn.close()
    # (0, 0) is the placeholder of a camera without a fix, as in read_exif_gps
    return {
        name: np.array([lat, lon, height])
        for name, lat, lon, height in rows
        if not (lat == 0 and lon == 0)
    }


def wgs84_dict_to_projected(gps_dict, epsg):
    """GPS (lat, lon, height) per name -> (easting, northing, height) in EPSG:epsg."""
    # always_xy: (lon, lat) in and (easting, northing) out, whatever the CRS axis order
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    projected = {}
    for name, (lat, lon, height) in gps_dict.items():
        easting, northing = transformer.transform(lon, lat)
        projected[name] = np.array([easting, northing, height])
    return projected


def extract_colmap_camera_centers(sparse_dir):
    images = read_colmap_images_binary(os.path.join(sparse_dir, "images.bin"))
    return {img.name: -qvec2rotmat(img.qvec).T @ img.tvec for img in images.values()}


def extract_vggt_camera_centers_from_extrinsics(vggt_dir):
    centers = []
    idx = 0
    while True:
        path = os.path.join(vggt_dir, f"extrinsic_{idx:03d}.npy")
        if not os.path.exists(path):
            return centers
        extrinsic = np.load(path)  # (3, 4) world -> camera
        centers.append(-extrinsic[:3, :3].T @ extrinsic[:3, 3])
        idx += 1


def umeyama_alignment(source, target):
    """Similarity transform (s, R, t) with target ~ s * R @ source + t, least squares
    (Umeyama 1991)."""
    assert source.shape == target.shape
    assert source.shape[0] >= 3, "Need at least 3 points"
    n = source.shape[0]
    mu_target = target.mean(axis=0)
    mu_source = source.mean(axis=0)
    target_zc = target - mu_target
    source_zc = source - mu_source

    C = (1.0 / n) * target_zc.T @ source_zc
    sigma2 = (1.0 / n) * np.sum(source_zc**2)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt.T) < 0:
        S[2, 2] = -1  # a reflection would fit better; force a rotation

    R = U @ S @ Vt
    s = (1.0 / sigma2) * np.trace(np.diag(D) @ S)
    t = mu_target - s * R @ mu_source
    return s, R, t


def apply_umeyama(points, s, R, t):
    """s * R @ p + t for an (N, 3) array or a single (3,) point."""
    if points.ndim == 1:
        return (s * (points.reshape(1, 3) @ R.T) + t).flatten()
    return s * (points @ R.T) + t


def compute_umeyama_residuals(source, target, s, R, t):
    """RMS and maximum of the horizontal and vertical residuals of the transformed
    source against target, and the RMS of the 3D residual length."""
    residuals = apply_umeyama(source, s, R, t) - target
    return {
        "xy_rms": np.sqrt(np.mean(np.sum(residuals[:, :2] ** 2, axis=1))),
        "z_rms": np.sqrt(np.mean(residuals[:, 2] ** 2)),
        "total_rms": np.sqrt(np.mean(np.sum(residuals**2, axis=1))),
        "max_xy": np.max(np.linalg.norm(residuals[:, :2], axis=1)),
        "max_z": np.max(np.abs(residuals[:, 2])),
    }


def save_umeyama(path, s, R, t, **metadata):
    # converted before np.savez opens the file, so a bad value leaves no partial npz
    arrays = {
        k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in metadata.items()
    }
    np.savez(path, scale=np.array(s), rotation=R, translation=t, **arrays)


def load_umeyama(path):
    """The arrays of a transform from 01; numbers as float, the mode as str."""
    result = {}
    with np.load(path) as data:
        for key in data.files:
            value = data[key]
            if value.ndim == 0:
                value = value.item()
                if isinstance(value, (int, float, np.integer, np.floating)):
                    value = float(value)
            result[key] = value
    return result


def discover_nerfrac_views(testset_dir, depth_type="mean"):
    """Xs_ptsNNN.npy, rays_NNN.npy and *NNN_depth[_mode].npy (None if missing) of each
    view of a render folder of run_nerfrac.py, up to the first view with neither Xs
    nor rays."""
    depth_suffix = "_depth_mode.npy" if depth_type == "mode" else "_depth.npy"
    views = []
    idx = 0
    while True:
        xs_path = os.path.join(testset_dir, f"Xs_pts{idx:03d}.npy")
        rays_path = os.path.join(testset_dir, f"rays_{idx:03d}.npy")
        xs_path = xs_path if os.path.exists(xs_path) else None
        rays_path = rays_path if os.path.exists(rays_path) else None
        if xs_path is None and rays_path is None:
            return views

        pattern = os.path.join(glob.escape(testset_dir), f"*{idx:03d}{depth_suffix}")
        depth_paths = sorted(glob.glob(pattern))
        views.append(
            {
                "xs": xs_path,
                "rays": rays_path,
                "depth": depth_paths[0] if depth_paths else None,
                "index": idx,
            }
        )
        idx += 1


def load_nerfrac_coord_metadata(log_folder, data_dir=None, bd_factor=0.75):
    """c2w_recenter (c2w.npy) and sc, the LLFF scale (llff_scale.npy), of an experiment
    folder; None for what is missing.

    Older experiment folders have no llff_scale.npy; sc is then recomputed from
    poses_bounds.npy in data_dir, as run_nerfrac.py computes llff_scale.npy.
    """
    c2w_path = os.path.join(log_folder, "c2w.npy")
    c2w_recenter = None
    if os.path.exists(c2w_path):
        c2w_recenter = np.load(c2w_path)
    else:
        print(f"No c2w.npy in {log_folder}")

    sc = None
    sc_path = os.path.join(log_folder, "llff_scale.npy")
    pb_path = None if data_dir is None else os.path.join(data_dir, "poses_bounds.npy")
    if os.path.exists(sc_path):
        sc = float(np.load(sc_path).flat[0])
    elif pb_path is not None and os.path.exists(pb_path):
        bds_raw = np.load(pb_path)[:, -2:]
        sc = 1.0 / (bds_raw.min() * bd_factor)
        print(f"No llff_scale.npy in {log_folder}, scale {sc:.8f} from {pb_path}")
    else:
        print(f"No llff_scale.npy in {log_folder}")
    return {"c2w_recenter": c2w_recenter, "sc": sc}


def load_focal_from_poses_bounds(poses_bounds_path, factor=1):
    """H, W and focal of the images downsampled by factor.

    H and W are rounded down, while LLFF rounds the size of images_<factor>
    (5460 / 8 -> 683 px), so 07 and 08 take H from the rendered pixels.
    """
    poses = np.load(poses_bounds_path)[:, :15].reshape(-1, 3, 5)
    H_full, W_full, focal_full = poses[0, :, 4]
    H = int(H_full) // factor
    W = int(W_full) // factor
    focal = focal_full * W / W_full
    return H, W, focal


def ndc_to_nerf_world(P_ndc, H, W, focal, near=NEAR):
    """Inverse of the NDC warp of ndc_rays() in run_nerfrac_helpers.py."""
    nx, ny, nz = P_ndc[:, 0], P_ndc[:, 1], P_ndc[:, 2]
    z = 2.0 * near / (nz - 1.0 + 1e-10)
    x = -(W * nx * z) / (2.0 * focal)
    y = -(H * ny * z) / (2.0 * focal)
    return np.stack([x, y, z], axis=-1)


def nerf_world_to_colmap(P_nerf, c2w_recenter, sc):
    """Undo the recentering (c2w.npy) and the scaling (llff_scale.npy) of
    load_llff_data."""
    R = c2w_recenter[:3, :3]
    t = c2w_recenter[:3, 3]
    P_scaled = (P_nerf @ R.T) + t
    return P_scaled / sc


def ndc_to_colmap(P_ndc, H, W, focal, c2w_recenter, sc, near=NEAR):
    P_nerf = ndc_to_nerf_world(P_ndc, H, W, focal, near)
    return nerf_world_to_colmap(P_nerf, c2w_recenter, sc)


def load_vggt_image_names(vggt_dir):
    meta_path = os.path.join(vggt_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f).get("image_names")


def discover_vggt_views(vggt_dir):
    """world_points_NNN.npy and depth_conf_NNN.npy (None if missing) of each view of an
    output folder of run_vggt_inference.py, up to the first missing world_points file
    and at most as many views as metadata.json has image names."""
    names = load_vggt_image_names(vggt_dir)
    n_views = 0
    while os.path.exists(os.path.join(vggt_dir, f"world_points_{n_views:03d}.npy")):
        n_views += 1
    if n_views and names is not None and n_views > len(names):
        # left from an earlier, larger run into the same folder
        print(
            f"{n_views - len(names)} view(s) after the {len(names)} of metadata.json "
            "left out"
        )
        n_views = len(names)

    views = []
    for idx in range(n_views):
        conf_path = os.path.join(vggt_dir, f"depth_conf_{idx:03d}.npy")
        views.append(
            {
                "world_points": os.path.join(vggt_dir, f"world_points_{idx:03d}.npy"),
                "depth_conf": conf_path if os.path.exists(conf_path) else None,
                "index": idx,
            }
        )
    return views


def sample_wsm_batch(wsm_path, eastings, northings):
    """WSM elevation at each point; NaN off the raster and on nodata."""
    eastings = np.asarray(eastings, dtype=np.float64)
    northings = np.asarray(northings, dtype=np.float64)
    elevations = np.full(len(eastings), np.nan)

    with rasterio.open(wsm_path) as src:
        raster = src.read(1)
        col_f, row_f = ~src.transform * (eastings, northings)
        cols = np.floor(col_f).astype(int)
        rows = np.floor(row_f).astype(int)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if inside.any():
            values = raster[rows[inside], cols[inside]]
            if src.nodata is not None:
                # compared in the raster's dtype, which the nodata value was written in
                is_nodata = values == src.nodata
                values = values.astype(np.float64)
                values[is_nodata] = np.nan
            elevations[inside] = values
    return elevations


def write_ply_binary(filepath, points, colors=None, scalar_fields=None):
    """Binary PLY of (N, 3) points with optional uint8 colors and float32 fields.

    x and y are float64: at a UTM northing of 5.3e6 m float32 steps are 0.5 m.
    """
    N = points.shape[0]
    if scalar_fields is None:
        scalar_fields = {}
    fields = [("x", "<f8", "double"), ("y", "<f8", "double"), ("z", "<f4", "float")]
    if colors is not None:
        fields += [(name, "u1", "uchar") for name in ("red", "green", "blue")]
    fields += [(name, "<f4", "float") for name in scalar_fields]

    vertices = np.empty(N, dtype=[(name, dtype) for name, dtype, _ in fields])
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2].astype(np.float32)
    if colors is not None:
        for i, name in enumerate(("red", "green", "blue")):
            vertices[name] = colors[:, i]
    for name, values in scalar_fields.items():
        vertices[name] = values.astype(np.float32)

    properties = "".join(f"property {kind} {name}\n" for name, _, kind in fields)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n{properties}end_header\n"
    )
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def compute_metrics(gt, pred, label=""):
    """N, Pearson r, Spearman rho, RMSE, MAE and bias of pred against gt, scale_ratio
    (the median of pred / gt over the points deeper than SCALE_RATIO_MIN_DEPTH_M),
    and label if given. Fewer than 3 points give NaN for everything but N."""
    N = len(gt)
    if N < 3:
        out = {
            "N": N,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "scale_ratio": np.nan,
        }
    else:
        err = pred - gt
        pearson_r = float(np.corrcoef(gt, pred)[0, 1])
        rho = spearmanr(gt, pred).correlation
        deep = gt > SCALE_RATIO_MIN_DEPTH_M
        out = {
            "N": N,
            "pearson_r": pearson_r,
            "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "scale_ratio": (
                float(np.median(pred[deep] / gt[deep])) if np.sum(deep) >= 2 else np.nan
            ),
        }
    if label:
        out["label"] = label
    return out
