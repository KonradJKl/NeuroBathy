"""
Shared coordinate transform and I/O utilities for the post-processing pipeline.

"""

import collections
import glob
import json
import os
import sqlite3
import struct

import numpy as np
import rasterio
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
from scipy import stats as sp_stats


WATER_MASK_THRESHOLD = 127  # uint8 mask: pixel > threshold = water
NEAR = 1.0  # NDC near plane distance



_ColmapImage = collections.namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name"]
)

_ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"]
)

_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}

_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_colmap_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            # Read null-terminated name
            name = ""
            ch = _read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = _read_next_bytes(fid, 1, "c")[0]
            # Skip 2D points
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            _read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            images[image_id] = _ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def read_colmap_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id = props[0]
            model_id = props[1]
            width = props[2]
            height = props[3]
            num_params = _CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(_read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = _ColmapCamera(
                id=camera_id,
                model=_CAMERA_MODEL_NAMES.get(model_id, f"UNKNOWN_{model_id}"),
                width=width,
                height=height,
                params=params,
            )
    return cameras


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def extract_colmap_camera_centers(sparse_dir):
    images = read_colmap_images_binary(os.path.join(sparse_dir, "images.bin"))
    centers = {}
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        center = -R.T @ img.tvec
        centers[img.name] = center
    return centers


def _dms_to_decimal(dms, ref):
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def read_exif_gps(image_path):
    try:
        img = PILImage.open(image_path)
        exif = img._getexif()
        if exif is None:
            return None
    except Exception:
        return None

    gps_raw = None
    for key, val in exif.items():
        if TAGS.get(key) == "GPSInfo":
            gps_raw = val
            break

    if gps_raw is None:
        return None

    gps = {}
    for t in gps_raw:
        gps[GPSTAGS.get(t, t)] = gps_raw[t]

    try:
        lat = _dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        alt = float(gps.get("GPSAltitude", 0))
        alt_ref = gps.get("GPSAltitudeRef", b"\x00")
        if isinstance(alt_ref, bytes) and alt_ref == b"\x01":
            alt = -alt
        elif isinstance(alt_ref, int) and alt_ref == 1:
            alt = -alt
        return (lat, lon, alt)
    except (KeyError, TypeError, ValueError):
        return None


def read_all_exif_gps(image_dir, extensions=(".JPG", ".jpg", ".jpeg", ".png")):
    gps_dict = {}
    for fname in sorted(os.listdir(image_dir)):
        if not any(fname.endswith(ext) for ext in extensions):
            continue
        result = read_exif_gps(os.path.join(image_dir, fname))
        if result is not None:
            gps_dict[fname] = np.array(result)
    return gps_dict


def extract_gps_from_colmap_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, prior_tx, prior_ty, prior_tz FROM images "
        "WHERE prior_tx IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    gps = {}
    for name, lat, lon, alt in rows:
        if lat is not None and lon is not None and alt is not None:
            gps[name] = np.array([lat, lon, alt])
    return gps

_wgs84_to_utm33_transformer = None


def _get_wgs84_to_utm33():
    global _wgs84_to_utm33_transformer
    if _wgs84_to_utm33_transformer is None:
        _wgs84_to_utm33_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:25833", always_xy=False
        )
    return _wgs84_to_utm33_transformer


def wgs84_to_utm33(lat, lon, alt=0.0):
    t = _get_wgs84_to_utm33()
    easting, northing = t.transform(lat, lon)
    return np.array([easting, northing, alt])


def wgs84_dict_to_utm33(gps_dict):
    t = _get_wgs84_to_utm33()
    utm = {}
    for name, (lat, lon, alt) in gps_dict.items():
        easting, northing = t.transform(lat, lon)
        utm[name] = np.array([easting, northing, alt])
    return utm



_wgs84_to_utm34_transformer = None


def _get_wgs84_to_utm34():
    global _wgs84_to_utm34_transformer
    if _wgs84_to_utm34_transformer is None:
        _wgs84_to_utm34_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:32634", always_xy=False
        )
    return _wgs84_to_utm34_transformer


def wgs84_to_utm34(lat, lon, alt=0.0):
    t = _get_wgs84_to_utm34()
    easting, northing = t.transform(lat, lon)
    return np.array([easting, northing, alt])


def wgs84_dict_to_utm34(gps_dict):
    t = _get_wgs84_to_utm34()
    utm = {}
    for name, (lat, lon, alt) in gps_dict.items():
        easting, northing = t.transform(lat, lon)
        utm[name] = np.array([easting, northing, alt])
    return utm



_wgs84_to_utm36_transformer = None


def _get_wgs84_to_utm36():
    global _wgs84_to_utm36_transformer
    if _wgs84_to_utm36_transformer is None:
        _wgs84_to_utm36_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:32636", always_xy=False
        )
    return _wgs84_to_utm36_transformer


def wgs84_to_utm36(lat, lon, alt=0.0):
    t = _get_wgs84_to_utm36()
    easting, northing = t.transform(lat, lon)
    return np.array([easting, northing, alt])


def wgs84_dict_to_utm36(gps_dict):
    t = _get_wgs84_to_utm36()
    utm = {}
    for name, (lat, lon, alt) in gps_dict.items():
        easting, northing = t.transform(lat, lon)
        utm[name] = np.array([easting, northing, alt])
    return utm


def ndc_to_nerf_world(P_ndc, H, W, focal, near=1.0):
    nx, ny, nz = P_ndc[:, 0], P_ndc[:, 1], P_ndc[:, 2]
    z = 2.0 * near / (nz - 1.0 + 1e-10)
    x = -(W * nx * z) / (2.0 * focal)
    y = -(H * ny * z) / (2.0 * focal)
    return np.stack([x, y, z], axis=-1)


def nerf_world_to_colmap(P_nerf, c2w_recenter, sc):
    R = c2w_recenter[:3, :3]
    t = c2w_recenter[:3, 3]
    P_scaled = (P_nerf @ R.T) + t
    return P_scaled / sc


def ndc_to_colmap(P_ndc, H, W, focal, c2w_recenter, sc, near=1.0):
    P_nerf = ndc_to_nerf_world(P_ndc, H, W, focal, near)
    return nerf_world_to_colmap(P_nerf, c2w_recenter, sc)


def load_nerfrac_coord_metadata(log_folder, data_dir=None, bd_factor=0.75):
    c2w_recenter = None
    sc = None

    c2w_path = os.path.join(log_folder, "c2w.npy")
    sc_path = os.path.join(log_folder, "llff_scale.npy")

    if os.path.exists(c2w_path):
        c2w_recenter = np.load(c2w_path)
    else:
        print(f"[WARN] c2w.npy not found in {log_folder}")

    if os.path.exists(sc_path):
        sc = float(np.load(sc_path).flat[0])
    elif data_dir is not None:
        pb_path = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(pb_path):
            pb = np.load(pb_path)
            bds_raw = pb[:, -2:]
            sc = 1.0 / (bds_raw.min() * bd_factor)
            print(f"[INFO] Recomputed sc={sc:.8f} from {pb_path}")
    else:
        print(f"[WARN] llff_scale.npy not found in {log_folder}")

    return {"c2w_recenter": c2w_recenter, "sc": sc}


def load_focal_from_poses_bounds(poses_bounds_path, factor=1):
    pb = np.load(poses_bounds_path)
    poses = pb[:, :15].reshape(-1, 3, 5)
    H_full, W_full, focal_full = poses[0, :, 4]
    H = int(H_full) // factor
    W = int(W_full) // factor
    focal = focal_full * W / W_full
    return H, W, focal


def umeyama_alignment(source, target):
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
    V = Vt.T

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (1.0 / sigma2) * np.trace(np.diag(D) @ S)
    t = mu_target - s * R @ mu_source

    return s, R, t


def apply_umeyama(points, s, R, t):
    squeeze = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        squeeze = True

    result = s * (points @ R.T) + t

    if squeeze:
        return result.flatten()
    return result


def compute_umeyama_residuals(source, target, s, R, t):
    transformed = apply_umeyama(source, s, R, t)
    residuals = transformed - target

    xy_rms = np.sqrt(np.mean(residuals[:, :2] ** 2))
    z_rms = np.sqrt(np.mean(residuals[:, 2] ** 2))
    total_rms = np.sqrt(np.mean(residuals**2))
    max_xy = np.max(np.linalg.norm(residuals[:, :2], axis=1))
    max_z = np.max(np.abs(residuals[:, 2]))

    return {
        "xy_rms": xy_rms,
        "z_rms": z_rms,
        "total_rms": total_rms,
        "max_xy": max_xy,
        "max_z": max_z,
        "residuals": residuals,
        "transformed": transformed,
    }



def sample_wsm(wsm_path, easting, northing):
    with rasterio.open(wsm_path) as src:
        row, col = src.index(easting, northing)
        if 0 <= row < src.height and 0 <= col < src.width:
            val = src.read(1)[row, col]
            if val != src.nodata:
                return float(val)
    return None


def sample_wsm_batch(wsm_path, eastings, northings):
    eastings = np.asarray(eastings, dtype=np.float64)
    northings = np.asarray(northings, dtype=np.float64)
    elevations = np.full(len(eastings), np.nan)

    with rasterio.open(wsm_path) as src:
        raster = src.read(1)
        nodata = src.nodata
        inv_transform = ~src.transform
        # Vectorized: convert geographic coords to pixel coords
        col_f, row_f = inv_transform * (eastings, northings)
        cols = np.floor(col_f).astype(int)
        rows = np.floor(row_f).astype(int)
        # Bounds check
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if valid.any():
            vals = raster[rows[valid], cols[valid]]
            if nodata is not None:
                nodata_mask = vals == nodata
                vals = vals.astype(np.float64)
                vals[nodata_mask] = np.nan
            elevations[valid] = vals.astype(np.float64)

    return elevations


def extract_vggt_camera_centers_from_extrinsics(vggt_dir):
    centers = []
    idx = 0
    while True:
        path = os.path.join(vggt_dir, f"extrinsic_{idx:03d}.npy")
        if not os.path.exists(path):
            break
        ext = np.load(path)  # (3, 4)
        R = ext[:3, :3]
        t = ext[:3, 3]
        center = -R.T @ t
        centers.append(center)
        idx += 1
    return centers


def load_vggt_image_names(vggt_dir):
    meta_path = os.path.join(vggt_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta.get("image_names")


def save_umeyama(path, s, R, t, **metadata):
    np.savez(
        path,
        scale=np.array(s),
        rotation=R,
        translation=t,
        **{
            k: np.array(v) if not isinstance(v, np.ndarray) else v
            for k, v in metadata.items()
        },
    )


def load_umeyama(path):
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        if val.ndim == 0:
            item = val.item()
            if isinstance(item, (int, float, np.integer, np.floating)):
                result[key] = float(item)
            else:
                result[key] = item
        else:
            result[key] = val
    return result


def compute_metrics(gt, pred, label=""):
    N = len(gt)
    if N < 3:
        out = {"N": N, "pearson_r": np.nan, "spearman_rho": np.nan,
               "rmse": np.nan, "mae": np.nan, "bias": np.nan,
               "scale_ratio": np.nan}
        if label:
            out["label"] = label
        return out

    err = pred - gt
    pearson_r = float(np.corrcoef(gt, pred)[0, 1])
    sp = sp_stats.spearmanr(gt, pred)
    spearman_rho = float(sp.correlation) if not np.isnan(sp.correlation) else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # median per-point ratio where gt is large enough to avoid division noise
    valid = gt > 0.1
    if np.sum(valid) >= 2:
        scale_ratio = float(np.median(pred[valid] / gt[valid]))
    else:
        scale_ratio = np.nan

    out = {"N": N, "pearson_r": pearson_r, "spearman_rho": spearman_rho,
           "rmse": rmse, "mae": mae, "bias": bias, "scale_ratio": scale_ratio}
    if label:
        out["label"] = label
    return out




def write_ply_binary(filepath, points, colors=None, scalar_fields=None):
    N = points.shape[0]
    if scalar_fields is None:
        scalar_fields = {}


    props = "property float x\nproperty float y\nproperty float z\n"
    dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]

    if colors is not None:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    for name in scalar_fields:
        props += f"property float {name}\n"
        dtype_fields.append((name, "<f4"))

    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {N}\n{props}end_header\n")

    vertex_dtype = np.dtype(dtype_fields)
    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)

    if colors is not None:
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

    for name, arr in scalar_fields.items():
        vertices[name] = arr.astype(np.float32)

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)



def discover_nerfrac_views(testset_dir, depth_type="mean"):
    depth_suffix = "_depth_mode.npy" if depth_type == "mode" else "_depth.npy"
    if depth_type != "mean":
        print(f"  [discover] Using depth_type='{depth_type}' (suffix: {depth_suffix})")

    views = []
    idx = 0
    while True:
        # surface points — try both naming conventions
        xs_path = None
        for pattern in [f"Xs_pts{idx:03d}.npy", f"Xs_pts_{idx:03d}.npy"]:
            p = os.path.join(testset_dir, pattern)
            if os.path.exists(p):
                xs_path = p
                break

        # refracted ray directions
        rays_path = os.path.join(testset_dir, f"rays_{idx:03d}.npy")
        if not os.path.exists(rays_path):
            rays_path2 = os.path.join(testset_dir, f"rays{idx:03d}.npy")
            if os.path.exists(rays_path2):
                rays_path = rays_path2
            elif xs_path is None:
                break
            else:
                rays_path = None

        depth_candidates = glob.glob(
            os.path.join(testset_dir, f"*{idx:03d}{depth_suffix}")
        )
        depth_path = depth_candidates[0] if depth_candidates else None

        if xs_path is None and rays_path is None and depth_path is None:
            break

        views.append({"xs": xs_path, "rays": rays_path,
                       "depth": depth_path, "index": idx})
        idx += 1

    return views


def discover_vggt_views(vggt_dir):
    views = []
    idx = 0
    while True:
        wp_path = os.path.join(vggt_dir, f"world_points_{idx:03d}.npy")
        if not os.path.exists(wp_path):
            break
        conf_path = os.path.join(vggt_dir, f"depth_conf_{idx:03d}.npy")
        if not os.path.exists(conf_path):
            conf_path = None

        views.append({"world_points": wp_path, "depth_conf": conf_path,
                       "index": idx})
        idx += 1

    return views
