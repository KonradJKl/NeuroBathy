"""
Run VGGT on a folder of images and save per-view depth, point maps, confidences
and predicted cameras for the postprocessing pipeline (01, 02, 04, 07).

Usage examples:
    # Depth unprojection (default), 9 nadir images
    python models/vggt/run_vggt_inference.py \\
        --image_dir /path/to/pielach_data/images \\
        --output_dir /path/to/vggt_nadir

    # Point-map head, with separate water and land point clouds
    python models/vggt/run_vggt_inference.py \\
        --image_dir /path/to/pielach_data/images \\
        --output_dir /path/to/vggt_nadir_pointmap \\
        --mask_dir /path/to/pielach_data/masks \\
        --use_point_map

Per-view outputs (NNN = index in sorted filename order):
    depth_NNN.npy              camera z-depth [H, W]
    depth_conf_NNN.npy         depth confidence [H, W]
    world_points_NNN.npy       world coordinates [H, W, 3], from unprojected depth
                               or from the point map with --use_point_map
    world_points_conf_NNN.npy  point-map confidence [H, W]
    extrinsic_NNN.npy          camera-from-world [3, 4], OpenCV convention
    intrinsic_NNN.npy          pinhole intrinsics [3, 3]
    rgb_NNN.png                preprocessed input image
"""

import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT inference and save per-view depth, point maps and cameras."
    )

    # Input
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory with the input images (jpg, png or tif)",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="'crop' resizes to width 518 and center-crops the height if it exceeds "
        "518, 'pad' pads to 518x518 (default: crop)",
    )

    # Reconstruction
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Use the point-map head instead of unprojected depth for "
        "world_points_NNN.npy and the .ply files, filtered on world_points_conf. "
        "The point map is consistent across views, while unprojected depth is more "
        "accurate but overlaps between views",
    )

    # Point cloud filtering
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=50.0,
        help="Confidence percentile (0-100) for filtering the .ply point clouds, "
        "not the .npy files (default: 50)",
    )

    # Water masks
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory with water masks named like the images, for "
        "pointcloud_water.ply and pointcloud_land.ply (optional)",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Mask value above which a pixel counts as water (default: 127)",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for all outputs"
    )

    args = parser.parse_args()
    if not 0 <= args.conf_threshold <= 100:
        parser.error(
            f"--conf_threshold must be in [0, 100], got {args.conf_threshold:g}"
        )
    if args.mask_dir is not None and not os.path.isdir(args.mask_dir):
        parser.error(f"--mask_dir not found: {args.mask_dir}")
    return args


def find_images(image_dir):
    paths = sorted(
        p
        for p in glob.glob(os.path.join(glob.escape(image_dir), "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    return paths


def load_masks(mask_dir, image_names, height, width, threshold):
    """Return (S, H, W) bool water masks matched to the images by filename stem.

    Masks may use a different extension than the images. A view without a mask
    counts as all land.
    """
    mask_paths = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(mask_dir), "*"))):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in IMAGE_EXTENSIONS:
            mask_paths.setdefault(stem, path)

    masks = np.zeros((len(image_names), height, width), dtype=bool)
    n_matched = 0
    for i, name in enumerate(image_names):
        path = mask_paths.get(os.path.splitext(name)[0])
        if path is None:
            print(f"  No mask for {name}, treating it as land")
            continue
        # nearest keeps the mask binary
        mask = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
        masks[i] = np.array(mask) > threshold
        n_matched += 1

    if n_matched == 0:
        print(f"ERROR: No mask in {mask_dir} matches an image name")
        sys.exit(1)
    print(f"  Masks matched: {n_matched}/{len(image_names)}")
    return masks


def save_ply(filepath, points, colors):
    """Write a binary little-endian PLY with float32 xyz and uint8 rgb."""
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for i, field in enumerate(("x", "y", "z")):
        vertices[field] = points[:, i]
    for i, field in enumerate(("red", "green", "blue")):
        vertices[field] = colors[:, i]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)
    print(f"  Saved {filepath} ({len(points):,} points)")


def load_model(device):
    model = VGGT()
    state_dict = torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL)
    model.load_state_dict(state_dict)
    # free the CPU copy of the 5 GB checkpoint before the model moves to the GPU
    del state_dict
    gc.collect()
    model.eval()
    return model.to(device)


def run_inference(model, images, device, dtype):
    """Run VGGT on preprocessed images of shape (S, 3, H, W).

    :return: dict of per-view numpy arrays, inference time in seconds
    """
    images = images.to(device)
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
        predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    if device == "cuda":
        # kernels run asynchronously, wait for them before stopping the clock
        torch.cuda.synchronize()
    inference_time_s = time.time() - t0

    outputs = {
        "depth": predictions["depth"].squeeze(0).squeeze(-1),
        "depth_conf": predictions["depth_conf"].squeeze(0),
        "world_points": predictions["world_points"].squeeze(0),
        "world_points_conf": predictions["world_points_conf"].squeeze(0),
        "extrinsic": extrinsic.squeeze(0),
        "intrinsic": intrinsic.squeeze(0),
    }
    return {k: v.cpu().numpy() for k, v in outputs.items()}, inference_time_s


def filter_points(points, conf, depth, percentile):
    """Select finite points with positive depth and confidence above the percentile.

    :return: boolean mask over all S*H*W pixels, confidence value at the percentile
    """
    points = points.reshape(-1, 3)
    conf = conf.reshape(-1)
    conf_value = np.percentile(conf, percentile)
    keep = (
        (conf >= conf_value)
        & (depth.reshape(-1) > 1e-4)
        & np.isfinite(points).all(axis=1)
        & (np.abs(points) < 1e6).all(axis=1)
    )
    return keep, conf_value


def format_summary(meta, depth, depth_conf, device, dtype):
    kept = meta["points_after_filter"]
    total = meta["total_pixels"]
    orig_w, orig_h = meta["original_resolution"]
    vggt_w, vggt_h = meta["vggt_resolution"]
    if meta["use_point_map"]:
        source, conf_name = "point map", "world_points_conf"
    else:
        source, conf_name = "depth unprojection", "depth_conf"

    lines = [
        "VGGT inference summary",
        "=" * 40,
        f"Images:            {meta['num_images']} from {meta['image_dir']}",
        f"Original res:      {orig_w} x {orig_h}",
        f"VGGT res:          {vggt_w} x {vggt_h} ({meta['preprocess_mode']})",
        f"Point source:      {source}",
        f"Inference time:    {meta['inference_time_s']:.2f} s on {device} ({dtype})",
        "",
        f"Depth range:       [{depth.min():.4f}, {depth.max():.4f}]",
        f"Depth conf range:  [{depth_conf.min():.4f}, {depth_conf.max():.4f}]",
        f"Conf threshold:    {meta['conf_threshold_value']:.4f} "
        f"({meta['conf_threshold_percentile']:g}th percentile of {conf_name})",
        f"Points kept:       {kept:,} of {total:,} ({100.0 * kept / total:.1f}%)",
    ]
    if "water_points" in meta:
        lines.append(f"Water points:      {meta['water_points']:,}")
        lines.append(f"Land points:       {meta['land_points']:,}")

    lines += ["", "Camera positions (VGGT world frame, arbitrary scale):"]
    for i, name in enumerate(meta["image_names"]):
        x, y, z = meta["camera_positions_world"][i]
        k = meta["predicted_intrinsics"][f"{i:03d}"]
        lines.append(
            f"  [{i:03d}] {name:45s}  pos=({x:8.3f}, {y:8.3f}, {z:8.3f})  "
            f"fx={k['fx']:.1f} fy={k['fy']:.1f}"
        )
    lines += ["", "Georeference with 01_compute_umeyama.py --mode vggt."]
    return lines


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # bfloat16 needs compute capability 8.0 (Ampere) or newer
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: {torch.cuda.get_device_name()} ({gpu_mem_gb:.1f} GB)")
    else:
        dtype = torch.float32
        print("Device: cpu")

    print("\n--- Loading images ---")
    image_paths = find_images(args.image_dir)
    image_names = [os.path.basename(p) for p in image_paths]
    image_sizes = []
    for p in image_paths:
        with Image.open(p) as img:
            image_sizes.append(img.size)
    orig_w, orig_h = image_sizes[0]
    if args.mask_dir and (
        args.preprocess_mode == "pad" or len(set(image_sizes)) > 1 or orig_h > orig_w
    ):
        # masks are only resized, while VGGT also crops or pads these images
        print("ERROR: --mask_dir needs landscape images of one size in crop mode")
        sys.exit(1)

    images = load_and_preprocess_images(image_paths, mode=args.preprocess_mode)
    S, _, vggt_h, vggt_w = images.shape
    print(f"{S} images, {orig_w} x {orig_h} -> {vggt_w} x {vggt_h}")
    masks = None
    if args.mask_dir:
        masks = load_masks(
            args.mask_dir, image_names, vggt_h, vggt_w, args.mask_threshold
        )

    print("\n--- Loading VGGT-1B ---")
    t0 = time.time()
    model = load_model(device)
    print(f"Loaded in {time.time() - t0:.1f} s")

    print("\n--- Running inference ---")
    out, inference_time_s = run_inference(model, images, device, dtype)
    print(f"{S} views in {inference_time_s:.2f} s ({dtype})")

    if args.use_point_map:
        points, points_conf = out["world_points"], out["world_points_conf"]
    else:
        points = unproject_depth_map_to_point_map(
            out["depth"][..., np.newaxis], out["extrinsic"], out["intrinsic"]
        )
        points_conf = out["depth_conf"]

    print("\n--- Saving per-view outputs ---")
    os.makedirs(args.output_dir, exist_ok=True)
    # VGGT preprocessing returns floats in [0, 1]
    rgb = (images.numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    per_view = {
        "depth": out["depth"],
        "depth_conf": out["depth_conf"],
        "world_points": points,
        "world_points_conf": out["world_points_conf"],
        "extrinsic": out["extrinsic"],
        "intrinsic": out["intrinsic"],
    }
    for i in range(S):
        for name, arr in per_view.items():
            np.save(os.path.join(args.output_dir, f"{name}_{i:03d}.npy"), arr[i])
        Image.fromarray(rgb[i]).save(os.path.join(args.output_dir, f"rgb_{i:03d}.png"))
    print(f"  Saved {S} views to {args.output_dir}")

    print("\n--- Building point clouds ---")
    keep, conf_value = filter_points(
        points, points_conf, out["depth"], args.conf_threshold
    )
    kept_pts = points.reshape(-1, 3)[keep]
    kept_rgb = rgb.reshape(-1, 3)[keep]
    total_pixels = S * vggt_h * vggt_w
    print(
        f"  Kept {len(kept_pts):,} of {total_pixels:,} points "
        f"(confidence >= {conf_value:.4f}, {args.conf_threshold:g}th percentile)"
    )
    save_ply(
        os.path.join(args.output_dir, "pointcloud_combined.ply"), kept_pts, kept_rgb
    )

    is_water = None
    if masks is not None:
        is_water = masks.reshape(-1)[keep]
        for name, subset in (("water", is_water), ("land", ~is_water)):
            if subset.any():
                path = os.path.join(args.output_dir, f"pointcloud_{name}.ply")
                save_ply(path, kept_pts[subset], kept_rgb[subset])
            else:
                print(f"  No {name} points, skipping pointcloud_{name}.ply")

    # camera center C = -R^T t (extrinsic is camera-from-world)
    cam_positions = [(-E[:3, :3].T @ E[:3, 3]).tolist() for E in out["extrinsic"]]
    metadata = {
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "mask_dir": os.path.abspath(args.mask_dir) if args.mask_dir else None,
        "num_images": S,
        "image_names": image_names,
        "original_resolution": [orig_w, orig_h],
        "vggt_resolution": [vggt_w, vggt_h],
        "preprocess_mode": args.preprocess_mode,
        "use_point_map": args.use_point_map,
        "conf_threshold_percentile": args.conf_threshold,
        "conf_threshold_value": float(conf_value),
        "mask_threshold": args.mask_threshold,
        "inference_time_s": round(inference_time_s, 2),
        "total_pixels": total_pixels,
        "points_after_filter": len(kept_pts),
        "coordinate_system": "VGGT world frame (arbitrary, anchored near first camera)",
        "extrinsic_convention": "camera-from-world, OpenCV (x-right, y-down, z-forward)",
        "depth_type": "camera z-depth (not Euclidean ray distance)",
        "camera_positions_world": cam_positions,
        "predicted_intrinsics": {
            f"{i:03d}": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            }
            for i, K in enumerate(out["intrinsic"])
        },
    }
    if is_water is not None:
        metadata["water_points"] = int(is_water.sum())
        metadata["land_points"] = int((~is_water).sum())
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {metadata_path}")

    summary_lines = format_summary(
        metadata, out["depth"], out["depth_conf"], device, dtype
    )
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Saved {summary_path}")
    print()
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
