import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(_THIS_DIR)
)

# VGGT repo root
VGGT_REPO_DIR = os.environ.get(
    "VGGT_REPO_DIR",
    "/path/to/vggt",
)
if VGGT_REPO_DIR not in sys.path:
    sys.path.insert(0, VGGT_REPO_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT dense inference and save depth maps, point clouds, and camera parameters."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images (jpg/png).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to write all outputs."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Optional directory with binary water masks (uint8 PNG, >127 = water). "
        "Filenames must match image filenames (ignoring extension).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=50.0,
        help="Confidence percentile threshold (0-100). Points below this percentile are "
        "excluded from the point cloud. Default: 50 (median).",
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Use direct point map prediction instead of unprojecting depth. "
        "The direct point map is natively multi-view consistent (less overlapping), "
        "while depth unprojection is denser but produces layered per-view points.",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Image preprocessing mode. 'crop' (default) sets width=518 and center-crops; "
        "'pad' pads to square 518x518.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Threshold for binary mask (pixel > threshold = water). Default: 127.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def load_and_resize_masks(mask_dir, image_names, target_h, target_w, threshold=127):
    masks = []
    matched = 0
    for img_name in image_names:
        stem = Path(img_name).stem
        mask_path = None
        for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".tif"]:  # Need all exts because some datsets are diffrent...
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f"  [WARN] No mask found for {img_name}, treating as all-land")
            masks.append(np.zeros((target_h, target_w), dtype=bool))
            continue

        mask_img = Image.open(mask_path).convert("L")
        # Nearest-neighbor, should work?
        mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.uint8)
        masks.append(mask_arr > threshold)
        matched += 1

    print(f"  Masks loaded: {matched}/{len(image_names)} matched")
    return np.stack(masks, axis=0)


def save_ply(filepath, points, colors=None):
    N = points.shape[0]
    has_color = colors is not None and len(colors) == N

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(line + "\n")

    print(f"  Saved PLY: {filepath} ({N:,} points)")


def save_ply_binary(filepath, points, colors=None):
    N = points.shape[0]
    has_color = colors is not None and len(colors) == N

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        if has_color:
            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
            vertices = np.empty(N, dtype=vertex_dtype)
            vertices["x"] = points[:, 0]
            vertices["y"] = points[:, 1]
            vertices["z"] = points[:, 2]
            vertices["red"] = colors[:, 0]
            vertices["green"] = colors[:, 1]
            vertices["blue"] = colors[:, 2]
            vertices.tofile(f)
        else:
            points.astype(np.float32).tofile(f)

    print(f"  Saved PLY: {filepath} ({N:,} points)")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            dtype = torch.float16
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        dtype = torch.float32
        print(f"Device: {device}")
    print(f"Dtype for autocast: {dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Loading images ===")  # Search for images, should set global var
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".tif",
        ".tiff",
    }
    image_paths = sorted(
        [
            p
            for p in glob.glob(os.path.join(args.image_dir, "*"))
            if os.path.splitext(p)[1] in image_extensions
        ]
    )
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    image_names = [os.path.basename(p) for p in image_paths]
    S = len(image_paths)
    print(f"Found {S} images in {args.image_dir}")
    for i, name in enumerate(image_names):
        print(f"  [{i:03d}] {name}")

    with Image.open(image_paths[0]) as img:
        orig_w, orig_h = img.size
    print(f"Original resolution: {orig_w} x {orig_h}")

    images = load_and_preprocess_images(image_paths, mode=args.preprocess_mode)
    # images shape seems to be (S, 3, H_vggt, W_vggt)?
    vggt_h, vggt_w = images.shape[2], images.shape[3]
    print(f"Preprocessed shape: {images.shape} (S={S}, H={vggt_h}, W={vggt_w})")

    print("\n=== Loading VGGT model ===")
    t0 = time.time()
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    model.load_state_dict(state_dict)
    del state_dict  # free CPU memory, learned from Nerfrac
    import gc

    gc.collect()

    model.eval()
    model = model.to(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print("\n=== Running inference ===")
    images_device = images.to(device)
    t0 = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_device)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_device.shape[-2:]
    )

    inference_time = time.time() - t0
    print(f"Inference completed in {inference_time:.2f}s for {S} frames")

    print("\n=== Processing outputs ===")

    # Has batch dimension, need to Remove
    depth = predictions["depth"].squeeze(0).squeeze(-1).cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

    world_points_direct = predictions["world_points"].squeeze(0).cpu().numpy()
    world_points_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()

    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()


    world_points_from_depth = unproject_depth_map_to_point_map(
        depth[..., np.newaxis],  # needs [S, H, W, 1]
        extrinsic_np,
        intrinsic_np,
    )

    # Need the vggt resolution for coloring point cloud
    images_np = images.numpy().transpose(0, 2, 3, 1)
    images_uint8 = (images_np * 255).clip(0, 255).astype(np.uint8)

    print(f"Depth shape: {depth.shape}, range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(
        f"Depth conf shape: {depth_conf.shape}, range: [{depth_conf.min():.4f}, {depth_conf.max():.4f}]"
    )
    print(f"World points (depth-based) shape: {world_points_from_depth.shape}")
    print(f"World points (direct) shape: {world_points_direct.shape}")

    # Free GPU memory, or it will crash
    del predictions, images_device, extrinsic, intrinsic
    if device == "cuda":
        torch.cuda.empty_cache()

    if args.use_point_map:
        pts_source = world_points_direct
        conf_source = world_points_conf
        print("\n=== Using direct point map for 3D outputs ===")
    else:
        pts_source = world_points_from_depth
        conf_source = depth_conf
        print("\n=== Using depth unprojection for 3D outputs ===")

    print("\n=== Saving per-view outputs ===")
    for i in range(S):
        idx_str = f"{i:03d}"

        np.save(os.path.join(args.output_dir, f"depth_{idx_str}.npy"), depth[i])
        np.save(
            os.path.join(args.output_dir, f"depth_conf_{idx_str}.npy"), depth_conf[i]
        )
        np.save(
            os.path.join(args.output_dir, f"world_points_conf_{idx_str}.npy"),
            world_points_conf[i],
        )
        np.save(
            os.path.join(args.output_dir, f"world_points_{idx_str}.npy"),
            pts_source[i],
        )
        np.save(
            os.path.join(args.output_dir, f"extrinsic_{idx_str}.npy"), extrinsic_np[i]
        )
        np.save(
            os.path.join(args.output_dir, f"intrinsic_{idx_str}.npy"), intrinsic_np[i]
        )

        # Save preprocessed RGB as PNG
        rgb_img = Image.fromarray(images_uint8[i])
        rgb_img.save(os.path.join(args.output_dir, f"rgb_{idx_str}.png"))

    print(f"  Saved {S} sets of per-view files")

    print(f"\n=== Building point cloud (conf threshold: {args.conf_threshold}%) ===")

    all_points = pts_source.reshape(-1, 3)
    all_conf = conf_source.reshape(-1)
    all_rgb = images_uint8.reshape(-1, 3)
    all_depth = depth.reshape(-1)

    view_indices = np.repeat(np.arange(S), vggt_h * vggt_w)

    conf_val = np.percentile(all_conf, args.conf_threshold)
    conf_mask = (all_conf >= conf_val) & (all_depth > 1e-4)

    valid_mask = np.isfinite(all_points).all(axis=1) & (np.abs(all_points) < 1e6).all(
        axis=1
    )
    combined_mask = conf_mask & valid_mask

    pts_filtered = all_points[combined_mask]
    rgb_filtered = all_rgb[combined_mask]
    view_filtered = view_indices[combined_mask]

    total_pixels = S * vggt_h * vggt_w
    print(f"  Total pixels: {total_pixels:,}")
    print(
        f"  Confidence threshold value: {conf_val:.4f} ({args.conf_threshold}th percentile)"
    )
    print(
        f"  After conf+valid filter: {pts_filtered.shape[0]:,} points "
        f"({100.0 * pts_filtered.shape[0] / total_pixels:.1f}%)"
    )

    #Never used...
    water_mask_flat = None
    has_masks = args.mask_dir is not None and os.path.isdir(args.mask_dir)

    if has_masks:
        print(f"\n=== Loading water masks from {args.mask_dir} ===")
        water_masks = load_and_resize_masks(
            args.mask_dir, image_names, vggt_h, vggt_w, threshold=args.mask_threshold
        )
        water_mask_flat = water_masks.reshape(-1)[combined_mask]

        n_water = water_mask_flat.sum()
        n_land = (~water_mask_flat).sum()
        print(f"  Water points: {n_water:,}")
        print(f"  Land points:  {n_land:,}")

    print("\n=== Saving point clouds ===")

    save_ply_binary(
        os.path.join(args.output_dir, "pointcloud_combined.ply"),
        pts_filtered.astype(np.float32),
        rgb_filtered,
    )

    # Water only
    if has_masks and water_mask_flat is not None:
        if water_mask_flat.sum() > 0:
            save_ply_binary(
                os.path.join(args.output_dir, "pointcloud_water.ply"),
                pts_filtered[water_mask_flat].astype(np.float32),
                rgb_filtered[water_mask_flat],
            )
        else:
            print("  [WARN] No water points found — skipping pointcloud_water.ply")

        # Land only
        if (~water_mask_flat).sum() > 0:
            save_ply_binary(
                os.path.join(args.output_dir, "pointcloud_land.ply"),
                pts_filtered[~water_mask_flat].astype(np.float32),
                rgb_filtered[~water_mask_flat],
            )
        else:
            print("  [WARN] No land points found — skipping pointcloud_land.ply")

    print("\n=== Saving metadata ===")

    # Need to get the correct camera postions
    cam_positions = []
    for i in range(S):
        R = extrinsic_np[i, :3, :3]
        t = extrinsic_np[i, :3, 3]
        # Camera position in world should be -R^T @ t
        pos = -R.T @ t
        cam_positions.append(pos.tolist())

    metadata = {
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "mask_dir": os.path.abspath(args.mask_dir) if args.mask_dir else None,
        "num_images": S,
        "image_names": image_names,
        "original_resolution": [orig_w, orig_h],
        "vggt_resolution": [vggt_w, vggt_h],
        "preprocess_mode": args.preprocess_mode,
        "conf_threshold_percentile": args.conf_threshold,
        "conf_threshold_value": float(conf_val),
        "mask_threshold": args.mask_threshold,
        "inference_time_s": round(inference_time, 2),
        "total_pixels": total_pixels,
        "points_after_filter": int(pts_filtered.shape[0]),
        "coordinate_system": "VGGT world frame (arbitrary, anchored near first camera)",
        "extrinsic_convention": "camera-from-world, OpenCV (x-right, y-down, z-forward)",
        "depth_type": "camera z-depth (not Euclidean ray distance)",
        "camera_positions_world": cam_positions,
        "predicted_intrinsics": {
            f"{i:03d}": {
                "fx": float(intrinsic_np[i, 0, 0]),
                "fy": float(intrinsic_np[i, 1, 1]),
                "cx": float(intrinsic_np[i, 0, 2]),
                "cy": float(intrinsic_np[i, 1, 2]),
            }
            for i in range(S)
        },
    }

    if has_masks and water_mask_flat is not None:
        metadata["water_points"] = int(water_mask_flat.sum())
        metadata["land_points"] = int((~water_mask_flat).sum())

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Prints for Debugging
    summary_lines = [
        "VGGT Dense Inference Summary",
        "=" * 40,
        f"Images:             {S} from {args.image_dir}",
        f"Original res:       {orig_w} x {orig_h}",
        f"VGGT res:           {vggt_w} x {vggt_h}",
        f"Preprocess mode:    {args.preprocess_mode}",
        f"Inference time:     {inference_time:.2f}s",
        f"Device:             {device}",
        f"Autocast dtype:     {dtype}",
        "",
        f"Depth range:        [{depth.min():.4f}, {depth.max():.4f}]",
        f"Depth conf range:   [{depth_conf.min():.4f}, {depth_conf.max():.4f}]",
        f"Conf threshold:     {args.conf_threshold}% -> value {conf_val:.4f}",
        "",
        f"Total pixels:       {total_pixels:,}",
        f"Points after filter:{pts_filtered.shape[0]:,} ({100.0 * pts_filtered.shape[0] / total_pixels:.1f}%)",
    ]

    if has_masks and water_mask_flat is not None:
        summary_lines += [
            f"Water points:       {int(water_mask_flat.sum()):,}",
            f"Land points:        {int((~water_mask_flat).sum()):,}",
        ]

    summary_lines += [
        "",
        "Camera positions (VGGT world frame):",
    ]
    for i in range(S):
        pos = cam_positions[i]
        fx = intrinsic_np[i, 0, 0]
        fy = intrinsic_np[i, 1, 1]
        summary_lines.append(
            f"  [{i:03d}] {image_names[i]:45s}  "
            f"pos=({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f})  "
            f"fx={fx:.1f} fy={fy:.1f}"
        )

    summary_lines += [
        "",
        "Coordinate system: VGGT arbitrary world frame",
        "  First camera is approximately at the origin.",
        "  Scale is self-consistent but not metric.",
        "  Use Umeyama alignment with known camera GPS positions to georeference.",
        "",
        "Per-view files saved:",
        "  depth_NNN.npy          - camera z-depth [H, W]",
        "  depth_conf_NNN.npy     - depth confidence [H, W]",
        "  world_points_conf_NNN.npy - direct point map confidence [H, W]",
        "  world_points_NNN.npy   - world coords [H, W, 3]  (either direct point_map or unprojected depth)",
        "  extrinsic_NNN.npy      - camera-from-world [3, 4] (OpenCV convention)",
        "  intrinsic_NNN.npy      - pinhole intrinsics [3, 3]",
        "  rgb_NNN.png            - preprocessed input image",
    ]

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
