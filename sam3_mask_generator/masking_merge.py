"""
Generate binary water masks with SAM3 text prompts.

Usage examples:
    # Pielach (river)
    python sam3_mask_generator/masking_merge.py \\
        --input_dir data/model_formatted/nerfrac/pielach_data/images_8 \\
        --output_dir data/model_formatted/nerfrac/pielach_data/masks_8 \\
        --scene river

    # Coastal scene with seagrass
    python sam3_mask_generator/masking_merge.py \\
        --input_dir /path/to/coast/images_4 \\
        --output_dir /path/to/coast/masks_4 \\
        --scene coastal \\
        --hsv_recovery
"""

import argparse
import gc
import glob
import os
import sys
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# scene: (primary prompts, fallback prompts)
PROMPTS = {
    "river": (["river"], ["dark river", "stream", "creek", "flowing river"]),
    "coastal": (["ocean", "sea", "shallow water"], ["coastal water", "clear water"]),
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

HSV_MIN_SATURATION = 30  # below this a pixel counts as gray, not colored
HSV_MIN_WATER_PCT = 5.0  # skip recovery if SAM3 found less water than this
HSV_GROW_KERNEL_SIZE = 5
HSV_GROW_MAX_ITERS = 50
HSV_HOLE_MIN_SAT_RATIO = 0.3  # holes need saturation > max(20, ratio * water median)
HSV_CLOSE_KERNEL_SIZE = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate binary water masks with SAM3 text prompts."
    )

    # Data
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with the .jpg/.png images (e.g. pielach_data/images_8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the full-size masks, saved as <image stem>.png "
        "with 255 = water (e.g. pielach_data/masks_8)",
    )

    # Segmentation
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        choices=sorted(PROMPTS),
        help="Prompt set from PROMPTS: 'river' (e.g. Pielach) or 'coastal' "
        "(e.g. Santa Maria)",
    )
    parser.add_argument(
        "--primary_conf",
        type=float,
        default=0.5,
        help="Confidence threshold of the primary prompts (default: 0.5)",
    )
    parser.add_argument(
        "--fallback_conf",
        type=float,
        default=0.3,
        help="Confidence threshold of the fallback prompts (default: 0.3)",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=768,
        help="Downscale images to this longest side in pixels before SAM3 "
        "(default: 768)",
    )

    # HSV recovery
    parser.add_argument(
        "--hsv_recovery",
        action="store_true",
        help="Grow the mask into hue-matching pixels and fill enclosed holes "
        "(seagrass, mixed sediment)",
    )
    return parser.parse_args()


def hsv_recover_water(image_rgb, binary_mask):
    """Add water pixels SAM3 missed by hue-guided region growing and hole filling.

    Growth only follows connected pixels, so water-colored streets or roofs that
    don't touch the water are not reached. Holes touching the image border are
    filled only if their hue is in the water range.

    :return: (mask, n_recovered), mask uint8 with 255 = water
    """
    H, W = binary_mask.shape
    seed = binary_mask > 127
    n_seed = int(seed.sum())
    if n_seed / (H * W) * 100 < HSV_MIN_WATER_PCT:
        return binary_mask, 0

    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # region growing into pixels with the water hue
    water_hues = img_hsv[seed, 0]
    median_hue = np.median(water_hues)
    q25, q75 = np.percentile(water_hues, [25, 75])
    half_range = max(10, min(30, 1.5 * (q75 - q25)))
    hue_min = max(0, int(median_hue - half_range))
    hue_max = min(179, int(median_hue + half_range))
    candidate = (
        (img_hsv[:, :, 0] >= hue_min)
        & (img_hsv[:, :, 0] <= hue_max)
        & (img_hsv[:, :, 1] > HSV_MIN_SATURATION)
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (HSV_GROW_KERNEL_SIZE, HSV_GROW_KERNEL_SIZE)
    )
    current = seed.astype(np.uint8)
    for _ in range(HSV_GROW_MAX_ITERS):
        grown = (cv2.dilate(current, kernel) & candidate) | current
        if np.array_equal(grown, current):
            break
        current = grown
    grown = current > 0

    # fill enclosed holes, e.g. submerged seagrass
    labeled_land, n_components = ndimage.label(~grown)
    border = np.zeros((H, W), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    sat_thresh = max(20, HSV_HOLE_MIN_SAT_RATIO * np.median(img_hsv[grown, 1]))
    for comp_id in range(1, n_components + 1):
        comp = labeled_land == comp_id
        if img_hsv[comp, 1].mean() <= sat_thresh:
            continue
        on_border = (comp & border).any()
        if not on_border or hue_min <= np.median(img_hsv[comp, 0]) <= hue_max:
            grown[comp] = True

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (HSV_CLOSE_KERNEL_SIZE, HSV_CLOSE_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed * 255, int(closed.sum()) - n_seed


def resize_image(image, max_size):
    W, H = image.size
    if max(W, H) <= max_size:
        return image
    scale = max_size / max(W, H)
    return image.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def segment(processor, image, prompts, primary_conf, fallback_conf):
    """Masks of all primary prompts, or of the first fallback prompt that finds water.

    :return: (masks, fallback prompt or None), empty list if no prompt found water
    """
    primary_prompts, fallback_prompts = prompts
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)

        processor.set_confidence_threshold(primary_conf)
        masks = []
        for prompt in primary_prompts:
            masks.extend(processor.set_text_prompt(state=state, prompt=prompt)["masks"])
        if masks:
            return masks, None

        processor.set_confidence_threshold(fallback_conf)
        for prompt in fallback_prompts:
            masks = processor.set_text_prompt(state=state, prompt=prompt)["masks"]
            if len(masks) > 0:
                return list(masks), prompt
    return [], None


def union_masks(masks, H, W):
    """Merge SAM3 masks into one uint8 mask at the full image size, 255 = water."""
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    for mask in masks:
        # SAM3 returns masks at the resized image size
        mask = cv2.resize(
            mask[0].cpu().numpy().astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )
        binary_mask[mask > 0.5] = 255
    return binary_mask


def main():
    args = parse_args()

    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(glob.escape(args.input_dir), "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_paths:
        print(f"ERROR: No .jpg or .png images in {args.input_dir}")
        sys.exit(1)
    stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    if len(set(stems)) < len(stems):
        print("ERROR: Several images share a file stem, their masks would collide")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(image_paths)} images. Loading SAM3...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    processor = Sam3Processor(build_sam3_image_model())

    no_water, failed = [], []
    for path in tqdm(image_paths, desc="Segmenting"):
        name = os.path.basename(path)
        try:
            image = Image.open(path).convert("RGB")
            W, H = image.size
            masks, fallback_prompt = segment(
                processor,
                resize_image(image, args.max_image_size),
                PROMPTS[args.scene],
                args.primary_conf,
                args.fallback_conf,
            )
            binary_mask = union_masks(masks, H, W)
            if not masks:
                no_water.append(name)
                tqdm.write(f"  {name}: no water found, saved an all-land mask")
            elif fallback_prompt is not None:
                tqdm.write(f"  {name}: fallback prompt '{fallback_prompt}'")

            if args.hsv_recovery:
                binary_mask, n_recovered = hsv_recover_water(
                    np.array(image), binary_mask
                )
                if n_recovered > 0:
                    tqdm.write(
                        f"  {name}: HSV recovered {n_recovered:,} px "
                        f"(+{100 * n_recovered / (H * W):.1f}%)"
                    )

            mask_name = os.path.splitext(name)[0] + ".png"
            Image.fromarray(binary_mask).save(os.path.join(args.output_dir, mask_name))
        except Exception:
            # load_llff.py drops all masks and 07_make_pointcloud.py pairs them by
            # position if one is missing, so record it and exit 1 after the loop
            failed.append(name)
            tqdm.write(f"  {name}: failed\n{traceback.format_exc()}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nSaved {len(image_paths) - len(failed):,} masks to {args.output_dir}")
    if no_water:
        print(f"No water found in {len(no_water)} images: {', '.join(no_water)}")
    if failed:
        print(f"ERROR: No mask for {len(failed)} images: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
