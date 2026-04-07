import os
import torch
import numpy as np
import cv2
from glob import glob
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import gc

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

#INPUT_DIR = "/path/to/images_4/"
#OUTPUT_DIR = "/path/to/masks_4_opt_off/"
INPUT_DIR = "/path/to/images/"
OUTPUT_DIR = "/path/to/masks/"
MAX_IMAGE_SIZE = 4096

# --- HSV Water Recovery Settings ---
# After SAM3 segmentation, recover water pixels that were missed (e.g. seagrass areas that SAM3 classifies as land despite being submerged).
# Stage 1: Region-grow from SAM3 seed into hue-matching pixels.
# Stage 2: Fill enclosed holes in the water mask.
# Stage 3: Morphological closing to smooth edges.
HSV_RECOVERY_ENABLED = False
HSV_MIN_SATURATION = 30  # minimum saturation to be considered "colored" (not gray)
HSV_MIN_WATER_PCT = 5.0  # only apply recovery if SAM3 detected at least this % water
HSV_GROW_KERNEL_SIZE = 5  # region-grow dilation kernel size (pixels)
HSV_GROW_MAX_ITERS = 50  # max region-grow iterations (converges much earlier)
HSV_HOLE_FILL_ENABLED = True  # fill enclosed non-water regions (brown seagrass)
HSV_HOLE_MIN_SAT_RATIO = 0.3  # hole sat threshold = max(20, ratio * water_sat_median)
HSV_MORPH_CLOSE_KERNEL = 15  # morphological closing kernel size (0 to disable)


def hsv_recover_water(image_bgr, binary_mask):
    """Recovers water pixels missed by SAM3 using HSV-guided region growing and hole filling.

    Three-stage recovery:
      1. Region-grow from SAM3 seed into adjacent hue-matching pixels. Uses an
         adaptive IQR-based hue range learned from the seed. Because growth is
         connectivity-constrained, disconnected features (streets, buildings)
         cannot be reached.
      2. Fill enclosed holes: non-water regions completely surrounded by water
         are likely submerged features (brown seagrass, dark water). A saturation
         filter rejects concrete/asphalt holes.
      3. Morphological closing to smooth edges and fill small gaps.

    :param image_bgr: Original image in BGR format (numpy array, H x W x 3).
    :param binary_mask: SAM3 binary mask (numpy uint8, H x W, 255=water 0=land).
    :return: Tuple of (recovered_mask, n_recovered) where recovered_mask is the
             expanded binary mask and n_recovered is the number of pixels added.
    """
    h, w = binary_mask.shape
    total_pixels = h * w
    seed = binary_mask > 127
    seed_count = int(seed.sum())
    water_pct = seed_count / total_pixels * 100

    if water_pct < HSV_MIN_WATER_PCT:
        return binary_mask, 0

    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # --- Stage 1: Region-grow from SAM3 seed into hue-matching pixels ---
    water_hues = img_hsv[seed, 0]
    median_hue = np.median(water_hues)
    q25 = np.percentile(water_hues, 25)
    q75 = np.percentile(water_hues, 75)
    iqr = q75 - q25

    half_range = max(10, min(30, 1.5 * iqr))
    hue_min = max(0, int(median_hue - half_range))
    hue_max = min(179, int(median_hue + half_range))

    candidate = (
        (img_hsv[:, :, 0] >= hue_min)
        & (img_hsv[:, :, 0] <= hue_max)
        & (img_hsv[:, :, 1] > HSV_MIN_SATURATION)
    )

    grow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (HSV_GROW_KERNEL_SIZE, HSV_GROW_KERNEL_SIZE)
    )
    current = seed.astype(np.uint8)
    candidate_uint8 = candidate.astype(np.uint8)
    for _ in range(HSV_GROW_MAX_ITERS):
        dilated = cv2.dilate(current, grow_kernel, iterations=1)
        new_mask = dilated & candidate_uint8
        new_mask = new_mask | current
        if np.array_equal(new_mask, current):
            break
        current = new_mask
    grown = current > 0

    # --- Stage 2: Fill enclosed holes (catches brown seagrass) ---
    if HSV_HOLE_FILL_ENABLED:
        land = ~grown
        labeled_land, n_components = ndimage.label(land.astype(np.uint8))

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        water_sat_median = np.median(img_hsv[grown, 1]) if grown.sum() > 0 else 50
        sat_thresh = max(20, HSV_HOLE_MIN_SAT_RATIO * water_sat_median)

        for comp_id in range(1, n_components + 1):
            comp = labeled_land == comp_id
            comp_sat = img_hsv[comp, 1].mean()
            touches_border = (comp & border).any()

            if not touches_border:
                if comp_sat > sat_thresh:
                    grown[comp] = True
            else:
                comp_hue_median = np.median(img_hsv[comp, 0])
                if hue_min <= comp_hue_median <= hue_max and comp_sat > sat_thresh:
                    grown[comp] = True

    # --- Stage 3: Morphological closing ---
    if HSV_MORPH_CLOSE_KERNEL > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (HSV_MORPH_CLOSE_KERNEL, HSV_MORPH_CLOSE_KERNEL)
        )
        grown = cv2.morphologyEx(
            grown.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    recovered_mask = grown.astype(np.uint8) * 255
    n_recovered = int(grown.sum()) - seed_count
    return recovered_mask, n_recovered


def resize_image(image_pil, max_size):
    """Resize image"""
    w, h = image_pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image_pil.resize((new_w, new_h), Image.LANCZOS), (w, h)
    return image_pil, (w, h)


def try_prompts_with_merging(
    processor,
    image_resized,
    primary_prompts,
    fallback_prompts,
    primary_conf,
    fallback_conf,
):

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(image_resized)

    all_masks = []

    if isinstance(primary_prompts, str):
        primary_prompts = [primary_prompts]

    processor.confidence_threshold = primary_conf

    for prompt in primary_prompts:
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        if masks is not None and len(masks) > 0:
            all_masks.extend(masks)

    if len(all_masks) > 0:
        return all_masks, str(primary_prompts), primary_conf

    processor.confidence_threshold = fallback_conf

    for prompt in fallback_prompts:
        torch.cuda.empty_cache()

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        if masks is not None and len(masks) > 0:
            return masks, prompt, fallback_conf

    return None, None, None


def generate_masks():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = sorted(glob(os.path.join(INPUT_DIR, "*")))
    image_paths = [
        p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_paths)} images. Loading SAM3...")

    torch.cuda.empty_cache()
    gc.collect()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    #primary_prompts = ["river", "stream", "creek", "shallow water"]
    primary_prompts = ["ocean", "sea", "shallow water"]
    #fallback_prompts = ["dark river", "stream", "creek", "flowing river"]
    fallback_prompts = ["shallow water", "coastal water", "clear water", "sea", "ocean"]
    primary_confidence = 0.5  # Slightly lower to catch seagrass
    fallback_confidence = 0.3

    failed_images = []

    for img_path in tqdm(image_paths, desc="Segmenting"):
        try:
            torch.cuda.empty_cache()
            gc.collect()

            image_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image_pil.size
            image_resized, _ = resize_image(image_pil, MAX_IMAGE_SIZE)

            masks, used_prompt, used_conf = try_prompts_with_merging(
                processor,
                image_resized,
                primary_prompts,
                fallback_prompts,
                primary_confidence,
                fallback_confidence,
            )

            binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            if masks is not None and len(masks) > 0:
                for mask in masks:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()

                    if mask.ndim == 3:
                        mask = mask.squeeze(0)

                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    binary_mask = np.maximum(
                        binary_mask, (mask_resized > 0.5).astype(np.uint8) * 255
                    )

                if str(used_prompt) != str(primary_prompts):
                    tqdm.write(
                        f"  {os.path.basename(img_path)}: used fallback '{used_prompt}' @ {used_conf}"
                    )
            else:
                failed_images.append(os.path.basename(img_path))
                tqdm.write(f"  {os.path.basename(img_path)}: NO MASKS FOUND")

            if HSV_RECOVERY_ENABLED:
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            if HSV_RECOVERY_ENABLED:
                binary_mask, n_recovered = hsv_recover_water(image_bgr, binary_mask)
                if n_recovered > 0:
                    pct_recovered = n_recovered / (orig_h * orig_w) * 100
                    tqdm.write(
                        f"  {os.path.basename(img_path)}: HSV recovered {n_recovered:,} px (+{pct_recovered:.1f}%)"
                    )

            # Save
            filename = os.path.basename(img_path)
            save_name = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, binary_mask)

            del masks, image_resized
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            # Print all of the error (need to see what's happening if it fails again...)
            import traceback

            traceback.print_exc()
            print(f"\nError processing {img_path}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print("\nDone! Masks saved to:", OUTPUT_DIR)
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for f in failed_images:
            print(f"  - {f}")


if __name__ == "__main__":
    generate_masks()
