"""
Online Feature Computation — HTR (Handwriting Text Recognition) Path

Computes features for real-time HTR inference at document upload time:
  1. Load a page image
  2. Detect handwritten regions (simple threshold-based detection)
  3. Crop each region
  4. Preprocess crops into model-ready feature arrays
  5. Output feature dict for each region

This module is integrate-able with the Paperless-ngx upload pipeline.
"""

import os
import io
import json
import logging
import uuid
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
TARGET_HEIGHT = 128
MIN_REGION_WIDTH = 50
MIN_REGION_HEIGHT = 15
BINARIZATION_THRESHOLD = 180


class HTRFeaturePipeline:
    """
    Online feature computation for HTR.
    Takes a page image, detects handwritten regions, crops them,
    and produces model-ready feature arrays.
    """

    def __init__(self):
        log.info("HTR feature pipeline initialized")

    def detect_handwritten_regions(self, page_image: Image.Image) -> list[dict]:
        """
        Detect handwritten regions in a page image.

        Uses horizontal projection to find text lines:
        1. Convert to grayscale, binarize
        2. Project ink density per row
        3. Find contiguous ink-heavy rows → region bounding boxes
        4. Filter by minimum size

        In production, this would be replaced by a detection model.
        """
        gray = page_image.convert("L")
        arr = np.array(gray)

        binary = (arr < BINARIZATION_THRESHOLD).astype(np.uint8)
        h_proj = binary.sum(axis=1)

        regions = []
        in_region = False
        y_start = 0

        for y, count in enumerate(h_proj):
            if count > arr.shape[1] * 0.02 and not in_region:
                in_region = True
                y_start = y
            elif count <= arr.shape[1] * 0.01 and in_region:
                in_region = False
                if y - y_start >= MIN_REGION_HEIGHT:
                    region_slice = binary[y_start:y, :]
                    v_proj = region_slice.sum(axis=0)
                    x_coords = np.where(v_proj > 0)[0]
                    if len(x_coords) > 0:
                        x_start = max(0, x_coords[0] - 5)
                        x_end = min(arr.shape[1], x_coords[-1] + 5)
                        if x_end - x_start >= MIN_REGION_WIDTH:
                            regions.append({
                                "bbox": [int(x_start), int(y_start), int(x_end), int(y)],
                                "width": int(x_end - x_start),
                                "height": int(y - y_start),
                            })

        if in_region and arr.shape[0] - y_start >= MIN_REGION_HEIGHT:
            region_slice = binary[y_start:, :]
            v_proj = region_slice.sum(axis=0)
            x_coords = np.where(v_proj > 0)[0]
            if len(x_coords) > 0:
                x_start = max(0, x_coords[0] - 5)
                x_end = min(arr.shape[1], x_coords[-1] + 5)
                if x_end - x_start >= MIN_REGION_WIDTH:
                    regions.append({
                        "bbox": [int(x_start), int(y_start), int(x_end), int(arr.shape[0])],
                        "width": int(x_end - x_start),
                        "height": int(arr.shape[0] - y_start),
                    })

        log.info(f"Detected {len(regions)} handwritten regions")
        return regions

    def crop_region(self, page_image: Image.Image, bbox: list[int]) -> Image.Image:
        """Crop a region from the page image."""
        x1, y1, x2, y2 = bbox
        return page_image.crop((x1, y1, x2, y2))

    def preprocess_crop(self, crop: Image.Image) -> dict:
        """
        Preprocess a cropped handwriting region into model-ready features.

        Steps:
        1. Convert to grayscale
        2. Resize to fixed height (128px), maintaining aspect ratio
        3. Normalize pixel values to [-1, 1]
        4. Return array metadata (shape, stats)

        In production, the numpy array would be converted to a torch tensor
        and passed directly to the HTR model.
        """
        gray = crop.convert("L")

        w, h = gray.size
        aspect = w / h
        new_w = int(TARGET_HEIGHT * aspect)
        resized = gray.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

        # Normalize to [-1, 1] (same as Normalize(mean=0.5, std=0.5))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5

        return {
            "array_shape": [1, arr.shape[0], arr.shape[1]],  # [C, H, W]
            "original_size": [w, h],
            "resized_size": [new_w, TARGET_HEIGHT],
            "pixel_mean": round(float(arr.mean()), 4),
            "pixel_std": round(float(arr.std()), 4),
        }

    def compute_features(self, page_image: Image.Image, document_id: str = None) -> dict:
        """
        Full online feature computation for one page.
        Returns a feature dict for each detected region.
        """
        start = time.time()

        if document_id is None:
            document_id = str(uuid.uuid4())

        page_w, page_h = page_image.size

        regions = self.detect_handwritten_regions(page_image)

        region_features = []
        for idx, region in enumerate(regions):
            region_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:region:{idx}"))

            crop = self.crop_region(page_image, region["bbox"])
            features = self.preprocess_crop(crop)

            region_features.append({
                "region_id": region_id,
                "region_index": idx,
                "bbox": region["bbox"],
                "crop_width": region["width"],
                "crop_height": region["height"],
                "array_shape": features["array_shape"],
                "original_size": features["original_size"],
                "resized_size": features["resized_size"],
                "pixel_mean": features["pixel_mean"],
                "pixel_std": features["pixel_std"],
            })

        elapsed_ms = round((time.time() - start) * 1000, 1)

        output = {
            "document_id": document_id,
            "page_size": [page_w, page_h],
            "num_regions_detected": len(regions),
            "regions": region_features,
            "inference_time_ms": elapsed_ms,
        }

        return output


def create_sample_page() -> Image.Image:
    """Create a synthetic page image with text for demo purposes."""
    from PIL import ImageDraw
    from faker import Faker
    import random

    fake = Faker()
    random.seed(42)

    width, height = 850, 1100
    img = Image.new("RGB", (width, height), color=(252, 251, 248))
    draw = ImageDraw.Draw(img)

    draw.text((50, 40), "ACADEMIC DEPARTMENT MEMO", fill=(30, 30, 30))
    draw.text((50, 70), "From: Office of the Dean", fill=(80, 80, 80))
    draw.line([(50, 100), (800, 100)], fill=(150, 150, 150), width=2)

    y = 130
    for _ in range(15):
        line = fake.sentence(nb_words=random.randint(8, 14))
        draw.text((50, y), line, fill=(40, 40, 40))
        y += 24

    # Simulated handwritten annotations (darker ink, thicker)
    draw.text((500, 200), "Approved - JW", fill=(20, 20, 150))
    for i in range(20):
        draw.point((500 + i * 6, 218 + random.randint(-2, 2)), fill=(20, 20, 150))

    draw.text((50, 600), "See revised numbers below", fill=(180, 30, 30))
    for i in range(30):
        draw.point((50 + i * 5, 618 + random.randint(-2, 2)), fill=(180, 30, 30))

    draw.text((400, 900), "Return to sender - wrong dept", fill=(20, 80, 20))

    return img


def demo():
    """Run an end-to-end demo of the HTR feature pipeline."""
    pipeline = HTRFeaturePipeline()

    log.info("=" * 60)
    log.info("HTR Online Feature Computation Demo")
    log.info("=" * 60)

    log.info("\nSTEP 1: Creating sample document page...")
    page = create_sample_page()

    log.info("\nSTEP 2: Running feature computation pipeline...")
    log.info("  - Detecting handwritten regions")
    log.info("  - Cropping detected regions")
    log.info("  - Preprocessing crops → model-ready arrays")

    features = pipeline.compute_features(page, document_id="demo-doc-001")

    log.info(f"\nSTEP 3: Results")
    log.info(f"  Document ID: {features['document_id']}")
    log.info(f"  Page size: {features['page_size']}")
    log.info(f"  Regions detected: {features['num_regions_detected']}")
    log.info(f"  Total inference time: {features['inference_time_ms']}ms")

    for region in features["regions"]:
        log.info(f"\n  Region {region['region_index']}:")
        log.info(f"    ID:          {region['region_id']}")
        log.info(f"    Bounding box: {region['bbox']}")
        log.info(f"    Crop size:    {region['crop_width']}x{region['crop_height']}")
        log.info(f"    Array shape:  {region['array_shape']}")
        log.info(f"    Pixel mean:   {region['pixel_mean']}")
        log.info(f"    Pixel std:    {region['pixel_std']}")

    log.info(f"\nFull feature output (JSON):")
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    demo()
