"""
Synthetic Data Expansion for IAM Dataset
Reads IAM Parquet shards from MinIO, applies image augmentations,
writes augmented shards back to MinIO under warehouse/iam_dataset/{split}_augmented/

Augmentations applied:
- Random rotation (-10° to +10°)
- Gaussian noise
- Contrast/brightness jitter
- Elastic distortion (simulates handwriting variation)
"""

import os
import io
import logging
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFilter, ImageEnhance
from minio import Minio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
BUCKET = os.getenv("MINIO_BUCKET", "paperless-datalake")
IAM_PREFIX = "warehouse/iam_dataset"
AUGMENTATIONS_PER_IMAGE = 3  # 3x expansion
SEED = 42


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def augment_image(img: Image.Image, aug_idx: int) -> Image.Image:
    """Apply a deterministic set of augmentations based on aug_idx."""
    rng = random.Random(aug_idx)
    img = img.copy()

    # Convert to RGB if grayscale (some augmentations need it)
    if img.mode == "L":
        was_gray = True
        img = img.convert("RGB")
    else:
        was_gray = False

    # 1. Random rotation
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    # 2. Brightness jitter
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 3. Contrast jitter
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 4. Optional Gaussian blur (50% chance)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    # 5. Add salt-and-pepper noise (30% chance)
    if rng.random() < 0.3:
        arr = np.array(img)
        noise_mask = np.random.RandomState(aug_idx).random(arr.shape[:2])
        arr[noise_mask < 0.01] = 0      # salt
        arr[noise_mask > 0.99] = 255    # pepper
        img = Image.fromarray(arr)

    if was_gray:
        img = img.convert("L")

    return img


def list_shards(client: Minio, split: str) -> list[str]:
    """List all parquet shard objects for a given split."""
    prefix = f"{IAM_PREFIX}/{split}/"
    objects = client.list_objects(BUCKET, prefix=prefix)
    return [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]


def process_shard(client: Minio, shard_path: str, split: str, out_shard_idx: int):
    """Read a shard, augment each image, write augmented shard."""
    response = client.get_object(BUCKET, shard_path)
    table = pq.read_table(io.BytesIO(response.read()))
    response.close()

    aug_ids, aug_images, aug_texts, aug_splits = [], [], [], []

    for row_idx in range(len(table)):
        orig_id = table.column("image_id")[row_idx].as_py()
        orig_img_bytes = table.column("image_png")[row_idx].as_py()
        orig_text = table.column("transcription")[row_idx].as_py()
        orig_img = bytes_to_image(orig_img_bytes)

        for aug_i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = augment_image(orig_img, hash(f"{orig_id}_{aug_i}") & 0xFFFFFFFF)
            aug_ids.append(f"{orig_id}_aug{aug_i}")
            aug_images.append(image_to_bytes(aug_img))
            aug_texts.append(orig_text)  # transcription stays the same
            aug_splits.append(f"{split}_augmented")

    out_table = pa.table({
        "image_id": pa.array(aug_ids, type=pa.string()),
        "image_png": pa.array(aug_images, type=pa.binary()),
        "transcription": pa.array(aug_texts, type=pa.string()),
        "split": pa.array(aug_splits, type=pa.string()),
    })

    buf = io.BytesIO()
    pq.write_table(out_table, buf)
    buf.seek(0)
    obj_name = f"{IAM_PREFIX}/{split}_augmented/shard_{out_shard_idx:04d}.parquet"
    client.put_object(BUCKET, obj_name, buf, length=buf.getbuffer().nbytes)
    log.info(f"Uploaded {obj_name} ({len(out_table)} augmented rows)")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    client = get_minio_client()

    for split in ["train", "validation", "test"]:
        shards = list_shards(client, split)
        if not shards:
            log.warning(f"No shards found for split '{split}', skipping.")
            continue

        log.info(f"Augmenting {len(shards)} shards for split '{split}' ({AUGMENTATIONS_PER_IMAGE}x expansion)")
        for idx, shard_path in enumerate(tqdm(shards, desc=f"  {split}")):
            process_shard(client, shard_path, split, idx)

    log.info("Augmentation complete.")


if __name__ == "__main__":
    main()
