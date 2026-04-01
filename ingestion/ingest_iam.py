"""
Ingest IAM Handwriting Dataset → MinIO
Downloads from HuggingFace, extracts line images + transcriptions,
writes as Parquet shards to paperless-datalake/warehouse/iam_dataset/{split}/
"""

import os
import io
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from minio import Minio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
BUCKET = "paperless-datalake"
PREFIX = "warehouse/iam_dataset"
SHARD_SIZE = 500  # rows per parquet file


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def image_to_bytes(img) -> bytes:
    """Convert PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def upload_parquet_shard(client: Minio, table: pa.Table, split: str, shard_idx: int):
    """Write a PyArrow table as Parquet and upload to MinIO."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    obj_name = f"{PREFIX}/{split}/shard_{shard_idx:04d}.parquet"
    client.put_object(BUCKET, obj_name, buf, length=buf.getbuffer().nbytes)
    log.info(f"Uploaded {obj_name} ({len(table)} rows)")


def ingest_split(client: Minio, dataset, split: str):
    """Process one split of the IAM dataset into Parquet shards."""
    log.info(f"Processing split: {split} ({len(dataset)} samples)")

    batch_images = []
    batch_texts = []
    batch_ids = []
    shard_idx = 0

    for i, sample in enumerate(tqdm(dataset, desc=f"  {split}")):
        img_bytes = image_to_bytes(sample["image"])
        text = sample.get("text", sample.get("transcription", ""))
        sample_id = sample.get("id", f"{split}_{i:06d}")

        batch_images.append(img_bytes)
        batch_texts.append(text)
        batch_ids.append(str(sample_id))

        if len(batch_images) >= SHARD_SIZE:
            table = pa.table({
                "image_id": pa.array(batch_ids, type=pa.string()),
                "image_png": pa.array(batch_images, type=pa.binary()),
                "transcription": pa.array(batch_texts, type=pa.string()),
                "split": pa.array([split] * len(batch_ids), type=pa.string()),
            })
            upload_parquet_shard(client, table, split, shard_idx)
            shard_idx += 1
            batch_images, batch_texts, batch_ids = [], [], []

    # Upload remaining rows
    if batch_images:
        table = pa.table({
            "image_id": pa.array(batch_ids, type=pa.string()),
            "image_png": pa.array(batch_images, type=pa.binary()),
            "transcription": pa.array(batch_texts, type=pa.string()),
            "split": pa.array([split] * len(batch_ids), type=pa.string()),
        })
        upload_parquet_shard(client, table, split, shard_idx)


def upload_metadata(client: Minio, ds_info):
    """Upload dataset metadata as JSON."""
    meta = {
        "source": "HuggingFace: Teklia/IAM",
        "description": "IAM Handwriting Database - line-level images with transcriptions",
        "splits": {name: {"num_rows": ds_info[name].num_rows} for name in ds_info},
        "schema": ["image_id", "image_png", "transcription", "split"],
    }
    buf = io.BytesIO(json.dumps(meta, indent=2).encode())
    client.put_object(BUCKET, f"{PREFIX}/metadata.json", buf, length=buf.getbuffer().nbytes)
    log.info("Uploaded metadata.json")


def main():
    log.info("Loading IAM dataset from HuggingFace...")
    ds = load_dataset("Teklia/IAM", "lines", trust_remote_code=True)

    client = get_minio_client()

    # Ensure bucket exists
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    for split in ds:
        ingest_split(client, ds[split], split)

    upload_metadata(client, ds)
    log.info("IAM ingestion complete.")


if __name__ == "__main__":
    main()
