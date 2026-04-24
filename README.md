# Paperless-ngx ML Data Platform

**Team:** Dongting Gao (Training), Yikai Sun (Serving), Elnath Zhao (Data)

## Overview

Data infrastructure for a Paperless-ngx document management platform extended with HTR (handwriting text recognition) and semantic search ML features. Deployed on Chameleon Cloud with persistent object storage on CHI@TACC.

## Architecture

| Component | Purpose |
|---|---|
| PostgreSQL | Application state — documents, corrections, queries, feedback |
| CHI@TACC Object Storage | **Persistent** S3 bucket — training datasets, ingested data (survives VM deletion) |
| MinIO | Internal object storage for Docker services |
| Redpanda | Event streaming — decouples API from downstream consumers |
| Qdrant | Vector index for semantic search |

## Quick Start

### 1. Provision on Chameleon

Open `provision_chameleon.ipynb` in the Chameleon Jupyter environment and run all cells. This:
- Provisions a VM on KVM@TACC
- Creates a persistent bucket on CHI@TACC
- Generates S3 credentials
- Starts all Docker services

### 2. Run pipelines

SSH into the VM and run:

```bash
cd ~/paperless_data

# Ingest datasets to persistent storage (CHI@TACC)
source .env.chi
make chi-ingest

# Generate synthetic production data
make generate
make generate-stop

# Run batch pipeline to persistent storage
source .env.chi
make chi-batch

# Demo online features
make demo-htr
make demo-retrieval
```

### 3. Browse persistent storage

Go to https://chi.tacc.chameleoncloud.org → Object Store → Containers → your bucket.

## Repository Structure

```
├── docker/                  # Docker Compose + init SQL
├── ingestion/               # IAM ingestion + augmentation
├── data_generator/          # Stub API + synthetic traffic generator
├── online_features/         # HTR + retrieval feature computation
├── batch_pipeline/          # Versioned training data compilation
├── docs/                    # Design doc + JSON samples
├── provision_chameleon.ipynb # Chameleon setup notebook
└── Makefile                 # All pipeline targets
```

## Make Targets

| Target | Description |
|---|---|
| `make up` / `make down` | Start/stop Docker services |
| `make ingest` | Ingest to local MinIO (dev) |
| `make chi-ingest` | Ingest to CHI@TACC persistent storage |
| `make generate` | Run data generator (5 min of synthetic traffic) |
| `make demo-htr` | Demo HTR online feature path |
| `make demo-retrieval` | Demo retrieval online feature path |
| `make batch` | Batch pipeline to local MinIO |
| `make chi-batch` | Batch pipeline to CHI@TACC persistent storage |

## Persistent Storage

Training data is stored in CHI@TACC object storage (S3-compatible), which persists independently of VM instances. Course staff can browse the bucket in the Horizon GUI without needing SSH access.

Bucket contents:
```
warehouse/
  iam_dataset/         # IAM handwriting line images + transcriptions
  htr_training/        # Versioned HTR training datasets (from batch pipeline)
  retrieval_training/  # Versioned retrieval training datasets (from batch pipeline)
```

## Credentials

- **Adminer (PostgreSQL):** System=PostgreSQL, Server=postgres, User=user, Password=paperless_postgres, DB=paperless
- **MinIO Console:** admin / paperless_minio
- **CHI@TACC bucket:** See `.env.chi` on the VM (generated during provisioning)
