# Paperless-ngx ML Platform — Data Layer

## Services

| Service | Port | Credentials |
|---|---|---|
| PostgreSQL | 5432 | user / paperless_postgres |
| Adminer (PG UI) | 5050 | System: PostgreSQL, Server: postgres |
| MinIO | 9000 (API), 9001 (UI) | admin / paperless_minio |
| Redpanda | 19092 (Kafka) | — |
| Redpanda Console | 8090 | — |
| Qdrant | 6333 (HTTP), 6334 (gRPC) | — |

## Buckets

| Bucket | Purpose |
|---|---|
| `paperless-images` | Raw page images and cropped handwritten regions |
| `paperless-datalake` | Iceberg tables (IAM, SQuAD, training datasets) |
| `paperless-staging` | Temporary ETL staging area |

## Redpanda Topics

| Topic | Written by | Consumed by |
|---|---|---|
| `paperless.uploads` | API on document upload | HTR service, indexing service |
| `paperless.corrections` | API on correction submit | Re-indexing, Airflow |
| `paperless.queries` | API on search query | Airflow retrieval DAG |
| `paperless.feedback` | API on thumbs/click | Airflow retrieval DAG |

## Quick Start

```bash
# On Chameleon VM after SSH:
git clone <your-repo-url>
cd paperless_data
make up
make status
```

## Chameleon Security Groups Required

Open these ports in Horizon GUI:

| Port | Service |
|---|---|
| 22 | SSH |
| 5050 | Adminer |
| 9001 | MinIO Console |
| 8090 | Redpanda Console |
| 6333 | Qdrant |

## Directory Structure

```
.
├── docker/
│   ├── docker-compose.yaml
│   └── init_sql/
│       ├── 00_create_app_tables.sql
│       └── 01_create_iceberg_catalog.sql
├── ingestion/          <- milestone 2
├── data_generator/     <- milestone 3
├── online_features/    <- milestone 3
├── batch_pipeline/     <- milestone 4
├── docs/
│   └── data_design_document.md
├── scripts/
│   └── chameleon_setup.sh
├── Makefile
└── README.md
```
