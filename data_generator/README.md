# Data Generator

Synthetic production-style traffic for the Paperless-ngx + ml_hooks stack.

Simulates realistic user behavior against the live Paperless REST API:

| Action | Endpoint | Weight (early/mid/late) |
|---|---|---|
| Upload a synthetic scanned page | `POST /api/documents/post_document/` | 70% / 25% / 10% |
| Keyword + semantic search | `GET  /api/search/?query=...` | 15% / 35% / 40% |
| HTR correction feedback | `POST /api/ml/feedback/` (kind=`htr_correction`) | 10% / 20% / 15% |
| Search result feedback | `POST /api/ml/feedback/` (kind=`search_click` or `search_rating`) | 5% / 20% / 35% |

All traffic authenticated via `Authorization: Token <paperless-api-token>`.

## Usage

### Locally against the integrated stack

```bash
export PAPERLESS_TOKEN=<your-paperless-api-token>

python generator.py \
    --paperless-url   http://localhost:8000 \
    --paperless-token $PAPERLESS_TOKEN \
    --rate            2.0 \
    --duration        300
```

### In Docker (attached to the integrated network)

The generator image is shipped as the default entrypoint in this folder's Dockerfile.

```bash
docker build -t paperless-data-generator .

docker run --rm \
    --network paperless_ml_net \
    -e PAPERLESS_TOKEN=$PAPERLESS_TOKEN \
    paperless-data-generator \
    --paperless-url   http://paperless-webserver-1:8000 \
    --rate            2.0 \
    --duration        300
```

## Arguments

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--paperless-url` | `PAPERLESS_URL` | `http://localhost:8000` | Base URL of the Paperless instance. |
| `--paperless-token` | `PAPERLESS_TOKEN` | *(required)* | Paperless API token. Generate with `python manage.py shell` or via the Django admin. |
| `--rate` | — | `2.0` | Average events per second. |
| `--duration` | — | `300` | Total runtime in seconds. |

## What the traffic exercises

- **Upload path:** every upload fires a Paperless Celery `post_save` signal → `paperless_ml.signals` publishes `paperless.uploads.v1` → `htr_consumer` picks it up → region slicing + crop upload → ml-gateway `/htr` calls → writes to `documents / document_pages / handwritten_regions`.
- **Search path:** every search hits Yikai's `MlGlobalSearchView` in `ml_hooks.views`, which merges Paperless keyword hits with ml-gateway `/search/query` semantic hits and publishes `paperless.queries.v1`.
- **Correction path:** `kind=htr_correction` writes a `Feedback` row in `ml_hooks`, publishes `paperless.corrections.v1`, and fires a `/metrics/correction-recorded` metric hook at ml-gateway. This is the retraining signal.
- **Feedback path:** `kind=search_click` or `search_rating` writes `Feedback` + publishes `paperless.feedback.v1`. CTR and thumbs-down rates are the rollback-trigger signals for retrieval.

## Traffic profile

The weight curve is piecewise:
- **First 20% of runtime:** upload-heavy (no docs to search yet)
- **Middle 60%:** balanced
- **Last 20%:** search + feedback heavy (realistic steady-state usage)

This is intended to produce recognizable daily-usage patterns on the Grafana dashboards (upload spikes early, ingestion lag tapering, sustained search traffic).

## Stub API (legacy)

This folder also ships `stub_api.py` — the original local stub that the generator targeted before the integrated stack existed. It's no longer used for the integrated demo, but remains in the repo so the generator code is self-testable offline:

```bash
# Terminal 1
uvicorn stub_api:app --host 0.0.0.0 --port 8001

# Terminal 2 — point the (old) generator contract at it
# (requires reverting generator.py to the pre-integration version)
```

The stub is not network-accessible from the integrated stack; for live demos always point at the real Paperless.
