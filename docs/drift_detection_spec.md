# Production Drift Detection — Integration Spec

**Owner:** Data (Elnath) writes this spec + provides reference data.
**Implementer:** Serving (Yikai) adds the detector to `ml-gateway`.
**Rubric point:** Data-role deliverable #3 — "live inference data quality
and drift in production."

## What we're monitoring

Whether HTR inputs arriving in production look statistically different
from the IAM training distribution the HTR model saw. If yes, the model's
outputs on that data are suspect and we want to (a) know about it, (b) roll
back if it's bad enough.

This is the exact pattern from the course's online-evaluation lab
(`eval-online-chi`), adapted from Food-11/AI-images to IAM/handwriting
crops.

## Where the detector runs

Inside `ml-gateway`, on the `/predict/htr` (or `/htr`) handler, run
asynchronously so it doesn't add user-visible latency. Same thread-pool
pattern the lab uses:

```python
# imports
from concurrent.futures import ThreadPoolExecutor
from alibi_detect.saving import load_detector
from prometheus_client import Counter, Histogram

# at module load
cd = load_detector("cd")          # persisted detector; see "Reference data" below
_drift_pool = ThreadPoolExecutor(max_workers=2)

drift_events_total = Counter(
    "drift_events_total",
    "MMD drift test triggered (is_drift=1)",
)
drift_test_stat = Histogram(
    "drift_test_stat",
    "MMD test statistic per inference",
)

def _detect(x_np):
    pred = cd.predict(x_np)["data"]
    drift_test_stat.observe(pred["test_stat"])
    if pred["is_drift"]:
        drift_events_total.inc()

# inside /predict/htr, right before return:
_drift_pool.submit(_detect, preprocessed_crop.cpu().numpy())
```

## Reference data

I'll stage 500 IAM line crops, preprocessed the same way production crops
are (grayscale, resized/padded to `(IAM_HEIGHT, IAM_WIDTH)`), at:

```
s3://paperless-datalake/warehouse/drift_reference/htr_v1/
    reference.npy      # float32 array shape (500, C, H, W)
    manifest.json      # preprocessing recipe + source shard ids
```

The detector is built offline (on my side) by running:

```python
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch import HiddenOutput, preprocess_drift
from alibi_detect.saving import save_detector
from functools import partial

feature_model = HiddenOutput(htr_model, layer=-1)   # TrOCR encoder's last hidden state
preprocess_fn = partial(preprocess_drift, model=feature_model)

cd_online = MMDDriftOnline(
    x_ref=reference_array,
    ert=300,          # expected run-time between false positives
    window_size=20,   # reasonable for a sequence-level input
    backend="pytorch",
    preprocess_fn=preprocess_fn,
)
save_detector(cd_online, "cd")
```

and I'll hand you the resulting `cd/` directory. It bakes in the TrOCR
encoder reference, so when you load it, you only need alibi-detect and
`torch` in the image.

## Metrics → Prometheus → Grafana

- `drift_events_total` — Counter, increments when `is_drift == 1`.
- `drift_test_stat` — Histogram, raw MMD test statistic.

Prometheus scrape config: same as the existing `ml-gateway` entry — no
new target needed.

Grafana dashboard: "Paperless Drift Monitoring" with two panels
(matches the online-eval lab JSON exactly, substitute `drift_` metric names):

1. `rate(drift_events_total[1m])` — drift events per second
2. `histogram_quantile(0.5, rate(drift_test_stat_bucket[1m]))` — median MMD stat

## Rollback trigger (the piece the rubric wants automated)

Alert rule (Grafana or Prometheus, your call):

```
expr:    rate(drift_events_total[5m]) > 0.2     # > 1 drift hit per 5s sustained
for:     2m
labels:  severity=critical, action=rollback
```

When this fires, your existing `rollback-ctrl` webhook flips the active
HTR model version (or just flips a `PAPERLESS_HTR_FALLBACK=tesseract` env
var — whichever is cheaper to wire up).

## Dependencies I'm adding (for your awareness)

On my side only — nothing new in `ml-gateway`:

- `alibi-detect` (building the reference detector offline)

On your side:

- `alibi-detect` in `ml-gateway`'s `requirements.txt`
- `torch` (probably already there for TrOCR)
- `prometheus-client` (probably already there for existing metrics)

## Timeline

- Reference `.npy` + saved detector ready: I'll drop them in MinIO within
  **2 hours of you sending the ml-gateway token + VM IP**, since I need
  the TrOCR encoder loaded to build the feature extractor the detector
  uses.
- Your integration: aim for one commit that adds the ~30 lines above +
  alibi-detect to requirements.txt.

## Open questions (ping me)

- Does `ml-gateway` already run on GPU or CPU? MMD preprocessing is
  cheaper on GPU but runs fine on CPU for line-crop-sized inputs.
- Do you want drift-by-region or drift-by-document? Region is more
  sensitive; document is noisier but closer to user experience. I'd
  suggest region.
