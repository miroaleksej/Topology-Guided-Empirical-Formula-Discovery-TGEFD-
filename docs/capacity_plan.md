# Capacity Plan (P0)

## Goal
Estimate sustainable QPS and latency for `discover`/`evaluate` and async job
submissions; define worker scaling targets.

## Inputs to collect
- p50/p95/p99 latency for `/v1/discover` and `/v1/evaluate`
- CPU/memory per worker pod under load
- Error rate and rejection rate under peak

## Load test plan (k6)
1) Deploy into test cluster with production-like limits.
2) Run k6 with low → medium → high load, record metrics.
3) Derive scaling rules (e.g., 1 worker per X QPS).

### Test profiles (example)
- **Baseline:** VUS=5, DURATION=2m
- **Medium:** VUS=20, DURATION=5m
- **High:** VUS=50, DURATION=5m

### Example k6 commands (with summary export)
```bash
k6 run scripts/load_test_k6.js \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e API_KEY=change-me \
  -e VUS=5 \
  -e DURATION=2m \
  --summary-export runs/k6_baseline.json

k6 run scripts/load_test_k6.js \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e API_KEY=change-me \
  -e VUS=20 \
  -e DURATION=5m \
  --summary-export runs/k6_medium.json

k6 run scripts/load_test_k6.js \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e API_KEY=change-me \
  -e VUS=50 \
  -e DURATION=5m \
  --summary-export runs/k6_high.json
```

### Extract p95/p99 from summary (Python)
```bash
K6_SUMMARY=runs/k6_baseline.json python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["K6_SUMMARY"])
data = json.loads(path.read_text())
metric = data["metrics"]["http_req_duration"]
print(path.name, "p95=", metric["percentiles"]["p(95)"], "p99=", metric["percentiles"]["p(99)"])
PY
```

## Recorded results (fill after run)
| Profile | VUS | Duration | p50 | p95 | p99 | Error rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5 | 2m | TODO | TODO | TODO | TODO | |
| Medium | 20 | 5m | TODO | TODO | TODO | TODO | |
| High | 50 | 5m | TODO | TODO | TODO | TODO | |

## Scaling guidance
- Start with 1 API pod + N worker pods.
- Scale API pods by CPU (>70%) and p95 latency (>2s).
- Scale workers by queue depth / async job wait time.
- Use concurrency limits to cap heavy endpoints if p95 degrades.
  - `TGEFD_GLOBAL_CONCURRENCY_LIMIT` (global in-flight cap)
  - `TGEFD_HEAVY_CONCURRENCY_LIMIT` (discover/evaluate only)
  - `TGEFD_CONCURRENCY_TIMEOUT_SEC` (optional queue timeout)

## Deliverables
- Baseline report: max stable QPS, p95 latency, CPU/memory per pod.
- Autoscaling thresholds for HPA.
