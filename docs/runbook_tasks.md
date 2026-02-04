# TGEFD Runbook Tasks (60)

All tasks assume you run commands from the repo root:
`/Users/miro-aleksejyandex.ru/Desktop/Gemini/Topology-Guided Empirical Formula Discovery (TGEFD)`

Task 01 - Install base package (editable)
```bash
python -m pip install -e .
```

Task 02 - Run the full test suite
```bash
python -m pytest
```

Task 03 - Run the CLI demo (built-in data)
```bash
python -m tgefd.cli --demo
```

Task 04 - Run the basic example script (auto-bootstraps sys.path)
```bash
python examples/basic_run.py
```

Task 05 - Synthetic truth benchmark (with negative control)
```bash
python examples/synthetic_truth_demo.py --negative-control
```

Task 06 - Synthetic truth with stricter H0 filtering
```bash
python examples/synthetic_truth_demo.py --negative-control --h0-threshold 0.5 --min-component-size 2
```

Task 07 - Validate config
```bash
tgefd validate config.yaml
```

Config topology fields (current config.yaml)
```yaml
topology:
  persistent_homology:
    max_dim: 1
    metric: "euclidean"
  persistence_image:
    birth_range: [0.0, 1.0]
    pers_range: [0.0, 1.0]
    pixel_size: 0.05
    weight:
      type: "persistence"
      params: {n: 2.0}
    kernel:
      type: "gaussian"
      params: {sigma: [[0.05, 0.0], [0.0, 0.05]]}
  stability_threshold: 0.2
  h1_threshold: 0.1
  min_component_size: 5
  point_scale: "none"
```

Config compute_budget fields (guardrails)
```yaml
compute_budget:
  max_hypotheses: 50000
  max_noise_trials: 32
  max_total_runs: 256
  max_model_evals: 2000000
  symbolic_regression_timeout_sec: 10.0
  ph_timeout_sec: 10.0
  max_request_wall_time_sec: 60.0
```

Config dataset limits fields (data/format guardrails)
```yaml
dataset:
  source: "inline"  # inline, file, or uri
  limits:
    max_points: 50000
    max_features: 256
    max_total_values: 5000000
    max_metadata_entries: 64
    max_metadata_value_chars: 2048
```

Optional output.store reliability fields
```yaml
output:
  store:
    upload_timeout_sec: 30.0
    retry_max_attempts: 3
    retry_backoff_sec: 0.2
    retry_max_backoff_sec: 2.0
```

Task 08 - Generate JSON Schema from Pydantic models
```bash
python scripts/generate_schema.py
```

Task 08a - Validate config + print topology fields (quick sync check)
```bash
tgefd validate config.yaml
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("config.yaml"))
topo = cfg.get("topology", {})
print("topology:", topo)
PY
```

Task 08b - Trigger budget violation with explicit explanation
```bash
python - <<'PY'
import yaml
from tgefd.config import parse_config
from tgefd.api_v1 import discover, BudgetExceededError

cfg = yaml.safe_load(open("config.yaml"))
cfg["compute_budget"]["max_hypotheses"] = 1
parsed = parse_config(cfg)
try:
    discover(
        parsed.to_request(),
        evaluation=parsed.to_evaluation_policy(),
        adaptive=parsed.to_adaptive_config(),
        budget=parsed.to_compute_budget(),
    )
except BudgetExceededError as exc:
    print("budget rejection:", exc)
PY
```

Task 09 - Run discovery from config (reuse if already exists)
```bash
tgefd discover config.yaml --reuse
```

Task 10 - Run discovery into a new base dir
```bash
tgefd discover config.yaml --base-dir runs/exp2
```

Task 11 - Show manifest for the artifact derived from config.yaml
```bash
artifact_id=$(python - <<'PY'
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_id_from_hash
cfg = yaml.safe_load(open("config.yaml"))
print(artifact_id_from_hash(canonical_config_hash(cfg)))
PY
)
tgefd show "$artifact_id"
```

Task 12 - Verify integrity for the same artifact
```bash
artifact_id=$(python - <<'PY'
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_id_from_hash
cfg = yaml.safe_load(open("config.yaml"))
print(artifact_id_from_hash(canonical_config_hash(cfg)))
PY
)
tgefd verify "$artifact_id"
```

Task 13 - Print metrics + noise report for the same artifact
```bash
python - <<'PY'
import json
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_path
cfg = yaml.safe_load(open("config.yaml"))
config_hash = canonical_config_hash(cfg)
path = artifact_path("./runs", config_hash)
print("artifact:", path)
print(json.loads((path / "metrics.json").read_text()))
PY
```

Task 13a - Print method_version and decision for the same artifact
```bash
python - <<'PY'
import json
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_path
cfg = yaml.safe_load(open("config.yaml"))
path = artifact_path("./runs", canonical_config_hash(cfg))
result = json.loads((path / "results.json").read_text())
decision = json.loads((path / "decision.json").read_text())
manifest = json.loads((path / "manifest.json").read_text())
print("response method_version:", result["reproducibility"]["method_version"])
print("response seed:", result["reproducibility"]["seed"])
print("response deterministic:", result["reproducibility"]["deterministic"])
print("decision:", decision)
print("manifest run method_version:", manifest["run"]["method_version"])
print("manifest run seed:", manifest["run"]["seed"])
print("manifest run deterministic:", manifest["run"]["deterministic"])
PY
```

Task 13b - Print library versions and evidence files from artifact
```bash
python - <<'PY'
import json
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_path
cfg = yaml.safe_load(open("config.yaml"))
path = artifact_path("./runs", canonical_config_hash(cfg))
manifest = json.loads((path / "manifest.json").read_text())
print("evidence:", (path / "evidence.json").exists())
print("libraries ref:", manifest["reproducibility"]["library_versions"])
if manifest["reproducibility"]["library_versions"]:
    print(json.loads((path / manifest["reproducibility"]["library_versions"]).read_text()))
PY
```

Task 13c - Run discovery with file dataset (sample_measurements.csv) + show provenance
```bash
tgefd discover examples/config_file_dataset.yaml --reuse
python - <<'PY'
import json
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_path
cfg = yaml.safe_load(open("examples/config_file_dataset.yaml"))
path = artifact_path("./runs", canonical_config_hash(cfg))
manifest = json.loads((path / "manifest.json").read_text())
print("dataset provenance:", manifest["dataset"])
PY
```

Task 13d - Run discovery with URI dataset (file://)
```bash
tgefd discover examples/config_uri_dataset.yaml --reuse
```

Task 13e - Run golden experiment + update JSON (ECDSA + Dilithium2)
```bash
python -m pip install -e ".[benchmarks]"
python scripts/run_golden_experiment.py --scheme both --n-values 8,16,32,64,128,256
```

Task 13f - Render benchmark curves (AUC by default)
```bash
python scripts/plot_benchmark_curves.py \
  benchmarks/curves/ecdsa_detection_vs_n.json \
  benchmarks/curves/pq_dilithium_detection_vs_n.json
```
Optional: plot TPR@FPR
```bash
python scripts/plot_benchmark_curves.py --metric "TPR@FPR=0.05" \
  benchmarks/curves/ecdsa_detection_vs_n.json \
  benchmarks/curves/pq_dilithium_detection_vs_n.json
```

Task 13g - Review benchmark value story report
```bash
cat docs/benchmarks_value_story.md
```

Task 14 - Start REST API server (FastAPI)
```bash
python -m pip install -e ".[server]"
uvicorn tgefd.server:app --reload
```

Task 14a - Start API with auth, body limit, rate limit, and audit log
```bash
export TGEFD_API_KEY="change-me"
export TGEFD_MAX_BODY_BYTES=1048576
export TGEFD_RATE_LIMIT_PER_MINUTE=120
export TGEFD_BURST_LIMIT=30
export TGEFD_AUDIT_LOG_PATH="./runs/api-audit.jsonl"
uvicorn tgefd.server:app --reload
```

Task 15 - Call REST /v1/discover using config.yaml
```bash
python - <<'PY' | curl -s -X POST "http://127.0.0.1:8000/v1/discover" -H "Content-Type: application/json" -H "x-api-key: ${TGEFD_API_KEY:-}" -d @-
import json
import yaml
print(json.dumps(yaml.safe_load(open("config.yaml"))))
PY
```

Task 16 - Call REST /v1/evaluate (same config, mode=evaluate)
```bash
python - <<'PY' | curl -s -X POST "http://127.0.0.1:8000/v1/evaluate" -H "Content-Type: application/json" -H "x-api-key: ${TGEFD_API_KEY:-}" -d @-
import json
import yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["run"]["mode"] = "evaluate"
print(json.dumps(cfg))
PY
```

Task 17 - Launch Streamlit UI
```bash
python -m pip install -e ".[ui]"
python -m streamlit run ui/streamlit_app.py
```

Task 18 - List stored artifacts on disk
```bash
ls runs/v1
```

Task 19 - Inspect the first model family in the artifact
```bash
python - <<'PY'
import json
import yaml
from tgefd.artifacts import canonical_config_hash, artifact_path
cfg = yaml.safe_load(open("config.yaml"))
path = artifact_path("./runs", canonical_config_hash(cfg))
models = sorted((path / "models").glob("*.json"))
print("models:", [m.name for m in models])
if models:
    print(json.loads(models[0].read_text()))
PY
```

Task 20 - Create a new config with a different seed and run discovery
```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["run"]["seed"] = 7
with open("config_seed7.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
tgefd discover config_seed7.yaml
```

Task 21 - Metrics and monitoring checklist (production readiness)
- Track service latency (p50/p90/p99), throughput (req/min), and error rate (4xx/5xx).
- Track rejection vs acceptance counts by reason (budget exceeded, validation failure, rate limit, auth).
- Emit structured logs (JSON) with request_id, artifact_id, config_hash, seed, timings, and decision.
- Optional: distributed tracing for discover/evaluate spans (e.g., OpenTelemetry).
- Alert on anomalies: error spikes, latency regressions, rejection rate spikes, throughput drops, timeouts.

Task 22 - Documentation checklist
- OpenAPI / Swagger reflects current request/response models and error codes.
- Runbooks exist for install, run, and debug (local + server).
- Acceptance vs rejection logic is documented with examples and reasons.
- Example configs and pipeline walkthroughs are provided.
- FAQ includes guidance on negative results and how to interpret them.

Task 23 - Readiness assessment
- See docs/readiness.md for current stage, capabilities, and production gap checklist.

Task 24 - ECDSA golden experiment (RNG detectability)
- See docs/ecdsa_golden_experiment.md for the benchmark design and curves vs N.

Task 25 - Security hardening (production)
- See docs/security.md for auth, rate limits by trusted identity, secrets, and audit log policy.

Task 26 - Reliability hardening (production)
- See docs/reliability.md for background jobs, idempotency, and retries.

Task 27 - Run Temporal worker (async jobs)
```bash
python -m pip install -e ".[temporal]"
python -m tgefd.temporal_worker
```

Task 28 - Observability plan (production)
- See docs/observability.md for Prometheus/OTel, tracing, SLOs, and alerts.

Task 29 - RBAC policies
- See docs/rbac.md for role mappings, scopes, and token examples.

Task 30 - Grafana dashboards + Prometheus alerts
- Alert rules: docs/alerts/prometheus_alerts.yaml
- Dashboard JSON: docs/dashboards/tgefd_overview.json

Task 31 - Production checklist (Kubernetes-first)
- See docs/production_checklist.md for Done/Partial/TODO status.

Task 32 - Apply minimal Kubernetes manifests (test cluster)
```bash
kubectl apply -k k8s
kubectl rollout status deployment/tgefd
```

Task 33 - Smoke test K8s deployment
```bash
kubectl port-forward service/tgefd 8000:80
curl -s http://127.0.0.1:8000/healthz
```

Task 34 - Install Helm chart (preferred)
```bash
helm install tgefd helm/tgefd \
  --set image.repository=YOUR_IMAGE \
  --set image.tag=YOUR_TAG
```

Task 35 - Enable Ingress + TLS in Helm
```bash
helm upgrade --install tgefd helm/tgefd \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=tgefd.local \
  --set ingress.tls[0].secretName=tgefd-tls \
  --set ingress.tls[0].hosts[0]=tgefd.local
```

Task 36 - Enable ExternalSecrets (SecretManager)
```bash
helm upgrade --install tgefd helm/tgefd \
  --set externalSecrets.enabled=true \
  --set externalSecrets.secretStoreRef.name=vault-backend \
  --set externalSecrets.secretStoreRef.kind=ClusterSecretStore
```

Task 37 - Review OND methodology addendum
```bash
cat docs/ond_methodology.md
```

Task 38 - Render OND class mapping diagram
```bash
python scripts/render_ond_mapping.py
```

Task 39 - Review OND-RNG-Bench roadmap
```bash
cat docs/ond_rng_bench_roadmap.md
```

Task 40 - Review A/B/C → OND mapping diagram
```bash
cat docs/benchmarks_value_story.md
```

Task 41 - Review OND-RNG-Bench spec
```bash
cat docs/ond_rng_bench_spec.md
```

Task 42 - Dry-run OND-RNG-Bench config validation
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml --dry-run
```

Task 43 - Run OND-RNG-Bench (ECDSA baseline)
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml
```

Task 44 - Run OND-RNG-Bench with DRBG families
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_drbg.yaml
```

Task 45 - Run OND-RNG-Bench with DRBG (Dilithium2)
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/dilithium2_drbg.yaml
```

Task 46 - Auto-render SVG curves from bench output
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_drbg.yaml
```

Task 47 - Apply monitoring manifests (Prometheus/Grafana)
```bash
kubectl apply -f k8s/monitoring/servicemonitor.yaml
kubectl apply -f k8s/monitoring/prometheusrule.yaml
kubectl apply -f k8s/monitoring/grafana_dashboard_configmap.yaml
```

Task 48 - Review production runbook (deploy/smoke/rollback)
```bash
cat docs/ops/production_runbook.md
```

Task 49 - Review ingress trust boundary guidance
```bash
cat docs/ingress_trust_boundary.md
```

Task 50 - Review capacity plan and load test
```bash
cat docs/capacity_plan.md
```

Task 51 - Run k6 load test (example)
```bash
k6 run scripts/load_test_k6.js -e BASE_URL=http://127.0.0.1:8000 -e API_KEY=change-me
```

Task 52 - Run DRBG reseed benchmark (ECDSA)
```bash
python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_drbg_reseed.yaml
```

Task 53 - Review audit/retention/PII policy
```bash
cat docs/audit_policy.md
```

Task 54 - Review ops install/run/debug
```bash
cat docs/ops/install_run_debug.md
```

Task 55 - Review on-call guide
```bash
cat docs/ops/on_call.md
```

Task 56 - Review secret management guide
```bash
cat docs/secret_management.md
```

Task 57 - Apply ExternalSecret (example)
```bash
kubectl apply -f k8s/externalsecret.yaml
```

Task 58 - Review nightly benchmark automation
```bash
cat docs/nightly_benchmarking.md
```

Task 59 - Run nightly OND-RNG-Bench schedule (dry run)
```bash
python scripts/run_nightly_bench.py benchmarks/ond_rng_bench/nightly.yaml --dry-run
```

Task 60 - Review observability integrations (OTel + tenant dashboards)
```bash
cat docs/observability_integrations.md
```

Task 61 - Configure optional tenant labels for metrics
```bash
export TGEFD_METRICS_TENANT_LABEL=true
```

Task 62 - Configure concurrency limits (optional)
```bash
export TGEFD_GLOBAL_CONCURRENCY_LIMIT=50
export TGEFD_HEAVY_CONCURRENCY_LIMIT=10
export TGEFD_CONCURRENCY_TIMEOUT_SEC=0.5
```

Task 63 - Run regression check (example)
```bash
python scripts/check_bench_regressions.py \
  benchmarks/ond_rng_bench/baselines/ecdsa_baseline.json \
  benchmarks/nightly/<bench_id>/curves.json \
  --metric AUC \
  --max-drop 0.05
```
