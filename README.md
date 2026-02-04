# TGEFD - Topology-Guided Empirical Formula Discovery

TGEFD is a lightweight research framework that searches over a hypercube of hypothesis parameters, fits sparse models, and filters results using persistent homology on the hypothesis landscape.

Tags
- tgefd, topology-guided, empirical-formula-discovery, sparse-regression, sindy, persistent-homology, tda, topology, hypercube-search, ond, rng-bench, ecdsa, dilithium2, pq-signatures, observability, k8s, prometheus, grafana, opentelemetry

Highlights
- Hypothesis hypercube over feature-shape parameters (e.g., exponents).
- Sparse regression (SINDy-style sequential thresholded least squares).
- Topology-guided filtering via persistent homology on the hypothesis landscape.
- Persistent landscapes/images for vectorized topology.
- Topology-regularized SINDy for stability-aware optimization.

Raw-data defaults
- Feature normalization is off by default (raw coefficients and scales).
- Topology point-cloud scaling is off by default.
- Enable normalization with `--normalize` in the CLI or `normalize=True` in library calls.
- Enable point scaling with `--scale standard` (CLI) or `topology.point_scale: standard` in config.

Quick start

1) Install deps
```
python -m pip install -U numpy ripser pydantic pyyaml pytest
```

2) Install this package (editable)
```
python -m pip install -e .
```

3) Run the demo
```
python -m tgefd.cli --demo
```

4) Example script
```
python examples/basic_run.py
```

5) Synthetic truth benchmark
```
python examples/synthetic_truth_demo.py --negative-control
```

Config schema
- JSON Schema: `schemas/tgefd_config_v1.json` (generated via `python scripts/generate_schema.py`)
- Python validation: `from tgefd.config import parse_config, load_config`
- CLI validation: `tgefd validate config.yaml`

Immutable artifacts
- Run discovery from config: `tgefd discover config.yaml`
- Verify artifact: `tgefd verify sha256-<hash>`
- If the artifact already exists: use `--reuse` to reuse or `--force` to overwrite.
- Each artifact includes immutable `config.yaml`, `results.json`, and `evidence.json`.
- Reproducibility metadata includes config hash, seed, deterministic mode, method version, and library versions (`libraries.json` when enabled).

Evaluation policy
- `evaluation` block in config controls acceptance (min stable components, H1 requirement, PI energy stability, min PI energy, coeff pattern tolerance).
- Rejection is a valid policy outcome (`status: rejected`), not an execution error.

Method & trust
- Acceptance/rejection logic is formalized in `tgefd/evaluation.py` and documented in `docs/theory.md`.
- Negative outcomes are stored as first-class artifacts (`decision.json` and `negative_result.json` for rejected runs).
- Every run records `reproducibility.method_version` for method traceability.
- Trust contract summary: `docs/method_trust.md`.
- Reproducibility/immutability contract: `docs/reproducibility_immutability.md`.

Correctness guardrails
- `compute_budget` enforces runtime limits (hypercube size, noise trials, total runs, model evals).
- Timeouts are enforced for symbolic regression and persistent homology stages.
- Budget violations fail fast with explicit `budget exceeded: ...` explanations.

Data and format safety
- `dataset.source` accepts `inline | file | uri` with CSV/JSON/YAML ingest (YAML requires `PyYAML`).
- Datasets are validated for shape compatibility (`x`/`y` length), finite numeric values, and consistent feature width.
- Dataset limits are configurable via `dataset.limits` (`max_points`, `max_features`, `max_total_values`, metadata limits).
- Dataset provenance (hash/size/source) is recorded in artifact manifests for traceability.
- API-level hard safety limits are also enforced in request validation.

Adaptive hypercube
- `hypothesis_space.adaptive` enables coarse→fine refinement using stable model families.

REST API (FastAPI)
- Install: `python -m pip install -e .[server]`
- Run: `uvicorn tgefd.server:app --reload`
- Health probes: `/healthz` and `/readyz`
- Optional auth/limits:
  - `TGEFD_API_KEY=...` (set `x-api-key` header)
  - `TGEFD_MAX_BODY_BYTES=...` (reject large payloads)
  - `TGEFD_RATE_LIMIT_PER_MINUTE=...` + `TGEFD_BURST_LIMIT=...` (server-side token-bucket rate limiting; also supported via `TGEFD_RATE_LIMIT`)
  - `TGEFD_GLOBAL_CONCURRENCY_LIMIT=...` + `TGEFD_HEAVY_CONCURRENCY_LIMIT=...` (cap concurrent requests)
  - `TGEFD_CONCURRENCY_TIMEOUT_SEC=...` (queue timeout before 429)
  - `TGEFD_AUDIT_LOG_PATH=...` (write JSONL audit events to file; access logs go to standard output)
  - `TGEFD_REQUEST_TIMEOUT_SEC=...` (fail long API requests with `504 request timeout`)
  - `TGEFD_METRICS_TENANT_LABEL=...` (add tenant label to Prometheus metrics)
  - Error responses are sanitized (`invalid request payload` / `internal server error`) to avoid leaking internal details.
  - Compute budget violations return explicit `budget exceeded: ...` explanations (HTTP 422).
- OpenAPI: `http://127.0.0.1:8000/docs`

Artifact storage (S3/MinIO)
- Install: `python -m pip install -e .[cloud]`

Kubernetes (test deploy)
- Manifests: `k8s/`
- Apply: `kubectl apply -k k8s`
- Helm chart: `helm/tgefd/`
- Monitoring integration: `docs/monitoring_k8s.md`
- Production runbook: `docs/ops/production_runbook.md`
- Capacity plan: `docs/capacity_plan.md`
- Audit policy: `docs/audit_policy.md`
- Observability integrations: `docs/observability_integrations.md`
- Ops install/run/debug: `docs/ops/install_run_debug.md`
- On-call guide: `docs/ops/on_call.md`
- Secret management: `docs/secret_management.md`

Benchmarks (ECDSA + Dilithium2 golden experiment)
- Install: `python -m pip install -e ".[benchmarks]"`
- Run: `python scripts/run_golden_experiment.py --scheme both --n-values 8,16,32,64,128,256`
- Plot: `python scripts/plot_benchmark_curves.py benchmarks/curves/ecdsa_detection_vs_n.json benchmarks/curves/pq_dilithium_detection_vs_n.json`
- Methodology addendum: `docs/ond_methodology.md`
- OND-RNG-Bench roadmap: `docs/ond_rng_bench_roadmap.md`

OND-RNG-Bench (industrial)
- Spec: `docs/ond_rng_bench_spec.md`
- Configs: `benchmarks/ond_rng_bench/configs/`
- Run: `python scripts/run_ond_rng_bench.py benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml`
- Nightly + regressions: `docs/nightly_benchmarking.md`
- Configure in `output.store` (backend: s3, bucket, prefix, endpoint_url for MinIO).
- Retries/backoff and upload timeout are configurable in `output.store` (`retry_max_attempts`, `retry_backoff_sec`, `retry_max_backoff_sec`, `upload_timeout_sec`).
- Partial upload failures trigger best-effort rollback of uploaded objects.

Artifact storage (GCS)
- Install: `python -m pip install -e .[gcs]`
- Configure in `output.store` (backend: gcs, bucket, prefix).
- Retries/backoff and upload timeout are configurable in `output.store`.

UI (Streamlit)
- Install: `python -m pip install -e .[ui]`
- Run: `streamlit run ui/streamlit_app.py`

Theory
- See `docs/theory.md` for formal acceptance criteria and Algorithm 1.

Design notes
- The system searches for stable manifolds of solutions, not a single best formula.
- Persistent homology (H0/H1/...) is used to distinguish noise from durable structure.
- Stable candidates are selected from H0 components at a persistence threshold.

Optional visualization
- For plotting persistence diagrams: `python -m pip install persim`

Hashtags
- #tgefd #topologyguided #empiricalformulas #sparselregression #sindy #tda #persistenthomology #ond #rngbench #ecdsa #dilithium2 #pqsignatures #k8s #prometheus #grafana #opentelemetry

Topology vectors & regularization
- Persistent landscapes/images and topology-regularized SINDy require `persim` in addition to `ripser`.
- Persistence images are configured with fixed birth/persistence ranges and pixel size by default (see `topo_sparse_regression`).
