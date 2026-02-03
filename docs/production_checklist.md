# Production Checklist (Kubernetes-First)

Statuses: Done / Partial / TODO

## Security
- Auth modes (API key/JWT/mTLS): Done
- JWKS/RS256 support + retries/timeout + alg allowlist: Done
- RBAC scopes on endpoints: Done
- Secret management (K8s Secrets/Vault) integration policy: Done
- Audit log retention policy and storage backend: Done
- Ingress trust boundary for identity headers: TODO

## Reliability
- Async jobs via Temporal: Done
- Job status/result endpoint: Done
- Idempotency on results store+reuse: Done
- Explicit retry policy for storage/backends: Partial
- Worker failure recovery + SLA alerts: Partial

## Observability
- Prometheus /metrics endpoint: Done
- OTel tracing to OTLP: Done
- Prometheus alert rules: Done
- Grafana dashboard JSON: Done
- Optional tenant dashboards: Partial
- SLOs adopted in ops tooling: Partial

## Data Handling
- Inline dataset validation + size limits: Done
- File/URI ingestion with validation: Done
- Dataset provenance/versioning: Done

## Performance
- Compute budgets + timeouts + sampling for topology: Done
- Horizontal scaling strategy (workers/queue): Partial
- Load testing and capacity plan: Partial
- Nightly benchmarks + regression gates: Partial

## Documentation & Runbooks
- Security/Reliability/Observability docs: Done
- Production runbook (deploy/rollback/incident): Done
- Acceptance/rejection policy doc: Partial
- FAQ for negative results: Partial

## Kubernetes Fit
- Deployment manifest templates: Done
- Config via env/ConfigMap: Done
- Readiness/liveness probes: Done
- Resource requests/limits: Done
- Helm chart templates: Done
- Ingress + TLS templates: Done
- ExternalSecrets templates: Done
- Monitoring manifests (ServiceMonitor/PrometheusRule/Grafana CM): Done
