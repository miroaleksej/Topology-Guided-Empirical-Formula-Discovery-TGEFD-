# Observability Plan (Production)

This plan describes how to export metrics and traces, and how to define SLOs and alerts.

## Metrics (Prometheus)
- Expose a `/metrics` endpoint for Prometheus scraping (separate from `/v1/metrics`).
- Export:
  - Request count, error count, latency histograms.
  - Acceptance/rejection counters by reason.
  - Queue depth and worker runtime (for async jobs).
- Add labels for endpoint, status code, and identity hash (if allowed).
- Env:
  - `TGEFD_PROMETHEUS_ENABLED=true|false`
  - `TGEFD_METRICS_TENANT_LABEL=true|false` (optional)

## Tracing (OpenTelemetry)
- Instrument request lifecycle for discover/evaluate.
- Propagate trace IDs from inbound requests; generate if missing.
- Export traces to OTLP collector.
- Env:
  - `TGEFD_OTEL_ENABLED=true|false`
  - `TGEFD_OTEL_ENDPOINT=http://collector:4318/v1/traces`
  - `TGEFD_OTEL_SERVICE_NAME=tgefd`
  - `TGEFD_OTEL_SAMPLE_RATIO=1.0`
  - `TGEFD_OTEL_HEADERS=key=value,another=value`

## SLOs and Alerts
- Define SLOs:
  - p95 latency (discover/evaluate) under target.
  - Error rate below threshold (4xx/5xx).
  - Rejection rate within expected bounds.
- Alerts:
  - Error rate spikes.
  - Latency regression.
  - Queue backlog growth.
  - Worker failure rate.

## Dashboards
- Latency and error rate by endpoint.
- Acceptance vs rejection trends with reason breakdown.
- Queue depth and worker throughput.
- Artifact write success/fail metrics.

## Templates
- See:
  - `docs/observability_templates.md` for PromQL alerts and dashboard suggestions.
  - `docs/alerts/prometheus_alerts.yaml` for ready alert rules.
  - `docs/dashboards/tgefd_overview.json` for a ready Grafana dashboard.
  - `docs/observability_integrations.md` for OTel backends + tenant dashboard tips.
