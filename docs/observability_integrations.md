# Observability Integrations (OTel + Multi-tenant Dashboards)

## OpenTelemetry exporters
TGEFD emits OTLP/HTTP traces when enabled. Use an OTel Collector to route to
Jaeger, Tempo, Zipkin, or vendor backends.

### Server env vars
- `TGEFD_OTEL_ENABLED=true`
- `TGEFD_OTEL_ENDPOINT=http://collector:4318/v1/traces`
- `TGEFD_OTEL_HEADERS=Authorization=Bearer <token>` (optional)
- `TGEFD_OTEL_SAMPLE_RATIO=1.0`
- `TGEFD_OTEL_SERVICE_NAME=tgefd`

### Collector routing (example)
Point the endpoint to the collector and configure exporters there. Example
high-level flow:
- receiver: `otlp` (http)
- processors: batch + resource
- exporters: `jaeger` / `otlp` / `logging`

## Per-tenant dashboards
If you want dashboards sliced by tenant identity:

1) Enable tenant labels on Prometheus metrics:
   - `TGEFD_METRICS_TENANT_LABEL=true`
2) Ensure identity is set by auth (JWT/mTLS/API key).
3) Add a Grafana variable `tenant` and filter queries with `tenant=~"$tenant"`.

Example PromQL (requests per tenant):
```
sum by (tenant) (rate(tgefd_http_requests_total[5m]))
```

Example PromQL (p95 latency by tenant):
```
histogram_quantile(0.95,
  sum by (le, tenant) (rate(tgefd_http_request_duration_seconds_bucket[5m]))
)
```

## Notes
- Enabling tenant labels increases metric cardinality; use for controlled tenant sets.
- If tenant labels are disabled, dashboards still work without tenant filtering.
