# Observability Templates

This file provides starter PromQL alert rules and dashboard panel ideas.

## Prometheus Alert Rules (PromQL)

### High error rate (5xx)
```
sum(rate(tgefd_http_requests_total{status_code=~"5.."}[5m]))
/
sum(rate(tgefd_http_requests_total[5m]))
> 0.05
```

### High p95 latency (discover/evaluate)
```
histogram_quantile(
  0.95,
  sum(rate(tgefd_http_request_duration_seconds_bucket{path=~"/v1/(discover|evaluate)"}[5m]))
  by (le)
)
> 2.0
```

### Rejection rate spike
```
sum(rate(tgefd_decisions_total{decision="rejected"}[10m]))
/
sum(rate(tgefd_decisions_total[10m]))
> 0.8
```

### Async queue backlog (if exported)
```
tgefd_queue_depth > 100
```

## Dashboard Panels (Suggestions)
- Request rate by endpoint (stacked).
- Error rate by endpoint (4xx/5xx).
- p50/p95 latency (discover/evaluate).
- Acceptance vs rejection with reason breakdown.
- Async job runtime (p95) and queue depth.
- Artifact write success/fail ratio.

