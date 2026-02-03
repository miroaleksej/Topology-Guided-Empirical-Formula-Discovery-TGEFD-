# On-Call Guide

## What to monitor (priority)
- Error rate > 5% (5xx)
- p95 latency for discover/evaluate > 2s
- Rejection rate spikes
- Job backlog / worker failures

## First-response checklist
1) Check dashboards (Grafana) and recent alerts.
2) Confirm service health: `/healthz`, `/readyz`.
3) Inspect logs for the affected window (API + workers).
4) Identify whether issue is systemic (all requests) or per-tenant.

## Mitigation actions
- Scale API pods (HPA) if CPU or latency is high.
- Scale worker pool if queue backlog grows.
- Temporarily increase rate limits for trusted clients if needed.
- If storage errors, verify S3/GCS credentials and network.

## Escalation
- If sustained 5xx or timeouts > 10m, page infra owner.
- If data integrity is in question, pause ingest and notify security.

## Post-incident
- Record timeline, root cause, and remediation.
- Update runbooks or alert thresholds if necessary.
