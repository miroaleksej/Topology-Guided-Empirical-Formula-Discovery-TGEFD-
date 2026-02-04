# TGEFD Readiness Matrix

This document summarizes what the system can do today and the remaining work
needed to reach higher readiness stages.

## Current Status
- Overall stage: **MVP / internal prototype**.
- Fit: research workflows, demos, and controlled internal use.
- Not production-ready without additional platform and operational work.

## Capability Snapshot
- Hypothesis hypercube search + sparse regression + topology-guided filtering.
- CLI, config-driven runs, and REST API endpoints.
- Artifact bundles with integrity checks and reproducibility metadata.
- Guardrails: dataset limits, compute budgets, timeouts, and topology sampling.
- Structured audit/access logs and basic metrics endpoint.

## Readiness Matrix

Legend: Ready / Partial / Not Ready

| Area | R&D | Internal | Beta | Production |
| --- | --- | --- | --- | --- |
| Security (authn/authz, secrets) | Partial | Partial | Not Ready | Not Ready |
| Reliability (timeouts, retries, error handling) | Partial | Partial | Not Ready | Not Ready |
| Performance (scaling, async, queueing) | Partial | Partial | Not Ready | Not Ready |
| Observability (metrics, logs, tracing) | Partial | Partial | Not Ready | Not Ready |
| Data handling (inputs, validation, storage) | Partial | Partial | Not Ready | Not Ready |
| Reproducibility (seed, versions, artifacts) | Ready | Ready | Partial | Partial |
| Documentation (runbooks, API, acceptance logic) | Partial | Partial | Not Ready | Not Ready |

## Production Gap Checklist (Concrete Steps)

### Security
- Add authn/authz beyond static API key (tokens, RBAC, or mTLS).
- Trust boundary for X-Forwarded-For and rate limiting behind a known proxy.
- Secret management and audit log storage policy (PII controls).

### Reliability
- Add request queueing / background jobs for long runs.
- Idempotency keys for discover/evaluate to prevent duplicate work.
- Explicit retry strategy for transient storage/network errors.

### Performance
- Async worker pool or job system (Celery/RQ/Temporal).
- GPU/parallelization strategy if needed for topology steps.
- Better topology scaling (sampling strategies, approximate methods).

### Observability
- Export metrics to Prometheus/OTel; add tracing for discover/evaluate spans.
- Dashboards and alerting with SLOs for latency/error/rejection rate.

### Data Handling
- File/URI inputs with validation and size limits (Done).
- Dataset provenance tracking in artifacts (Done).

### Reproducibility
- Persist library versions in all runs by default (not optional).
- Record environment hash and runtime configuration in artifacts.

### Documentation
- OpenAPI spec alignment + contract tests.
- Acceptance/rejection policy doc with examples.
- FAQ for negative results and interpretation.
