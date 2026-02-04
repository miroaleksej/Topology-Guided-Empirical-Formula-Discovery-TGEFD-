# Reliability Hardening Plan (Production)

This checklist outlines the minimal reliability upgrades needed for production
usage, with a focus on long-running discovery/evaluation workloads.

## Background Jobs
- Move `discover`/`evaluate` to a job queue (RQ/Celery/Temporal).
- Return a job ID immediately; provide `/status` and `/result` endpoints.
- Add job cancellation and timeout enforcement at the worker level.
- Use separate worker pools for heavy and light workloads.

### Temporal (current path)
- Install: `pip install -e ".[temporal]"`.
- Start worker: `python -m tgefd.temporal_worker`.
- Server env:
  - `TGEFD_JOB_BACKEND=temporal`
  - `TGEFD_TEMPORAL_TARGET=host:port`
  - `TGEFD_TEMPORAL_NAMESPACE=default`
  - `TGEFD_TEMPORAL_TASK_QUEUE=tgefd`
  - `TGEFD_TEMPORAL_WORKFLOW_TIMEOUT_SEC=3600`
  - `TGEFD_TEMPORAL_ACTIVITY_TIMEOUT_SEC=600`
  - `TGEFD_TEMPORAL_ACTIVITY_MAX_ATTEMPTS=3`
  - `TGEFD_JOB_RESULT_TIMEOUT_SEC=2`
  - `TGEFD_JOB_CACHE_SIZE=256`
  - `TGEFD_JOB_CACHE_TTL_SEC=300`
- Async endpoints:
  - `POST /v1/discover_async`
  - `POST /v1/evaluate_async`
  - `GET /v1/jobs/{job_id}?include_result=true`

## Idempotency
- Require an idempotency key for discover/evaluate (optional but recommended).
- Store request hashes and results keyed by idempotency key + identity.
- On retry, return the previous result rather than re-executing.
  - Header: `x-idempotency-key`.
- Persist idempotency records with TTL and cleanup (avoid unbounded growth).
- Reuse existing artifacts when the request hash matches (store-level reuse is implemented).
- Control reuse via `TGEFD_STORE_REUSE=true|false`.

## Retries
- Define retry policy for transient errors (storage uploads, network timeouts).
- Use exponential backoff and bounded retry attempts.
- Ensure retries are safe and idempotent.
- Apply explicit retry policy for storage backends (S3/GCS/local).
- Log retry attempts and final failure reason in audit logs.

## Failure Handling
- Persist partial results for debugging.
- Add structured error codes and actionable messages.
- Track job failure rates and alert on spikes.
- Add worker SLA alerts (queue backlog, job runtime p95, failure rate).
- Add worker liveness checks and automated restarts.
