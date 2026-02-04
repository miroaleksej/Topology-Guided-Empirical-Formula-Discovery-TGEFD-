# Audit Log Policy (Retention & PII)

## Scope
Applies to API access logs and audit events emitted by TGEFD (`/v1/*`).

## Data captured
- Timestamp, event type, method, path, status code
- Request ID / trace ID
- Hashed identity (not raw tokens)
- Artifact identifiers (artifact_id/config_hash) when available

## PII / Secrets handling
- **Do not store raw API keys, JWTs, or client certificates** in logs.
- Identity is **hashed** before storage (see server hashing logic).
- Payloads are not logged by default.
- Audit log MUST avoid personal data and secrets (PII redaction if extended).

## Retention
- Default: 30 days hot storage, 90 days cold storage.
- Align with your regulatory requirements.
- Rotate logs daily and enforce size-based rollover.

## Storage backend (recommended)
1) **Stdout + centralized logging** (preferred):
   - Ship logs to Loki/ELK/Splunk.
   - Use cluster-level retention policies.
2) **File + object storage** (simple):
   - Set `TGEFD_AUDIT_LOG_PATH=/var/log/tgefd/audit.jsonl`.
   - Use Fluent Bit / Vector to ship to S3/GCS with retention policy.

## Compliance notes
- Audit logs are diagnostic only (not cryptographic proof).
- Ensure access controls on log storage and encryption at rest.
