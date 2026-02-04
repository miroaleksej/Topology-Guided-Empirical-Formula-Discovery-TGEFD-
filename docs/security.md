# Security Hardening Plan (Production)

This is a practical checklist for moving from a basic API-key deployment to a
production-grade security posture in Kubernetes or other environments.

## Authentication & Authorization
- Use JWT (HS256 for dev only, JWKS/RS256 in prod) or mTLS end-to-end.
- Support key rotation and revoke lists (token blacklisting or short TTLs).
- Add RBAC or scoped roles for discover/evaluate/metrics endpoints.
- Preserve API key support only for local/dev mode.
  - `TGEFD_AUTH_MODE=api_key|jwt|mtls|none`.

### JWT (HS256)
- Set `TGEFD_AUTH_MODE=jwt`.
- Configure:
  - `TGEFD_JWT_SECRET`
  - `TGEFD_JWT_HEADER` (default: `authorization`)
  - `TGEFD_JWT_BEARER_PREFIX` (default: `bearer `)
  - `TGEFD_JWT_ISSUER` / `TGEFD_JWT_AUDIENCE` (optional)
  - `TGEFD_JWT_SUB_CLAIM` (default: `sub`)
  - `TGEFD_JWT_CLOCK_SKEW_SEC` (default: 60)
  - `TGEFD_JWT_ALLOWED_ALGS` (default: `HS256,RS256`)

### JWT (JWKS / RS256) - Production
- Switch to JWKS-backed validation (public keys, rotation support).
- Enforce `iss` and `aud` claims.
- Cache JWKS with TTL and allow key rotation.
- Reject unsigned or weak algorithms.
  - `TGEFD_JWKS_URL`
  - `TGEFD_JWKS_TTL_SEC`
  - `TGEFD_JWKS_TIMEOUT_SEC`
  - `TGEFD_JWKS_MAX_ATTEMPTS`
  - `TGEFD_JWKS_BACKOFF_SEC`
  - `TGEFD_JWKS_MAX_BACKOFF_SEC`

### mTLS
- Set `TGEFD_AUTH_MODE=mtls`.
- Configure:
  - `TGEFD_MTLS_IDENTITY_HEADER` (default: `x-ssl-client-subject`).
  - Terminate TLS at the edge and forward verified identity only from trusted ingress.

## RBAC / Scopes
- Define scopes for `discover`, `evaluate`, `metrics`, and `jobs`.
- Enforce scope checks per endpoint.
- Store scopes in JWT claims or mTLS policy mapping.
  - `TGEFD_RBAC_ENABLED=true|false`
  - `TGEFD_SCOPE_DISCOVER`, `TGEFD_SCOPE_EVALUATE`, `TGEFD_SCOPE_METRICS`, `TGEFD_SCOPE_JOBS`
  - `TGEFD_JWT_SCOPES_CLAIM` (optional claim override)
  - `TGEFD_MTLS_SCOPES_HEADER` (optional header with scopes)
  - `TGEFD_ROLE_CLAIM` / `TGEFD_ROLES_CLAIM` (optional role claim names)
  - `TGEFD_MTLS_ROLE_HEADER` (optional header with roles)
  - `TGEFD_API_KEY_SCOPES` (default: `*`)
  - Role scope mapping:
    - `TGEFD_ROLE_ADMIN_SCOPES` (default `*`)
    - `TGEFD_ROLE_ANALYST_SCOPES` (default `discover evaluate jobs metrics`)
    - `TGEFD_ROLE_READER_SCOPES` (default `metrics jobs`)

## Rate Limiting by Trusted Identity
- Prefer a trusted identity source (JWT subject, mTLS client cert CN/SAN).
- If behind a proxy/ingress, only trust `x-forwarded-for` from known sources.
- Define an explicit ingress trust boundary and identity header mapping:
  - Use a dedicated header (e.g., `x-auth-request-user`) set only by the ingress.
  - Set `TGEFD_IDENTITY_HEADER` to that header.
  - Do not forward user-supplied identity headers from the public edge.
  - See `docs/ingress_trust_boundary.md`.
- Add per-identity quotas and burst limits.
- Separate budgets for heavy endpoints (discover/evaluate) vs read-only endpoints.
- Configure trusted identity header (e.g., `TGEFD_IDENTITY_HEADER=x-client-id`).
  - Heavy tier: `TGEFD_HEAVY_RATE_LIMIT_PER_MINUTE`, `TGEFD_HEAVY_BURST_LIMIT`.

## Secret Management
- Move all secrets to K8s Secrets or Vault (no plaintext env in manifests).
- Define audit retention/PII policy and storage backend: see `docs/audit_policy.md`.
- Use short-lived credentials for external storage backends (S3/GCS).
- Track secret access and rotation policy.
  - Define rotation cadence and incident response for key compromise.

## Audit Log Policy
- Define what is logged (request_id, identity, action, status, latency).
- Define retention, storage, and redaction policy (PII and sensitive config).
- Ensure audit logs are immutable and access-controlled.
  - Retention example: hot 7 days, warm 30 days, cold 180 days.

## Transport & Network
- Enforce TLS everywhere (ingress + internal).
- Optionally require mTLS for internal service calls.
- Restrict egress to known endpoints where possible.
