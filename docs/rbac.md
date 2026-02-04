# RBAC Policies (Roles and Scopes)

This document defines role-to-scope mappings and provides example tokens and checks.

## Default Scopes
- `discover`: `/v1/discover`, `/v1/discover_async`
- `evaluate`: `/v1/evaluate`, `/v1/evaluate_async`
- `metrics`: `/v1/metrics`, `/metrics`
- `jobs`: `/v1/jobs/{job_id}`

## Default Role Mapping
- **admin**: `*` (all scopes)
- **analyst**: `discover evaluate jobs metrics`
- **reader**: `metrics jobs`

Override with:
- `TGEFD_ROLE_ADMIN_SCOPES`
- `TGEFD_ROLE_ANALYST_SCOPES`
- `TGEFD_ROLE_READER_SCOPES`

## JWT Role/Scope Claims
Scopes can come from:
- `scope`, `scopes`, or `scp`
- `TGEFD_JWT_SCOPES_CLAIM`
Roles can come from:
- `role` or `roles`
- `TGEFD_ROLE_CLAIM` / `TGEFD_ROLES_CLAIM`

If scopes are missing but roles are present, scopes are derived from role mapping.

## mTLS Role/Scope Headers
- `TGEFD_MTLS_SCOPES_HEADER` for explicit scopes.
- `TGEFD_MTLS_ROLE_HEADER` for roles.

## Example: HS256 Dev Token (Python)
```python
import time
import jwt

token = jwt.encode(
    {"sub": "user-123", "role": "reader", "exp": time.time() + 3600},
    "dev-secret",
    algorithm="HS256",
)
print(token)
```

## Example: RS256 JWKS Token (Python)
```python
import time
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
token = jwt.encode(
    {"sub": "user-123", "role": "analyst", "exp": time.time() + 3600},
    private_key,
    algorithm="RS256",
    headers={"kid": "key-1"},
)
print(token)
```

## Example: Curl Check
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/metrics
```
