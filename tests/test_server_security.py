import json
import base64
import hashlib
import hmac
import json
import logging
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import tgefd.server as server


def _valid_request_config() -> dict:
    return {
        "tgefd_config_version": "1.0",
        "run": {"mode": "discover", "seed": 0, "deterministic": True},
        "dataset": {
            "source": "inline",
            "x": {"format": "array", "values": [[0.0], [0.1], [0.2], [0.3]]},
            "y": {"values": [0.0, 0.1, 0.2, 0.3]},
        },
        "hypothesis_space": {
            "features": [
                {"name": "const", "expression": "1"},
                {"name": "x", "expression": "x"},
                {"name": "x_pow", "expression": "x^p", "parameters": ["p"]},
            ],
            "parameters": {"p": {"type": "float", "values": [1.0, 2.0]}},
            "regularization": {"lambda": [0.01]},
        },
        "topology": {
            "persistent_homology": {"max_dim": 1, "metric": "euclidean"},
            "persistence_image": {
                "birth_range": [0.0, 1.0],
                "pers_range": [0.0, 1.0],
                "pixel_size": 0.05,
                "weight": {"type": "persistence", "params": {"n": 2.0}},
                "kernel": {"type": "gaussian", "params": {"sigma": [[0.05, 0.0], [0.0, 0.05]]}},
            },
            "stability_threshold": 0.2,
        },
        "noise": {"enabled": True, "type": "gaussian", "levels": [0.0], "trials_per_level": 1},
        "evaluation": {
            "acceptance": {"min_stable_components": 1, "require_h1": False},
            "stability_score": {"method": "pi_energy", "aggregation": "mean", "tolerance": 0.1},
            "rejection_reasons": ["no_stable_components"],
        },
        "output": {
            "verbosity": "summary",
            "save_artifacts": False,
            "formats": ["json"],
            "paths": {"base_dir": "./runs"},
        },
        "reproducibility": {
            "tgefd_version": "1.0.0",
            "library_versions": True,
            "hash_algorithm": "sha256",
        },
    }


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _hs256_token(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


@pytest.fixture(autouse=True)
def _reset_security_state(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", None)
    monkeypatch.setattr(server, "_AUTH_MODE", "none")
    monkeypatch.setattr(server, "_MAX_BODY_BYTES", 0)
    monkeypatch.setattr(server, "_RATE_LIMIT_PER_MINUTE", 0.0)
    monkeypatch.setattr(server, "_BURST_LIMIT", 0)
    monkeypatch.setattr(server, "_HEAVY_RATE_LIMIT_PER_MINUTE", 0.0)
    monkeypatch.setattr(server, "_HEAVY_BURST_LIMIT", 0)
    monkeypatch.setattr(server, "_AUDIT_LOG_PATH", None)
    monkeypatch.setattr(server, "_REQUEST_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr(server, "_RATE_LIMITER", server._TokenBucketLimiter())
    monkeypatch.setattr(server, "_IDENTITY_HEADER", None)
    monkeypatch.setattr(server, "_REQUIRE_ASYNC_IDENTITY", False)
    monkeypatch.setattr(server, "_JWT_HEADER", "authorization")
    monkeypatch.setattr(server, "_JWT_BEARER_PREFIX", "bearer ")
    monkeypatch.setattr(server, "_JWT_SECRET", None)
    monkeypatch.setattr(server, "_JWT_ISSUER", None)
    monkeypatch.setattr(server, "_JWT_AUDIENCE", None)
    monkeypatch.setattr(server, "_JWT_SUB_CLAIM", "sub")
    monkeypatch.setattr(server, "_JWT_CLOCK_SKEW_SEC", 60.0)
    monkeypatch.setattr(server, "_JWT_SCOPES_CLAIM", None)
    monkeypatch.setattr(server, "_JWKS_URL", None)
    monkeypatch.setattr(server, "_JWKS_TTL_SEC", 300.0)
    monkeypatch.setattr(server, "_JWKS_TIMEOUT_SEC", 0.1)
    monkeypatch.setattr(server, "_JWKS_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(server, "_JWKS_BACKOFF_SEC", 0.0)
    monkeypatch.setattr(server, "_JWKS_MAX_BACKOFF_SEC", 0.0)
    monkeypatch.setattr(server, "_JWT_ALLOWED_ALGS", "HS256,RS256")
    monkeypatch.setattr(server, "_JWKS_CACHE", None)
    monkeypatch.setattr(server, "_JWKS_CACHE_AT", 0.0)
    monkeypatch.setattr(server, "_ROLE_CLAIM", "role")
    monkeypatch.setattr(server, "_ROLES_CLAIM", "roles")
    monkeypatch.setattr(server, "_MTLS_IDENTITY_HEADER", "x-ssl-client-subject")
    monkeypatch.setattr(server, "_MTLS_ROLE_HEADER", None)
    monkeypatch.setattr(server, "_MTLS_SCOPES_HEADER", None)
    monkeypatch.setattr(server, "_RBAC_ENABLED", False)
    monkeypatch.setattr(server, "_SCOPE_DISCOVER", "discover")
    monkeypatch.setattr(server, "_SCOPE_EVALUATE", "evaluate")
    monkeypatch.setattr(server, "_SCOPE_METRICS", "metrics")
    monkeypatch.setattr(server, "_SCOPE_JOBS", "jobs")
    monkeypatch.setattr(server, "_API_KEY_SCOPES", "*")
    monkeypatch.setattr(server, "_ROLE_ADMIN_SCOPES", "*")
    monkeypatch.setattr(server, "_ROLE_ANALYST_SCOPES", "discover evaluate jobs metrics")
    monkeypatch.setattr(server, "_ROLE_READER_SCOPES", "metrics jobs")


def test_server_requires_api_key(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", "top-secret")
    monkeypatch.setattr(server, "_AUTH_MODE", "api_key")
    client = TestClient(server.app)

    response = client.post("/v1/discover", json={})
    assert response.status_code == 401
    assert response.json()["detail"] == "unauthorized"


def test_health_endpoints_bypass_auth(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", "top-secret")
    monkeypatch.setattr(server, "_AUTH_MODE", "api_key")
    client = TestClient(server.app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_rejects_large_body(monkeypatch):
    monkeypatch.setattr(server, "_MAX_BODY_BYTES", 8)
    client = TestClient(server.app)

    response = client.post("/v1/discover", content="x" * 128, headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"] == "request too large"


def test_server_rate_limit_with_burst(monkeypatch):
    monkeypatch.setattr(server, "_RATE_LIMIT_PER_MINUTE", 60.0)
    monkeypatch.setattr(server, "_BURST_LIMIT", 1)
    monkeypatch.setattr(server, "_RATE_LIMITER", server._TokenBucketLimiter())
    client = TestClient(server.app)

    first = client.get("/openapi.json")
    second = client.get("/openapi.json")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "too many requests"


def test_server_writes_access_and_audit_logs(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", "top-secret")
    monkeypatch.setattr(server, "_AUTH_MODE", "api_key")
    audit_handler = _CaptureHandler()
    access_handler = _CaptureHandler()
    server._AUDIT_LOGGER.addHandler(audit_handler)
    server._ACCESS_LOGGER.addHandler(access_handler)
    client = TestClient(server.app)

    try:
        response = client.post("/v1/evaluate", json={})
    finally:
        server._AUDIT_LOGGER.removeHandler(audit_handler)
        server._ACCESS_LOGGER.removeHandler(access_handler)

    assert response.status_code == 401
    assert any('"event":"auth_failed"' in msg for msg in audit_handler.messages)
    access_payloads = []
    for msg in access_handler.messages:
        try:
            access_payloads.append(json.loads(msg))
        except json.JSONDecodeError:
            continue
    assert any(
        payload.get("event") == "access"
        and payload.get("status_code") == 401
        and payload.get("path") == "/v1/evaluate"
        for payload in access_payloads
    )


def test_server_error_response_does_not_leak_value_error(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise ValueError("internal path: /tmp/secret")

    monkeypatch.setattr(server, "discover", _boom)
    client = TestClient(server.app)
    response = client.post("/v1/discover", json=_valid_request_config())

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid request payload"
    assert "secret" not in response.text


def test_server_error_response_does_not_leak_internal_exception(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("db password leaked")

    monkeypatch.setattr(server, "discover", _boom)
    client = TestClient(server.app)
    response = client.post("/v1/discover", json=_valid_request_config())

    assert response.status_code == 500
    assert response.json()["detail"] == "internal server error"
    assert "password" not in response.text


def test_server_budget_violation_returns_explanation(monkeypatch):
    def _budget_fail(*_args, **_kwargs):
        raise server.BudgetExceededError("budget exceeded: total runs 1000 > max 256")

    monkeypatch.setattr(server, "discover", _budget_fail)
    client = TestClient(server.app)
    response = client.post("/v1/discover", json=_valid_request_config())

    assert response.status_code == 422
    assert "budget exceeded" in response.json()["detail"]


def test_server_request_timeout(monkeypatch):
    def _slow_discover(*_args, **_kwargs):
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(server, "discover", _slow_discover)
    monkeypatch.setattr(server, "_REQUEST_TIMEOUT_SEC", 0.001)
    client = TestClient(server.app)
    response = client.post("/v1/discover", json=_valid_request_config())

    assert response.status_code == 504
    assert response.json()["detail"] == "request timeout"


def test_rate_limit_uses_identity_header(monkeypatch):
    monkeypatch.setattr(server, "_IDENTITY_HEADER", "x-client-id")
    monkeypatch.setattr(server, "_RATE_LIMIT_PER_MINUTE", 60.0)
    monkeypatch.setattr(server, "_BURST_LIMIT", 1)
    monkeypatch.setattr(server, "_RATE_LIMITER", server._TokenBucketLimiter())
    client = TestClient(server.app)

    first = client.get("/openapi.json", headers={"x-client-id": "alpha"})
    second = client.get("/openapi.json", headers={"x-client-id": "beta"})

    assert first.status_code == 200
    assert second.status_code == 200


def test_jwt_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWT_SECRET", "secret")
    token = _hs256_token({"sub": "user-1", "exp": time.time() + 60}, "secret")
    client = TestClient(server.app)

    response = client.get("/v1/metrics", headers={"authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_jwt_auth_rejects_missing_token(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWT_SECRET", "secret")
    client = TestClient(server.app)

    response = client.get("/v1/metrics")
    assert response.status_code == 401


def test_jwt_rejects_disallowed_algorithm(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWT_SECRET", "secret")
    monkeypatch.setattr(server, "_JWT_ALLOWED_ALGS", "RS256")
    token = _hs256_token({"sub": "user-1", "exp": time.time() + 60}, "secret")
    client = TestClient(server.app)

    response = client.get("/v1/metrics", headers={"authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_mtls_auth_requires_header(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "mtls")
    monkeypatch.setattr(server, "_MTLS_IDENTITY_HEADER", "x-ssl-client-subject")
    client = TestClient(server.app)

    missing = client.get("/v1/metrics")
    assert missing.status_code == 401

    ok = client.get("/v1/metrics", headers={"x-ssl-client-subject": "CN=test"})
    assert ok.status_code == 200


def test_heavy_rate_limit_applies_to_async_endpoints(monkeypatch):
    monkeypatch.setattr(server, "_HEAVY_RATE_LIMIT_PER_MINUTE", 60.0)
    monkeypatch.setattr(server, "_HEAVY_BURST_LIMIT", 1)
    monkeypatch.setattr(server, "_RATE_LIMITER", server._TokenBucketLimiter())
    client = TestClient(server.app)

    first = client.post("/v1/discover_async", json=_valid_request_config())
    second = client.post("/v1/discover_async", json=_valid_request_config())

    assert first.status_code == 503
    assert second.status_code == 429


def test_jwks_rs256_jwt_allows_metrics(monkeypatch):
    pytest.importorskip("jwt")
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import jwt as pyjwt

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(public_key)
    jwk_data = json.loads(jwk)
    jwk_data["kid"] = "key-1"
    jwks = {"keys": [jwk_data]}

    def _fake_fetch():
        return jwks

    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWKS_URL", "https://example.invalid/jwks")
    monkeypatch.setattr(server, "_fetch_jwks", _fake_fetch)
    token = pyjwt.encode(
        {"sub": "user-1", "scope": "metrics", "exp": time.time() + 60},
        private_key,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )

    client = TestClient(server.app)
    response = client.get("/v1/metrics", headers={"authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_rbac_rejects_missing_scope(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWT_SECRET", "secret")
    monkeypatch.setattr(server, "_RBAC_ENABLED", True)
    token = _hs256_token({"sub": "user-1", "scope": "metrics", "exp": time.time() + 60}, "secret")
    client = TestClient(server.app)

    response = client.post("/v1/discover", json=_valid_request_config(), headers={"authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_rbac_role_mapping_allows_metrics(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_MODE", "jwt")
    monkeypatch.setattr(server, "_JWT_SECRET", "secret")
    monkeypatch.setattr(server, "_RBAC_ENABLED", True)
    token = _hs256_token({"sub": "user-1", "role": "reader", "exp": time.time() + 60}, "secret")
    client = TestClient(server.app)

    ok = client.get("/v1/metrics", headers={"authorization": f"Bearer {token}"})
    assert ok.status_code == 200

    blocked = client.post("/v1/discover", json=_valid_request_config(), headers={"authorization": f"Bearer {token}"})
    assert blocked.status_code == 403
