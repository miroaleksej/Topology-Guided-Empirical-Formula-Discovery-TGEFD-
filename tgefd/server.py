from __future__ import annotations

import asyncio
import base64
from collections import deque, OrderedDict
from contextlib import nullcontext
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import urllib.request
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, Response

from .api_v1 import BudgetExceededError, discover, evaluate
from .artifacts import (
    artifact_id_from_hash,
    canonical_config_hash,
    write_artifact_bundle,
)
from .config.models import TGEFDConfig
from .storage import build_store

_PROM_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
    from prometheus_client.exposition import CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover - optional dependency
    CollectorRegistry = None
    Counter = None
    Gauge = None
    Histogram = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"
else:
    _PROM_AVAILABLE = True

_OTEL_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from opentelemetry import propagate, trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace import SpanKind, Status, StatusCode
except Exception:  # pragma: no cover - optional dependency
    propagate = None
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    TraceIdRatioBased = None
    SpanKind = None
    Status = None
    StatusCode = None
else:
    _OTEL_AVAILABLE = True

_JWT_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import jwt
    from jwt.algorithms import RSAAlgorithm
except Exception:  # pragma: no cover - optional dependency
    jwt = None
    RSAAlgorithm = None
else:
    _JWT_AVAILABLE = True

app = FastAPI(title="TGEFD API", version="1.0")


def _read_non_negative_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger("tgefd.server").warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    return max(0, value)


def _read_non_negative_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.getLogger("tgefd.server").warning("Invalid %s=%r; using %.2f", name, raw, default)
        return default
    return max(0.0, value)


def _read_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logging.getLogger("tgefd.server").warning("Invalid %s=%r; using %s", name, raw, default)
    return default


def _read_probability(name: str, default: float) -> float:
    return min(1.0, _read_non_negative_float(name, default))


_API_KEY = os.getenv("TGEFD_API_KEY")
_AUTH_MODE = os.getenv("TGEFD_AUTH_MODE", "").strip().lower()
if not _AUTH_MODE:
    _AUTH_MODE = "api_key" if _API_KEY else "none"
_MAX_BODY_BYTES = _read_non_negative_int("TGEFD_MAX_BODY_BYTES", 0)
_RATE_LIMIT_PER_MINUTE = _read_non_negative_float(
    "TGEFD_RATE_LIMIT_PER_MINUTE",
    _read_non_negative_float("TGEFD_RATE_LIMIT", 0.0),
)
_BURST_LIMIT = _read_non_negative_int("TGEFD_BURST_LIMIT", 0)
if _RATE_LIMIT_PER_MINUTE > 0.0 and _BURST_LIMIT == 0:
    _BURST_LIMIT = max(1, int(_RATE_LIMIT_PER_MINUTE))
_HEAVY_RATE_LIMIT_PER_MINUTE = _read_non_negative_float("TGEFD_HEAVY_RATE_LIMIT_PER_MINUTE", 0.0)
_HEAVY_BURST_LIMIT = _read_non_negative_int("TGEFD_HEAVY_BURST_LIMIT", 0)
if _HEAVY_RATE_LIMIT_PER_MINUTE > 0.0 and _HEAVY_BURST_LIMIT == 0:
    _HEAVY_BURST_LIMIT = max(1, int(_HEAVY_RATE_LIMIT_PER_MINUTE))
_AUDIT_LOG_PATH = os.getenv("TGEFD_AUDIT_LOG_PATH")
_REQUEST_TIMEOUT_SEC = _read_non_negative_float("TGEFD_REQUEST_TIMEOUT_SEC", 0.0)
_TRACE_ENABLED = _read_bool("TGEFD_TRACE_ENABLED", False)
_GLOBAL_CONCURRENCY_LIMIT = _read_non_negative_int("TGEFD_GLOBAL_CONCURRENCY_LIMIT", 0)
_HEAVY_CONCURRENCY_LIMIT = _read_non_negative_int("TGEFD_HEAVY_CONCURRENCY_LIMIT", 0)
_CONCURRENCY_TIMEOUT_SEC = _read_non_negative_float("TGEFD_CONCURRENCY_TIMEOUT_SEC", 0.0)
_METRICS_TENANT_LABEL = _read_bool("TGEFD_METRICS_TENANT_LABEL", False)
_ALERTS_ENABLED = _read_bool("TGEFD_ALERTS_ENABLED", True)
_ALERT_MIN_SAMPLES = max(1, _read_non_negative_int("TGEFD_ALERT_MIN_SAMPLES", 20))
_ALERT_ERROR_RATE_THRESHOLD = _read_probability("TGEFD_ALERT_ERROR_RATE_THRESHOLD", 0.25)
_ALERT_REJECTION_RATE_THRESHOLD = _read_probability("TGEFD_ALERT_REJECTION_RATE_THRESHOLD", 0.8)
_ALERT_P95_LATENCY_MS_THRESHOLD = _read_non_negative_float("TGEFD_ALERT_P95_LATENCY_MS_THRESHOLD", 3000.0)
_ALERT_COOLDOWN_SEC = _read_non_negative_float("TGEFD_ALERT_COOLDOWN_SEC", 60.0)
_JOB_RESULT_TIMEOUT_SEC = _read_non_negative_float("TGEFD_JOB_RESULT_TIMEOUT_SEC", 2.0)
_IDENTITY_HEADER = os.getenv("TGEFD_IDENTITY_HEADER")
_REQUIRE_ASYNC_IDENTITY = _read_bool("TGEFD_REQUIRE_ASYNC_IDENTITY", False)
_JOB_CACHE_SIZE = _read_non_negative_int("TGEFD_JOB_CACHE_SIZE", 256)
_JOB_CACHE_TTL_SEC = _read_non_negative_float("TGEFD_JOB_CACHE_TTL_SEC", 300.0)
_STORE_REUSE_ENABLED = _read_bool("TGEFD_STORE_REUSE", True)
_JWT_HEADER = os.getenv("TGEFD_JWT_HEADER", "authorization")
_JWT_BEARER_PREFIX = os.getenv("TGEFD_JWT_BEARER_PREFIX", "bearer ")
_JWT_SECRET = os.getenv("TGEFD_JWT_SECRET")
_JWT_ISSUER = os.getenv("TGEFD_JWT_ISSUER")
_JWT_AUDIENCE = os.getenv("TGEFD_JWT_AUDIENCE")
_JWT_SUB_CLAIM = os.getenv("TGEFD_JWT_SUB_CLAIM", "sub")
_JWT_CLOCK_SKEW_SEC = _read_non_negative_float("TGEFD_JWT_CLOCK_SKEW_SEC", 60.0)
_JWT_SCOPES_CLAIM = os.getenv("TGEFD_JWT_SCOPES_CLAIM")
_JWKS_URL = os.getenv("TGEFD_JWKS_URL")
_JWKS_TTL_SEC = _read_non_negative_float("TGEFD_JWKS_TTL_SEC", 300.0)
_JWKS_TIMEOUT_SEC = _read_non_negative_float("TGEFD_JWKS_TIMEOUT_SEC", 3.0)
_JWKS_MAX_ATTEMPTS = _read_non_negative_int("TGEFD_JWKS_MAX_ATTEMPTS", 3)
_JWKS_BACKOFF_SEC = _read_non_negative_float("TGEFD_JWKS_BACKOFF_SEC", 0.2)
_JWKS_MAX_BACKOFF_SEC = _read_non_negative_float("TGEFD_JWKS_MAX_BACKOFF_SEC", 2.0)
_JWT_ALLOWED_ALGS = os.getenv("TGEFD_JWT_ALLOWED_ALGS", "HS256,RS256")
_ROLE_CLAIM = os.getenv("TGEFD_ROLE_CLAIM", "role")
_ROLES_CLAIM = os.getenv("TGEFD_ROLES_CLAIM", "roles")
_MTLS_IDENTITY_HEADER = os.getenv("TGEFD_MTLS_IDENTITY_HEADER", "x-ssl-client-subject")
_MTLS_SCOPES_HEADER = os.getenv("TGEFD_MTLS_SCOPES_HEADER")
_MTLS_ROLE_HEADER = os.getenv("TGEFD_MTLS_ROLE_HEADER")
_RBAC_ENABLED = _read_bool("TGEFD_RBAC_ENABLED", False)
_SCOPE_DISCOVER = os.getenv("TGEFD_SCOPE_DISCOVER", "discover")
_SCOPE_EVALUATE = os.getenv("TGEFD_SCOPE_EVALUATE", "evaluate")
_SCOPE_METRICS = os.getenv("TGEFD_SCOPE_METRICS", "metrics")
_SCOPE_JOBS = os.getenv("TGEFD_SCOPE_JOBS", "jobs")
_API_KEY_SCOPES = os.getenv("TGEFD_API_KEY_SCOPES", "*")
_ROLE_ADMIN_SCOPES = os.getenv("TGEFD_ROLE_ADMIN_SCOPES", "*")
_ROLE_ANALYST_SCOPES = os.getenv("TGEFD_ROLE_ANALYST_SCOPES", "discover evaluate jobs metrics")
_ROLE_READER_SCOPES = os.getenv("TGEFD_ROLE_READER_SCOPES", "metrics jobs")
_PROMETHEUS_ENABLED = _read_bool("TGEFD_PROMETHEUS_ENABLED", True)
_OTEL_ENABLED = _read_bool("TGEFD_OTEL_ENABLED", False)
_OTEL_ENDPOINT = os.getenv("TGEFD_OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
_OTEL_SERVICE_NAME = os.getenv("TGEFD_OTEL_SERVICE_NAME", "tgefd")
_OTEL_SAMPLE_RATIO = min(1.0, _read_non_negative_float("TGEFD_OTEL_SAMPLE_RATIO", 1.0))
_OTEL_HEADERS = os.getenv("TGEFD_OTEL_HEADERS", "")

_ACCESS_LOGGER = logging.getLogger("tgefd.access")
_AUDIT_LOGGER = logging.getLogger("tgefd.audit")
_SERVER_LOGGER = logging.getLogger("tgefd.server")
_BAD_REQUEST_MESSAGE = "invalid request payload"
_INTERNAL_ERROR_MESSAGE = "internal server error"

if not _ACCESS_LOGGER.handlers:
    _access_handler = logging.StreamHandler()
    _access_handler.setFormatter(logging.Formatter("%(message)s"))
    _ACCESS_LOGGER.addHandler(_access_handler)
if not _AUDIT_LOGGER.handlers:
    _audit_handler = logging.StreamHandler()
    _audit_handler.setFormatter(logging.Formatter("%(message)s"))
    _AUDIT_LOGGER.addHandler(_audit_handler)

_ACCESS_LOGGER.setLevel(logging.INFO)
_AUDIT_LOGGER.setLevel(logging.INFO)
_ACCESS_LOGGER.propagate = False
_AUDIT_LOGGER.propagate = False


def _log_json(logger: logging.Logger, payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _parse_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


_PROM_REGISTRY = None
_PROM_REQUESTS = None
_PROM_LATENCY = None
_PROM_INFLIGHT = None
_PROM_DECISIONS = None
_PROM_JOB_CACHE_ENTRIES = None
if _PROM_AVAILABLE and _PROMETHEUS_ENABLED:
    _PROM_REQUEST_LABELS = ["method", "path", "status_code"]
    _PROM_LATENCY_LABELS = ["method", "path"]
    _PROM_DECISION_LABELS = ["decision", "reason"]
    if _METRICS_TENANT_LABEL:
        _PROM_REQUEST_LABELS.append("tenant")
        _PROM_LATENCY_LABELS.append("tenant")
        _PROM_DECISION_LABELS.append("tenant")
    _PROM_REGISTRY = CollectorRegistry()
    _PROM_REQUESTS = Counter(
        "tgefd_http_requests_total",
        "Total HTTP requests",
        _PROM_REQUEST_LABELS,
        registry=_PROM_REGISTRY,
    )
    _PROM_LATENCY = Histogram(
        "tgefd_http_request_duration_seconds",
        "HTTP request latency in seconds",
        _PROM_LATENCY_LABELS,
        registry=_PROM_REGISTRY,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _PROM_INFLIGHT = Gauge(
        "tgefd_http_inflight_requests",
        "In-flight HTTP requests",
        ["path"],
        registry=_PROM_REGISTRY,
    )
    _PROM_DECISIONS = Counter(
        "tgefd_decisions_total",
        "Decision counts by reason",
        _PROM_DECISION_LABELS,
        registry=_PROM_REGISTRY,
    )
    _PROM_JOB_CACHE_ENTRIES = Gauge(
        "tgefd_job_cache_entries",
        "Job cache entries",
        registry=_PROM_REGISTRY,
    )


_TRACER = None
if _OTEL_ENABLED and _OTEL_AVAILABLE:
    resource = Resource.create({"service.name": _OTEL_SERVICE_NAME})
    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(_OTEL_SAMPLE_RATIO),
    )
    exporter = OTLPSpanExporter(
        endpoint=_OTEL_ENDPOINT,
        headers=_parse_headers(_OTEL_HEADERS) if _OTEL_HEADERS else None,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("tgefd.server")


_GLOBAL_SEMAPHORE = asyncio.Semaphore(_GLOBAL_CONCURRENCY_LIMIT) if _GLOBAL_CONCURRENCY_LIMIT > 0 else None
_HEAVY_SEMAPHORE = asyncio.Semaphore(_HEAVY_CONCURRENCY_LIMIT) if _HEAVY_CONCURRENCY_LIMIT > 0 else None


def _metric_path(request: Request) -> str:
    path = request.url.path
    if path.startswith("/v1/jobs/"):
        return "/v1/jobs/{job_id}"
    return path


def _metric_tenant(request: Request) -> str:
    if not _METRICS_TENANT_LABEL:
        return ""
    identity = getattr(request.state, "identity", None)
    if identity:
        return str(identity)
    return "anonymous"


def _labels_with_tenant(base: dict[str, str], request: Request) -> dict[str, str]:
    if not _METRICS_TENANT_LABEL:
        return base
    labels = dict(base)
    labels["tenant"] = _metric_tenant(request)
    return labels


_HEAVY_PATHS = {
    "/v1/discover",
    "/v1/evaluate",
    "/v1/discover_async",
    "/v1/evaluate_async",
}


async def _acquire_concurrency_limit(semaphore: asyncio.Semaphore | None, request: Request, label: str) -> None:
    if semaphore is None:
        return
    try:
        if _CONCURRENCY_TIMEOUT_SEC > 0.0:
            await asyncio.wait_for(semaphore.acquire(), timeout=_CONCURRENCY_TIMEOUT_SEC)
        else:
            await semaphore.acquire()
    except asyncio.TimeoutError as exc:
        _audit_log("concurrency_limit", request, 429, f"{label} concurrency limit")
        raise HTTPException(status_code=429, detail="too many concurrent requests") from exc


_JWKS_CACHE: dict[str, Any] | None = None
_JWKS_CACHE_AT = 0.0
_JWKS_LOCK = Lock()


def _split_scopes(raw: str | None) -> set[str]:
    if not raw:
        return set()
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return {p for p in parts if p}


def _allowed_jwt_algs() -> set[str]:
    return {alg.strip().upper() for alg in _JWT_ALLOWED_ALGS.replace(",", " ").split() if alg.strip()}


def _role_scope_map() -> dict[str, set[str]]:
    return {
        "admin": _split_scopes(_ROLE_ADMIN_SCOPES),
        "analyst": _split_scopes(_ROLE_ANALYST_SCOPES),
        "reader": _split_scopes(_ROLE_READER_SCOPES),
    }


def _scopes_from_roles(roles: set[str]) -> set[str]:
    mapping = _role_scope_map()
    scopes: set[str] = set()
    for role in roles:
        scopes |= mapping.get(role.lower(), set())
    return scopes


def _roles_from_payload(payload: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for key in (_ROLE_CLAIM, _ROLES_CLAIM, "role", "roles"):
        if not key:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            roles.update(str(v) for v in value)
        elif isinstance(value, str):
            roles.update(_split_scopes(value))
    return {r for r in roles if r}


def _scopes_from_payload(payload: dict[str, Any]) -> set[str]:
    if _JWT_SCOPES_CLAIM:
        value = payload.get(_JWT_SCOPES_CLAIM)
        if isinstance(value, list):
            return {str(v) for v in value}
        if isinstance(value, str):
            return _split_scopes(value)
    for key in ("scope", "scopes", "scp"):
        value = payload.get(key)
        if isinstance(value, list):
            return {str(v) for v in value}
        if isinstance(value, str):
            return _split_scopes(value)
    roles = _roles_from_payload(payload)
    if roles:
        return _scopes_from_roles(roles)
    return set()


def _fetch_jwks() -> dict[str, Any]:
    if not _JWKS_URL:
        raise ValueError("jwks url not configured")
    attempt = 0
    last_exc = None
    max_attempts = max(1, _JWKS_MAX_ATTEMPTS)
    while attempt < max_attempts:
        attempt += 1
        try:
            with urllib.request.urlopen(_JWKS_URL, timeout=_JWKS_TIMEOUT_SEC) as response:
                payload = response.read()
            data = json.loads(payload.decode("utf-8"))
            if "keys" not in data:
                raise ValueError("jwks response missing keys")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = min(_JWKS_MAX_BACKOFF_SEC, _JWKS_BACKOFF_SEC * (2 ** (attempt - 1)))
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError(f"jwks fetch failed after {max_attempts} attempts: {last_exc}")


def _get_jwks() -> dict[str, Any]:
    global _JWKS_CACHE, _JWKS_CACHE_AT
    now = time.time()
    with _JWKS_LOCK:
        if _JWKS_CACHE is None or (_JWKS_TTL_SEC > 0 and now - _JWKS_CACHE_AT > _JWKS_TTL_SEC):
            _JWKS_CACHE = _fetch_jwks()
            _JWKS_CACHE_AT = now
        return _JWKS_CACHE


def _jwt_payload_hs256(token: str) -> dict[str, Any]:
    allowed = _allowed_jwt_algs()
    if "HS256" not in allowed:
        raise ValueError("jwt algorithm not allowed")
    if _JWT_AVAILABLE:
        if not _JWT_SECRET:
            raise ValueError("jwt secret not configured")
        header = jwt.get_unverified_header(token)
        if str(header.get("alg", "")).upper() != "HS256":
            raise ValueError("unsupported jwt algorithm")
        return jwt.decode(
            token,
            _JWT_SECRET,
            algorithms=["HS256"],
            issuer=_JWT_ISSUER,
            audience=_JWT_AUDIENCE,
            leeway=_JWT_CLOCK_SKEW_SEC,
            options={"verify_aud": bool(_JWT_AUDIENCE)},
        )
    # fallback to local HS256 decoder
    return _jwt_payload_local(token)


def _jwt_payload_rs256(token: str) -> dict[str, Any]:
    if not _JWT_AVAILABLE or RSAAlgorithm is None:
        raise ValueError("jwt rs256 support not available")
    allowed = _allowed_jwt_algs()
    if "RS256" not in allowed:
        raise ValueError("jwt algorithm not allowed")
    jwks = _get_jwks()
    header = jwt.get_unverified_header(token)
    if str(header.get("alg", "")).upper() != "RS256":
        raise ValueError("unsupported jwt algorithm")
    kid = header.get("kid")
    keys = jwks.get("keys", [])
    key = None
    if kid:
        for candidate in keys:
            if candidate.get("kid") == kid:
                key = candidate
                break
    if key is None and len(keys) == 1:
        key = keys[0]
    if key is None:
        raise ValueError("jwks key not found")
    public_key = RSAAlgorithm.from_jwk(json.dumps(key))
    return jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        issuer=_JWT_ISSUER,
        audience=_JWT_AUDIENCE,
        leeway=_JWT_CLOCK_SKEW_SEC,
        options={"verify_aud": bool(_JWT_AUDIENCE)},
    )


def _configure_audit_logger() -> None:
    if not _AUDIT_LOG_PATH:
        return
    abspath = os.path.abspath(_AUDIT_LOG_PATH)
    for handler in _AUDIT_LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abspath:
            return
    parent_dir = os.path.dirname(abspath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_handler = logging.FileHandler(abspath)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    _AUDIT_LOGGER.addHandler(file_handler)


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


@dataclass
class _EndpointStats:
    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0


class _TokenBucketLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def allow(self, key: str, *, rate_per_minute: float, burst: int) -> bool:
        if rate_per_minute <= 0.0 or burst <= 0:
            return True

        now = time.monotonic()
        refill_rate = rate_per_minute / 60.0

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(burst), updated_at=now)
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(float(burst), bucket.tokens + elapsed * refill_rate)
                bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                allowed = True
            else:
                allowed = False

            self._buckets[key] = bucket
            if len(self._buckets) > 4096:
                cutoff = now - 600.0
                stale_keys = [k for k, state in self._buckets.items() if state.updated_at < cutoff]
                for stale_key in stale_keys[:512]:
                    self._buckets.pop(stale_key, None)

        return allowed


_RATE_LIMITER = _TokenBucketLimiter()


class _ServerMonitoring:
    def __init__(self, window_sec: float = 60.0) -> None:
        self._window_sec = max(1.0, window_sec)
        self._started_at = time.monotonic()
        self._lock = Lock()
        self._recent: deque[tuple[float, float, int]] = deque()
        self._total_requests = 0
        self._total_client_errors = 0
        self._total_server_errors = 0
        self._endpoints: dict[str, _EndpointStats] = {}
        self._accepted = 0
        self._rejected = 0
        self._last_alert_at: dict[str, float] = {}

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window_sec
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        q = max(0.0, min(1.0, q))
        rank = q * (len(ordered) - 1)
        lower = int(rank)
        upper = min(len(ordered) - 1, lower + 1)
        if upper == lower:
            return ordered[lower]
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _should_emit_alert_locked(self, key: str, now: float) -> bool:
        if not _ALERTS_ENABLED:
            return False
        last = self._last_alert_at.get(key)
        if last is None:
            self._last_alert_at[key] = now
            return True
        if _ALERT_COOLDOWN_SEC <= 0.0 or now - last >= _ALERT_COOLDOWN_SEC:
            self._last_alert_at[key] = now
            return True
        return False

    def _build_snapshot_locked(self, now: float) -> dict[str, Any]:
        self._prune_locked(now)
        window_count = len(self._recent)
        window_errors = sum(1 for _, _, status in self._recent if status >= 400)
        latencies = [duration for _, duration, _ in self._recent]
        p50 = self._percentile(latencies, 0.50)
        p95 = self._percentile(latencies, 0.95)
        mean_latency = (sum(latencies) / window_count) if window_count else 0.0
        throughput_rpm = (window_count / self._window_sec) * 60.0
        total_errors = self._total_client_errors + self._total_server_errors
        decision_total = self._accepted + self._rejected
        endpoint_stats = {}
        for path, stats in sorted(self._endpoints.items()):
            endpoint_stats[path] = {
                "requests": stats.requests,
                "errors": stats.errors,
                "error_rate": (stats.errors / stats.requests) if stats.requests else 0.0,
                "latency_ms_mean": (stats.total_latency_ms / stats.requests) if stats.requests else 0.0,
            }
        return {
            "uptime_sec": max(0.0, now - self._started_at),
            "requests": {
                "total": self._total_requests,
                "client_errors": self._total_client_errors,
                "server_errors": self._total_server_errors,
                "error_rate": (total_errors / self._total_requests) if self._total_requests else 0.0,
                "window_count": window_count,
                "window_error_rate": (window_errors / window_count) if window_count else 0.0,
                "throughput_rpm": throughput_rpm,
                "latency_ms": {
                    "p50": p50,
                    "p95": p95,
                    "mean": mean_latency,
                },
            },
            "decisions": {
                "accepted": self._accepted,
                "rejected": self._rejected,
                "acceptance_rate": (self._accepted / decision_total) if decision_total else 0.0,
                "rejection_rate": (self._rejected / decision_total) if decision_total else 0.0,
                "total": decision_total,
            },
            "endpoints": endpoint_stats,
        }

    def _detect_alerts_locked(self, now: float, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        req = snapshot["requests"]
        decisions = snapshot["decisions"]

        if req["window_count"] >= _ALERT_MIN_SAMPLES:
            if req["window_error_rate"] > _ALERT_ERROR_RATE_THRESHOLD and self._should_emit_alert_locked(
                "request_error_rate", now
            ):
                alerts.append(
                    {
                        "name": "request_error_rate_high",
                        "threshold": _ALERT_ERROR_RATE_THRESHOLD,
                        "value": req["window_error_rate"],
                        "samples": req["window_count"],
                    }
                )
            p95 = req["latency_ms"]["p95"]
            if p95 > _ALERT_P95_LATENCY_MS_THRESHOLD and self._should_emit_alert_locked("request_latency_p95", now):
                alerts.append(
                    {
                        "name": "request_latency_p95_high",
                        "threshold": _ALERT_P95_LATENCY_MS_THRESHOLD,
                        "value": p95,
                        "samples": req["window_count"],
                    }
                )

        if (
            decisions["total"] >= _ALERT_MIN_SAMPLES
            and decisions["rejection_rate"] > _ALERT_REJECTION_RATE_THRESHOLD
            and self._should_emit_alert_locked("rejection_rate", now)
        ):
            alerts.append(
                {
                    "name": "rejection_rate_high",
                    "threshold": _ALERT_REJECTION_RATE_THRESHOLD,
                    "value": decisions["rejection_rate"],
                    "samples": decisions["total"],
                }
            )
        return alerts

    def record_request(self, *, path: str, status_code: int, duration_ms: float) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            self._total_requests += 1
            if status_code >= 500:
                self._total_server_errors += 1
            elif status_code >= 400:
                self._total_client_errors += 1
            stats = self._endpoints.get(path)
            if stats is None:
                stats = _EndpointStats()
                self._endpoints[path] = stats
            stats.requests += 1
            if status_code >= 400:
                stats.errors += 1
            stats.total_latency_ms += max(0.0, duration_ms)
            self._recent.append((now, max(0.0, duration_ms), status_code))
            snapshot = self._build_snapshot_locked(now)
            return self._detect_alerts_locked(now, snapshot)

    def record_decision(self, *, accepted: bool) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if accepted:
                self._accepted += 1
            else:
                self._rejected += 1
            snapshot = self._build_snapshot_locked(now)
            return self._detect_alerts_locked(now, snapshot)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return self._build_snapshot_locked(now)


_MONITORING = _ServerMonitoring()


@dataclass
class _JobCacheEntry:
    status_payload: dict[str, Any] | None
    result_payload: dict[str, Any] | None
    updated_at: float


class _JobCache:
    def __init__(self, *, max_size: int, ttl_sec: float) -> None:
        self._max_size = max(1, int(max_size))
        self._ttl_sec = max(0.0, float(ttl_sec))
        self._items: OrderedDict[str, _JobCacheEntry] = OrderedDict()
        self._lock = Lock()

    def _prune_locked(self, now: float) -> None:
        if self._ttl_sec <= 0:
            return
        stale = [
            key
            for key, entry in self._items.items()
            if now - entry.updated_at > self._ttl_sec
        ]
        for key in stale:
            self._items.pop(key, None)

    def _touch_locked(self, key: str) -> None:
        try:
            self._items.move_to_end(key)
        except KeyError:
            return

    def _evict_locked(self) -> None:
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)
        if _PROM_JOB_CACHE_ENTRIES is not None:
            _PROM_JOB_CACHE_ENTRIES.set(len(self._items))

    def get(self, job_id: str, *, include_result: bool) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            entry = self._items.get(job_id)
            if entry is None:
                return None
            payload = entry.result_payload if include_result else entry.status_payload
            if payload is None:
                return None
            self._touch_locked(job_id)
            return payload

    def set_status(self, job_id: str, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(job_id)
            if entry is None:
                entry = _JobCacheEntry(status_payload=None, result_payload=None, updated_at=now)
            entry.status_payload = payload
            entry.updated_at = now
            self._items[job_id] = entry
            self._touch_locked(job_id)
            self._evict_locked()
            if _PROM_JOB_CACHE_ENTRIES is not None:
                _PROM_JOB_CACHE_ENTRIES.set(len(self._items))

    def set_result(self, job_id: str, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(job_id)
            if entry is None:
                entry = _JobCacheEntry(status_payload=None, result_payload=None, updated_at=now)
            entry.result_payload = payload
            entry.status_payload = {
                "job_id": job_id,
                "status": payload.get("status", "completed"),
            }
            entry.updated_at = now
            self._items[job_id] = entry
            self._touch_locked(job_id)
            self._evict_locked()
            if _PROM_JOB_CACHE_ENTRIES is not None:
                _PROM_JOB_CACHE_ENTRIES.set(len(self._items))


_JOB_CACHE = None if _JOB_CACHE_SIZE == 0 else _JobCache(
    max_size=_JOB_CACHE_SIZE,
    ttl_sec=_JOB_CACHE_TTL_SEC,
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client = forwarded.split(",", 1)[0].strip()
        if client:
            return client
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _identity_value(request: Request) -> str | None:
    identity = getattr(request.state, "identity", None)
    if identity:
        return identity
    if not _IDENTITY_HEADER:
        return None
    value = request.headers.get(_IDENTITY_HEADER)
    if not value:
        return None
    value = value.strip()
    return value or None


def _hash_identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _request_subject(request: Request) -> str:
    identity = _identity_value(request)
    if identity:
        return f"id:{_hash_identity(identity)}"
    return _client_ip(request)


def _audit_subject(request: Request) -> str:
    identity = _identity_value(request)
    if identity:
        return f"id:{_hash_identity(identity)}"
    key = request.headers.get("x-api-key")
    if key:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return f"key:{digest}"
    return f"ip:{_client_ip(request)}"


def _audit_log(event: str, request: Request, status_code: int, detail: str | None = None) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status_code": status_code,
        "method": request.method,
        "path": request.url.path,
        "subject": _audit_subject(request),
        "request_id": getattr(request.state, "request_id", None),
    }
    trace_id = getattr(request.state, "trace_id", None)
    if trace_id:
        payload["trace_id"] = trace_id
    if detail:
        payload["detail"] = detail
    _log_json(_AUDIT_LOGGER, payload)


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _jwt_payload_local(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid token")
    header_b64, payload_b64, signature_b64 = parts
    header = json.loads(_b64url_decode(header_b64))
    payload = json.loads(_b64url_decode(payload_b64))
    alg = str(header.get("alg", "")).upper()
    if alg != "HS256":
        raise ValueError("unsupported jwt algorithm")
    if "HS256" not in _allowed_jwt_algs():
        raise ValueError("jwt algorithm not allowed")
    if not _JWT_SECRET:
        raise ValueError("jwt secret not configured")
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = hmac.new(_JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected, signature):
        raise ValueError("invalid jwt signature")
    now = time.time()
    exp = payload.get("exp")
    if exp is not None and now > float(exp) + _JWT_CLOCK_SKEW_SEC:
        raise ValueError("jwt expired")
    nbf = payload.get("nbf")
    if nbf is not None and now + _JWT_CLOCK_SKEW_SEC < float(nbf):
        raise ValueError("jwt not yet valid")
    if _JWT_ISSUER:
        if payload.get("iss") != _JWT_ISSUER:
            raise ValueError("jwt issuer mismatch")
    if _JWT_AUDIENCE:
        aud = payload.get("aud")
        if isinstance(aud, list):
            if _JWT_AUDIENCE not in aud:
                raise ValueError("jwt audience mismatch")
        elif aud != _JWT_AUDIENCE:
            raise ValueError("jwt audience mismatch")
    return payload


def _jwt_payload(token: str) -> dict[str, Any]:
    if _JWKS_URL:
        return _jwt_payload_rs256(token)
    return _jwt_payload_hs256(token)


def _jwt_token(request: Request) -> str:
    raw = request.headers.get(_JWT_HEADER)
    if not raw:
        raise HTTPException(status_code=401, detail="unauthorized")
    value = raw.strip()
    prefix = _JWT_BEARER_PREFIX.lower()
    if prefix and value.lower().startswith(prefix):
        value = value[len(prefix):].strip()
    return value


def _mtls_identity(request: Request) -> str:
    header = _MTLS_IDENTITY_HEADER
    value = request.headers.get(header) if header else None
    if not value:
        raise HTTPException(status_code=401, detail="unauthorized")
    return value.strip()


def _auth_context(request: Request) -> tuple[str | None, set[str]]:
    if _AUTH_MODE == "jwt":
        token = _jwt_token(request)
        try:
            payload = _jwt_payload(token)
        except Exception:
            raise HTTPException(status_code=401, detail="unauthorized")
        claim = payload.get(_JWT_SUB_CLAIM)
        if not claim:
            raise HTTPException(status_code=401, detail="unauthorized")
        return str(claim), _scopes_from_payload(payload)
    if _AUTH_MODE == "mtls":
        identity = _mtls_identity(request)
        scopes = _split_scopes(request.headers.get(_MTLS_SCOPES_HEADER)) if _MTLS_SCOPES_HEADER else set()
        if not scopes and _MTLS_ROLE_HEADER:
            roles = _split_scopes(request.headers.get(_MTLS_ROLE_HEADER))
            scopes = _scopes_from_roles(roles)
        return identity, scopes
    if _AUTH_MODE == "api_key":
        if not _API_KEY:
            return None, set()
        key = request.headers.get("x-api-key")
        if key != _API_KEY:
            raise HTTPException(status_code=401, detail="unauthorized")
        scopes = _split_scopes(_API_KEY_SCOPES)
        return key, scopes
    if _AUTH_MODE in {"none", ""}:
        return None, set()
    raise HTTPException(status_code=401, detail="unauthorized")


def _emit_monitor_alerts(request: Request, status_code: int, alerts: list[dict[str, Any]]) -> None:
    for alert in alerts:
        _audit_log(
            "monitor_alert",
            request,
            status_code,
            detail=json.dumps(alert, sort_keys=True, separators=(",", ":")),
        )


def _resolve_trace_id(request: Request) -> str | None:
    if not _TRACE_ENABLED:
        return None
    incoming = request.headers.get("x-trace-id")
    if incoming:
        return incoming[:128]
    return uuid.uuid4().hex


def _required_scopes_for_request(request: Request) -> set[str]:
    path = request.url.path
    if path in {"/v1/discover", "/v1/discover_async"}:
        return {_SCOPE_DISCOVER}
    if path in {"/v1/evaluate", "/v1/evaluate_async"}:
        return {_SCOPE_EVALUATE}
    if path.startswith("/v1/jobs/"):
        return {_SCOPE_JOBS}
    if path in {"/v1/metrics", "/metrics"}:
        return {_SCOPE_METRICS}
    return set()


def _authorize_request(request: Request) -> None:
    if not _RBAC_ENABLED:
        return
    required = _required_scopes_for_request(request)
    if not required:
        return
    scopes = getattr(request.state, "scopes", set())
    if "*" in scopes:
        return
    if required.issubset(scopes):
        return
    raise HTTPException(status_code=403, detail="forbidden")


def _guard_request(request: Request) -> None:
    if request.url.path in {"/healthz", "/readyz"}:
        return
    try:
        identity, scopes = _auth_context(request)
        request.state.identity = identity
        request.state.scopes = scopes
    except HTTPException:
        _audit_log("auth_failed", request, 401, "unauthorized")
        raise
    _authorize_request(request)
    if _MAX_BODY_BYTES > 0:
        length = request.headers.get("content-length")
        if length:
            try:
                content_length = int(length)
            except ValueError as exc:
                _audit_log("invalid_content_length", request, 400, "invalid content-length header")
                raise HTTPException(status_code=400, detail="invalid content-length header") from exc
            if content_length > _MAX_BODY_BYTES:
                _audit_log("request_too_large", request, 413, "request too large")
                raise HTTPException(status_code=413, detail="request too large")
    if _RATE_LIMIT_PER_MINUTE > 0.0 and _BURST_LIMIT > 0:
        subject = f"general:{_request_subject(request)}"
        allowed = _RATE_LIMITER.allow(
            subject,
            rate_per_minute=_RATE_LIMIT_PER_MINUTE,
            burst=_BURST_LIMIT,
        )
        if not allowed:
            _audit_log("rate_limited", request, 429, "too many requests")
            raise HTTPException(status_code=429, detail="too many requests")
    if _HEAVY_RATE_LIMIT_PER_MINUTE > 0.0 and _HEAVY_BURST_LIMIT > 0:
        if request.url.path in _HEAVY_PATHS:
            subject = f"heavy:{_request_subject(request)}"
            allowed = _RATE_LIMITER.allow(
                subject,
                rate_per_minute=_HEAVY_RATE_LIMIT_PER_MINUTE,
                burst=_HEAVY_BURST_LIMIT,
            )
            if not allowed:
                _audit_log("rate_limited", request, 429, "too many requests")
                raise HTTPException(status_code=429, detail="too many requests")


def _wrap_request_with_body_limit(request: Request) -> Request:
    if _MAX_BODY_BYTES <= 0:
        return request

    original_receive = request.receive
    bytes_seen = 0

    async def _limited_receive():
        nonlocal bytes_seen
        message = await original_receive()
        if message["type"] == "http.request":
            bytes_seen += len(message.get("body", b""))
            if bytes_seen > _MAX_BODY_BYTES:
                _audit_log("request_too_large", request, 413, "request too large")
                raise HTTPException(status_code=413, detail="request too large")
        return message

    return Request(request.scope, receive=_limited_receive)


def _job_backend() -> str:
    return os.getenv("TGEFD_JOB_BACKEND", "").strip().lower()


def _idempotency_key(request: Request) -> str | None:
    key = request.headers.get("x-idempotency-key")
    if not key:
        return None
    return key.strip() or None


def _require_async_identity(request: Request) -> None:
    if not _REQUIRE_ASYNC_IDENTITY:
        return
    if _identity_value(request):
        return
    raise HTTPException(status_code=401, detail="identity required")


def _maybe_reuse_artifact(config: TGEFDConfig, request: Request) -> JSONResponse | None:
    if not _STORE_REUSE_ENABLED:
        return None
    if not config.output.save_artifacts:
        return None
    try:
        store_config = config.to_store_config()
        store = build_store(store_config)
        config_hash = canonical_config_hash(config.model_dump(by_alias=True))
        artifact_id = artifact_id_from_hash(config_hash)
        if not store.exists(artifact_id):
            return None
        payload = store.load_json(artifact_id, "results.json")
        artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
        artifacts.setdefault("artifact_id", artifact_id)
        artifacts.setdefault("artifact_uri", store.uri_for(artifact_id))
        if isinstance(payload, dict):
            payload["artifacts"] = artifacts
        if isinstance(payload, dict):
            decision = payload.get("decision", {})
            accepted = bool(decision.get("accepted", False))
            request_alerts = _MONITORING.record_decision(accepted=accepted)
            _emit_monitor_alerts(request, 200, request_alerts)
            if _PROM_DECISIONS is not None:
                _PROM_DECISIONS.labels(
                    **_labels_with_tenant(
                        {
                            "decision": "accepted" if accepted else "rejected",
                            "reason": decision.get("reason", "reused"),
                        },
                        request,
                    )
                ).inc()
        return JSONResponse(content=payload, headers={"x-artifact-reused": "true"})
    except Exception:
        return None


async def _submit_temporal_job(config: TGEFDConfig, *, workflow_name: str, request: Request) -> dict[str, Any]:
    try:
        from . import temporal_backend as temporal
    except Exception as exc:
        raise HTTPException(status_code=503, detail="temporal backend not available") from exc

    try:
        payload = config.model_dump(by_alias=True)
        key = _idempotency_key(request)
        workflow_id = temporal.workflow_id_from_idempotency(key) if key else None
        job_id = await temporal.submit_workflow(
            payload,
            workflow_name=workflow_name,
            workflow_id=workflow_id,
        )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="temporal backend not available") from exc
    except Exception as exc:
        _SERVER_LOGGER.exception("Temporal submit failed")
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_MESSAGE) from exc

    response = {"job_id": job_id, "status": "queued"}
    if _JOB_CACHE is not None:
        _JOB_CACHE.set_status(job_id, response)
    return response


async def _temporal_status(job_id: str, *, include_result: bool) -> dict[str, Any]:
    try:
        from . import temporal_backend as temporal
    except Exception as exc:
        raise HTTPException(status_code=503, detail="temporal backend not available") from exc

    try:
        if include_result:
            return await temporal.get_workflow_result(job_id, timeout_sec=_JOB_RESULT_TIMEOUT_SEC)
        return await temporal.get_workflow_status(job_id)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="temporal backend not available") from exc
    except Exception as exc:
        _SERVER_LOGGER.exception("Temporal status failed")
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_MESSAGE) from exc


@app.middleware("http")
async def _security_access_middleware(request: Request, call_next):
    _configure_audit_logger()
    request.state.request_id = request.headers.get("x-request-id", str(time.time_ns()))
    request.state.trace_id = _resolve_trace_id(request)
    metric_path = _metric_path(request)
    span_cm = nullcontext()
    if _TRACER is not None and propagate is not None and SpanKind is not None:
        ctx = propagate.extract(request.headers)
        span_cm = _TRACER.start_as_current_span(
            f"{request.method} {metric_path}",
            context=ctx,
            kind=SpanKind.SERVER,
        )
    if _PROM_INFLIGHT is not None:
        _PROM_INFLIGHT.labels(path=metric_path).inc()
    start = time.perf_counter()
    with span_cm as span:
        if span is not None:
            trace_id = span.get_span_context().trace_id
            if trace_id:
                request.state.trace_id = f"{trace_id:032x}"
        global_acquired = False
        heavy_acquired = False
        try:
            _guard_request(request)
            if _GLOBAL_SEMAPHORE is not None:
                await _acquire_concurrency_limit(_GLOBAL_SEMAPHORE, request, "global")
                global_acquired = True
            if _HEAVY_SEMAPHORE is not None and request.url.path in _HEAVY_PATHS:
                await _acquire_concurrency_limit(_HEAVY_SEMAPHORE, request, "heavy")
                heavy_acquired = True
            wrapped_request = _wrap_request_with_body_limit(request)
            wrapped_request.state.request_id = request.state.request_id
            wrapped_request.state.trace_id = request.state.trace_id
            if _REQUEST_TIMEOUT_SEC > 0.0:
                response = await asyncio.wait_for(call_next(wrapped_request), timeout=_REQUEST_TIMEOUT_SEC)
            else:
                response = await call_next(wrapped_request)
        except asyncio.TimeoutError as exc:
            if span is not None and Status is not None and StatusCode is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
            _audit_log("request_timeout", request, 504, "request timeout")
            response = JSONResponse(status_code=504, content={"detail": "request timeout"})
        except HTTPException as exc:
            if span is not None and Status is not None and StatusCode is not None and exc.status_code >= 500:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
            response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        except Exception as exc:
            if span is not None and Status is not None and StatusCode is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
            _SERVER_LOGGER.exception("Unhandled server error for %s %s", request.method, request.url.path)
            _audit_log("internal_error", request, 500, "internal server error")
            raise
        finally:
            if heavy_acquired and _HEAVY_SEMAPHORE is not None:
                _HEAVY_SEMAPHORE.release()
            if global_acquired and _GLOBAL_SEMAPHORE is not None:
                _GLOBAL_SEMAPHORE.release()
            if _PROM_INFLIGHT is not None:
                _PROM_INFLIGHT.labels(path=metric_path).dec()

    duration_ms = (time.perf_counter() - start) * 1000.0
    if span is not None:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", metric_path)
        span.set_attribute("http.target", request.url.path)
        span.set_attribute("http.status_code", response.status_code)
        if Status is not None and StatusCode is not None and response.status_code >= 500:
            span.set_status(Status(StatusCode.ERROR))
    alerts = _MONITORING.record_request(
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    if _PROM_REQUESTS is not None:
        _PROM_REQUESTS.labels(
            **_labels_with_tenant(
                {
                    "method": request.method,
                    "path": metric_path,
                    "status_code": str(response.status_code),
                },
                request,
            )
        ).inc()
    if _PROM_LATENCY is not None:
        _PROM_LATENCY.labels(
            **_labels_with_tenant(
                {
                    "method": request.method,
                    "path": metric_path,
                },
                request,
            )
        ).observe(duration_ms / 1000.0)
    response.headers["x-request-id"] = request.state.request_id
    if request.state.trace_id:
        response.headers["x-trace-id"] = request.state.trace_id
    _log_json(
        _ACCESS_LOGGER,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "access",
            "request_id": request.state.request_id,
            "trace_id": request.state.trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 3),
            "client_ip": _client_ip(request),
            "identity": _hash_identity(_identity_value(request)) if _identity_value(request) else None,
        },
    )
    _emit_monitor_alerts(request, response.status_code, alerts)
    if request.url.path.startswith("/v1/"):
        _audit_log("api_call", request, response.status_code)
    return response


@app.get("/v1/metrics")
def metrics_endpoint():
    return _MONITORING.snapshot()


@app.get("/metrics")
def prometheus_metrics_endpoint():
    if _PROM_REGISTRY is None or generate_latest is None:
        raise HTTPException(status_code=503, detail="prometheus not available")
    data = generate_latest(_PROM_REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ok"}


@app.post("/v1/discover")
def discover_endpoint(config: TGEFDConfig, request: Request):
    reused = _maybe_reuse_artifact(config, request)
    if reused is not None:
        return reused
    try:
        evaluation = config.to_evaluation_policy()
        adaptive = config.to_adaptive_config()
        budget = config.to_compute_budget()
        discover_request = config.to_request()
        response = discover(
            discover_request,
            evaluation=evaluation,
            adaptive=adaptive,
            budget=budget,
        )

        if config.output.save_artifacts:
            store_config = config.to_store_config()
            store = build_store(store_config)
            config_hash = canonical_config_hash(config.model_dump(by_alias=True))
            artifact_id = artifact_id_from_hash(config_hash)
            artifact_uri = store.uri_for(artifact_id)
            response = replace(
                response,
                artifacts=replace(
                    response.artifacts,
                    artifact_id=artifact_id,
                    artifact_uri=artifact_uri,
                ),
                reproducibility=replace(
                    response.reproducibility,
                    config_hash=config_hash,
                ),
            )
            info = write_artifact_bundle(
                config.model_dump(by_alias=True),
                response,
                base_dir=store_config.base_dir,
                artifact_uri=artifact_uri,
                dataset_provenance=config.dataset_provenance(),
            )
            store.store(info.artifact_dir, info.artifact_id)

        request_alerts = _MONITORING.record_decision(accepted=response.decision.accepted)
        _emit_monitor_alerts(request, 200, request_alerts)
        if _PROM_DECISIONS is not None:
            _PROM_DECISIONS.labels(
                **_labels_with_tenant(
                    {
                        "decision": "accepted" if response.decision.accepted else "rejected",
                        "reason": response.decision.reason,
                    },
                    request,
                )
            ).inc()
        return response.to_dict()
    except HTTPException:
        raise
    except BudgetExceededError as exc:
        _SERVER_LOGGER.warning(
            "Discover budget violation request_id=%s: %s",
            getattr(request.state, "request_id", None),
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        _SERVER_LOGGER.warning(
            "Bad discover request request_id=%s: %s",
            getattr(request.state, "request_id", None),
            exc.__class__.__name__,
        )
        raise HTTPException(status_code=400, detail=_BAD_REQUEST_MESSAGE) from exc
    except Exception as exc:
        _SERVER_LOGGER.exception(
            "Discover endpoint failed request_id=%s",
            getattr(request.state, "request_id", None),
        )
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_MESSAGE) from exc


@app.post("/v1/evaluate")
def evaluate_endpoint(config: TGEFDConfig, request: Request):
    reused = _maybe_reuse_artifact(config, request)
    if reused is not None:
        return reused
    try:
        evaluation = config.to_evaluation_policy()
        adaptive = config.to_adaptive_config()
        budget = config.to_compute_budget()
        evaluate_request = config.to_request()
        response = evaluate(
            evaluate_request,
            evaluation=evaluation,
            adaptive=adaptive,
            budget=budget,
        )
        request_alerts = _MONITORING.record_decision(accepted=response.decision.accepted)
        _emit_monitor_alerts(request, 200, request_alerts)
        if _PROM_DECISIONS is not None:
            _PROM_DECISIONS.labels(
                **_labels_with_tenant(
                    {
                        "decision": "accepted" if response.decision.accepted else "rejected",
                        "reason": response.decision.reason,
                    },
                    request,
                )
            ).inc()
        return response.to_dict()
    except HTTPException:
        raise
    except BudgetExceededError as exc:
        _SERVER_LOGGER.warning(
            "Evaluate budget violation request_id=%s: %s",
            getattr(request.state, "request_id", None),
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        _SERVER_LOGGER.warning(
            "Bad evaluate request request_id=%s: %s",
            getattr(request.state, "request_id", None),
            exc.__class__.__name__,
        )
        raise HTTPException(status_code=400, detail=_BAD_REQUEST_MESSAGE) from exc
    except Exception as exc:
        _SERVER_LOGGER.exception(
            "Evaluate endpoint failed request_id=%s",
            getattr(request.state, "request_id", None),
        )
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_MESSAGE) from exc


@app.post("/v1/discover_async")
async def discover_async_endpoint(config: TGEFDConfig, request: Request):
    if _job_backend() != "temporal":
        raise HTTPException(status_code=503, detail="job backend not configured")
    _require_async_identity(request)
    return await _submit_temporal_job(config, workflow_name="DiscoverWorkflow", request=request)


@app.post("/v1/evaluate_async")
async def evaluate_async_endpoint(config: TGEFDConfig, request: Request):
    if _job_backend() != "temporal":
        raise HTTPException(status_code=503, detail="job backend not configured")
    _require_async_identity(request)
    return await _submit_temporal_job(config, workflow_name="EvaluateWorkflow", request=request)


@app.get("/v1/jobs/{job_id}")
async def job_status_endpoint(job_id: str, include_result: bool = Query(default=False)):
    if _job_backend() != "temporal":
        raise HTTPException(status_code=503, detail="job backend not configured")
    if _JOB_CACHE is not None:
        cached = _JOB_CACHE.get(job_id, include_result=include_result)
        if cached is not None:
            return cached
    payload = await _temporal_status(job_id, include_result=include_result)
    if _JOB_CACHE is not None:
        if include_result and payload.get("status") == "completed" and "result" in payload:
            _JOB_CACHE.set_result(job_id, payload)
        else:
            _JOB_CACHE.set_status(job_id, {"job_id": job_id, "status": payload.get("status", "unknown")})
    return payload
