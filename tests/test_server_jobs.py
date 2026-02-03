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


def test_async_endpoints_require_temporal_backend(monkeypatch):
    monkeypatch.setenv("TGEFD_JOB_BACKEND", "temporal")
    client = TestClient(server.app)

    response = client.post("/v1/discover_async", json=_valid_request_config())
    assert response.status_code == 503
    assert "temporal" in response.json()["detail"]

    status = client.get("/v1/jobs/fake-job")
    assert status.status_code == 503
    assert "temporal" in status.json()["detail"]


def test_async_requires_identity_when_enabled(monkeypatch):
    monkeypatch.setenv("TGEFD_JOB_BACKEND", "temporal")
    monkeypatch.setattr(server, "_IDENTITY_HEADER", "x-client-id")
    monkeypatch.setattr(server, "_REQUIRE_ASYNC_IDENTITY", True)
    client = TestClient(server.app)

    response = client.post("/v1/discover_async", json=_valid_request_config())
    assert response.status_code == 401
    assert response.json()["detail"] == "identity required"


def test_job_status_uses_cache(monkeypatch):
    monkeypatch.setenv("TGEFD_JOB_BACKEND", "temporal")
    calls = {"count": 0}

    async def _fake_status(job_id: str, include_result: bool):
        calls["count"] += 1
        if include_result:
            return {"job_id": job_id, "status": "completed", "result": {"ok": True}}
        return {"job_id": job_id, "status": "running"}

    monkeypatch.setattr(server, "_temporal_status", _fake_status)
    client = TestClient(server.app)

    response = client.get("/v1/jobs/job-1?include_result=true")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert calls["count"] == 1

    second = client.get("/v1/jobs/job-1?include_result=true")
    assert second.status_code == 200
    assert second.json()["status"] == "completed"
    assert calls["count"] == 1
