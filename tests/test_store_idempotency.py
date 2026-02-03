import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from tgefd.artifacts import canonical_config_hash, artifact_id_from_hash, artifact_path
import tgefd.server as server
from tgefd.config import parse_config


def _base_config(base_dir: str) -> dict:
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
            "save_artifacts": True,
            "formats": ["json"],
            "paths": {"base_dir": base_dir},
        },
        "reproducibility": {
            "tgefd_version": "1.0.0",
            "library_versions": True,
            "hash_algorithm": "sha256",
        },
    }


@pytest.fixture(autouse=True)
def _reset_store_reuse(monkeypatch):
    monkeypatch.setattr(server, "_STORE_REUSE_ENABLED", True)


def test_store_reuse_returns_existing_artifact(tmp_path: Path):
    cfg = _base_config(str(tmp_path))
    parsed = parse_config(cfg)
    config_hash = canonical_config_hash(parsed.model_dump(by_alias=True))
    artifact_id = artifact_id_from_hash(config_hash)
    artifact_dir = artifact_path(tmp_path, config_hash)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "run_id": "reused",
        "status": "completed",
        "summary": {},
        "decision": {"accepted": True, "reason": "reused"},
        "models": [],
        "noise_report": [],
        "topology_report": {},
        "artifacts": {"artifact_id": artifact_id, "artifact_uri": str(artifact_dir)},
        "reproducibility": {
            "config_hash": config_hash,
            "seed": 0,
            "deterministic": True,
            "version": "tgefd-test",
            "method_version": "tgefd-test",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps({"config": {"hash": config_hash}}), encoding="utf-8")
    (artifact_dir / "results.json").write_text(json.dumps(results_payload), encoding="utf-8")

    client = TestClient(server.app)
    response = client.post("/v1/discover", json=cfg)
    assert response.status_code == 200
    assert response.headers.get("x-artifact-reused") == "true"
    body = response.json()
    assert body.get("artifacts", {}).get("artifact_id") == artifact_id
