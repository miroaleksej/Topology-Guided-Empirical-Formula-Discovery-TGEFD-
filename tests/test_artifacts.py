import json
from pathlib import Path

import pytest

from tgefd.api_v1 import discover
from tgefd.config import parse_config
from tgefd.artifacts import verify_artifact, write_artifact_bundle
from tgefd.evaluation import AcceptancePolicy, EvaluationPolicy, StabilityScorePolicy
import tgefd.artifacts as artifacts_mod


def _base_config():
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
            "parameters": {"p": {"type": "float", "values": [1.0, 2.0, 3.0]}},
            "regularization": {"lambda": [0.01, 0.05]},
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
            "paths": {"base_dir": "./runs"},
        },
        "reproducibility": {
            "tgefd_version": "1.0.0",
            "library_versions": True,
            "hash_algorithm": "sha256",
        },
    }


def test_artifact_bundle(tmp_path: Path):
    pytest.importorskip("ripser")

    cfg = _base_config()
    parsed = parse_config(cfg)
    response = discover(parsed.to_request(), evaluation=parsed.to_evaluation_policy())

    dataset_provenance = {"source": "inline", "sha256": "sha256:test", "size_bytes": 64}
    info = write_artifact_bundle(
        cfg,
        response,
        base_dir=tmp_path,
        dataset_provenance=dataset_provenance,
    )
    assert (info.artifact_dir / "_LOCK").exists()
    assert verify_artifact(info.artifact_dir)

    manifest = json.loads((info.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["hash"].startswith("sha256:")
    assert manifest["run"]["method_version"] == response.reproducibility.method_version
    assert manifest["run"]["deterministic"] is True
    assert manifest["artifacts"]["config"] == "config.yaml"
    assert manifest["artifacts"]["results"] == "results.json"
    assert manifest["artifacts"]["evidence"] == "evidence.json"
    assert manifest["dataset"] == dataset_provenance
    assert (info.artifact_dir / "decision.json").exists()
    assert (info.artifact_dir / "evidence.json").exists()
    assert manifest["reproducibility"]["library_versions"] == "libraries.json"
    assert (info.artifact_dir / "libraries.json").exists()
    libs = json.loads((info.artifact_dir / "libraries.json").read_text(encoding="utf-8"))
    assert "python" in libs
    assert "numpy" in libs["packages"]


def test_artifact_bundle_reuse_and_overwrite(tmp_path: Path):
    pytest.importorskip("ripser")

    cfg = _base_config()
    parsed = parse_config(cfg)
    response = discover(parsed.to_request(), evaluation=parsed.to_evaluation_policy())

    info = write_artifact_bundle(cfg, response, base_dir=tmp_path)
    reused = write_artifact_bundle(cfg, response, base_dir=tmp_path, if_exists="reuse")
    assert reused.artifact_dir == info.artifact_dir

    overwritten = write_artifact_bundle(cfg, response, base_dir=tmp_path, if_exists="overwrite")
    assert overwritten.artifact_dir == info.artifact_dir
    assert (overwritten.artifact_dir / "_LOCK").exists()


def test_negative_result_saved_as_artifact(tmp_path: Path):
    pytest.importorskip("ripser")

    cfg = _base_config()
    parsed = parse_config(cfg)
    strict_policy = EvaluationPolicy(
        acceptance=AcceptancePolicy(min_stable_components=999, require_h1=False),
        stability_score=StabilityScorePolicy(tolerance=1.0),
    )
    rejected = discover(parsed.to_request(), evaluation=strict_policy)
    assert rejected.status == "rejected"

    info = write_artifact_bundle(cfg, rejected, base_dir=tmp_path)
    decision = json.loads((info.artifact_dir / "decision.json").read_text(encoding="utf-8"))
    negative = json.loads((info.artifact_dir / "negative_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((info.artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert decision["accepted"] is False
    assert decision["reason"] == "no_stable_components"
    assert decision["method_version"] == rejected.reproducibility.method_version
    assert negative["status"] == "rejected"
    assert negative["method_version"] == rejected.reproducibility.method_version
    assert manifest["result"]["status"] == "rejected"
    assert manifest["result"]["accepted"] is False
    assert manifest["result"]["reason"] == "no_stable_components"
    assert manifest["artifacts"]["negative_result"] == "negative_result.json"
    assert (info.artifact_dir / "negative_result.json").exists()


def test_artifact_bundle_atomic_cleanup_on_failure(tmp_path: Path, monkeypatch):
    pytest.importorskip("ripser")

    cfg = _base_config()
    parsed = parse_config(cfg)
    response = discover(parsed.to_request(), evaluation=parsed.to_evaluation_policy())

    def _fail_models(*_args, **_kwargs):
        raise RuntimeError("model write failed")

    monkeypatch.setattr(artifacts_mod, "_write_models", _fail_models)
    with pytest.raises(RuntimeError, match="model write failed"):
        write_artifact_bundle(cfg, response, base_dir=tmp_path)

    artifact_dir = artifacts_mod.artifact_path(tmp_path, artifacts_mod.canonical_config_hash(cfg))
    assert not artifact_dir.exists()
    tmp_dirs = [p for p in artifact_dir.parent.glob(".tmp-*") if p.is_dir()]
    assert tmp_dirs == []
