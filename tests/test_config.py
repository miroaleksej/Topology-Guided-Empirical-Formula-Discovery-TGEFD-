import numpy as np
import pytest

from pydantic import ValidationError

from tgefd.config import TGEFDConfig, config_to_request, parse_config


def _base_config():
    return {
        "tgefd_config_version": "1.0",
        "run": {"mode": "discover", "seed": 0, "deterministic": True},
        "dataset": {
            "source": "inline",
            "x": {"format": "array", "values": [[0.0], [0.1], [0.2], [0.3]]},
            "y": {"values": [0.0, 0.1, 0.2, 0.3]},
            "metadata": {"domain": "synthetic"},
        },
        "hypothesis_space": {
            "features": [
                {"name": "const", "expression": "1"},
                {"name": "x", "expression": "x"},
                {"name": "sinx", "expression": "sin(x)"},
                {"name": "x_pow", "expression": "x^p", "parameters": ["p"]},
            ],
            "parameters": {"p": {"type": "float", "values": [1.0, 2.0]}},
            "regularization": {"lambda": [0.001, 0.01]},
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
            "acceptance": {"min_stable_components": 1, "require_h1": True},
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


def test_parse_config_roundtrip_request():
    cfg = _base_config()
    parsed = parse_config(cfg)
    assert isinstance(parsed, TGEFDConfig)
    req = config_to_request(cfg)
    assert req.seed == 0
    assert np.asarray(req.dataset.x).shape[0] == 4


def test_missing_parameter_detected():
    cfg = _base_config()
    cfg["hypothesis_space"]["parameters"] = {}
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_evaluate_mode_rejects_noise():
    cfg = _base_config()
    cfg["run"]["mode"] = "evaluate"
    cfg["noise"]["enabled"] = True
    cfg["noise"]["levels"] = [0.01]
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_run_deterministic_must_be_true():
    cfg = _base_config()
    cfg["run"]["deterministic"] = False
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_compute_budget_validation():
    cfg = _base_config()
    cfg["compute_budget"] = {"max_hypotheses": 0}
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_dataset_source_file_requires_path():
    cfg = _base_config()
    cfg["dataset"]["source"] = "file"
    cfg["dataset"].pop("x", None)
    cfg["dataset"].pop("y", None)
    with pytest.raises(ValidationError, match="dataset\\.path"):
        parse_config(cfg)


def test_dataset_source_uri_requires_uri():
    cfg = _base_config()
    cfg["dataset"]["source"] = "uri"
    cfg["dataset"].pop("x", None)
    cfg["dataset"].pop("y", None)
    with pytest.raises(ValidationError, match="dataset\\.uri"):
        parse_config(cfg)


def test_dataset_limits_max_points():
    cfg = _base_config()
    cfg["dataset"]["limits"] = {"max_points": 2}
    with pytest.raises(ValidationError, match="max_points"):
        parse_config(cfg)


def test_dataset_x_row_width_mismatch():
    cfg = _base_config()
    cfg["dataset"]["x"]["values"] = [[0.0], [0.1, 0.2], [0.3]]
    with pytest.raises(ValidationError, match="consistent width"):
        parse_config(cfg)


def test_dataset_metadata_too_large():
    cfg = _base_config()
    cfg["dataset"]["metadata"] = {"note": "x" * 3000}
    with pytest.raises(ValidationError, match="max_metadata_value_chars"):
        parse_config(cfg)


def test_parameter_values_must_be_non_empty():
    cfg = _base_config()
    cfg["hypothesis_space"]["parameters"]["p"]["values"] = []
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_regularization_lambda_must_be_non_empty():
    cfg = _base_config()
    cfg["hypothesis_space"]["regularization"]["lambda"] = []
    with pytest.raises(ValidationError):
        parse_config(cfg)
