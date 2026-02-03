import numpy as np
import pytest

from tgefd.config import parse_config


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


def test_file_dataset_loads_from_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n0.0,0.1\n0.2,0.3\n", encoding="utf-8")
    cfg = _base_config()
    cfg["dataset"] = {"source": "file", "path": str(csv_path)}
    parsed = parse_config(cfg)
    req = parsed.to_request()
    assert np.asarray(req.dataset.x).shape == (2, 1)
    assert np.asarray(req.dataset.y).shape == (2,)
    provenance = parsed.dataset_provenance()
    assert provenance
    assert provenance["source"] == "file"


def test_uri_dataset_loads_from_file_uri(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("0.0,0.1\n0.2,0.3\n", encoding="utf-8")
    cfg = _base_config()
    cfg["dataset"] = {"source": "uri", "uri": f"file://{csv_path}"}
    parsed = parse_config(cfg)
    req = parsed.to_request()
    assert np.asarray(req.dataset.x).shape == (2, 1)
    assert np.asarray(req.dataset.y).shape == (2,)
    provenance = parsed.dataset_provenance()
    assert provenance
    assert provenance["source"] == "file"
