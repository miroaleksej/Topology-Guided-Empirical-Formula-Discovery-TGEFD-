import pytest

from tgefd.bench_regressions import compare_curves


def test_compare_curves_detects_drop():
    baseline = {
        "n_values": [8, 16],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "reference", "values": [0.9, 0.8]},
            ]
        },
    }
    candidate = {
        "n_values": [8, 16],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "reference", "values": [0.9, 0.6]},
            ]
        },
    }
    regressions = compare_curves(baseline, candidate, max_drop=0.1)
    assert len(regressions) == 1
    assert regressions[0].kind == "metric_drop"
    assert regressions[0].n_value == 16


def test_compare_curves_ignores_small_drop():
    baseline = {
        "n_values": [8],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "reference", "values": [0.7]},
            ]
        },
    }
    candidate = {
        "n_values": [8],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "reference", "values": [0.65]},
            ]
        },
    }
    regressions = compare_curves(baseline, candidate, max_drop=0.1)
    assert regressions == []


def test_compare_curves_missing_label():
    baseline = {
        "n_values": [8],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "reference", "values": [0.7]},
            ]
        },
    }
    candidate = {
        "n_values": [8],
        "default_metric": "AUC",
        "metrics": {
            "AUC": [
                {"label": "other", "values": [0.7]},
            ],
        },
    }
    regressions = compare_curves(baseline, candidate)
    assert regressions
    assert regressions[0].kind == "missing_label"
