import json
from pathlib import Path

import pytest

from tgefd.ond_rng_bench import BenchConfig, bench_config_hash, compute_ond_metrics, load_bench_config


def test_bench_config_loads():
    cfg = load_bench_config("benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml")
    parsed = BenchConfig.model_validate(cfg)
    assert parsed.protocol.name == "ecdsa_secp256k1"
    assert parsed.n_values


def test_bench_config_hash_stable():
    cfg = load_bench_config("benchmarks/ond_rng_bench/configs/ecdsa_baseline.yaml")
    h1 = bench_config_hash(cfg)
    h2 = bench_config_hash(cfg)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_ond_metrics_basic():
    pytest.importorskip("sklearn")
    u_series = [[0.1, 0.2], [0.2, 0.3], [0.25, 0.35], [0.3, 0.4]]
    metrics = compute_ond_metrics(__import__("numpy").array(u_series, dtype=float))
    assert set(metrics.keys()) == {"H_rank", "H_sub", "H_branch"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
