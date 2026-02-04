import numpy as np
import pytest
import time

from tgefd.api_v1 import (
    BudgetExceededError,
    ComputeBudget,
    Dataset,
    DiscoverRequest,
    HypothesisSpace,
    NoiseProfile,
    PHConfig,
    PersistenceImageConfig,
    TopologyConfig,
    _run_with_timeout,
    _sample_results_for_topology,
    discover,
)
from tgefd.evaluation import EvaluationPolicy
from tgefd.evaluation import AcceptancePolicy, METHOD_VERSION, StabilityScorePolicy
from tgefd.data import make_synthetic_truth_data
from tgefd.models import HypothesisParams, ModelResult


def _slow_search(*_args, **_kwargs):
    time.sleep(0.03)
    return []


def test_api_v1_validation_feature_params():
    dataset = Dataset(id="d1", x=np.arange(5), y=np.arange(5, dtype=float))
    space = HypothesisSpace(
        features=["1", "x^p"],
        parameters={},
        regularization={"lambda": [0.01]},
    )
    topo = TopologyConfig(ph=PHConfig(), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=0)
    with pytest.raises(ValueError):
        req.validate()


def test_api_v1_dataset_non_finite_values():
    dataset = Dataset(id="bad", x=np.array([0.0, np.nan, 1.0]), y=np.array([0.0, 1.0, 2.0]))
    space = HypothesisSpace(
        features=["1", "x"],
        parameters={},
        regularization={"lambda": [0.01]},
    )
    topo = TopologyConfig(ph=PHConfig(), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=0)
    with pytest.raises(ValueError, match="finite"):
        req.validate()


def test_api_v1_dataset_hard_point_limit(monkeypatch):
    monkeypatch.setattr("tgefd.api_v1._DATASET_HARD_MAX_POINTS", 2)
    dataset = Dataset(id="large", x=np.arange(3), y=np.arange(3, dtype=float))
    space = HypothesisSpace(
        features=["1", "x"],
        parameters={},
        regularization={"lambda": [0.01]},
    )
    topo = TopologyConfig(ph=PHConfig(), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=0)
    with pytest.raises(ValueError, match="hard safety limit"):
        req.validate()


def test_api_v1_discover_deterministic_summary():
    pytest.importorskip("ripser")

    x, y = make_synthetic_truth_data(n=60, noise=0.0, seed=0)
    dataset = Dataset(id="syn", x=x, y=y)
    space = HypothesisSpace(
        features=["1", "x", "sin(x)", "x^p", "sin(x)^q"],
        parameters={"p": [1.0, 2.0, 3.0], "q": [1.0, 2.0, 3.0]},
        regularization={"lambda": [0.01, 0.05]},
    )
    topo = TopologyConfig(ph=PHConfig(max_dim=1), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)

    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=1)
    policy = EvaluationPolicy()
    resp1 = discover(req, evaluation=policy)
    resp2 = discover(req, evaluation=policy)

    assert resp1.reproducibility.seed == 1
    assert resp1.reproducibility.deterministic is True
    assert resp1.reproducibility.config_hash.startswith("sha256:")
    assert resp1.reproducibility.method_version == METHOD_VERSION
    assert resp1.summary == resp2.summary


def test_api_v1_rejection_is_valid_result():
    pytest.importorskip("ripser")

    x, y = make_synthetic_truth_data(n=40, noise=0.0, seed=2)
    dataset = Dataset(id="syn-reject", x=x, y=y)
    space = HypothesisSpace(
        features=["1", "x", "sin(x)", "x^p", "sin(x)^q"],
        parameters={"p": [1.0, 2.0, 3.0], "q": [1.0, 2.0, 3.0]},
        regularization={"lambda": [0.01, 0.05]},
    )
    topo = TopologyConfig(ph=PHConfig(max_dim=1), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=3)
    strict_policy = EvaluationPolicy(
        acceptance=AcceptancePolicy(min_stable_components=999, require_h1=False),
        stability_score=StabilityScorePolicy(tolerance=1.0),
    )

    response = discover(req, evaluation=strict_policy)

    assert response.status == "rejected"
    assert response.decision.accepted is False
    assert response.decision.reason == "no_stable_components"


def test_api_v1_budget_rejects_large_hypercube():
    dataset = Dataset(id="d2", x=np.arange(6), y=np.arange(6, dtype=float))
    space = HypothesisSpace(
        features=["1", "x", "x^p", "sin(x)^q"],
        parameters={"p": [1.0, 2.0, 3.0], "q": [1.0, 2.0]},
        regularization={"lambda": [0.01, 0.1]},
    )
    topo = TopologyConfig(ph=PHConfig(max_dim=1), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=0)
    budget = ComputeBudget(max_hypotheses=4)

    with pytest.raises(BudgetExceededError, match="hypothesis hypercube size"):
        discover(req, budget=budget)


def test_api_v1_budget_rejects_noise_trials():
    dataset = Dataset(id="d3", x=np.arange(6), y=np.arange(6, dtype=float))
    space = HypothesisSpace(
        features=["1", "x"],
        parameters={},
        regularization={"lambda": [0.01]},
    )
    topo = TopologyConfig(ph=PHConfig(max_dim=1), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0, 0.1], trials=4)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=0)
    budget = ComputeBudget(max_noise_trials=2)

    with pytest.raises(BudgetExceededError, match="noise trials"):
        discover(req, budget=budget)


def test_api_v1_budget_timeout_on_symbolic_regression(monkeypatch):
    pytest.importorskip("ripser")

    x, y = make_synthetic_truth_data(n=20, noise=0.0, seed=0)
    dataset = Dataset(id="syn-timeout", x=x, y=y)
    space = HypothesisSpace(
        features=["1", "x"],
        parameters={},
        regularization={"lambda": [0.01]},
    )
    topo = TopologyConfig(ph=PHConfig(max_dim=1), persistence_image=PersistenceImageConfig())
    noise = NoiseProfile(levels=[0.0], trials=1)
    req = DiscoverRequest(dataset=dataset, hypothesis_space=space, topology=topo, noise=noise, seed=1)

    monkeypatch.setattr("tgefd.api_v1.search_models", _slow_search)
    budget = ComputeBudget(symbolic_regression_timeout_sec=0.001)

    with pytest.raises(BudgetExceededError, match="symbolic regression timeout"):
        discover(req, budget=budget)


def test_run_with_timeout_enforces_deadline():
    with pytest.raises(BudgetExceededError, match="sleep timeout"):
        _run_with_timeout(
            time.sleep,
            timeout_sec=0.05,
            stage="sleep",
            args=(0.2,),
        )


def test_sample_results_for_topology_is_deterministic():
    results = []
    for i in range(10):
        params = HypothesisParams(p=float(i), q=1.0, lam=0.1)
        results.append(
            ModelResult(
                params=params,
                coeffs=np.array([1.0]),
                error=float(i),
                l0=1,
                l1=1.0,
            )
        )

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    sampled1 = _sample_results_for_topology(results, 4, rng1)
    sampled2 = _sample_results_for_topology(results, 4, rng2)

    sampled_ps1 = [r.params.p for r in sampled1]
    sampled_ps2 = [r.params.p for r in sampled2]
    input_ps = [r.params.p for r in results]

    assert sampled_ps1 == sampled_ps2
    assert len(sampled_ps1) == 4
    assert all(p in input_ps for p in sampled_ps1)
