from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import HypothesisParams, ModelResult
from .search import hypothesis_hypercube, rank_results, search_models
from .topology import build_point_cloud, persistent_components, select_stable_models


@dataclass(frozen=True)
class AdaptiveHypercubeConfig:
    enabled: bool = True
    refine_factor: float = 2.0
    top_k: int = 5
    min_component_size: int = 2
    h0_threshold: float | None = None


def adaptive_search_models(
    x: np.ndarray,
    y: np.ndarray,
    hypercube: Iterable[HypothesisParams],
    topology,
    adaptive: AdaptiveHypercubeConfig,
    iters: int = 10,
    normalize: bool = False,
    features: list[str] | None = None,
) -> list[ModelResult]:
    results = search_models(x, y, hypercube, iters=iters, normalize=normalize, features=features)
    if not adaptive.enabled:
        return results

    threshold = adaptive.h0_threshold
    if threshold is None:
        threshold = topology.stability_threshold

    X = build_point_cloud(results, scale=topology.point_scale)
    components = persistent_components(
        X,
        persistence_threshold=threshold,
        min_component_size=adaptive.min_component_size,
    )
    stable = select_stable_models(results, components)
    if not stable:
        return results

    seeds = rank_results(stable, top_k=adaptive.top_k)
    p_values, q_values, lam_values = _param_values(hypercube)
    refined = _refine_params(p_values, q_values, lam_values, seeds, adaptive.refine_factor)
    if refined is None:
        return results

    fine_hypercube = hypothesis_hypercube(*refined)
    fine_results = search_models(x, y, fine_hypercube, iters=iters, normalize=normalize, features=features)
    return results + fine_results


def _param_values(
    hypercube: Iterable[HypothesisParams],
) -> tuple[list[float], list[float], list[float]]:
    p_vals, q_vals, lam_vals = set(), set(), set()
    for params in hypercube:
        p_vals.add(float(params.p))
        q_vals.add(float(params.q))
        lam_vals.add(float(params.lam))
    return sorted(p_vals), sorted(q_vals), sorted(lam_vals)


def _refine_params(
    p_values: list[float],
    q_values: list[float],
    lam_values: list[float],
    seeds: list[ModelResult],
    refine_factor: float,
) -> tuple[list[float], list[float], list[float]] | None:
    if refine_factor <= 1.0:
        return None

    p_step = _min_step(p_values)
    q_step = _min_step(q_values)
    lam_step = _min_step(lam_values)

    p_min, p_max = min(p_values), max(p_values)
    q_min, q_max = min(q_values), max(q_values)
    lam_min, lam_max = min(lam_values), max(lam_values)

    p_new = set(p_values)
    q_new = set(q_values)
    lam_new = set(lam_values)

    for seed in seeds:
        p_new.update(_neighbors(seed.params.p, p_step, refine_factor, p_min, p_max))
        q_new.update(_neighbors(seed.params.q, q_step, refine_factor, q_min, q_max))
        lam_new.update(_neighbors(seed.params.lam, lam_step, refine_factor, lam_min, lam_max, positive=True))

    return sorted(p_new), sorted(q_new), sorted(lam_new)


def _min_step(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = sorted(abs(b - a) for a, b in zip(values[:-1], values[1:]) if abs(b - a) > 0)
    return diffs[0] if diffs else 0.0


def _neighbors(
    value: float,
    step: float,
    refine_factor: float,
    min_value: float,
    max_value: float,
    positive: bool = False,
) -> list[float]:
    if step <= 0:
        return [value]
    delta = step / refine_factor
    candidates = [value - delta, value, value + delta]
    out = []
    for v in candidates:
        if v < min_value or v > max_value:
            continue
        if positive and v <= 0:
            continue
        out.append(float(v))
    return out
