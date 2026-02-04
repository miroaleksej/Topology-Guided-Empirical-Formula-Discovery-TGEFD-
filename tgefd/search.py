from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .features import feature_library
from .metrics import l0_norm, l1_norm, l2_error
from .models import HypothesisParams, ModelResult
from .sparse import sparse_regression


def hypothesis_hypercube(
    p_range: Sequence[float],
    q_range: Sequence[float],
    lam_range: Sequence[float],
) -> list[HypothesisParams]:
    return [HypothesisParams(p, q, lam) for p, q, lam in product(p_range, q_range, lam_range)]


def search_models(
    x: np.ndarray,
    y: np.ndarray,
    hypercube: Iterable[HypothesisParams],
    iters: int = 10,
    normalize: bool = False,
    features: Sequence[str] | None = None,
) -> list[ModelResult]:
    results: list[ModelResult] = []

    for params in hypercube:
        Phi = feature_library(x, params, features=features)
        coeffs = sparse_regression(Phi, y, lam=params.lam, iters=iters, normalize=normalize)
        error = l2_error(Phi, coeffs, y)
        results.append(
            ModelResult(
                params=params,
                coeffs=coeffs,
                error=error,
                l0=l0_norm(coeffs),
                l1=l1_norm(coeffs),
            )
        )

    return results


def rank_results(results: Iterable[ModelResult], top_k: int = 10) -> list[ModelResult]:
    return sorted(results, key=lambda r: (r.error, r.l0, r.l1))[:top_k]
