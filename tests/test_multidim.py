import numpy as np

from tgefd.api_v1 import HypothesisSpace
from tgefd.search import rank_results, search_models


def test_multidim_emc2_recovery():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.5, 2.0, size=80)
    c = rng.uniform(0.5, 2.0, size=80)
    X = np.column_stack([m, c])
    y = m * (c**2)

    features = ["1", "x0", "x1", "x1^2", "x0*x1^2"]
    space = HypothesisSpace(features=features, parameters={}, regularization={"lambda": [1e-6]})
    space.validate()

    results = search_models(X, y, space.hypercube(), iters=20, normalize=True, features=features)
    best = rank_results(results, top_k=1)[0]

    coeffs = best.coeffs
    idx = features.index("x0*x1^2")
    assert abs(coeffs[idx] - 1.0) < 1e-2
    others = [abs(c) for i, c in enumerate(coeffs) if i != idx]
    assert max(others) < 1e-2
