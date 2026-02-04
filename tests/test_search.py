from tgefd.data import make_demo_data
from tgefd.search import hypothesis_hypercube, search_models


def test_search_models_min_error_small():
    x, y = make_demo_data(n=120, noise=0.0, seed=1)
    hypercube = hypothesis_hypercube([2.0], [2.0], [0.001, 0.01])
    results = search_models(x, y, hypercube, iters=8)

    errors = [r.error for r in results]
    assert len(results) == 2
    assert min(errors) < 1e-5
