import numpy as np
import pytest

from tgefd.data import make_demo_data
from tgefd.search import hypothesis_hypercube, search_models
from tgefd.topology import (
    build_point_cloud,
    persistent_components,
    persistent_homology,
    select_stable_models,
    significant_features,
)


def test_persistent_components_simple():
    X = np.array([[0.0], [0.05], [1.0], [1.05]])
    components = persistent_components(X, persistence_threshold=0.2, min_component_size=1)
    sizes = sorted(comp.size for comp in components)
    assert sizes == [2, 2]


def test_topology_ph_pipeline():
    pytest.importorskip("ripser")

    x, y = make_demo_data(n=80, noise=0.01, seed=2)
    hypercube = hypothesis_hypercube([1.0, 2.0], [2.0, 3.0], [0.001, 0.01])
    results = search_models(x, y, hypercube, iters=5)

    X = build_point_cloud(results)
    diagrams = persistent_homology(X, maxdim=1)
    h0 = significant_features(diagrams[0], persistence_threshold=0.1)
    assert isinstance(h0, list)

    components = persistent_components(X, persistence_threshold=0.1, min_component_size=2)
    stable = select_stable_models(results, components)
    assert len(stable) <= len(results)
