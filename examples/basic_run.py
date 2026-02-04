from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgefd.data import make_demo_data
from tgefd.search import hypothesis_hypercube, rank_results, search_models
from tgefd.topology import (
    build_point_cloud,
    persistent_homology,
    persistent_components,
    select_stable_models,
    significant_features,
)


def main() -> None:
    x, y = make_demo_data(n=200, noise=0.05, seed=7)

    p_range = [0.5, 1.0, 1.5, 2.0, 2.5]
    q_range = [1.0, 2.0, 3.0, 4.0]
    lam_range = [0.001, 0.005, 0.01]

    hypercube = hypothesis_hypercube(p_range, q_range, lam_range)
    results = search_models(x, y, hypercube)

    X = build_point_cloud(results)
    diagrams = persistent_homology(X, maxdim=2)
    h0_sig = significant_features(diagrams[0], persistence_threshold=0.2)
    print(f"significant H0: {len(h0_sig)}")

    components = persistent_components(X, persistence_threshold=0.2, min_component_size=5)
    stable = select_stable_models(results, components)

    ranked = rank_results(stable or results, top_k=5)

    for r in ranked:
        print(r)


if __name__ == "__main__":
    main()
