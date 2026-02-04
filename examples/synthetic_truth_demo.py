from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgefd.data import make_synthetic_truth_data
from tgefd.search import hypothesis_hypercube, rank_results, search_models
from tgefd.topology import (
    build_point_cloud,
    persistent_components,
    persistent_homology,
    select_stable_models,
    significant_features,
)
from tgefd.topo_vectors import image_energy, persistent_image


FEATURE_NAMES = ["1", "x", "x^p", "sin(x)", "sin(x)^q", "exp(-x)"]
_HAS_PERSIM = True


def _parse_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _format_coeffs(coeffs: np.ndarray, precision: int = 4) -> str:
    fmt = f"{{:.{precision}g}}"
    return "[" + ", ".join(fmt.format(c) for c in coeffs) + "]"


def _summarize_coeffs(coeffs: np.ndarray) -> str:
    parts = []
    for name, coef in zip(FEATURE_NAMES, coeffs):
        parts.append(f"{name}={coef:+.4g}")
    return ", ".join(parts)


def _topo_energy(diagram: np.ndarray) -> float | None:
    global _HAS_PERSIM
    if diagram.size == 0:
        return 0.0
    try:
        img = persistent_image(
            diagram,
            birth_range=(0.0, 1.0),
            pers_range=(0.0, 1.0),
            pixel_size=0.05,
            weight="persistence",
            weight_params={"n": 2.0},
            kernel="gaussian",
            kernel_params={"sigma": [[0.05, 0.0], [0.0, 0.05]]},
            skew=True,
        )
    except ImportError:
        _HAS_PERSIM = False
        return None
    return image_energy(img)


def run_case(
    x: np.ndarray,
    y: np.ndarray,
    p_range: Iterable[float],
    q_range: Iterable[float],
    lam_range: Iterable[float],
    *,
    iters: int,
    scale: str,
    h0_threshold: float,
    h1_threshold: float,
    min_component_size: int,
) -> dict[str, object]:
    hypercube = hypothesis_hypercube(list(p_range), list(q_range), list(lam_range))
    results = search_models(x, y, hypercube, iters=iters)

    X = build_point_cloud(results, scale=scale)
    diagrams = persistent_homology(X, maxdim=2)

    h0_sig = significant_features(diagrams[0], persistence_threshold=h0_threshold)
    h1_sig = []
    if len(diagrams) > 1:
        h1_sig = significant_features(diagrams[1], persistence_threshold=h1_threshold)

    components = persistent_components(
        X,
        persistence_threshold=h0_threshold,
        min_component_size=min_component_size,
    )
    stable = select_stable_models(results, components)

    ranked = rank_results(stable or results, top_k=1)
    top = ranked[0] if ranked else None

    energy = None
    if len(diagrams) > 1:
        energy = _topo_energy(diagrams[1])

    return {
        "results": results,
        "stable": stable,
        "h0_sig": h0_sig,
        "h1_sig": h1_sig,
        "top": top,
        "energy": energy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic truth benchmark for TGEFD")
    parser.add_argument("--n", type=int, default=200, help="Number of points")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--noise-levels", default="0,0.01,0.05", help="Noise levels list")
    parser.add_argument("--p-range", default="1,2,3", help="p range list")
    parser.add_argument("--q-range", default="1,2", help="q range list")
    parser.add_argument("--lam-range", default="1e-3,1e-2,5e-2", help="lambda list")
    parser.add_argument("--iters", type=int, default=8, help="STLSQ iterations")
    parser.add_argument("--scale", choices=["none", "standard"], default="none")
    parser.add_argument("--h0-threshold", type=float, default=0.2)
    parser.add_argument("--h1-threshold", type=float, default=0.1)
    parser.add_argument("--min-component-size", type=int, default=5)
    parser.add_argument("--negative-control", action="store_true", help="Run random y control")

    args = parser.parse_args()

    noise_levels = _parse_list(args.noise_levels)
    p_range = _parse_list(args.p_range)
    q_range = _parse_list(args.q_range)
    lam_range = _parse_list(args.lam_range)

    print("Synthetic truth: y = x + 2*sin(x)")
    print(f"noise levels: {noise_levels}")
    print(f"hypercube: p={p_range} q={q_range} lam={lam_range}")
    print(f"scale={args.scale} | h0={args.h0_threshold} h1={args.h1_threshold}")
    print("")

    for noise in noise_levels:
        x, y = make_synthetic_truth_data(n=args.n, noise=noise, seed=args.seed)
        out = run_case(
            x,
            y,
            p_range,
            q_range,
            lam_range,
            iters=args.iters,
            scale=args.scale,
            h0_threshold=args.h0_threshold,
            h1_threshold=args.h1_threshold,
            min_component_size=args.min_component_size,
        )

        top = out["top"]
        energy = out["energy"]
        energy_str = "n/a" if energy is None else f"{energy:.4g}"

        print(f"[noise={noise}]")
        print(f"  significant H0: {len(out['h0_sig'])} | significant H1: {len(out['h1_sig'])}")
        print(f"  stable models: {len(out['stable'])} | topo energy (H1 PI): {energy_str}")
        if top is not None:
            print(f"  top error={top.error:.4g} l0={top.l0} l1={top.l1:.4g}")
            print(f"  coeffs: {_format_coeffs(top.coeffs)}")
            print(f"  coeffs by name: {_summarize_coeffs(top.coeffs)}")
        print("")

    if args.negative_control:
        rng = np.random.default_rng(args.seed)
        x = np.linspace(0.0, 10.0, args.n)
        y = rng.standard_normal(len(x))
        out = run_case(
            x,
            y,
            p_range,
            q_range,
            lam_range,
            iters=args.iters,
            scale=args.scale,
            h0_threshold=args.h0_threshold,
            h1_threshold=args.h1_threshold,
            min_component_size=args.min_component_size,
        )
        energy = out["energy"]
        energy_str = "n/a" if energy is None else f"{energy:.4g}"

        print("[negative control: random y]")
        print(f"  significant H0: {len(out['h0_sig'])} | significant H1: {len(out['h1_sig'])}")
        print(f"  stable models: {len(out['stable'])} | topo energy (H1 PI): {energy_str}")

    if not _HAS_PERSIM:
        print("")
        print("Note: persim not installed; persistent image energy is skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
