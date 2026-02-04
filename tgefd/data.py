from __future__ import annotations

import numpy as np


def make_grid(n: int = 200, x_min: float = 0.0, x_max: float = 6.0) -> np.ndarray:
    return np.linspace(x_min, x_max, n)


def make_demo_data(n: int = 200, noise: float = 0.05, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = make_grid(n)

    # Ground-truth formula for demo only.
    y_clean = 1.5 + 0.8 * x + 2.0 * (np.sin(x) ** 2.0) - 0.3 * np.exp(-x)
    y = y_clean + noise * rng.normal(size=x.shape)
    return x, y


def make_synthetic_truth_data(
    n: int = 200,
    noise: float = 0.0,
    seed: int = 0,
    x_min: float = 0.0,
    x_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, n)
    y_clean = x + 2.0 * np.sin(x)
    if noise > 0:
        y_clean = y_clean + noise * rng.standard_normal(len(x))
    return x, y_clean
