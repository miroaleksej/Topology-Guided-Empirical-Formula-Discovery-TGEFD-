from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .topology import persistent_homology
from .topo_vectors import (
    image_energy,
    landscape_norm,
    persistent_image,
    persistent_landscape,
)

TopoVectorMethod = Literal["landscape", "image"]


@dataclass(frozen=True)
class TopoPenaltyResult:
    penalty: float
    diagram: np.ndarray
    vector: np.ndarray | None
    method: TopoVectorMethod | None


def perturb_coeffs(
    coeffs: np.ndarray,
    sigma: float = 1e-3,
    n_samples: int = 20,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    coeffs = np.asarray(coeffs, dtype=float)
    return [coeffs + sigma * rng.normal(size=coeffs.shape) for _ in range(n_samples)]


def hypothesis_cloud(Phi: np.ndarray, coeffs_list: Sequence[np.ndarray]) -> np.ndarray:
    Phi = np.asarray(Phi, dtype=float)
    return np.asarray([Phi @ coeffs for coeffs in coeffs_list], dtype=float)


def topological_penalty(
    hyp_cloud: np.ndarray,
    method: TopoVectorMethod = "landscape",
    maxdim: int = 1,
    transpose_cloud: bool = True,
    num_landscapes: int = 5,
    resolution: int = 100,
    birth_range: Sequence[float] = (0.0, 1.0),
    pers_range: Sequence[float] = (0.0, 1.0),
    image_pixel_size: float | None = 0.05,
    image_resolution: Sequence[int] | None = None,
    image_weight: str = "persistence",
    image_weight_params: dict | None = None,
    image_kernel: str = "gaussian",
    image_kernel_params: dict | None = None,
    image_skew: bool = True,
    p: int | float = 2,
) -> TopoPenaltyResult:
    hyp_cloud = np.asarray(hyp_cloud, dtype=float)
    if hyp_cloud.ndim != 2:
        raise ValueError("hyp_cloud must be a 2D array [n_models, n_points]")
    if transpose_cloud:
        hyp_cloud = hyp_cloud.T
    diagrams = persistent_homology(hyp_cloud, maxdim=maxdim)
    if len(diagrams) < 2:
        return TopoPenaltyResult(0.0, np.zeros((0, 2)), None, None)

    h1 = diagrams[1]
    if h1.size == 0:
        return TopoPenaltyResult(0.0, h1, None, method)

    if method == "landscape":
        pl = persistent_landscape(h1, num_landscapes=num_landscapes, resolution=resolution)
        penalty = landscape_norm(pl, p=p)
        vector = np.asarray(pl.values, dtype=float)
        return TopoPenaltyResult(penalty, h1, vector, method)
    if method == "image":
        img = persistent_image(
            h1,
            birth_range=birth_range,
            pers_range=pers_range,
            pixel_size=image_pixel_size,
            resolution=image_resolution,
            weight=image_weight,
            weight_params=image_weight_params,
            kernel=image_kernel,
            kernel_params=image_kernel_params,
            skew=image_skew,
        )
        penalty = image_energy(img)
        return TopoPenaltyResult(penalty, h1, img, method)

    raise ValueError("method must be 'landscape' or 'image'")
