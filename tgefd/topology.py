from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from .models import ModelResult

try:
    from ripser import ripser
except ImportError as exc:  # pragma: no cover - exercised via error path in runtime usage
    ripser = None
    _ripser_import_error = exc
else:
    _ripser_import_error = None


@dataclass(frozen=True)
class PersistenceFeature:
    birth: float
    death: float
    persistence: float


@dataclass(frozen=True)
class PersistenceComponent:
    indices: np.ndarray
    birth: float
    death: float
    persistence: float
    size: int


def build_point_cloud(
    results: Iterable[ModelResult],
    scale: Literal["none", "standard"] = "none",
) -> np.ndarray:
    rows = []
    for r in results:
        rows.append([r.params.p, r.params.q, r.params.lam, r.error, r.l0])
    X = np.asarray(rows, dtype=float)

    if scale == "none":
        return X
    if scale == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std
    raise ValueError("scale must be 'none' or 'standard'")


def persistent_homology(
    X: np.ndarray,
    maxdim: int = 2,
    metric: str = "euclidean",
) -> list[np.ndarray]:
    if ripser is None:
        raise ImportError(
            "ripser is required for persistent homology. "
            "Install with `pip install ripser`."
        ) from _ripser_import_error

    X = np.asarray(X, dtype=float)
    return ripser(X, maxdim=maxdim, metric=metric)["dgms"]


def significant_features(
    diagram: np.ndarray,
    persistence_threshold: float,
    include_infinite: bool = True,
) -> list[PersistenceFeature]:
    features = []
    for birth, death in diagram:
        birth = float(birth)
        death = float(death)
        if np.isinf(death):
            if include_infinite:
                features.append(PersistenceFeature(birth, death, float("inf")))
            continue
        persistence = float(death - birth)
        if persistence > persistence_threshold:
            features.append(PersistenceFeature(birth, death, persistence))
    return features


def _pairwise_distances(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "precomputed":
        distances = np.asarray(X, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("precomputed metric requires a square distance matrix")
        return distances
    if metric != "euclidean":
        raise ValueError("only 'euclidean' and 'precomputed' metrics are supported")
    diffs = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def _component_labels(distances: np.ndarray, radius: float) -> np.ndarray:
    n = distances.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    rows, cols = np.where(distances <= radius)
    for i, j in zip(rows, cols):
        if i < j:
            union(i, j)

    roots = [find(i) for i in range(n)]
    label_map: dict[int, int] = {}
    labels = np.empty(n, dtype=int)
    next_label = 0
    for idx, root in enumerate(roots):
        if root not in label_map:
            label_map[root] = next_label
            next_label += 1
        labels[idx] = label_map[root]
    return labels


def _component_death(distances: np.ndarray, indices: np.ndarray) -> float:
    if distances.size == 0:
        return float("inf")
    n = distances.shape[0]
    if len(indices) == n:
        return float("inf")
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    inter = distances[np.ix_(mask, ~mask)]
    if inter.size == 0:
        return float("inf")
    return float(np.min(inter))


def persistent_components(
    X: np.ndarray,
    persistence_threshold: float,
    metric: str = "euclidean",
    min_component_size: int = 1,
    include_infinite: bool = False,
) -> list[PersistenceComponent]:
    """Return H0 components that persist beyond the given threshold."""
    if persistence_threshold < 0:
        raise ValueError("persistence_threshold must be non-negative")
    if min_component_size < 1:
        raise ValueError("min_component_size must be at least 1")

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return []

    distances = _pairwise_distances(X, metric=metric)
    labels = _component_labels(distances, persistence_threshold)

    components = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        size = int(len(indices))
        if size < min_component_size:
            continue
        death = _component_death(distances, indices)
        if np.isinf(death) and not include_infinite:
            continue
        components.append(
            PersistenceComponent(
                indices=indices.astype(int),
                birth=0.0,
                death=float(death),
                persistence=float(death),
                size=size,
            )
        )
    return components


def select_stable_models(
    results: Sequence[ModelResult],
    components: Sequence[PersistenceComponent],
) -> list[ModelResult]:
    if not components:
        return []

    indices = sorted({int(i) for comp in components for i in comp.indices})
    if indices and indices[-1] >= len(results):
        raise ValueError("component indices out of range for results")
    return [results[i] for i in indices]
