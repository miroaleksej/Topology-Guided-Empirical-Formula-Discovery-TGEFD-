from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from persim import PersLandscapeApprox, PersistenceImager
except ImportError as exc:  # pragma: no cover - exercised via error path in runtime usage
    PersLandscapeApprox = None
    PersistenceImager = None
    _persim_import_error = exc
else:
    _persim_import_error = None


def _require_persim() -> None:
    if PersLandscapeApprox is None or PersistenceImager is None:
        raise ImportError(
            "persim is required for landscapes/images. Install with `pip install persim`."
        ) from _persim_import_error


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    finite = np.isfinite(diagram[:, 1])
    return diagram[finite]


def persistent_landscape(
    diagram: np.ndarray,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> "PersLandscapeApprox":
    _require_persim()
    diagram = _finite_diagram(diagram)
    if diagram.size == 0:
        diagram = np.zeros((0, 2), dtype=float)
    pl = PersLandscapeApprox(
        dgms=[diagram],
        hom_deg=0,
        num_steps=resolution,
    )
    values = np.asarray(pl.values, dtype=float)
    if num_landscapes is not None:
        if values.shape[0] >= num_landscapes:
            values = values[:num_landscapes]
        else:
            pad = np.zeros((num_landscapes - values.shape[0], values.shape[1]), dtype=float)
            values = np.vstack([values, pad])
    pl.values = values
    return pl


def landscape_norm(pl: "PersLandscapeApprox", p: int | float = 2) -> float:
    return float(pl.p_norm(p=p))


def persistent_image(
    diagram: np.ndarray,
    birth_range: Sequence[float] = (0.0, 1.0),
    pers_range: Sequence[float] = (0.0, 1.0),
    pixel_size: float | None = 0.05,
    resolution: Sequence[int] | None = None,
    weight: str = "persistence",
    weight_params: dict | None = None,
    kernel: str = "gaussian",
    kernel_params: dict | None = None,
    skew: bool = True,
) -> np.ndarray:
    _require_persim()
    diagram = _finite_diagram(diagram)
    if weight_params is None:
        weight_params = {"n": 2.0}
    if kernel_params is None:
        kernel_params = {"sigma": [[0.05, 0.0], [0.0, 0.05]]}

    birth_range = tuple(float(v) for v in birth_range)
    pers_range = tuple(float(v) for v in pers_range)
    birth_span = birth_range[1] - birth_range[0]
    pers_span = pers_range[1] - pers_range[0]
    if birth_span <= 0 or pers_span <= 0:
        raise ValueError("birth_range and pers_range must have positive span")

    if resolution is not None:
        target_resolution = tuple(int(v) for v in resolution)
        if pixel_size is None:
            pixel_size = min(
                birth_span / max(target_resolution[0], 1),
                pers_span / max(target_resolution[1], 1),
            )
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
    elif pixel_size is None:
        raise ValueError("pixel_size or resolution must be provided")

    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")

    pimgr = PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=float(pixel_size),
        weight=weight,
        weight_params=weight_params,
        kernel=kernel,
        kernel_params=kernel_params,
    )
    if diagram.size == 0:
        if resolution is not None:
            return np.zeros(target_resolution, dtype=float)
        return np.zeros(pimgr.resolution, dtype=float)

    img = np.asarray(pimgr.transform(diagram, skew=skew), dtype=float)

    if resolution is not None and img.shape != target_resolution:
        raise ValueError(
            "persistence image resolution does not match requested size; "
            "adjust pixel_size or ranges"
        )

    return img


def image_energy(img: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(img, dtype=float)))
