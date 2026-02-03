from __future__ import annotations

import numpy as np


def sparse_regression(
    Phi: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    iters: int = 10,
    normalize: bool = True,
) -> np.ndarray:
    """Sequential thresholded least squares (SINDy-style)."""
    Phi = np.asarray(Phi)
    y = np.asarray(y).reshape(-1)

    if normalize:
        norms = np.linalg.norm(Phi, axis=0)
        norms[norms == 0] = 1.0
        Phi_scaled = Phi / norms
    else:
        norms = np.ones(Phi.shape[1])
        Phi_scaled = Phi

    coeffs = np.linalg.lstsq(Phi_scaled, y, rcond=None)[0]

    for _ in range(iters):
        small = np.abs(coeffs) < lam
        coeffs[small] = 0.0
        if np.any(~small):
            coeffs[~small] = np.linalg.lstsq(
                Phi_scaled[:, ~small], y, rcond=None
            )[0]

    return coeffs / norms


class SparseRegressor:
    def __init__(self, lam: float = 1e-3, iters: int = 10, normalize: bool = True):
        self.lam = lam
        self.iters = iters
        self.normalize = normalize

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        return sparse_regression(Phi, y, self.lam, self.iters, self.normalize)


def topo_sparse_regression(
    Phi: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    mu: float = 1e-2,
    iters: int = 5,
    normalize: bool = True,
    topo_method: str = "landscape",
    topo_samples: int = 64,
    topo_sigma: float = 1e-3,
    topo_maxdim: int = 1,
    topo_transpose_cloud: bool = True,
    topo_landscapes: int = 5,
    topo_resolution: int = 100,
    topo_birth_range: tuple[float, float] = (0.0, 1.0),
    topo_pers_range: tuple[float, float] = (0.0, 1.0),
    topo_image_pixel_size: float | None = 0.05,
    topo_image_resolution: tuple[int, int] | None = None,
    topo_image_weight: str = "persistence",
    topo_image_weight_params: dict | None = None,
    topo_image_kernel: str = "gaussian",
    topo_image_kernel_params: dict | None = None,
    topo_image_skew: bool = True,
    topo_p: int | float = 2,
    random_state: int | None = None,
) -> np.ndarray:
    """SINDy-style STLSQ with topology regularization on perturbed hypothesis clouds."""
    from .topo_regularization import hypothesis_cloud, perturb_coeffs, topological_penalty

    Phi = np.asarray(Phi)
    y = np.asarray(y).reshape(-1)

    if normalize:
        norms = np.linalg.norm(Phi, axis=0)
        norms[norms == 0] = 1.0
        Phi_scaled = Phi / norms
    else:
        norms = np.ones(Phi.shape[1])
        Phi_scaled = Phi

    coeffs = np.linalg.lstsq(Phi_scaled, y, rcond=None)[0]
    rng = np.random.default_rng(random_state)
    n_points = Phi_scaled.shape[0]
    if topo_samples < 1:
        raise ValueError("topo_samples must be at least 1")
    if topo_transpose_cloud:
        topo_samples = min(topo_samples, max(n_points - 1, 1))

    for _ in range(iters):
        small = np.abs(coeffs) < lam
        coeffs[small] = 0.0
        if np.any(~small):
            coeffs[~small] = np.linalg.lstsq(
                Phi_scaled[:, ~small], y, rcond=None
            )[0]

        if mu > 0:
            coeffs_list = perturb_coeffs(
                coeffs,
                sigma=topo_sigma,
                n_samples=topo_samples,
                rng=rng,
            )
            cloud = hypothesis_cloud(Phi_scaled, coeffs_list)
            penalty = topological_penalty(
                cloud,
                method=topo_method,
                maxdim=topo_maxdim,
                transpose_cloud=topo_transpose_cloud,
                num_landscapes=topo_landscapes,
                resolution=topo_resolution,
                birth_range=topo_birth_range,
                pers_range=topo_pers_range,
                image_pixel_size=topo_image_pixel_size,
                image_resolution=topo_image_resolution,
                image_weight=topo_image_weight,
                image_weight_params=topo_image_weight_params,
                image_kernel=topo_image_kernel,
                image_kernel_params=topo_image_kernel_params,
                image_skew=topo_image_skew,
                p=topo_p,
            ).penalty
            coeffs = coeffs / (1.0 + mu * penalty)

    return coeffs / norms


class TopoSparseRegressor:
    def __init__(
        self,
        lam: float = 1e-3,
        mu: float = 1e-2,
        iters: int = 5,
        normalize: bool = True,
        topo_method: str = "landscape",
        topo_samples: int = 64,
        topo_sigma: float = 1e-3,
        topo_maxdim: int = 1,
        topo_transpose_cloud: bool = True,
        topo_landscapes: int = 5,
        topo_resolution: int = 100,
        topo_birth_range: tuple[float, float] = (0.0, 1.0),
        topo_pers_range: tuple[float, float] = (0.0, 1.0),
        topo_image_pixel_size: float | None = 0.05,
        topo_image_resolution: tuple[int, int] | None = None,
        topo_image_weight: str = "persistence",
        topo_image_weight_params: dict | None = None,
        topo_image_kernel: str = "gaussian",
        topo_image_kernel_params: dict | None = None,
        topo_image_skew: bool = True,
        topo_p: int | float = 2,
        random_state: int | None = None,
    ):
        self.lam = lam
        self.mu = mu
        self.iters = iters
        self.normalize = normalize
        self.topo_method = topo_method
        self.topo_samples = topo_samples
        self.topo_sigma = topo_sigma
        self.topo_maxdim = topo_maxdim
        self.topo_transpose_cloud = topo_transpose_cloud
        self.topo_landscapes = topo_landscapes
        self.topo_resolution = topo_resolution
        self.topo_birth_range = topo_birth_range
        self.topo_pers_range = topo_pers_range
        self.topo_image_pixel_size = topo_image_pixel_size
        self.topo_image_resolution = topo_image_resolution
        self.topo_image_weight = topo_image_weight
        self.topo_image_weight_params = topo_image_weight_params
        self.topo_image_kernel = topo_image_kernel
        self.topo_image_kernel_params = topo_image_kernel_params
        self.topo_image_skew = topo_image_skew
        self.topo_p = topo_p
        self.random_state = random_state

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        return topo_sparse_regression(
            Phi,
            y,
            lam=self.lam,
            mu=self.mu,
            iters=self.iters,
            normalize=self.normalize,
            topo_method=self.topo_method,
            topo_samples=self.topo_samples,
            topo_sigma=self.topo_sigma,
            topo_maxdim=self.topo_maxdim,
            topo_transpose_cloud=self.topo_transpose_cloud,
            topo_landscapes=self.topo_landscapes,
            topo_resolution=self.topo_resolution,
            topo_birth_range=self.topo_birth_range,
            topo_pers_range=self.topo_pers_range,
            topo_image_pixel_size=self.topo_image_pixel_size,
            topo_image_resolution=self.topo_image_resolution,
            topo_image_weight=self.topo_image_weight,
            topo_image_weight_params=self.topo_image_weight_params,
            topo_image_kernel=self.topo_image_kernel,
            topo_image_kernel_params=self.topo_image_kernel_params,
            topo_image_skew=self.topo_image_skew,
            topo_p=self.topo_p,
            random_state=self.random_state,
        )
