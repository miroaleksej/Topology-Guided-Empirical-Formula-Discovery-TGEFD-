from __future__ import annotations

import numpy as np


def residuals(Phi: np.ndarray, coeffs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return Phi @ coeffs - y


def l0_norm(coeffs: np.ndarray) -> int:
    return int(np.count_nonzero(coeffs))


def l1_norm(coeffs: np.ndarray) -> float:
    return float(np.sum(np.abs(coeffs)))


def l2_error(Phi: np.ndarray, coeffs: np.ndarray, y: np.ndarray) -> float:
    res = residuals(Phi, coeffs, y)
    return float(np.linalg.norm(res))


def rmse(Phi: np.ndarray, coeffs: np.ndarray, y: np.ndarray) -> float:
    res = residuals(Phi, coeffs, y)
    return float(np.sqrt(np.mean(res ** 2)))
