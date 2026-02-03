import numpy as np
import pytest

from tgefd.sparse import sparse_regression, topo_sparse_regression


def test_sparse_regression_recovers_coeffs():
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(200, 3))
    Q, _ = np.linalg.qr(Phi)
    Phi = Q

    coeffs_true = np.array([1.5, 0.0, -2.0])
    y = Phi @ coeffs_true

    coeffs = sparse_regression(Phi, y, lam=0.05, iters=5, normalize=False)
    assert np.allclose(coeffs, coeffs_true, atol=1e-2)


def test_topo_sparse_regression_runs():
    pytest.importorskip("ripser")
    pytest.importorskip("persim")

    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(80, 4))
    Q, _ = np.linalg.qr(Phi)
    Phi = Q
    y = Phi @ np.array([1.0, 0.0, -0.5, 0.0])

    coeffs = topo_sparse_regression(
        Phi,
        y,
        lam=0.05,
        mu=1e-2,
        iters=2,
        normalize=False,
        topo_samples=10,
        topo_sigma=1e-3,
        random_state=0,
    )
    assert coeffs.shape == (4,)
