import numpy as np

from tgefd.data import make_synthetic_truth_data


def test_make_synthetic_truth_data_deterministic():
    x1, y1 = make_synthetic_truth_data(n=5, noise=0.0, seed=1, x_min=0.0, x_max=2.0)
    x2, y2 = make_synthetic_truth_data(n=5, noise=0.0, seed=999, x_min=0.0, x_max=2.0)
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_make_synthetic_truth_data_noise_changes():
    x1, y1 = make_synthetic_truth_data(n=20, noise=0.05, seed=1)
    x2, y2 = make_synthetic_truth_data(n=20, noise=0.05, seed=2)
    assert np.allclose(x1, x2)
    assert not np.allclose(y1, y2)
