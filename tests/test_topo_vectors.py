import numpy as np
import pytest

from tgefd.topo_vectors import (
    image_energy,
    landscape_norm,
    persistent_image,
    persistent_landscape,
)


def test_persistent_vectors_shapes():
    pytest.importorskip("persim")

    diagram = np.array([[0.0, 1.0], [0.2, 0.8]])
    pl = persistent_landscape(diagram, num_landscapes=3, resolution=50)
    assert pl.values.shape == (3, 50)
    assert landscape_norm(pl, p=2) >= 0

    img = persistent_image(
        diagram,
        birth_range=(0.0, 1.0),
        pers_range=(0.0, 1.0),
        pixel_size=0.05,
    )
    assert img.shape == (20, 20)
    assert image_energy(img) >= 0
