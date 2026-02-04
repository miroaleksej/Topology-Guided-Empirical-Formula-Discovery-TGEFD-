import inspect

from tgefd.config.models import TopologyConfig
from tgefd.search import search_models
from tgefd.sparse import sparse_regression, topo_sparse_regression


def _default_of(func, name: str):
    return inspect.signature(func).parameters[name].default


def test_defaults_raw_mode():
    assert _default_of(sparse_regression, "normalize") is False
    assert _default_of(topo_sparse_regression, "normalize") is False
    assert _default_of(search_models, "normalize") is False


def test_topology_point_scale_default_none():
    assert TopologyConfig.model_fields["point_scale"].default == "none"
