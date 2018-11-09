import pytest

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.TensorProductMesh import TensorProductMesh


@pytest.fixture
def T():
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3],
        [0, 0, 1, 2, 2]
    ]
    d = [2, 1]
    dim = 2

    return TensorProductMesh(d, knots, dim)


def test_hierarchical_mesh_init(T):
    H = HierarchicalMesh(T)

    assert H.physical_dim == 1
    assert H.parametric_dim == 2

    assert H.number_of_elements == 6
    assert H.number_of_levels == 1
    assert H.number_of_elements_per_level[0] == 6
    assert H.active_elements_per_level[0] == {0, 1, 2, 3, 4, 5}


def test_hierarchical_mesh_add_level(T):
    H = HierarchicalMesh(T)

    H.add_new_level()

    assert H.number_of_levels == 2
    assert H.active_elements_per_level[0] == {0, 1, 2, 3, 4, 5}
    assert H.active_elements_per_level[1] == set()
    assert H.number_of_elements_per_level[0] == 6
    assert H.number_of_elements_per_level[1] == 0
