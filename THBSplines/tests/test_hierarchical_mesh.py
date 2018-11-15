import numpy as np
import pytest

from THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplines.THBSplines.TensorProductMesh import TensorProductMesh


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

    np.testing.assert_equal(H.cell_to_children[0], {
        0: {0, 1, 4, 5},
        1: {2, 3, 6, 7},
        2: {8, 9, 12, 13},
        3: {10, 11, 14, 15},
        4: {16, 17, 20, 21},
        5: {18, 19, 22, 23}
    })


def test_hierarchical_mesh_univariate():
    knots = [
        [0, 1, 2, 3, 4]
    ]
    d = [2]
    dim = 1

    H = HierarchicalMesh(TensorProductMesh(d, knots, dim))

    assert H.cell_to_children == {}

    H.add_new_level()

    assert H.cell_to_children == {0: {0: {0, 1}, 1: {2, 3}, 2: {4, 5}, 3: {6, 7}}}
