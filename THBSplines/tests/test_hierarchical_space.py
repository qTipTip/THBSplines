import pytest

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductMesh import TensorProductMesh
from THBSplines.src.TensorProductSpace import TensorProductSpace


@pytest.fixture
def SM():
    knots = [
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1]
    dim = 1

    S = TensorProductSpace(d, knots, dim)
    M = TensorProductMesh(d, knots, dim)

    return S, M


def test_hierarchical_space_init(SM):
    S, M = SM

    hierarchical_mesh = HierarchicalMesh(M)
    H = HierarchicalSpace(hierarchical_mesh, S)

    assert H.number_of_levels == 1
    assert H.number_of_active_functions_per_level == {0: 4}
    assert H.number_of_functions == 4
    assert H.active_functions_per_level == {0: {0, 1, 2, 3}}
    assert H.deactivated_functions_per_level == {0: set()}
    assert H.projectors == {}
    assert H.physical_dim == 1
    assert H.parametric_dim == 1

    H.mesh.add_new_level()
    H.add_new_level()

    assert H.number_of_levels == 2
    assert H.number_of_active_functions_per_level == {0: 4, 1: 0}
    assert H.active_functions_per_level == {0: {0, 1, 2, 3}, 1: set()}
    assert H.deactivated_functions_per_level == {0: set(), 1: set()}


def test_hierarchical_space_refine_hierarchical_space():
    knots = [[0, 0, 1, 2, 2]]
    d = [1]
    dim = 1

    S = TensorProductSpace(d, knots, dim)
    H = HierarchicalMesh(S.mesh)

    H = HierarchicalSpace(H, S)

    marked_cells = [{0}]

    NE = H.refine_hierarchical_mesh(marked_cells)

    assert NE == {0: set(), 1: {0, 1}}
    assert H.mesh.active_elements_per_level[0] == {1}
    assert H.mesh.active_elements_per_level[1] == {0, 1}

    MF = H.functions_to_deactivate_from_cells(marked_cells)
    assert MF == {0: {0}}

