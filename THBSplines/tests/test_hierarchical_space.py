import numpy as np
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def test_active_funcs_per_level():
    knots = [
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)

    np.testing.assert_equal(T.afunc_level, {0: [0, 1, 2, 3]})


def test_active_funcs_per_level_refine():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    cells = {0: [0]}
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {0: 8, 1: 4})


def test_functions_to_deactivate_from_cells():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0]}
    new_cells = T.mesh.refine(marked_cells)
    marked_functions = T.functions_to_deactivate_from_cells(marked_cells)

    np.testing.assert_equal(marked_functions, {0: [0]})


def test_projection_matrix_linear():
    knots = [
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)

    C = T.projections[0]

    assert C.shape == (7, 4)
    np.testing.assert_allclose(C, np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]]
    ))


def test_projection_matrix_bilinear():
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1, 1]
    dim = 2
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.projections[0]
    assert C.shape == (49, 16)


def test_subdivision_matrix_linear():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.create_subdivision_matrix('full')

    np.testing.assert_allclose(C[0].toarray(), np.eye(4))
    np.testing.assert_allclose(C[1].toarray(), np.array([
        [1, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 1, 0]
    ]))

def test_change_of_basis_matrix_linear():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.get_basis_conversion_matrix(0)
    np.testing.assert_allclose(C, np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]
    ]))