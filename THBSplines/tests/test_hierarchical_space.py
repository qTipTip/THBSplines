import numpy as np
from THBSplines import create_subdivision_matrix
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

    C = T.compute_full_projection_matrix(0)

    assert C.shape == (7, 4)
    np.testing.assert_allclose(C.toarray(), np.array([
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
    C = T.compute_full_projection_matrix(0)
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
    np.testing.assert_allclose(C.toarray(), np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]
    ]))


def test_full_mesh_refine():
    knots = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.float64)
    deg = [1, 1]
    dim = 2
    T = HierarchicalSpace(knots, deg, dim)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 4
    })

    cells = {}
    rectangle = np.array([[0, 1 + np.spacing(1)], [0, 1 + np.spacing(1)]], dtype=np.float64)
    cells[0] = T.refine_in_rectangle(rectangle, 0)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 9,
    })

    cells[1] = T.refine_in_rectangle(rectangle, 1)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 0,
        2: 25
    })

    cells[2] = T.refine_in_rectangle(rectangle, 2)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 0,
        2: 0,
        3: 81
    })


def test_partition_of_unity():
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3],
        [0, 0, 0, 1, 2, 3, 3, 3]
    ]
    d = 2
    degrees = [2, 2]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3]}
    T = refine(T, marked_cells)
    marked_cells = {0: [0, 1, 2, 3], 1: [0, 1, 2]}
    T = refine(T, marked_cells)
    C = create_subdivision_matrix(T)
    N = 5
    x = np.linspace(0, 3, N)
    y = np.linspace(0, 3, N)
    z = np.zeros((N, N))


    c = C[T.nlevels - 1]
    c = c.toarray()
    for i in range(T.nfuncs):
        u = np.zeros(T.nfuncs)
        u[i] = 1
        u_fine = c @ u

        f = T.spaces[T.nlevels - 1].construct_function(u_fine)
        for i in range(N):
            for j in range(N):
                z[i, j] += f(np.array([x[i], y[j]]))

    np.testing.assert_allclose(z, 1)