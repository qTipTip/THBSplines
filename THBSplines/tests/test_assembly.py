import numpy as np
import pytest
import scipy.integrate

from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def integrate(bi, bj, points, weights):
    I = 0
    for i in range(len(points)):
        I += weights[i] * bi(points[i]) * bj(points[i])

    return I


def integrate_smart(bi, bj, cell):
    dim = cell.shape[0]

    if dim == 1:
        return scipy.integrate.quad(lambda x: bi(x) * bj(x), cell[0, 0], cell[0, 1])[0]
    elif dim == 2:
        return scipy.integrate.dblquad(lambda y, x: bi(np.array([x, y])) * bj(np.array([x, y])), cell[0, 0], cell[0, 1],
                                       lambda x: cell[1, 0], lambda x: cell[1, 1])[0]


def translate_points(points, cell, weights):
    """
    Translates the gauss-quadrature points to the cell
    :param points:
    :param cell:
    :return:
    """
    n = len(points)
    dim = cell.shape[0]
    quad_points = np.zeros((dim, n))
    quad_weights = np.zeros((dim, n))
    quad_weights[:] = weights

    for i in range(dim):
        for j in range(n):
            quad_points[i, j] = 0.5 * (points[j] + 1) * (cell[i, 1] - cell[i, 0]) + cell[i, 0]
    weights = np.prod(np.stack(np.meshgrid(*quad_points), -1).reshape(-1, dim), axis=1)
    points = np.stack(np.meshgrid(*quad_points), -1).reshape(-1, dim)
    area_cell = np.prod(np.diff(cell[:]))

    return points, weights, area_cell


def local_mass_matrix(T, level):
    active_cells = T.mesh.meshes[level].cells[T.mesh.aelem_level[level]]

    ndofs_u = T.spaces[level].nfuncs
    ndofs_v = T.spaces[level].nfuncs

    M = np.zeros((ndofs_u, ndofs_v))

    for cell in active_cells:
        for i in range(ndofs_u):
            bi = T.spaces[level].basis[i]
            for j in range(ndofs_v):
                bj = T.spaces[level].basis[j]
                I = integrate_smart(bi, bj, cell)
                M[i, j] += I

    return M


def hierarchical_mass_matrix(T):
    mesh = T.mesh

    n = T.nfuncs
    M = np.zeros((n, n))

    ndofs_u = 0
    ndofs_v = 0
    C = T.create_subdivision_matrix('full')
    for level in range(mesh.nlevels):
        ndofs_u += T.nfuncs_level[level]
        ndofs_v += T.nfuncs_level[level]

        if mesh.nel_per_level[level] > 0:
            M_local = local_mass_matrix(T, level)

            dofs_u = range(ndofs_u)
            dofs_v = range(ndofs_v)

            ix = np.ix_(dofs_u, dofs_v)
            M[ix] += C[level].T @ M_local @ C[level]

    return M


def test_linear_mass_matrix():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    deg = [1]
    dim = 1

    T = HierarchicalSpace(knots, deg, dim)
    cells = {0: [1]}
    T = refine(T, cells)

    M = hierarchical_mass_matrix(T)

    assert T.nfuncs_level == {0: 4, 1: 1}
    assert T.nfuncs == 5
    assert M.shape == (5, 5)

    expected_M = np.array(
        [[0.11111111, 0.05555556, 0., 0., 0., ],
         [0.05555556, 0.16666667, 0., 0, 0.02777778],
         [0., 0., 0.16666667, 0.05555556, 0.02777778],
         [0., 0., 0.05555556, 0.11111111, 0.],
         [0., 0.02777778, 0.02777778, 0., 0.11111111]]
    )
    np.testing.assert_allclose(M, expected_M)


@pytest.mark.slow
def test_bilinear_mass_matrix():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1],
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    deg = [1, 1]
    dim = 2

    t = HierarchicalSpace(knots, deg, dim)
    cells = {0: [1]}
    t = refine(t, cells)

    m = hierarchical_mass_matrix(t)

    expected_m = np.array(
        [[0.012346, 0.006173, 0.000000, 0.000000, 0.006173, 0.003086, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.006173, 0.018519, 0.000000, 0.000000, 0.003086, 0.009452, 0.000193, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001929, 0.002315],
         [0.000000, 0.000000, 0.018519, 0.006173, 0.000000, 0.000193, 0.009452, 0.003086, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001929, 0.002315],
         [0.000000, 0.000000, 0.006173, 0.012346, 0.000000, 0.000000, 0.003086, 0.006173, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.006173, 0.003086, 0.000000, 0.000000, 0.024691, 0.012346, 0.000000, 0.000000, 0.006173, 0.003086, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.003086, 0.009452, 0.000193, 0.000000, 0.012346, 0.046682, 0.009645, 0.000000, 0.003086, 0.012346, 0.003086,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000386, 0.003858],
         [0.000000, 0.000193, 0.009452, 0.003086, 0.000000, 0.009645, 0.046682, 0.012346, 0.000000, 0.003086, 0.012346,
          0.003086, 0.000000, 0.000000, 0.000000, 0.000000, 0.000386, 0.003858],
         [0.000000, 0.000000, 0.003086, 0.006173, 0.000000, 0.000000, 0.012346, 0.024691, 0.000000, 0.000000, 0.003086,
          0.006173, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.006173, 0.003086, 0.000000, 0.000000, 0.024691, 0.012346, 0.000000,
          0.000000, 0.006173, 0.003086, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.012346, 0.003086, 0.000000, 0.012346, 0.049383, 0.012346,
          0.000000, 0.003086, 0.012346, 0.003086, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.012346, 0.003086, 0.000000, 0.012346, 0.049383,
          0.012346, 0.000000, 0.003086, 0.012346, 0.003086, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.006173, 0.000000, 0.000000, 0.012346,
          0.024691, 0.000000, 0.000000, 0.003086, 0.006173, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006173, 0.003086, 0.000000,
          0.000000, 0.012346, 0.006173, 0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.012346, 0.003086,
          0.000000, 0.006173, 0.024691, 0.006173, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.012346,
          0.003086, 0.000000, 0.006173, 0.024691, 0.006173, 0.000000, 0.000000],
         [.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086,
          0.006173, 0.000000, 0.000000, 0.006173, 0.012346, 0.000000, 0.000000
          ],
         [.000000, 0.001929, 0.001929, 0.000000, 0.000000, 0.000386, 0.000386, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006173, 0.003086
          ],
         [.000000, 0.002315, 0.002315, 0.000000, 0.000000, 0.003858, 0.003858, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.003086, 0.012346
          ], ])
    np.testing.assert_allclose(m, expected_m, atol=1.0e-5)


def test_local_mass_matrix_univariate():
    knots = [
        [0, 1, 2]
    ]
    deg = [1]
    dim = 1
    T = HierarchicalSpace(knots, deg, dim)

    M = local_mass_matrix(T, 0)

    assert M.shape == (1, 1)
    np.testing.assert_allclose(M, 2 / 3)


def test_local_mass_matrix_univariate_refined():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    deg = [1]
    dim = 1
    T = HierarchicalSpace(knots, deg, dim)
    cell = {0: [1]}
    T = refine(T, cell)

    m0 = local_mass_matrix(T, 0)
    m1 = local_mass_matrix(T, 1)
    assert m0.shape == (4, 4)
    assert m1.shape == (7, 7)