import numpy as np
import scipy.integrate
from src.assembly import hierarchical_mass_matrix, local_mass_matrix
from tqdm import tqdm

from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def integrate(bi, bj, points, weights):
    I = 0
    for i in range(len(points)):
        I += weights[i] * bi(points[i]) * bj(points[i])

    return I





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


def integrate_bivariate(bi, bj, gp, gw, a):
    i_vals = bi(gp)
    j_vals = bj(gp)
    vals = gw * i_vals * j_vals
    return sum(vals)







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


if __name__ == '__main__':
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3],
        [0, 0, 0, 1, 2, 3, 3, 3]
    ]
    deg = [2, 2]
    dim = 2

    t = HierarchicalSpace(knots, deg, dim)
    cells = {0: [4]}
    t = refine(t, cells)

    cells[1] = [0, 1, 2, 3, 4, 5, 6, 7]
    t = refine(t, cells)

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    z = np.zeros((100, 100))
    for b in t.spaces[0].basis:
        for i in range(100):
            for j in range(100):
                z[i, j] = b(np.array([x[i], y[j]]))
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        axs = Axes3D(fig)

        X, Y = np.meshgrid(x, y)
        axs.plot_surface(X, Y, z)
        plt.show()