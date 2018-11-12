import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductSpace import TensorProductSpace


def test_partition_of_unity_univariate():
    knots = [
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]
    ]

    d = [2]
    dim = 1

    S = TensorProductSpace(d, knots, dim)
    H = HierarchicalMesh(S.mesh)
    T = HierarchicalSpace(H, S)

    marked_cells = [{2, 3, 4}]
    T.refine(marked_cells)

    x = np.linspace(0, 8, 100)
    y = np.zeros(100)

    for b in T.get_truncated_basis():
        y += np.array([b(X) for X in x])

    np.testing.assert_allclose(np.ones(100), y)


def test_partition_of_unity_bivariate():
    knots = [
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
        [0, 0, 1, 2, 3, 4, 4]
    ]

    d = [2, 1]
    dim = 2

    S = TensorProductSpace(d, knots, dim)
    H = HierarchicalMesh(S.mesh)
    T = HierarchicalSpace(H, S)

    marked_cells = [{2, 3, 4, 4, 5, 6, 7, 8, 9, 11, 12, 13}]
    T.refine(marked_cells)

    num = 10
    x = np.linspace(0, 8, num)
    y = np.linspace(0, 4, num)
    z = np.zeros((num, num))

    for b in T.get_truncated_basis():
        for i in range(num):
            for j in range(num):
                z[i, j] += b(np.array((x[i], y[i])))

    np.testing.assert_allclose(np.ones((num, num)), z)

def test_partition_of_unity_double_refinement():
    knots = [
        [0, 0, 0, 1, 2, 3, 4, 4, 4],
        [0, 0, 0, 1, 2, 3, 4, 4, 4]
    ]

    d = [2, 2]
    dim = 2

    S = TensorProductSpace(d, knots, dim)
    H = HierarchicalMesh(S.mesh)
    T = HierarchicalSpace(H, S)

    marked_cells = [{0, 1, 2, 3, 4, 5, 7, 8}]
    T.refine(marked_cells)
    marked_cells = [{0, 1, 2, 3, 4, 5, 7, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]
    T.refine(marked_cells)

    num = 10
    x = np.linspace(0, 4, num)
    y = np.linspace(0, 4, num)
    z = np.zeros((num, num))
    for b in T.get_truncated_basis():
        for i in range(num):
            for j in range(num):
                z[i, j] += b(np.array((x[i], y[j])))

    np.testing.assert_allclose(z, np.ones((num, num)))