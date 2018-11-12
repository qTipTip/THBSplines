import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductSpace import TensorProductSpace


def test_partition_of_unity():
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