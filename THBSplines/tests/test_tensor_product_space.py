import numpy as np
from THBSplines.src.tensor_product_space import TensorProductSpace


def test_refine():
    knots = [
        [0, 1, 2],
        [0, 1, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = TensorProductSpace(knots, degrees, d)

    np.testing.assert_allclose(T.mesh.cells, np.array([
        [[0, 1], [0, 1]],
        [[1, 2], [0, 1]],
        [[0, 1], [1, 2]],
        [[1, 2], [1, 2]]
    ]))

    T1, proj, _ = T.refine()
    np.testing.assert_allclose(T1.mesh.cells, np.array([
        [[0, 0.5], [0, 0.5]],
        [[0.5, 1], [0, 0.5]],
        [[1, 1.5], [0, 0.5]],
        [[1.5, 2], [0, 0.5]],

        [[0, 0.5], [0.5, 1]],
        [[0.5, 1], [0.5, 1]],
        [[1, 1.5], [0.5, 1]],
        [[1.5, 2], [0.5, 1]],

        [[0, 0.5], [1, 1.5]],
        [[0.5, 1], [1, 1.5]],
        [[1, 1.5], [1, 1.5]],
        [[1.5, 2], [1, 1.5]],

        [[0, 0.5], [1.5, 2]],
        [[0.5, 1], [1.5, 2]],
        [[1, 1.5], [1.5, 2]],
        [[1.5, 2], [1.5, 2]],
    ]))


def test_get_cells():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    deg = [1, 1]
    T = TensorProductSpace(knots, deg, d)

    funcs_to_deact = np.array([0, 1, 3, 4])

    cells, cells_map = T.get_cells(funcs_to_deact)
    np.testing.assert_equal(cells_map, {0 : [0], 1: [0, 1], 3 : [0, 2], 4 : [0, 1, 2, 3]})