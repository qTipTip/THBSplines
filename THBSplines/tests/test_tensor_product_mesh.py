import numpy as np

from THBSplines.src.TensorProductMesh import TensorProductMesh


def test_tensor_product_mesh_init():
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3]
    ]
    d = [2]
    dim = 1

    M = TensorProductMesh(d, knots, dim)

    expected_cells = np.array([
        [[0, 1]],
        [[1, 2]],
        [[2, 3]]
    ])

    np.testing.assert_allclose(M.cells, expected_cells)

    expected_refined_cells = np.array([
        [[0, 0.5]],
        [[0.5, 1]],
        [[1, 1.5]],
        [[1.5, 2]],
        [[2, 2.5]],
        [[2.5, 3]]
    ])

    M2 = M.refine()

    np.testing.assert_allclose(M2.cells, expected_refined_cells)
