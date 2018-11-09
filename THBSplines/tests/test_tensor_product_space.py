import numpy as np

from THBSplines.src.TensorProductSpace import TensorProductSpace


def test_univariate_space():
    knots = [
        [0, 0, 1, 2, 3, 4, 4]
    ]

    d = [1]
    dim = 1

    S = TensorProductSpace(d, knots, dim)

    S2, proj = S.refine()

    expected_proj = np.array([
        [1, 0.5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.5, 1, 0.5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.5, 1, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.5, 1],
    ]).T

    np.testing.assert_allclose(proj, expected_proj)
