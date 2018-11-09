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


def test_bivariate_space():
    knots = [
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]

    d = [1, 1]
    dim = 2

    S = TensorProductSpace(d, knots, dim)
    S2, proj = S.refine()

    expected_proj = np.array([
        [1., 0., 0., 0.],
        [0.5, 0.5, 0., 0.],
        [0., 1., 0., 0.],
        [0.5, 0., 0.5, 0.],
        [0.25, 0.25, 0.25, 0.25],
        [0., 0.5, 0., 0.5],
        [0., 0., 1., 0.],
        [0., 0., 0.5, 0.5],
        [0., 0., 0., 1.]])
    np.testing.assert_allclose(proj, expected_proj)


def test_get_basis_functions():
    knots = [
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1]
    dim = 1
    S = TensorProductSpace(d, knots, dim)

    assert S.get_basis_functions(0) == {0, 1}
    assert S.get_basis_functions(1) == {1, 2}
    assert S.get_basis_functions(2) == {2, 3}
