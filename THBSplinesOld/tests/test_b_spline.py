import numpy as np

from THBSplinesOld.src.BSplines import UnivariateBSpline, BSpline


def test_univariate_evaluation_linear():
    d = 1
    k = np.array([0, 1, 2])
    B = UnivariateBSpline(d, k, endpoint=False)

    x = np.linspace(0, 2, 30)

    def exact(y):
        if 0 <= y < 1:
            return y
        elif 1 <= y < 2:
            return 2 - y
        else:
            return 0

    computed = [B(X) for X in x]
    expected = [exact(X) for X in x]

    np.testing.assert_allclose(computed, expected)


def test_univariate_evaluation_quadratic():
    d = 2
    k = np.array([0, 0, 0, 1])
    B = UnivariateBSpline(d, k, endpoint=False)

    x = np.linspace(0, 1, 30)

    def exact(y):
        if 0 <= y < 1:
            return (1 - y) ** 2
        else:
            return 0

    computed = [B(X) for X in x]
    expected = [exact(X) for X in x]

    np.testing.assert_allclose(computed, expected)


def test_bivariate_evaluation_linear():
    d = (1, 1)
    k = np.array([
        [0, 1, 2],
        [0, 1, 2]
    ])
    B = BSpline(d, k)

    def exact_uni(y):
        if 0 <= y < 1:
            return y
        elif 1 <= y < 2:
            return 2 - y
        else:
            return 0

    x = np.array([
        [X, Y]
        for X in np.linspace(0, 2, 100)
        for Y in np.linspace(0, 2, 100)
    ])

    def exact(x):
        return exact_uni(x[0]) * exact_uni(x[1])

    computed = [B(X) for X in x]
    expected = [exact(X) for X in x]

    np.testing.assert_allclose(computed, expected)


def test_b_spline_evaluation_mixed_degree():
    d = (1, 2)
    knots = np.array([
        [0, 1, 2],
        [0, 0, 0, 1]
    ])
    B = BSpline(d, knots)

    x = np.array([
        [X, Y]
        for X in np.linspace(0, 2, 30)
        for Y in np.linspace(0, 1, 30)
    ])

    def exact_uni_lin(y):
        if 0 <= y < 1:
            return y
        elif 1 <= y < 2:
            return 2 - y
        else:
            return 0

    def exact_uni_quad(y):
        if 0 <= y < 1:
            return (1 - y) ** 2
        else:
            return 0

    exact = lambda x: exact_uni_lin(x[0]) * exact_uni_quad(x[1])

    computed = [B(X) for X in x]
    expected = [exact(X) for X in x]

    np.testing.assert_allclose(computed, expected)


def test_b_spline_support():
    d = (1, 2)
    knots = np.array([
        [0, 1, 2],
        [0, 0, 0, 1]
    ])
    B = BSpline(d, knots)

    expected = np.array([
        [[0, 2], [0, 1]]
    ], dtype=np.float64)

    np.testing.assert_allclose(B.support, expected)
