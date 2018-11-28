import numpy as np
import pytest

from THBSplines.lib.BSpline import BSpline, TensorProductBSpline


@pytest.fixture
def x():
    return np.array([0, 0, 0, 1, 2, 3, 4, 4, 4], dtype=np.float64)


def test_b_spline_evaluation_linear_vectorized():
    knots = np.array([0, 1, 2, 3], dtype=np.float64)
    d = 2
    B = BSpline(d, knots)

    @np.vectorize
    def exact(x):
        if 0 <= x < 1:
            return x * x / 2
        elif 1 <= x < 2:
            return x / 2 * (2 - x) + (3 - x) / 2 * (x - 1)
        else:
            return (3 - x) * (3 - x) / 2

    x = np.linspace(0, 3, 20)
    y_computed = B(x)
    y_expected = exact(x)

    np.testing.assert_allclose(y_computed, y_expected)

def test_b_spline_evaluation_derivatives_vectorized():
    knots = np.array([0, 1, 2, 3], dtype=np.float64)

    d = 2
    B = BSpline(d, knots)
    Bl = BSpline(d-1, knots[:-1])
    Br = BSpline(d-1, knots[1:])

    @np.vectorize
    def exact(x):
        if 0 <= x < 1:
            return x
        elif 1 <= x < 2:
            return 1 - 2 * x
        else:
            return -3 + x

    x = np.linspace(0, 3, 20)
    y_computed = B.derivative(x, 1)
    y_expected = Bl(x) - Br(x)

    np.testing.assert_allclose(y_computed, y_expected)


def test_b_spline_equality():
    B1 = BSpline(2, np.array([0, 1, 2, 3], dtype=np.float64))
    B2 = BSpline(3, np.array([0, 1, 2, 3, 4], dtype=np.float64))
    B3 = BSpline(2, np.array([-1, 0, 1, 2], dtype=np.float64))

    assert B1 == B1
    assert B1 != B2
    assert B1 != B3
    assert B2 != B3


def test_tensor_product_b_splines():
    knots = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ], dtype=np.float64)
    degrees = np.array([3, 3], dtype=np.intc)

    B = TensorProductBSpline(degrees, knots)
    x = np.random.uniform(0, 4, size=(10, 2))

    assert B(x).shape == (10,)


def test_tensor_product_b_splines_quadratic():
    knots = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float64)
    d = 2
    degrees = np.array([2, 2], dtype=np.intc)

    @np.vectorize
    def exact(x):
        if 0 <= x < 1:
            return x * x / 2
        elif 1 <= x < 2:
            return x / 2 * (2 - x) + (3 - x) / 2 * (x - 1)
        else:
            return (3 - x) * (3 - x) / 2

    def eexact(x):
        return exact(x[:, 0]) * exact(x[:, 1])

    B = TensorProductBSpline(degrees, knots)
    x = np.random.uniform(0, 3, size=(10, 2))
    y_expected = eexact(x)
    y_computed = B(x)
    np.testing.assert_allclose(y_computed, y_expected)