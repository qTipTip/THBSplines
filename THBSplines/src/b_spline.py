from functools import lru_cache

import numpy as np


class BSpline(object):

    def __init__(self, degrees, knots):
        """
        A list of d = (d_1, ..., d_n) polynomial degrees, and a list of
        corresponding knots.
        
        :param degrees: np.array of degrees
        :param knots: list of knot-vectors where knot vector knots[i] has
        length d_i + 2.
        """
        self.degrees = degrees
        self.knots = knots
        self.basis_functions = [UnivariateBSpline(d, t) for d, t in zip(degrees, knots)]
        self.dimension = len(degrees)
        self.elements_of_support = np.array([], dtype=np.int64)
        self.support = np.array([
            [[k[0], k[-1]] for k in self.knots]
        ], dtype=np.float64)
        self.tensor_product_indices = None  # used for identifying the B-spline in a tensor product mesh.

    def __call__(self, x):
        """
        Evaluates the BSpline at the point x.

        :param x: np.ndarray
        :return: B(x_1, ..., x_d)
        """
        x = np.reshape(x, self.dimension)
        evaluated_b_splines = np.array([self.basis_functions[i](x[i]) for i in range(self.dimension)])

        return np.prod(evaluated_b_splines)

    def __str__(self):
        return """
        
        knots  = {}
        degrees = {}
        
        """.format(self.knots, self.degrees)

    __repr__ = __str__


class UnivariateBSpline(object):

    def __init__(self, degree, knots, endpoint=True):
        self.degree = degree
        self.knots = knots
        self._endpoint = endpoint
        self._augmented_knots = self.augment_knots()

    @lru_cache(maxsize=None, typed=False)
    def __call__(self, x):
        """
        Evaluates the univariate B-spline at the point x.

        :param x: point of evaluation
        :return: B(x)
        """

        i = self.knot_index(x)
        if i == -1:
            return 0
        t = self._augmented_knots
        i = i + self.degree + 1

        # evaluation loop
        c = np.zeros(len(t) - self.degree - 1)
        c[self.degree + 1] = 1
        c = c[i - self.degree: i + 1]
        for k in range(self.degree, 0, -1):
            t1 = t[i - k + 1: i + 1]
            t2 = t[i + 1: i + k + 1]
            omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

            a = np.multiply((1 - omega), c[:-1])
            b = np.multiply(omega, c[1:])
            c = a + b

        return float(c)

    @lru_cache(maxsize=None, typed=False)
    def knot_index(self, x):
        """
        Finds the knot-index such that x is contained in the corresponding knot-interval.

        :param x: point of interest
        :return: index
        """
        return find_knot_index(x, self.knots, self.endpoint)

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value):
        self._endpoint = value
        self.__call__.cache.clear()
        self.knot_index.cache.clear()

    def augment_knots(self):
        return augment_knots(self.knots, self.degree)


def find_knot_index(x, knots, endpoint=False):
    """
    Finds the index i such that t_i <= x < t_i+1.
    If endpoint is True, then we check for t_n-1 <= x <= t_n
    for the final index.
    
    :param x: point in question
    :param knots: knot vector
    :endpoint false: Boolean, whether to include the final knot
    :return: index i
    """
    # if we have requested end point, and are at the end, return
    # corresponding index.
    knots = knots
    if endpoint and (knots[-2] <= x <= knots[-1]):
        i = max(np.argmax(knots < x) - 1, 0)
        return len(knots) - i - 2
    # if we are outside the domain, return -1
    if x < knots[0] or x >= knots[-1]:
        return -1
    # otherwise, return the corresponding index
    return np.max(np.argmax(knots > x) - 1, 0)


def augment_knots(knots, degree):
    """
    Adds degree + 1 values to either end of the knot vector, in order to
    facilitate matrix based evaluation.

    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """

    return np.pad(knots, (degree + 1, degree + 1), 'constant',
                  constant_values=(knots[0] - 1, knots[-1] + 1))
