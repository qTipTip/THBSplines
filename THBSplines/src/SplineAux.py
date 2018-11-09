import itertools
from functools import lru_cache

import numpy as np


def find_knot_index(x, knots, endpoint=False):
    # if we have requested end point, and are at the end, return corresponding index.
    knots = knots
    if endpoint and (knots[-2] <= x <= knots[-1]):
        i = max(np.argmax(knots < x) - 1, 0)
        return len(knots) - i - 2
    # if we are outside the domain, return -1
    if x < knots[0] or x >= knots[-1]:
        return -1
    # otherwise, return the corresponding index
    return np.max(np.argmax(knots > x) - 1, 0)


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

    def augment_knots(self):
        """
        Adds degree + 1 values to either end of the knot vector, in order to facilitate matrix based evaluation.
        :param knots: knot vector
        :param degree: polynomial degree
        :return: padded knot vector
        """

        return np.pad(self.knots, (self.degree + 1, self.degree + 1), 'constant',
                      constant_values=(self.knots[0] - 1, self.knots[-1] + 1))

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value):
        self._endpoint = value
        self.__call__.cache.clear()
        self.knot_index.cache.clear()


class BSpline(object):

    def __init__(self, degrees, knots):
        """
        A list of d = (d_1, ..., d_n) polynomial degrees, and a list of corresponding knots.
        :param degrees: np.array of degrees
        :param knots: list of knot-vectors where knot vector knots[i] has length d_i + 2.
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


def compute_knot_insertion_matrix(degree, coarse, fine):
    """
    Computes the knot insertion matrix that write coarse B-splines as linear combinations
    of finer B-splines. Requires tau, t to be p+1 regular.
    :param degree: The degree
    :param coarse: p+1 regular coarse knot vector with common ends
    :param fine: p+1 regular fine knot vector with common ends
    :return: The knot insertion matrix A
    """
    # TODO: Enforce p+1 regularity properly
    # assert fine[0:degree + 1] == coarse[0:degree + 1]
    # assert fine[-(degree + 1):-1] == coarse[-(degree + 1):-1]

    m = len(fine) - (degree + 1)
    n = len(coarse) - (degree + 1)

    a = np.zeros(shape=(m, n))
    fine = np.array(fine, dtype=np.float64)
    coarse = np.array(coarse, dtype=np.float64)
    for i in range(m):
        mu = find_knot_index(fine[i], coarse)
        b = 1
        for k in range(1, degree + 1):
            tau1 = coarse[mu - k + 1:mu + 1]
            tau2 = coarse[mu + 1:mu + k + 1]
            omega = (fine[i + k] - tau1) / (tau2 - tau1)
            b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)
        a[i, mu - degree:mu + 1] = b

    return a


def generate_tensor_product_space(degrees, knots, dim):
    b_splines = []
    idx_start = [
        [j for j in range(len(knots[i]) - degrees[i] - 1)] for i in range(dim)
    ]
    idx_stop = [
        [j + degrees[i] + 2 for j in idx_start[i]] for i in range(dim)
    ]

    idx_start_perm = list(itertools.product(*idx_start))
    idx_stop_perm = list(itertools.product(*idx_stop))
    n = len(idx_start_perm)

    for i in range(n):
        new_knots = []
        for j in range(dim):
            new_knots.append(knots[j][idx_start_perm[i][j]: idx_stop_perm[i][j]])
        new_b_spline = BSpline(degrees, new_knots)
        new_b_spline.tensor_product_indices = idx_start_perm[i]
        b_splines.append(new_b_spline)

    return b_splines


def generate_cells(knots):
    dim = len(knots)
    cells = []

    unique_knots = [np.unique(knot) for knot in knots]
    idx_start = [
        [j for j in range(len(unique_knots[i]) - 1)] for i in range(dim)
    ]
    idx_stop = [
        [j + 1 for j in idx_start[i]] for i in range(dim)
    ]

    idx_start_perm = list(itertools.product(*idx_start))
    idx_stop_perm = list(itertools.product(*idx_stop))
    n = len(idx_start_perm)

    for i in range(n):
        new_cells = []
        for j in range(dim):
            new_cells.append(unique_knots[j][idx_start_perm[i][j]: idx_stop_perm[i][j] + 1])
        cells.append(np.array(new_cells))

    return np.array(cells)
