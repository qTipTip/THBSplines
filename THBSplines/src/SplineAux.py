import itertools
from collections import defaultdict
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


def augment_knots(knots, degree):
    """
    Adds degree + 1 values to either end of the knot vector, in order to facilitate matrix based evaluation.
    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """

    return np.pad(knots, (degree + 1, degree + 1), 'constant',
                  constant_values=(knots[0] - 1, knots[-1] + 1))


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


def compute_knot_insertion_matrix(degrees, coarse_knots, fine_knots):
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

    matrices = []
    for fine, coarse, degree in zip(fine_knots, coarse_knots, degrees):
        fine = augment_knots(fine, degree)
        coarse = augment_knots(coarse, degree)

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
        matrices.append(a[degree + 1:-degree - 1, degree + 1:-degree - 1])
    a = matrices[0]
    for matrix in matrices[1:]:
        a = np.kron(a, matrix)
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


def insert_midpoints(knots, p, s='pad'):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot vector.
    :param s:
    :param knots: p + 1 regular knot vector to be refined
    :param p: spline degree
    :return: refined_knots
    """

    if s == 'pad':
        knots = np.array(knots, dtype=np.float64)
        midpoints = (knots[p:-p - 1] + knots[p + 1:-p]) / 2
        new_array = np.zeros(len(knots) + len(midpoints), dtype=np.float64)

        new_array[:p + 1] = knots[:p + 1]
        new_array[-p - 1:] = knots[-p - 1]
        new_array[p + 1:p + 2 * len(midpoints):2] = midpoints
        new_array[p + 2:p + 2 * len(midpoints) - 1:2] = knots[p + 1:-p - 1]

        return new_array
    else:
        knots = np.array(knots, dtype=np.float64)
        midpoints = (knots[1:] + knots[:-1]) / 2
        new_array = np.zeros(len(knots) + len(midpoints), dtype=np.float64)
        n = len(new_array)
        new_array[0::2] = knots
        new_array[1:-1:2] = midpoints

        return new_array



def set_basis_support_cells(functions, cells):
    basis_to_cell_map = {}
    cell_to_basis_map = defaultdict(set)
    for n, b in enumerate(functions):
        i = np.flatnonzero(
            np.all((b.support[:, :, 0] <= cells[:, :, 0]) & (b.support[:, :, 1] >= cells[:, :, 1]), axis=1))
        basis_to_cell_map[n] = set(i)

        for cell in i:
            cell_to_basis_map[cell].add(n)
    return basis_to_cell_map, cell_to_basis_map


def set_children_of_cells(fine_mesh, coarse_mesh):
    cell_to_children_map = {}
    fine_cells = fine_mesh.cells
    coarse_cells = coarse_mesh.cells

    print('fine', fine_cells, 'coarse', coarse_cells)
    for i, cell in enumerate(coarse_cells):
        children = set(np.flatnonzero(
            np.all((cell[:, 0] <= fine_cells[:, :, 0]) & (cell[:, 1] >= fine_cells[:, :, 1]), axis=1)))
        cell_to_children_map[i] = children

    return cell_to_children_map
