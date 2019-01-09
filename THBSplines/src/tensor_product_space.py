from functools import lru_cache
from typing import Union, List, Tuple

import numpy as np
import scipy.sparse as sp
from THBSplines.lib.BSpline import TensorProductBSpline
from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import augment_knots, find_knot_index
from THBSplines.src.cartesian_mesh import CartesianMesh
from memory_profiler import profile


class TensorProductSpace(Space):

    def __init__(self, knots, degrees, dim):
        self.knots = np.array(knots, dtype=np.float64)
        self.degrees = np.array(degrees, dtype=np.intc)
        self.dim = dim
        self.mesh = CartesianMesh(knots, dim)
        self.basis_supports = None  # a list of supports
        self.basis = None

        self.construct_basis()
        self.nfuncs = len(self.basis)
        self.nfuncs_onedim = [len(k) - d - 1 for k, d in zip(self.knots, self.degrees)]
        self.cell_area = self.mesh.cell_area

    def cell_to_basis(self, cell_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        basis_idx = []
        for cell_idx in cell_indices:
            cell = self.mesh.cells[cell_idx]
            i = np.flatnonzero(
                np.all((self.basis_supports[:, :, 0] <= cell[:, 0]) & (self.basis_supports[:, :, 1] >= cell[:, 1]),
                       axis=1))
            basis_idx.append(i)
        return np.array(basis_idx)

    def basis_to_cell(self, basis_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        cell_indices = []
        for basis_idx in basis_indices:
            basis_supp = self.basis_supports[basis_idx]
            i = np.flatnonzero(
                np.all(
                    (basis_supp[:, 0] <= self.mesh.cells[:, :, 0]) & (basis_supp[:, 1] >= self.mesh.cells[:, :, 1]),
                    axis=1))
            cell_indices.append(i)
        return np.array(cell_indices)

    def construct_basis(self):
        degrees = self.degrees
        dim = self.dim
        idx_start = np.array([
            list(range(len(self.knots[j]) - self.degrees[j] - 1)) for j in range(dim)
        ])
        idx_stop = np.array([
            [j + self.degrees[i] + 2 for j in idx_start[i]] for i in range(dim)
        ])

        idx_start_perm = np.stack(np.meshgrid(*idx_start), -1).reshape(-1, self.dim)
        idx_stop_perm = np.stack(np.meshgrid(*idx_stop), -1).reshape(-1, self.dim)

        n = len(idx_start_perm)

        b_splines = []
        b_splines_end_evals = []
        b_support = np.zeros((n, self.dim, 2))

        for i in range(n):
            new_knots = []
            end_evals = []
            for j in range(dim):
                new_knots.append(self.knots[j][idx_start_perm[i, j]: idx_stop_perm[i, j]])
                end_evals.append(idx_stop_perm[i, j] == len(self.knots[j]))
            new_knots = np.array(new_knots, dtype=np.float64)
            end_evals = np.array(end_evals, dtype=np.intc).ravel()
            # new_b_spline = TensorProductBSpline(degrees, new_knots, end_evals)
            b_splines.append(new_knots)
            b_support[i] = [[new_knots[j][0], new_knots[j][-1]] for j in range(dim)]
            b_splines_end_evals.append(end_evals)
        self.basis = np.array(b_splines)
        self.basis_supports = b_support
        self.basis_end_evals = b_splines_end_evals
        self.nfuncs = len(self.basis)

    @profile
    def refine(self) -> Tuple["TensorProductSpace", np.ndarray, List]:
        """
        Refine the space by dyadically inserting midpoints in the knot vectors, and computing the knot-insertion
        matrix (the projection matrix form coarse to fine space).
        :return:
        """

        coarse_knots = self.knots
        fine_knots = [insert_midpoints(knot_vector, degree) for knot_vector, degree in zip(self.knots, self.degrees)]

        projection, projection_onedim = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace(fine_knots, self.degrees, self.dim)

        return fine_space, projection, projection_onedim

    @staticmethod
    def compute_projection_matrix(coarse_knots, fine_knots, degrees):
        matrices = []
        for fine, coarse, degree in zip(fine_knots, coarse_knots, degrees):
            coarse = augment_knots(coarse, degree)
            fine = augment_knots(fine, degree)
            m = len(fine) - (degree + 1)
            n = len(coarse) - (degree + 1)

            a = sp.lil_matrix((m, n), dtype=np.float64)
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
            a = sp.kron(a, matrix, format='lil')
        return a, matrices

    def get_basis_functions(self, cell_list: np.ndarray) -> np.ndarray:
        """
        Returns the indices of basis functions supported over the given list of cells.
        :param cell_list: Numpy array containing the indices of cells.
        :return: numpy array containing the indices of basis functions.
        """

        basis = np.array([], dtype=np.int)

        for cell_idx in cell_list:
            cell = self.mesh.cells[cell_idx]
            condition = (self.basis_supports[:, :, 0] <= cell[:, 0]) & (self.basis_supports[:, :, 1] >= cell[:, 1])
            i = np.flatnonzero(np.all(condition, axis=1))
            basis = np.union1d(basis, i)
        return basis

    def get_cells(self, basis_function_list: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Given a list of indices corresponding to basis functions, return the union of the support-cells,
        and a dictionary mapping basis_function to cell index.
        :param basis_function_list:
        :return:
        """

        cells = np.array([], dtype=np.int)
        cells_map = {}

        eps = np.spacing(1)
        for func in basis_function_list:
            supp = self.basis_supports[func]
            condition = (self.mesh.cells[:, :, 0] + eps >= supp[:, 0]) & (self.mesh.cells[:, :, 1] <= supp[:, 1] + eps)
            i = np.flatnonzero(np.all(condition, axis=1))
            cells = np.union1d(cells, i)
            cells_map[func] = i
        return cells, cells_map

    def construct_function(self, coefficients):

        assert len(coefficients) == len(self.basis)

        def f(x):
            return sum([c * self.construct_B_spline(i)(x) for i, c in enumerate(coefficients)])

        return f

    def get_functions_on_rectangle(self, rectangle):
        """
        Returns the indices of the supports whose support contains the rectangle.
        :param rectangle:
        :return:
        """
        condition = (self.basis_supports[:, :, 0] <= rectangle[:, 0]) & (
                self.basis_supports[:, :, 1] >= rectangle[:, 1])
        i = np.flatnonzero(np.all(condition, axis=1))
        return i

    @lru_cache()
    def construct_B_spline(self, i):
        """
        Return a Callable TensorProductBSpline
        :param i:
        :return:
        """

        return TensorProductBSpline(self.degrees, self.basis[i], self.basis_end_evals[i])


class TensorProductSpace2D(TensorProductSpace):


    def construct_basis(self):

        knots_u = self.knots[0]
        knots_v = self.knots[1]
        deg_u = self.degrees[0]
        deg_v = self.degrees[1]

        lenu = len(knots_u)
        n = lenu - deg_u - 1
        lenv = len(knots_v)
        m = lenv - deg_v - 1

        b_splines_end_evals = np.zeros((n * m, 2), dtype=np.intc)
        b_support = np.zeros((n * m, self.dim, 2))

        index = 0
        for j in range(m):
            for i in range(n):
                b_support[index] = [[knots_u[i], knots_u[i + deg_u + 1]], [knots_v[j], knots_v[j + deg_v + 1]]]
                offset_u = (i + deg_u + 2)
                offset_v = (j + deg_v + 2)
                b_splines_end_evals[index] = [lenu == offset_u, lenv == offset_v]

                index += 1
        self.basis_supports = b_support
        self.basis_end_evals = b_splines_end_evals
        self.nfuncs = n * m
        self.dim_u = n
        self.dim_v = m
        self.nfuncs_onedim = [n, m]
        self.basis = [0]*(n * m)

    @profile
    def refine(self) -> Tuple["TensorProductSpace2D", np.ndarray]:
        """
        Refine the space by dyadically inserting midpoints in the knot vectors, and computing the knot-insertion
        matrix (the projection matrix form coarse to fine space).
        :return:
        """

        coarse_knots = self.knots
        fine_knots = [insert_midpoints(knot_vector, degree) for knot_vector, degree in zip(self.knots, self.degrees)]

        projection_onedim = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace2D(fine_knots, self.degrees, self.dim)

        return fine_space, projection_onedim

    @lru_cache()
    def construct_B_spline(self, i):
        """
        Return a Callable TensorProductBSpline
        :param i:
        :return:
        """

        ind_v = i // self.dim_u
        ind_u = i % self.dim_u

        knots = np.array([self.knots[0][ind_u : ind_u + self.degrees[0] + 2],
            self.knots[1][ind_v : ind_v + self.degrees[1] + 2]], dtype=np.float64)
        return TensorProductBSpline(self.degrees, knots, self.basis_end_evals[i])


def insert_midpoints(knots, p):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot vector.
    :param s:
    :param knots: p + 1 regular knot vector to be refined
    :param p: spline degree
    :return: refined_knots
    """

    knots = np.array(knots, dtype=np.float64)
    unique_knots = np.unique(knots)
    midpoints = (unique_knots[:-1] + unique_knots[1:]) / 2

    return np.sort(np.concatenate((knots, midpoints)))



if __name__ == '__main__':
    knots = [
        [0, 0, 0, 1, 2, 3, 4, 5, 5, 5],
        [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    ]
    d = [2, 2]
    dim = 2

    T = TensorProductSpace2D(knots, d, dim)

    for i in range(T.nfuncs):
        T.construct_B_spline(i)

    assert T.nfuncs == 49