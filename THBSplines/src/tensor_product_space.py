from typing import Union, List, Tuple

import numpy as np

from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import BSpline, augment_knots, find_knot_index
from THBSplines.src.cartesian_mesh import CartesianMesh


class TensorProductSpace(Space):

    def __init__(self, knots, degrees, dim):
        self.knots = np.array(knots)
        self.degrees = np.array(degrees)
        self.dim = dim
        self.mesh = CartesianMesh(knots, dim)
        self.basis_supports = None  # a list of supports
        self.basis = None

        self.construct_basis()
        self.nfuncs = len(self.basis)

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
        b_support = np.zeros((n, self.dim, 2))
        for i in range(n):
            new_knots = []
            for j in range(dim):
                new_knots.append(self.knots[j][idx_start_perm[i, j]: idx_stop_perm[i, j]])
            new_b_spline = BSpline(degrees, new_knots)
            new_b_spline.tensor_product_indices = idx_start_perm[i]
            b_splines.append(new_b_spline)
            b_support[i] = [[new_knots[j][0], new_knots[j][-1]] for j in range(dim)]
        self.basis = np.array(b_splines)
        self.basis_supports = b_support

    def refine(self) -> Tuple["TensorProductSpace", np.ndarray]:
        """
        Refine the space by dyadically inserting midpoints in the knot vectors, and computing the knot-insertion
        matrix (the projection matrix form coarse to fine space).
        :return:
        """

        coarse_knots = self.knots
        fine_knots = [insert_midpoints(knot_vector, degree) for knot_vector, degree in zip(self.knots, self.degrees)]

        projection = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace(fine_knots, self.degrees, self.dim)

        return fine_space, projection

    @staticmethod
    def compute_projection_matrix(coarse_knots, fine_knots, degrees):
        matrices = []
        for fine, coarse, degree in zip(fine_knots, coarse_knots, degrees):
            coarse = augment_knots(coarse, degree)
            fine = augment_knots(fine, degree)
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

        for func in basis_function_list:
            supp = self.basis_supports[func]
            condition = (self.mesh.cells[:, :, 0] <= supp[:, 0]) & (self.mesh.cells[:, :, 1] >= supp[:, 1])
            i = np.flatnonzero(np.all(condition, axis=1))
            cells = np.union1d(cells, i)
            cells_map[func] = i
        return cells, cells_map


def insert_midpoints(knots, p):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot vector.
    :param s:
    :param knots: p + 1 regular knot vector to be refined
    :param p: spline degree
    :return: refined_knots
    """

    knots = np.array(knots, dtype=np.float64)
    midpoints = (knots[p:-p - 1] + knots[p + 1:-p]) / 2
    new_array = np.zeros(len(knots) + len(midpoints), dtype=np.float64)

    new_array[:p + 1] = knots[:p + 1]
    new_array[-p - 1:] = knots[-p - 1]
    new_array[p + 1:p + 2 * len(midpoints):2] = midpoints
    new_array[p + 2:p + 2 * len(midpoints) - 1:2] = knots[p + 1:-p - 1]
    return new_array


if __name__ == '__main__':
    knots = [
        [0, 1, 2, 3, 4, 5, 5],
        [0, 0, 1, 2, 3, 4, 5, 5]
    ]
    d = [1, 1]
    dim = 2

    T = TensorProductSpace(knots, d, dim)

    c = [0, 1, 2]
    print(T.cell_to_basis(c))
    print(T.basis_to_cell(c))
