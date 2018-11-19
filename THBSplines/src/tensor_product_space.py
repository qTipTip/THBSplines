from typing import Union, List

import numpy as np
from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import BSpline
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
