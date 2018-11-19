import numpy as np

from THBSplines.src.abstract_mesh import Mesh


class CartesianMesh(Mesh):

    def plot_cells(self) -> None:
        pass

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, knots, parametric_dimension):
        """
        Represents a regular cartesian mesh in ``parametric_dimension`` dimensions.
        :param knots:
        :param parametric_dimension:
        """
        self.knots = np.array([np.unique(knot_v) for knot_v in knots])
        self.dim = parametric_dimension
        self.cells = self.compute_cells()

    def compute_cells(self) -> np.ndarray:
        """
        Computes an array of cells, represented as AABBs with each cell as [[min1, max1], [min2, max2], ..., ]
        :return: a list of N cells of shape (N, dim, 2).
        """
        cells_bottom_left = np.stack(np.meshgrid(*self.knots[:, :-1]), -1).reshape(-1, self.dim)
        cells_top_right = np.stack(np.meshgrid(*self.knots[:, 1:]), -1).reshape(-1, self.dim)
        cells = np.concatenate((cells_bottom_left, cells_top_right), axis=1).reshape(-1, self.dim, 2)
        return cells

    def refine(self) -> 'CartesianMesh':
        """
        Dyadic refinement of the mesh, by inserting midpoints in each knot vector.
        :return: a refined CartesianMesh object.
        """
        refined_knots = np.array([
            np.sort(np.concatenate((knot_v, (knot_v[1:] + knot_v[:-1]) / 2))) for knot_v in self.knots
        ])
        return CartesianMesh(refined_knots, self.dim)


if __name__ == '__main__':
    knots = [
        [0, 1, 2],
        [0, 1, 2]
    ]
    C = CartesianMesh(knots, 2)
    C1 = C.refine().refine().refine()
