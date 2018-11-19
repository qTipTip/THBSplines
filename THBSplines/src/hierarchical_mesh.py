import numpy as np
from src.abstract_mesh import Mesh
from src.cartesian_mesh import CartesianMesh


class HierarchicalMesh(Mesh):
    def plot_cells(self) -> None:
        pass

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, knots, dim):
        self.meshes = [CartesianMesh(knots, dim)]
        self.nlevels = 1
        self.aelem_level = [np.array((range(self.meshes[0].nelems)))]
        self.delem_level = [np.array([])]
