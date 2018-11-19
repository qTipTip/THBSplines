import numpy as np

from abstract_mesh import Mesh
from cartesian_mesh import CartesianMesh


class HierarchicalMesh(Mesh):
    def plot_cells(self) -> None:
        import matplotlib.pyplot as plt

        for level in range(self.nlevels):
            active_cells = self.meshes[level].cells[self.aelem_level[level]]
            for cell in active_cells:
                x = cell[0, [0, 1, 1, 0, 0]]
                y = cell[1, [0, 0, 1, 1, 0]]
                plt.plot(x, y, color='black')
        plt.show()

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, knots, dim):
        self.meshes = [CartesianMesh(knots, dim)]
        self.nlevels = 1
        self.aelem_level = {0: np.array((range(self.meshes[0].nelems)), dtype=np.int)}  # active elements on level
        self.delem_level = {0: np.array([], dtype=np.int)}  # deactivated elements on level

    def add_level(self):
        """
        Adds a new level to the hierarchical mesh, by refining the finest mesh.
        """
        self.nlevels += 1
        self.meshes.append(self.meshes[-1].refine())

        self.aelem_level[self.nlevels - 1] = (np.array([], dtype=np.int))
        self.delem_level[self.nlevels - 1] = (np.array([], dtype=np.int))

    def refine(self, marked_cells: dict):
        """
        Refines the hierarchical mesh, and returns the global
        element indices of active elements for each level.
        :return:
        """

        while self.nlevels - 1 <= max(marked_cells.keys()):
            self.add_level()

        # old_active_cells = self.aelem_level
        new_elements = self.update_active_cells(marked_cells)

        return new_elements

    def update_active_cells(self, marked_cells):
        """
        Updates the set of active cells and deactivated cells.
        :param marked_cells:
        :return:
        """

        number_of_levels = len(marked_cells)
        new_cells = {}

        for level in range(number_of_levels):
            if level in marked_cells:
                marked_active_elements = np.where(np.in1d(marked_cells[level], self.aelem_level[level]))
                self.aelem_level[level] = np.setdiff1d(self.aelem_level[level], marked_active_elements).astype(np.int)
                self.delem_level[level] = np.union1d(self.delem_level[level], marked_cells[level]).astype(np.int)

                new_cells[level + 1] = self.get_children(level, marked_cells[level])
                self.aelem_level[level + 1] = np.union1d(self.aelem_level[level + 1], new_cells[level + 1]).astype(
                    np.int)

        self.nel_per_level = [len(self.aelem_level[level]) if level in self.aelem_level else 0 for level in
                              range(self.nlevels)]
        self.nel = sum(self.nel_per_level)

        return new_cells

    def get_children(self, level: int, marked_cells_at_level) -> np.ndarray:
        children = np.array([])
        fine_cells = self.meshes[level + 1].cells
        for cell_idx in marked_cells_at_level:
            cell = self.meshes[level].cells[cell_idx]
            i = np.flatnonzero(np.all((cell[:, 0] <= fine_cells[:, :, 0]) & (cell[:, 1] >= fine_cells[:, :, 1]),
                                      axis=1))
            children = np.union1d(children, i)
        return children


if __name__ == '__main__':
    knots = [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ]
    dim = 2
    M = HierarchicalMesh(knots, dim)
    marked_cells = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3, 4, 5, 6],
                    2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
    M.refine(marked_cells)
    M.plot_cells()
