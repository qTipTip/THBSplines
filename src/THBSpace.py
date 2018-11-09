import matplotlib.patches as plp
import matplotlib.pyplot as plt

from src.Mesh import Mesh
from src.Space import Space


class THBSpace(object):

    def __init__(self, degrees, knots, dim):
        self.mesh = Mesh(knots, degrees, dim)
        self.space = Space(degrees, knots, dim)
        self.depth = 1
        self.dim = dim

    def add_level(self):
        self.mesh.add_level()
        self.space.add_level()
        self.depth += 1

    def visualize_hierarchical_mesh(self):
        if self.dim != 2:
            raise ValueError('Can only visualize 2D-meshes')

        fig = plt.figure()
        axs = fig.gca()
        for level in range(self.depth):
            for i, cell in enumerate(self.mesh.cells[level]):
                if i in self.mesh.active_cell_indices[level]:
                    w, h = (cell[:, 1] - cell[:, 0])
                    rect = plp.Rectangle((cell[0, 0], cell[1, 0]), w, h, alpha=0.2, linewidth=2, fill=False,
                                         edgecolor='black')
                    axs.add_patch(rect)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.show()

    def visualize_level_mesh(self, level):
        fig = plt.figure()
        axs = fig.gca()

        for i, cell in enumerate(self.mesh.cells[level]):
            w, h = (cell[:, 1] - cell[:, 0])
            rect = plp.Rectangle((cell[0, 0], cell[1, 0]), w, h, alpha=0.2, linewidth=2, fill=False,
                                 edgecolor='black')
            axs.add_patch(rect)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.show()

    def refine_element(self, marked_elements):

        refinement_depth = len(marked_elements)

        print(refinement_depth, self.depth)
        while refinement_depth >= self.depth:
            self.add_level()
        print(refinement_depth, self.depth)


        new_elements = []
        for l in range(refinement_depth):
            marked_cells = set(marked_elements[l])
            active_cells, non_active_cells = self.get_active_and_nonactive_cells_at_level(l)
            active_cells = active_cells.difference(marked_cells)
            non_active_cells = non_active_cells.union(marked_cells)

            self.set_active_and_nonactive_cells(active_cells, non_active_cells, level=l)

            next_level_cells = self.get_children_of_cells(marked_cells, l)
            next_level_active_cells, _ = self.get_active_and_nonactive_cells_at_level(l + 1)
            next_level_active_cells = next_level_active_cells.union(next_level_cells)

            self.set_active_and_nonactive_cells(next_level_active_cells, level=l + 1)

            new_elements.append(next_level_cells)

        return new_elements

    def get_active_and_nonactive_cells_at_level(self, level):
        active_cells = self.mesh.active_cell_indices[level]
        non_active_cells = self.mesh.nonactive_cell_indices[level]

        return set(active_cells), set(non_active_cells)

    def get_children_of_cells(self, marked_cells, level):
        children = []
        for cell_index in marked_cells:
            children += list(self.mesh.get_children_of_cell(level, cell_index))
        return children

    def set_active_and_nonactive_cells(self, active_cells=None, non_active_cells=None, level=None):

        if active_cells:
            self.mesh.active_cell_indices[level] = active_cells
        if non_active_cells:
            self.mesh.nonactive_cell_indices[level] = non_active_cells
