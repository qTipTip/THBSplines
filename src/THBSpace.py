import numpy as np

from src.Mesh import Mesh
from src.Space import Space


import matplotlib.pyplot as plt
import matplotlib.patches as plp

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

    def visualize_mesh(self):
        if self.dim != 2:
            raise ValueError('Can only visualize 2D-meshes')

        fig = plt.figure()
        axs = fig.gca()
        for level in range(self.depth):
            for i, cell in enumerate(self.mesh.cells[level]):
                if i in self.mesh.active_cell_indices[level]:
                    w, h = (cell[:, 1] - cell[:, 0])
                    rect = plp.Rectangle([cell[0, 0], cell[1, 0]], w, h, alpha=0.2, linewidth=2, fill=False, edgecolor='black')
                    axs.add_patch(rect)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.show()

    def refine_element(self, marked_elements):

        refinement_depth = len(marked_elements) - 1

        if refinement_depth >= self.depth:
            self.add_level()

        new_elements = []
        for l in range(refinement_depth):
            active_cells = set(self.mesh.active_cell_indices[l])
            nactive_cells = set(self.mesh.nonactive_cell_indices[l])
            marked_cells = set(marked_elements[l])

            active_cells = active_cells.difference(marked_cells)
            nactive_cells = nactive_cells.union(marked_cells)

            new_elements.append(list(set(np.unique([self.mesh.get_children_of_cell(l, c) for c in marked_cells]))))

            self.mesh.active_cell_indices[l] = active_cells
            self.mesh.nonactive_cell_indices[l] = nactive_cells
            self.mesh.active_cell_indices[l+1] = list(set(self.mesh.active_cell_indices[l+1]).union(new_elements))

        return new_elements

