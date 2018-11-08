import itertools
from typing import List

import numpy as np

from src.Space import insert_midpoints


class Mesh(object):
    MESH_REFINEMENT_SCHEME = "dyadic"

    def __init__(self, initial_knots, initial_degrees, dim):
        self.active_cell_indices = []  # each element is a list of indices corresponding to all the active cells on level
        self.deactive_cell_indices = []  # each element is a list of indices corresponding to all the deactivated cells on level
        self.knot_vectors = [initial_knots]  # each element is a list of all the knot vectors on the corresponding level
        self.regions = []
        self.depth = 0
        self.cell_to_children_map = {}  # for each level is a dictionary mapping cell_index to the cell_indices of their children.
        self.dim = dim
        self.degrees = initial_degrees
        self.cells = self.generate_cells(initial_knots)

    def get_children_of_cell(self, level: int, cell_index: int) -> List[int]:
        """
        Given the cell_index of a cell on given level, return a list of indices corresponding to the
        childrne of Q_cell_index on the next level.
        :param level:
        :param cell_index:
        :return:
        """

        assert level <= self.depth - 1

        pass

    def get_parent_of_cell(self, level: int, cell_index: int) -> int:
        """
        Given the cell_index of a cell on a given leve, return the index corresponding to the cell on the previous level
        with this cell as a child.
        :param level:
        :param cell_index:
        :return:
        """

        assert 1 <= level

        for cell in range(len(self.cells[level - 1])):
            if cell_index in self.get_children_of_cell(level - 1, cell):
                return cell

        return -1

    def add_level(self) -> None:
        """
        Adds another level to the hierarchical mesh.
        """

        old_knots = self.knot_vectors[-1]
        new_knots = []
        for degree, knot_vector in zip(self.degrees, old_knots):
            new_knots.append(insert_midpoints(knot_vector, degree))

        # update_cell_to_children_map
        # self.depth = depth + 1
        # update region
        # update active and passive cells.
        pass

    def generate_cells(self, knots):
        cells = []

        unique_knots = [np.unique(knot) for knot in knots]
        idx_start = [
            [j for j in range(len(unique_knots[i]) - 1)] for i in range(self.dim)
        ]
        idx_stop = [
            [j + 1 for j in idx_start[i]] for i in range(self.dim)
        ]

        idx_start_perm = list(itertools.product(*idx_start))
        idx_stop_perm = list(itertools.product(*idx_stop))
        n = len(idx_start_perm)

        for i in range(n):
            new_cells = []
            for j in range(self.dim):
                new_cells.append(unique_knots[j][idx_start_perm[i][j]: idx_stop_perm[i][j] + 1])
            cells.append(np.array(new_cells))

        return cells
