from typing import List


class Mesh(object):
    MESH_REFINEMENT_SCHEME = "dyadic"

    def __init__(self):
        self.cells = []  # each element is a list of all the cells on the corresponding level
        self.active_cell_indices = []  # each element is a list of indices corresponding to all the active cells on level
        self.deactive_cell_indices = []  # each element is a list of indices corresponding to all the deactivated cells on level
        self.knot_vectors = []  # each element is a list of all the knot vectors on the corresponding level
        self.regions = []
        self.depth = 0
        self.cell_to_children_map = {}  # for each level is a dictionary mapping cell_index to the cell_indices of their children.

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

        # insert_midpoints(knots)
        # update_cell_to_children_map
        # self.depth = depth + 1
        # update region
        # update active and passive cells.
        pass
