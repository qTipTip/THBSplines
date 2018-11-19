from functools import reduce
from typing import Union, List

import numpy as np

from abstract_space import Space
from hierarchical_mesh import HierarchicalMesh
from tensor_product_space import TensorProductSpace


class HierarchicalSpace(Space):

    def cell_to_basis(self, cell_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def basis_to_cell(self, basis_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def __init__(self, knots, degrees, dim):
        """
        Initialize a hierarchical space with a base mesh and space over the given knot vectors
        :param knots:
        :param degrees:
        :param dim:
        """
        self.nlevels = 1
        self.spaces = [TensorProductSpace(knots, degrees, dim)]
        self.mesh = HierarchicalMesh(knots, dim)
        self.afunc_level = {0: np.array((range(self.spaces[0].nfuncs)), dtype=np.int)}  # active functions on level
        self.dfunc_level = {0: np.array([], dtype=np.int)}  # deactivated functions on level
        self.nfuncs_level = {0: self.spaces[0].nfuncs}
        self.nfuncs = self.nfuncs_level[0]
        self.projections = []
        self.afunc = np.array([], np.int)
        self.dfunc = np.array([], np.int)

    def refine(self, marked_entities: dict, new_cells: dict) -> np.ndarray:
        """
        Refine the hierarchical space, and return the projection matrix to pass from coarse space to refined space.
        :param marked_entities: marked functions/cells to refine
        :param new_cells: global indices of new active cells for each level, as returned by ``HierarchicalMesh.refine``.
        :return: np.ndarray
        """

        if len(self.spaces) < self.mesh.nlevels:
            self.add_level()
            marked_entities[self.mesh.nlevels] = np.array([], dtype=np.int)

        self.update_active_functions(marked_entities, new_cells)

    def add_level(self):
        if len(self.spaces) == self.mesh.nlevels - 1:
            refined_space, projector = self.spaces[self.mesh.nlevels - 2].refine()
            self.spaces.append(refined_space)
            self.projections.append(projector)
            self.nlevels += 1
            self.afunc_level[self.mesh.nlevels - 1] = np.array([], dtype=np.int)
            self.dfunc_level[self.mesh.nlevels - 1] = np.array([], dtype=np.int)
            self.nfuncs_level[self.mesh.nlevels - 1] = 0
        else:
            raise ValueError('Non-compatible mesh and space levels')

    def update_active_functions(self, marked_entities: dict, new_cells: dict):
        """
        Updates the set of active and deactivated functions.
        :param marked_entities: indices of active functions to deactivate
        :param new_cells: cells added to the new mesh, as returned by ``HierarchicalMesh.refine``.
        """

        afunc = self.afunc_level
        dfunc = self.dfunc_level

        for level in range(self.nlevels - 1):
            afunc[level] = np.setdiff1d(afunc[level], marked_entities[level])
            dfunc[level] = np.union1d(marked_entities[level], dfunc[level])

            children = self.get_children(level, marked_entities[level])

            active_and_deactive = np.union1d(afunc[level + 1], dfunc[level + 1])
            next_level_afunc = np.setdiff1d(children, active_and_deactive)
            afunc[level + 1] = np.union1d(afunc[level + 1], next_level_afunc)

            new_possible_afunc = self.spaces[level + 1].get_basis_functions(new_cells[level + 1])
            new_possible_afunc = np.setdiff1d(new_possible_afunc, afunc[level + 1])
            _, new_possible_cells = self.spaces[level + 1].get_cells(new_possible_afunc)
            aelem_and_delem = np.union1d(self.mesh.aelem_level[level + 1], self.mesh.delem_level[level + 1])

            new_functions = np.array([
                i for i in new_possible_cells if np.all(np.isin(new_possible_cells[i], aelem_and_delem))
            ], dtype=np.int)
            afunc[level + 1] = np.union1d(afunc[level + 1], new_functions)

        # TODO: Not sure if the two following lines are needed
        self.afunc = reduce(np.union1d, *afunc.values())
        self.dfunc = reduce(np.union1d, *dfunc.values())

        self.nfuncs_level = [len(self.afunc_level[level]) if level in self.afunc_level else 0 for level in
                             range(self.nlevels)]
        self.nfuncs = sum(self.nfuncs_level)

    def get_children(self, level, marked_functions_at_level):
        children = np.array([], dtype=np.int)
        projection = self.projections[level]
        for func_idx in marked_functions_at_level:
            c = np.flatnonzero(projection[:, func_idx])
            children = np.union1d(children, c)
        return children



if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3, 4]}
    new_cells = T.mesh.refine(marked_cells)
    T.mesh.plot_cells()
    marked_funcs = {0:[1, 2, 3, 4]}
    T.refine(marked_funcs, new_cells)
    T.mesh.plot_cells()
