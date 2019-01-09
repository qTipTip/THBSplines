from functools import reduce
from typing import Union, List

import numpy as np
import scipy.sparse as sp
from THBSplines.src.abstract_space import Space
from THBSplines.src.hierarchical_mesh import HierarchicalMesh
from THBSplines.src.tensor_product_space import TensorProductSpace


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
        self.projections_onedim = []
        self.afunc = np.array([], np.int)
        self.dfunc = np.array([], np.int)
        self.truncated = True
        self.degrees = degrees
        self.dim = dim

    def refine(self, marked_functions: dict, new_cells: dict) -> np.ndarray:
        """
        Refine the hierarchical space, and return the projection matrix to pass from coarse space to refined space.
        :param marked_functions: marked functions/cells to refine
        :param new_cells: global indices of new active cells for each level, as returned by ``HierarchicalMesh.refine``.
        :return: np.ndarray
        """

        if len(self.spaces) < self.mesh.nlevels:
            self.add_level()
            marked_functions[self.mesh.nlevels] = np.array([], dtype=np.int)

        self.update_active_functions(marked_functions, new_cells)

    def add_level(self):
        if len(self.spaces) == self.mesh.nlevels - 1:
            refined_space, projector, projector_onedim = self.spaces[self.mesh.nlevels - 2].refine()

            self.spaces.append(refined_space)
            self.projections.append(projector)
            self.projections_onedim.append(projector_onedim)
            self.nlevels += 1
            self.afunc_level[self.mesh.nlevels - 1] = np.array([], dtype=np.int)
            self.dfunc_level[self.mesh.nlevels - 1] = np.array([], dtype=np.int)
            self.nfuncs_level[self.mesh.nlevels - 1] = 0
        else:
            raise ValueError('Non-compatible mesh and space levels')

    def get_basis_conversion_matrix(self, level, coarse_indices=None):
        """
        :param level:
        :param coarse_indices: indices of the coarse active functions.
        :return:
        """
        if coarse_indices is None:
            c = self.projections[level]
        else:
            print('Space dimensions', self.nfuncs_level[level], self.spaces[level].nfuncs,
                  self.spaces[level].nfuncs_onedim)
            prod = np.prod(self.spaces[level + 1].degrees + 2) * len(coarse_indices)  # allocate space for data
            rows = np.zeros(prod)

            columns = np.zeros_like(rows)
            values = np.zeros_like(rows)

            ncounter = 0
            sub_coarse = np.unravel_index(coarse_indices, self.spaces[level].nfuncs_onedim)
            for i in range(len(coarse_indices)):
                C = 1
                for dim in range(self.dim):
                    C = sp.kron(self.projections_onedim[level][dim][:, sub_coarse[dim][i]], C)
                ir, ic, iv = sp.find(C)
                rows[ncounter: len(ir) + ncounter] = ir
                columns[ncounter: len(ir) + ncounter] = i
                values[ncounter: len(ir) + ncounter] = iv

                ncounter += len(ir)

            rows = rows[:ncounter]
            columns = columns[:ncounter]
            values = values[:ncounter]
            c = sp.coo_matrix((values, (rows, columns)),
                              shape=(self.spaces[level + 1].nfuncs, self.spaces[level].nfuncs)).tolil()

        if self.truncated:
            i = np.union1d(self.afunc_level[level + 1], self.dfunc_level[level + 1])
            c[i, :] = 0
        return c

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
        self.afunc = reduce(np.union1d, afunc.values())
        self.dfunc = reduce(np.union1d, dfunc.values())

        # TODO: Make sure this monkeypatch is not needed! In some cases, functions that should be deactivated are
        # reactivated by this algorithm.
        for key in dfunc:
            afunc[key] = np.setdiff1d(afunc[key], dfunc[key])

        self.nfuncs_level = {level: len(self.afunc_level[level]) if level in self.afunc_level else 0 for level in
                             range(self.nlevels)}
        self.nfuncs = sum(self.nfuncs_level.values())

    def get_children(self, level, marked_functions_at_level):
        children = np.array([], dtype=np.int)
        projection = sp.lil_matrix(self.projections[level])
        for func_idx in marked_functions_at_level:
            c = np.flatnonzero(projection[:, func_idx].toarray())
            children = np.union1d(children, c)
        return children

    def functions_to_deactivate_from_cells(self, marked_cells: dict):
        """
        Returns the indices of functions that have no active cells within their support.
        :param marked_cells: cell indices to check against
        :return: function indices
        """
        marked_functions = {}
        for level in range(self.nlevels):
            func_to_deact = self.spaces[level].get_basis_functions(marked_cells[level])
            func_to_deact = np.intersect1d(func_to_deact, self.afunc_level[level])

            func_to_keep = np.array([], dtype=np.int)
            _, func_cells_map = self.spaces[level].get_cells(func_to_deact)
            for f in func_to_deact:
                func_cells = func_cells_map[f]
                common_elements = np.intersect1d(func_cells, self.mesh.aelem_level[level])
                if common_elements.size != 0:
                    func_to_keep = np.append(func_to_keep, f)
            func_to_deact = np.setdiff1d(func_to_deact, func_to_keep)
            marked_functions[level] = func_to_deact
        return marked_functions

    def _get_all_cells(self):
        all_cells = []
        for i, mesh in enumerate(self.mesh.meshes):
            for cell in mesh.cells[self.mesh.aelem_level[i]]:
                all_cells.append(cell)
        return np.array(all_cells)

    def create_subdivision_matrix(self, mode='reduced') -> dict:
        """
        Returns hspace.nlevels-1 matrices used for representing coarse B-splines in terms of the finer B-splines.
        :param self: HierarchicalSpace containing the needed information
        :return: a dictionary mapping
        """

        mesh = self.mesh

        C = {}
        C[0] = sp.identity(self.spaces[0].nfuncs, format='lil')
        C[0] = C[0][:, self.afunc_level[0]]

        if mode == 'reduced':
            func_on_active_elements = self.spaces[0].get_basis_functions(mesh.aelem_level[0])
            func_on_deact_elements = self.spaces[0].get_basis_functions(mesh.delem_level[0])
            func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)
            for level in range(1, self.nlevels):
                I_row_idx = self.afunc_level[level]
                I_col_idx = list(range(self.nfuncs_level[level]))

                data = np.ones(len(I_col_idx))
                I = sp.coo_matrix((data, (I_row_idx, I_col_idx)),
                                  shape=(self.spaces[level].nfuncs, self.nfuncs_level[level]))
                aux = self.get_basis_conversion_matrix(level - 1, coarse_indices=func_on_deact_elements)
                func_on_active_elements = self.spaces[level].get_basis_functions(mesh.aelem_level[level])
                func_on_deact_elements = self.spaces[level].get_basis_functions(mesh.delem_level[level])
                func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)
                C[level] = sp.hstack([aux @ C[level - 1], I])
            return C
        else:
            for level in range(1, self.nlevels):
                I = sp.identity(self.spaces[level].nfuncs, format='lil')
                aux = self.get_basis_conversion_matrix(level - 1)
                C[level] = sp.hstack([aux @ C[level - 1], I[:, self.afunc_level[level]]])
            return C

    def _get_truncated_supports(self):
        """
        Returns the indices at the fine level that constitutes the supports of each truncated basis function.
        :return:
        """
        C = self.create_subdivision_matrix(mode='full')[self.nlevels - 1].toarray()
        trunc_basis_to_fine_idx = {}
        for i, trunc_coeff in enumerate(C.T):
            fine_active_basis = np.flatnonzero(trunc_coeff)
            basis_support_cells = self._get_fine_basis_support_cells(fine_active_basis)
            trunc_basis_to_fine_idx[i] = self.mesh.meshes[-1].cells[basis_support_cells]
        return trunc_basis_to_fine_idx

    def _get_fine_basis_support_cells(self, fine_active_basis):
        """
        Given a list of functions, returns the unique cells at the finest level in their common support.
        :param fine_active_basis:
        :return:
        """

        cells = np.array([], dtype=np.int)
        fine_cells = self.mesh.meshes[-1].cells
        for i in fine_active_basis:
            supp = self.spaces[-1].basis_supports[i]
            condition = np.all((supp[:, 0] <= fine_cells[:, :, 0]) & (supp[:, 1] >= fine_cells[:, :, 1]), axis=1)
            cells = np.union1d(cells, np.flatnonzero(condition))
        return cells

    def refine_in_rectangle(self, rectangle, level):
        """
        Returns the set active of indices marked for refinement contained in the given rectangle
        :param rectangle:
        :param level:
        :return:
        """
        eps = np.spacing(1)
        cells = self.mesh.meshes[level].cells
        cells_to_mark = np.all((rectangle[:, 0] <= cells[:, :, 0] + eps) & (eps + rectangle[:, 1] >= cells[:, :, 1]),
                               axis=1)
        return np.intersect1d(np.flatnonzero(cells_to_mark), self.mesh.aelem_level[level])

    def plot_overloading(self, filename=None, text=False, fontsize=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as plp
        fig = plt.figure()
        axs = fig.add_subplot(1, 1, 1)
        u_min = 0
        u_max = 1
        v_min = 0
        v_max = 1
        C = self.create_subdivision_matrix('full')
        max_degree = np.prod([d + 1 for d in self.degrees])
        for level in range(self.nlevels):
            indices = self.spaces[level].cell_to_basis(self.mesh.aelem_level[level])
            Csub = sp.lil_matrix(C[level])
            mesh = self.mesh.meshes[level]
            for cell, i_elem in zip(self.mesh.aelem_level[level], indices):
                _, col, _ = sp.find(Csub[i_elem, :])
                n = len(np.unique(col))
                if n > max_degree:
                    color = 'black'
                    fill = True
                    hatch = '//'
                    dotext = True
                else:
                    color = 'black'
                    hatch = None
                    fill = False
                    dotext = False

                e = mesh.cells[cell]
                w = e[0, 1] - e[0, 0]
                h = e[1, 1] - e[1, 0]
                mp1 = e[0, 0] + w / 2
                mp2 = e[1, 0] + h / 2

                axs.add_patch(plp.Rectangle((e[0, 0], e[1, 0]), w, h, fill=fill, color=color, alpha=0.2, hatch=None,
                                            linewidth=0.2))
                if text:
                    if not dotext:
                        continue
                    if fontsize is None:
                        fontsize = 50

                    axs.text(mp1, mp2, '{}'.format(n - max_degree), ha='center',
                             va='center', fontsize=fontsize * w)
                v_min = min(v_min, e[1, 0])
                v_max = max(v_max, e[1, 1])
                u_min = min(u_min, e[0, 0])
                u_max = max(u_max, e[0, 1])

        plt.xlim(u_min, u_max)
        plt.ylim(v_min, v_max)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_basis_weights(self, weights, filename=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as plp
        fig = plt.figure()
        axs = fig.add_subplot(1, 1, 1)
        u_min = 0
        u_max = 1
        v_min = 0
        v_max = 1
        C = self.create_subdivision_matrix('full')
        Csub_last = C[self.nlevels - 1].toarray()
        max_degree = np.prod([d + 1 for d in self.degrees])
        for level in range(self.nlevels):
            indices = self.spaces[level].cell_to_basis(self.mesh.aelem_level[level])
            print(indices)
            Csub = sp.lil_matrix(C[level])
            mesh = self.mesh.meshes[level]
            for cell, i_elem in zip(self.mesh.aelem_level[level], indices):
                _, col, _ = sp.find(Csub[i_elem, :])
                print(col)
                n = len(np.unique(col))
                print(Csub_last)
                if n > max_degree:
                    color = 'black'
                    fill = True
                    hatch = '//'
                    dotext = True
                else:
                    color = 'black'
                    hatch = None
                    fill = False
                    dotext = False

                e = mesh.cells[cell]
                w = e[0, 1] - e[0, 0]
                h = e[1, 1] - e[1, 0]

                axs.add_patch(plp.Rectangle((e[0, 0], e[1, 0]), w, h, fill=fill, color=color, alpha=0.2, hatch=None,
                                            linewidth=0.2))

                v_min = min(v_min, e[1, 0])
                v_max = max(v_max, e[1, 1])
                u_min = min(u_min, e[0, 0])
                u_max = max(u_max, e[0, 1])

        plt.xlim(u_min, u_max)
        plt.ylim(v_min, v_max)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3, 4]}
    T.mesh.plot_cells()
    T.mesh.plot_cells()
    weights = np.random.random(T.nfuncs)
    T.plot_basis_weights(weights)
