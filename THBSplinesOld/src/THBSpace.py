import matplotlib.patches as plp
import matplotlib.pyplot as plt
import numpy as np

from THBSplinesOld.src.Mesh import Mesh
from THBSplinesOld.src.Space import Space


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

    def refine_hierarchical_mesh(self, marked_elements):

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

    def refine(self, marked_entities, type='cells'):

        # refine the mesh
        if type == 'cells':
            marked_cells = marked_entities
        elif type == 'functions':
            marked_cells = self.compute_cells_to_refine(marked_entities)
        else:
            raise ValueError('Refinement type not supported: {}'.format(type))
        new_cells = self.refine_hierarchical_mesh(marked_cells)
        # refine the functions
        if type == 'cells':
            marked_functions = self.functions_to_deactivate_from_cells(marked_entities)
        elif type == 'functions':
            marked_functions = self.functions_to_deactivate_from_neighbours(marked_entities)

        self.refine_hierarchical_space(marked_cells, new_cells)

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

    def compute_cells_to_refine(self, marked_functions):
        marked_cells = []
        for level in range(len(marked_functions)):
            support_cells = []
            for idx in marked_functions[level]:
                support_cells += list(self.space.functions[level].get_cells(idx))

            support_cells = set(support_cells)
            active_cells, _ = self.get_active_and_nonactive_cells_at_level(level)
            marked_cells.append(support_cells.intersection(active_cells))

        return marked_cells

    def functions_to_deactivate_from_cells(self, marked_entities):
        functions_to_deactivate = []
        for level in range(len(marked_entities)):
            marked_functions = []
            for cell_idx in marked_entities[level]:
                marked_functions += list(self.space.functions[level].get_basis_functions(cell_idx))
            marked_functions = set(marked_functions)
            active_functions, _ = self.get_active_and_nonactive_functions_at_level(level)
            marked_functions = marked_functions.intersection(active_functions)

            active_cells, _ = self.get_active_and_nonactive_cells_at_level(level)
            marked_functions_to_remove = []
            for marked_function in marked_functions:
                if set(self.space.functions[level].get_cells(marked_function)).intersection(active_cells) == set():
                    marked_functions_to_remove.append(marked_function)
            marked_functions = marked_functions.difference(marked_functions_to_remove)
            functions_to_deactivate.append(marked_functions)
        return functions_to_deactivate

    def get_active_and_nonactive_functions_at_level(self, level):

        active_functions = self.space.active_function_indices[level]
        nonactive_functions = self.space.deactivated_function_indices[level]

        return set(active_functions), set(nonactive_functions)

    def refine_hierarchical_space(self, marked_functions, new_cells):

        for level in range(len(marked_functions) - 1):
            active_functions, nonactive_functions = self.get_active_and_nonactive_cells_at_level(level)
            active_functions = active_functions.difference(marked_functions[level])
            nonactive_functions = nonactive_functions.union(marked_functions[level])

            self.set_active_and_nonactive_functions(active_functions, nonactive_functions, level)

            function_candidates = self.get_basis_functions_at_level(new_cells[level + 1], level)

            next_active_functions, _ = self.get_active_and_nonactive_functions_at_level(level + 1)
            function_candidates = function_candidates.difference(next_active_functions)

            candidates_to_remove = []
            next_active_cells, next_non_active_cells = self.get_active_and_nonactive_cells_at_level(level + 1)
            for candidate in function_candidates:
                Qb = self.space.functions[level].get_cells(candidate)
                for Q in Qb:
                    if Q not in next_active_cells.union(next_non_active_cells):
                        candidates_to_remove.append(candidate)
                    break
            candidates = function_candidates.difference(candidates_to_remove)
            next_active_functions = next_active_functions.union(candidates)

            self.set_active_and_nonactive_functions(next_active_functions, nonactive_functions=None, level=level + 1)

    def set_active_and_nonactive_functions(self, active_functions=None, nonactive_functions=None, level=None):
        if active_functions:
            self.space.active_function_indices[level] = active_functions
        if nonactive_functions:
            self.space.deactivated_function_indices[level] = nonactive_functions

    def get_basis_functions_at_level(self, marked_elements, level):
        basis = []
        for cell in marked_elements:
            basis += list(self.space.functions[level].get_basis_functions(cell))
        return set(basis)

    def plot_hierarchical_basis(self):
        if self.dim != 1:
            raise ValueError('Plot only for 1D atm')

        x = np.linspace(0, 3, 100)

        for b in self.get_hierarchical_basis():
            y = [b(X) for X in x]
            plt.plot(x, y)
        plt.show()

    def get_hierarchical_basis(self):

        b = []

        for level in range(self.depth):
            active_functions, _ = self.get_active_and_nonactive_functions_at_level(level)
            for i in range(len(self.space.functions[level].basis)):
                if i in active_functions:
                    b.append(self.space.functions[level].basis[i])
        return b

    def functions_to_deactivate_from_neighbours(self, marked_entities):
        functions_to_deactivate = []
        for level in range(self.depth):
            marked_functions = self.space.functions[level].get_neighbors(marked_entities[level])
            active_functions, _ = self.get_active_and_nonactive_functions_at_level(level)
            marked_functions = (marked_functions.intersection(active_functions)).difference(marked_entities)

            active_cells, _ = self.get_active_and_nonactive_cells_at_level(level)
            functions_to_remove = []
            for func_idx in marked_functions:
                if self.space.functions[level].get_cells(func_idx).intersection(active_cells) != set():
                    functions_to_remove.append(func_idx)
            marked_functions = marked_functions.difference(functions_to_remove)
            marked_functions = marked_functions.union(marked_entities)
            functions_to_deactivate.append(marked_functions)
        return functions_to_deactivate