import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.TensorProductSpace import TensorProductSpace


class HierarchicalSpace(object):

    def __init__(self, hierarchical_mesh: HierarchicalMesh, tensor_product_space: TensorProductSpace):
        self.truncated = True
        self.number_of_levels = 1
        self.number_of_functions = tensor_product_space.nfunctions
        self.number_of_active_functions_per_level = {0: tensor_product_space.nfunctions}
        self.active_functions_per_level = {0: set(range(tensor_product_space.nfunctions))}
        self.deactivated_functions_per_level = {0: set()}
        self.tensor_product_space_per_level = [tensor_product_space]
        self.parametric_dim = tensor_product_space.parametric_dim
        self.physical_dim = tensor_product_space.physical_dim
        self.mesh = hierarchical_mesh
        self.projectors = {}

    def add_new_level(self):
        if self.number_of_levels == self.mesh.number_of_levels - 1:
            new_space, projection = self.tensor_product_space_per_level[self.mesh.number_of_levels - 2].refine()

            self.tensor_product_space_per_level.append(new_space)
            self.projectors[self.mesh.number_of_levels - 2] = projection
            self.number_of_levels = self.number_of_levels + 1
            self.active_functions_per_level[self.number_of_levels - 1] = set()
            self.deactivated_functions_per_level[self.number_of_levels - 1] = set()
            self.number_of_active_functions_per_level[self.number_of_levels - 1] = 0

    def refine(self, marked_entities, type='cells'):

        if type == 'cells':
            marked_cells = marked_entities
        elif type == 'functions':
            marked_cells = self.compute_cells_to_refine(marked_entities)

        NE = self.refine_hierarchical_mesh(marked_cells)

        if type == 'cells':
            marked_functions = self.functions_to_deactivate_from_cells(marked_entities)
        elif type == 'functions':
            marked_functions = self.functions_to_deactivate_from_neighbours(marked_entities)

        self.refine_hierarchical_space(marked_functions, NE)

    def get_children(self, marked_functions, level):
        children = set()
        for function in marked_functions:
            c = set(np.flatnonzero(self.projectors[level][:, function]))
            children = children.union(c)
        return children

    def get_parents(self):
        pass

    def refine_hierarchical_mesh(self, marked_cells):
        n = self.number_of_levels
        if marked_cells[n - 1] != set():
            self.mesh.add_new_level()

        return self.update_active_cells(marked_cells)

    def update_active_cells(self, marked_cells):

        number_of_levels = len(marked_cells)
        new_cells = {0: set()}

        for level in range(number_of_levels):
            if marked_cells[level] == set():  # no  marked cells at this level:
                continue

            common_cells = marked_cells[level].intersection(self.mesh.active_elements_per_level[level])
            self.mesh.active_elements_per_level[level] = self.mesh.active_elements_per_level[level].difference(
                common_cells)
            self.mesh.deactivated_elements_per_level[level] = self.mesh.deactivated_elements_per_level[level].union(
                marked_cells[level])

            new_cells[level + 1] = self.mesh.get_children_of_cell(marked_cells[level], level)
            self.mesh.active_elements_per_level[level + 1] = self.mesh.active_elements_per_level[level + 1].union(
                new_cells[level + 1])

        self.mesh.number_of_elements_per_level = {level: len(self.mesh.active_elements_per_level[level]) for level in
                                                  range(self.mesh.number_of_levels)}
        self.mesh.number_of_elements = sum(self.mesh.number_of_elements_per_level)

        return new_cells

    def functions_to_deactivate_from_cells(self, marked_entities):
        n = self.number_of_levels
        marked_functions_per_level = {}
        for l in range(n):
            marked_functions = self.tensor_product_space_per_level[l].get_basis_functions(marked_entities[l])
            marked_functions = marked_functions.intersection(self.active_functions_per_level[l])
            marked_cells = self.tensor_product_space_per_level[l].get_cells(marked_functions)
            functions_to_remove = set()
            for cells, function in zip(marked_cells, marked_functions):
                if cells.intersection(self.mesh.active_elements_per_level[l]) != set():
                    functions_to_remove.add(function)
            marked_functions = marked_functions.difference(functions_to_remove)
            marked_functions_per_level[l] = marked_functions
        return marked_functions_per_level

    def refine_hierarchical_space(self, marked_functions, new_cells):
        if self.mesh.number_of_levels > self.number_of_levels:
            self.add_new_level()
            marked_functions[self.mesh.number_of_levels] = set()

        self.update_active_functions(marked_functions, new_cells)

    def update_active_functions(self, marked_functions, new_cells):

        active = self.active_functions_per_level
        deactivated = self.deactivated_functions_per_level
        for l in range(self.number_of_levels):
            active[l] = active[l].difference(marked_functions[l])
            deactivated[l] = deactivated[l].union(marked_functions[l])

            children = self.get_children(marked_functions[l], l)
            active_and_deactive = active[l + 1].union(deactivated[l + 1])
            print(active_and_deactive)
            new_active = children.difference(active_and_deactive)
            active[l + 1] = active[l + 1].union(new_active)

            new_possible_active_functions = self.tensor_product_space_per_level[l + 1].get_basis_functions(
                new_cells[l + 1])
            new_possible_active_functions = new_possible_active_functions.difference(active[l + 1])

            new_cells_next = self.tensor_product_space_per_level[l + 1].get_cells(new_possible_active_functions)

            new_functions_to_remove = set()
            active_deactive_cells = self.mesh.active_elements_per_level[l + 1].union(
                self.mesh.deactivated_elements_per_level[l + 1])

            for cells, function in zip(new_cells_next, new_possible_active_functions):
                if not cells.issubset(active_deactive_cells):
                    new_functions_to_remove.add(function)

            new_possible_active_functions.difference(new_functions_to_remove)
            active[l + 1] = active[l + 1].union(new_possible_active_functions)
