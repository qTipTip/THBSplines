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
            self.projectors[self.mesh.number_of_levels - 1] = projection
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

    def get_children(self):
        pass

    def get_parents(self):
        pass

    def refine_hierarchical_mesh(self, marked_cells):
        n = self.number_of_levels
        if marked_cells[n - 1] != set():
            print('Adding new level to mesh')
            self.mesh.add_new_level()

        NE = {0: set()}
        for l in range(n):
            self.mesh.active_elements_per_level[l] = self.mesh.active_elements_per_level[l].difference(marked_cells[l])
            self.mesh.deactivated_elements_per_level[l] = self.mesh.deactivated_elements_per_level[l].union(
                marked_cells[l])
            NE[l + 1] = self.mesh.get_children_of_cell(marked_cells[l], l)
            self.mesh.active_elements_per_level[l + 1] = self.mesh.active_elements_per_level[l + 1].union(NE[l + 1])
        assert len(NE) == n + 1
        return NE

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

    def refine_hierarchical_space(self, marked_functions, NE):
        if self.mesh.number_of_levels > self.number_of_levels:
            self.add_new_level()
        n = self.number_of_levels - 1
        for l in range(n):
            self.active_functions_per_level[l] = self.active_functions_per_level[l].difference(marked_functions)
            self.deactivated_functions_per_level[l] = self.deactivated_functions_per_level[l].union(marked_functions)
            new = NE[l+1]
            F = self.tensor_product_space_per_level[l + 1].get_basis_functions(new)
            F = F.difference(self.active_functions_per_level[l + 1])

            function_cells = self.tensor_product_space_per_level[l+1].get_cells(F)
            functions_to_remove = set()
            for cells, function in zip(function_cells, F):
                if not cells.issubset(self.mesh.active_elements_per_level[l + 1].union(
                        self.mesh.deactivated_elements_per_level[l + 1])):
                    functions_to_remove.add(function)
            F = F.difference(functions_to_remove)
            self.active_functions_per_level[l + 1] = self.active_functions_per_level[l + 1].union(F)
