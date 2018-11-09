from THBSplines.src import TensorProductMesh
from THBSplines.src.SplineAux import set_children_of_cells


class HierarchicalMesh(object):

    def __init__(self, mesh: TensorProductMesh):
        self.parametric_dim = mesh.parametric_dim
        self.physical_dim = mesh.physical_dim

        self.number_of_levels = 1
        self.number_of_elements = mesh.number_of_elements
        self.number_of_elements_per_level = {0: mesh.number_of_elements}
        self.active_elements_per_level = {0: set(range(mesh.number_of_elements))}
        self.deactivated_elements_per_level = {0: set()}
        self.mesh_per_level = {0: mesh}
        self.cell_to_children = {}

    def add_new_level(self):
        self.number_of_levels = self.number_of_levels + 1
        self.active_elements_per_level[self.number_of_levels - 1] = set()
        self.deactivated_elements_per_level[self.number_of_levels - 1] = set()
        self.number_of_elements_per_level[self.number_of_levels - 1] = 0
        self.mesh_per_level[self.number_of_levels - 1] = self.mesh_per_level[self.number_of_levels - 2].refine()
        self.cell_to_children[self.number_of_levels - 2] = set_children_of_cells(
            self.mesh_per_level[self.number_of_levels - 1], self.mesh_per_level[self.number_of_levels - 2])

    def refine(self):
        pass

    def get_children(self):
        pass

    def get_parents(self):
        pass

    def get_children_of_cell(self, marked_cells, level):
        children = set()
        for cell in marked_cells:
            children = children.union(self.cell_to_children[level][cell])
        return children
