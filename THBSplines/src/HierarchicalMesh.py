from THBSplines.src import TensorProductMesh


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

    def add_new_level(self):
        self.number_of_levels = self.number_of_levels + 1
        self.active_elements_per_level[self.number_of_levels - 1] = set()
        self.deactivated_elements_per_level[self.number_of_levels - 1] = set()
        self.number_of_elements_per_level[self.number_of_levels - 1] = 0
        self.mesh_per_level[self.number_of_levels - 1] = self.mesh_per_level[self.number_of_levels - 2].refine()

    def refine(self):
        pass

    def get_children(self):
        pass

    def get_parents(self):
        pass
