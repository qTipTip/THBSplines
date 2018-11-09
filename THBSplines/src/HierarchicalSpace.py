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

    def refine(self):
        pass

    def get_children(self):
        pass

    def get_parents(self):
        pass
