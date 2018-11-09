class HierarchicalMesh(object):

    def __init__(self):
        self.parametric_dim = 0
        self.physical_dim = 0

        self.number_of_levels = 1
        self.number_of_elements = 0
        self.number_of_elements_per_level = {}
        self.active_elements_per_level = {}
        self.deactivated_elements_per_level = {}
        self.mesh_per_level = {}

    def add_new_level(self):

        self.number_of_levels = self.number_of_levels + 1
        self.active_elements_per_level[self.number_of_levels] = set()
        self.deactivated_elements_per_level[self.number_of_levels] = set()
        self.number_of_elements_per_level[self.number_of_levels] = 0
        self.mesh_per_level[self.number_of_levels] = self.refine(self.mesh_per_level[self.self.number_of_levels - 1])

    def refine(self):
        pass

    def get_childrne(self):
        pass

    def get_parents(self):
        pass
