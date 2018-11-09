class HierarchicalSpace(object):

    def __init__(self):
        self.truncated = True
        self.number_of_levels = 1
        self.number_of_functions = 0
        self.number_of_functions_per_level = None
        self.active_functions_per_level = None
        self.deactivated_functions_per_level = None
        self.tensor_product_space_per_level = []

    def add_new_level(self):
        pass

    def refine(self):
        pass

    def get_children(self):
        pass

    def get_parents(self):
        pass
