from THBSplines.src.SplineAux import generate_tensor_product_space, insert_midpoints, compute_knot_insertion_matrix, \
    set_basis_support_cells
from THBSplines.src.TensorProductMesh import TensorProductMesh


class TensorProductSpace(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.functions = generate_tensor_product_space(degrees, knots, parametric_dimension)
        self.nfunctions = len(self.functions)
        self.mesh = TensorProductMesh(degrees, knots, parametric_dimension, physical_dimension)
        self.basis_to_cell, self.cell_to_basis = set_basis_support_cells(self.functions, self.mesh.cells)

    def refine(self):
        refined_knots = [insert_midpoints(knots, d) for knots, d in zip(self.knots, self.degrees)]
        projection = compute_knot_insertion_matrix(self.degrees, self.knots, refined_knots)
        refined_space = TensorProductSpace(self.degrees, refined_knots, self.parametric_dim, self.physical_dim)
        return refined_space, projection

    def get_basis_functions(self, cell_indices):
        if isinstance(cell_indices, int):
            cell_indices = [cell_indices]

        functions = set()
        for cell in cell_indices:
            functions = functions.union(self.cell_to_basis[cell])
        return functions

    def get_cells(self, function_indices):
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        function_cells = []
        for function in function_indices:
            function_cells.append(self.basis_to_cell[function])
        return function_cells

    def get_neighbours(self, function_indices):
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        functions = []
        function_cells = self.get_cells(function_indices)
        for i, function in enumerate(function_indices):
            cells = function_cells[i]
            function_neighbours = self.get_basis_functions(cells)
            function_neighbours.discard(function)
            functions.append(function_neighbours)
        return functions

    def get_function(self, coefficients, index):
        def f(x):
            return sum([self.functions[i](x) * coefficients[i] for i in index])
        return f
