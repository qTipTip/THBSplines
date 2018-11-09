from THBSplines.src.SplineAux import generate_tensor_product_space, insert_midpoints, compute_knot_insertion_matrix


class TensorProductSpace(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.functions = generate_tensor_product_space(degrees, knots, parametric_dimension)
        self.nfunctions = len(self.functions)

    def refine(self):
        refined_knots = [insert_midpoints(knots, d) for knots, d in zip(self.knots, self.degrees)]
        projection = compute_knot_insertion_matrix(self.degrees, self.knots, refined_knots)
        refined_space = TensorProductSpace(self.degrees, refined_knots, self.parametric_dim, self.physical_dim)
        return refined_space, projection
