from THBSplines.src.SplineAux import generate_tensor_product_space


class TensorProductSpace(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.functions = generate_tensor_product_space(degrees, knots, parametric_dimension)
