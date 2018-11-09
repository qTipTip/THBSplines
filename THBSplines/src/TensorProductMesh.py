from THBSplines.src.SplineAux import generate_cells


class TensorProductMesh(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.cells = generate_cells(knots)
