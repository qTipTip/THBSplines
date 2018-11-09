from THBSplines.src.SplineAux import generate_cells, insert_midpoints


class TensorProductMesh(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.cells = generate_cells(knots)

    def refine(self):
        refined_knots = [insert_midpoints(knots, d) for knots, d in zip(self.knots, self.degrees)]
        return TensorProductMesh(self.degrees, refined_knots, self.parametric_dim, self.physical_dim)

