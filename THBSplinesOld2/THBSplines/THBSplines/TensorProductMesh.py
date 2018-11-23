import numpy as np

from THBSplinesOld2.THBSplines.THBSplines.SplineAux import generate_cells


def insert_midpoints_mesh(knots):
    knots = np.array(knots, dtype=np.float64)
    midpoints = (knots[1:] + knots[:-1]) / 2
    new_array = np.zeros(len(knots) + len(midpoints), dtype=np.float64)
    new_array[0::2] = knots
    new_array[1:-1:2] = midpoints

    return new_array


class TensorProductMesh(object):

    def __init__(self, degrees, knots, parametric_dimension, physical_dimension=1):
        self.degrees = degrees
        self.knots = knots
        self.parametric_dim = parametric_dimension
        self.physical_dim = physical_dimension
        self.cells = generate_cells(knots)
        self.number_of_elements = len(self.cells)

    def refine(self):
        refined_knots = [insert_midpoints_mesh(np.unique(knots)) for knots in self.knots]
        return TensorProductMesh(self.degrees, refined_knots, self.parametric_dim, self.physical_dim)
