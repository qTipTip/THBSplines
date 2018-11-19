import numpy as np

from THBSplines.src.abstract_space import Space


class TensorProductSpace(Space):

    def __init__(self, knots, degrees, dim):
        self.knots = np.array(knots)
        self.degrees = np.array(degrees)
        self.dim = dim
