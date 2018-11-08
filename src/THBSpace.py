from src.Mesh import Mesh
from src.Space import Space


class THBSpace(object):

    def __init__(self, degrees, knots, dim):

        self.mesh = Mesh(knots, degrees, dim)
        self.space = Space(degrees, knots, dim)


