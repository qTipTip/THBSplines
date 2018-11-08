from src.Mesh import Mesh
from src.Space import Space


class THBSpace(object):

    def __init__(self, degrees, knots, dim):
        self.mesh = Mesh(knots, degrees, dim)
        self.space = Space(degrees, knots, dim)

    def add_level(self):
        self.mesh.add_level()
        self.space.add_level()
