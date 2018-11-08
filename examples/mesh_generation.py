from src.Mesh import Mesh

knots = [
    [0, 0, 1, 2, 3, 3]
]

d = [1]
dim = 1

M = Mesh(knots, d, dim)
print(M.cells)