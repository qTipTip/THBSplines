import matplotlib.pyplot as plt
import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductSpace import TensorProductSpace

knots = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
]

d = [2]
dim = 1

S = TensorProductSpace(d, knots, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_cells = [[2, 3, 4]]
T.refine(marked_cells)

x = np.linspace(knots[0][0], knots[0][-1], 200)
for level in range(T.number_of_levels):
    for active in T.active_functions_per_level[level]:
        b = T.tensor_product_space_per_level[level].functions[active]
        y = [b(X) for X in x]
        plt.plot(x, y)
plt.show()