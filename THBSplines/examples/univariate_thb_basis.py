import numpy as np
import matplotlib.pyplot as plt
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

marked_cells = [{2, 3, 4}]
T.refine(marked_cells)


C = T.truncated_coefficients()
B = T.get_truncated_basis()

x = np.linspace(0, 8, 400)

for b in B:
    y = [b(X) for X in x]
    plt.plot(x, y)
plt.show()