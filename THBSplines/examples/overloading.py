from THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplines.TensorProductSpace import TensorProductSpace

knots_old = [
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8],
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
]

d = [3, 3]
dim = 2

S = TensorProductSpace(d, knots_old, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_elements = [{0, 1, 2, 3, 4}]
T.refine(marked_elements)

import matplotlib.pyplot as plt
T.visualize_hierarchical_mesh()
plt.show()