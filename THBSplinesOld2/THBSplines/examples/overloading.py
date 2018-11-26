import matplotlib.pyplot as plt
import numpy as np

from THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplines.THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

knots_old = [
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8],
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
]

d = [3, 3]
dim = 2

S = TensorProductSpace(d, knots_old, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_elements = [T.refine_in_rectangle(np.array([[3, 8], [0, 8]]), 0)]
T.refine(marked_elements)
marked_elements.append(T.refine_in_rectangle(np.array([[4, 8], [0, 8]]), 1))
T.refine(marked_elements)
#T.update_hierarchical_mesh()
T.visualize_hierarchical_mesh()
plt.show()
