import matplotlib.pyplot as plt
import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductSpace import TensorProductSpace

knots = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]
]

d = [3, 3]
dim = 2

S = TensorProductSpace(d, knots, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_cells = [{0, 1, 2, 4, 5, 6, 8, 9}]
T.refine(marked_cells)
num = 30
x = np.linspace(0, 4, num)
y = np.linspace(0, 4, num)
z = np.zeros((num, num))
X, Y = np.meshgrid(x, y)
for k, b in enumerate(T.get_truncated_basis()):
    for i in range(num):
        for j in range(num):
            z[i, j] = b(np.array((x[i], y[j])))
    fig = T.visualize_hierarchical_mesh()
    plt.contourf(X, Y, z, levels=30)
    plt.savefig('thb_{}.pdf'.format(k))
