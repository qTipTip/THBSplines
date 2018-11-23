import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplines.THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

knots = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]
]

d = [1, 1]
dim = 2

S = TensorProductSpace(d, knots, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_cells = [{0, 1, 2, 4, 5, 6, 8, 9}]
T.refine(marked_cells)
num = 30

marked_cells = [{0, 1, 2, 4, 5, 6, 8, 9}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}]
T.refine(marked_cells)
x = np.linspace(0, 4, num)
y = np.linspace(0, 4, num)
z = np.zeros((num, num))


X, Y = np.meshgrid(x, y)
basis = T.get_truncated_basis()

for k, b in enumerate(basis):
    for i in range(num):
        for j in range(num):
            z[i, j] = b(np.array((x[i], y[j])))
    fig = T.visualize_hierarchical_mesh()
    #fig = plt.figure()
    axs = Axes3D(fig)
    axs.plot_wireframe(X, Y, z)
    plt.show()
