import matplotlib.pyplot as plt
import numpy as np

from src.Space import Space
from src.BSplines import generate_tensor_product_space
from mpl_toolkits.mplot3d import Axes3D
knots = [
    [0, 0, 1, 2, 3, 4, 4],
    [0, 0, 1, 2, 3, 4, 4]
]

d = [1, 1]
dim = 2

S0 = Space(d, knots, dim)

N = 100
x = np.linspace(0, 3, N)
y = np.linspace(0, 4, N)

z = np.zeros((N, N))

X, Y = np.meshgrid(x, y)
S0.add_level()

print(S0.coefficient_matrices[0].shape)
print(len(S0.functions[0]))
print(len(S0.functions[1]))
for level in S0.function_to_child_map:
    print(S0.function_to_child_map[level])
print(S0.get_children_of_function(0, 2))
print(S0.get_parent_of_function(1, 3))
'''
for b in S0.functions[1]:
    fig = plt.figure()
    axs = Axes3D(fig)
    for i in range(N):
        for j in range(N):
            z[i, j] = b(np.array([x[i], y[j]]))
    axs.plot_wireframe(X, Y, z)
    plt.show()
'''