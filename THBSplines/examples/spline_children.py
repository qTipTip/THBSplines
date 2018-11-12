import numpy as np

from THBSplines.src.HierarchicalMesh import HierarchicalMesh
from THBSplines.src.HierarchicalSpace import HierarchicalSpace
from THBSplines.src.TensorProductSpace import TensorProductSpace
import matplotlib.pyplot as plt


d = [2]
dim = 1
knots = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]

S = TensorProductSpace(d, knots, dim)
S1, C = S.refine()
B = S.functions[2]
childs = [S1.functions[i] for i in [4, 5, 6, 7]]

x = np.linspace(0, 8, 500)


import matplotlib.pyplot as plt

y = [B(X) for X in x]
plt.plot(x, y)

print(S1.knots)
for c, i in zip(childs, [4, 5, 6, 7]):
    print(c.knots)
    y = [C[i, 2] * c(X) for X in x]
    plt.plot(x, y, '--')

plt.show()