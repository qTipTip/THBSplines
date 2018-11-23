import numpy as np

from THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

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

for c, i in zip(childs, [4, 5, 6, 7]):
    y = [c(X) for X in x]
