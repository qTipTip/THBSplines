import numpy as np

from THBSplines.TensorProductSpace import TensorProductSpace

knots = [
    [1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
]

T = TensorProductSpace([3], knots, 1)
import matplotlib.pyplot as plt
x = np.linspace(1, 8, 400)
for b in T.functions:
    y = [b(X) for X in x]
    plt.plot(x, y)

plt.show()