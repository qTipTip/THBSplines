import numpy as np
import matplotlib.pyplot as plt

from src.THBSpace import THBSpace

knots = [
    [0, 0, 0, 1, 2, 3, 3, 3],
    [0, 0, 0, 1, 2, 3, 3, 3]
]

d = [2, 2]
dim = 2

T = THBSpace(d, knots, dim)
#T.add_level()
#T.add_level()
#T.add_level()
#x = np.linspace(0, 1, 200)

T.visualize_mesh()
T.refine_element([[0, 1], [0, 1, 2, 3, 4, 5]])