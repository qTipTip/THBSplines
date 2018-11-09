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


T.refine_element([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]])
T.visualize_hierarchical_mesh()