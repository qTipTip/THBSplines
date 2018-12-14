import THBSplines as thb
import numpy as np
import matplotlib.pyplot as plt

# Initialize a biquadraic space of Truncated Hierarchical B-Splines
knots = [
  [0, 0, 1, 2, 3, 3],
  [0, 0, 1, 2, 3, 3]
]
d = 2
degrees = [2, 2]
T = thb.HierarchicalSpace(knots, degrees, d)
marked_cells = {0: [0, 3, 6]}
T = thb.refine(T, marked_cells)
T.plot_overloading()