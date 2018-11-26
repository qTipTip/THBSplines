import numpy as np
from src.hierarchical_space import HierarchicalSpace
from src.refinement import refine

knots = [
    [0, 0, 0, 1, 2, 3, 3, 3],
    [0, 0, 0, 1, 2, 3, 3, 3]
]
d = 2
degrees = [2, 2]
T = HierarchicalSpace(knots, degrees, d)
cells = {}
cells[0] = T.refine_in_rectangle(np.array([[1, 3], [1, 3]]), level=0)
T = refine(T, marked_entities=cells)
cells[1] = T.refine_in_rectangle(np.array([[2, 3], [2, 3]]), level=1)
T = refine(T, marked_entities=cells)
T.mesh.plot_cells()
