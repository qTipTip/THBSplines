from src.hierarchical_space import HierarchicalSpace
from src.refinement import refine

knots = [
    [0, 1, 2],
    [0, 1, 2]
]
d = 2
degrees = [1, 1]
T = HierarchicalSpace(knots, degrees, d)
marked_cells = {0: [0]}
T = refine(T, marked_cells)
marked_cells = {0: [0], 1: [0]}
T = refine(T, marked_cells)
marked_cells = {0: [0], 1: [0], 2: [0]}
T = refine(T, marked_cells)
T.plot_overloading()
