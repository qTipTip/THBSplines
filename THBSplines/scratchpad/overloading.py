from src.hierarchical_space import HierarchicalSpace
from src.refinement import refine

knots = [
    [0, 0, 1, 2, 2],
    [0, 0, 1, 2, 2]
]
d = 2
degrees = [1, 1]
T = HierarchicalSpace(knots, degrees, d)
marked_cells = {0: [0]}
T = refine(T, marked_cells)
print(T.afunc_level)

T.plot_overloading()