import THBSplines as thb
import numpy as np
import matplotlib.pyplot as plt

# Initialize a biquadraic space of Truncated Hierarchical B-Splines
knots = [
  [0, 0, 1/3, 2/3, 1, 1],
  [0, 0, 1/3, 2/3, 1, 1]
]
degrees = [2, 2]
dimension = 2
T = thb.HierarchicalSpace(knots, degrees, dimension)

# Select cells to refine at each level, either by explicitly marking the elements, or by choosing a rectangular region.
cells_to_refine = {}
cells_to_refine[0] = [0, 1, 2, 3, 4, 5, 6]
T = thb.refine(T, cells_to_refine)

rect = np.array([[0, 1 / 3], [0, 2 / 3]])
cells_to_refine[1] = T.refine_in_rectangle(rect, level = 1)
T = thb.refine(T, cells_to_refine)
T.mesh.plot_cells()

# If no integration order is specified, exact gauss quadrature suitable for the given basis is used.
mass_matrix = thb.hierarchical_mass_matrix(T)
stiffness_matrix = thb.hierarchical_stiffness_matrix(T)

plt.spy(mass_matrix, markersize=1)
plt.show()