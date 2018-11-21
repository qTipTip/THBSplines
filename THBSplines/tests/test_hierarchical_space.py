import numpy as np
from src.hierarchical_space import HierarchicalSpace
from src.refinement import refine


def test_active_funcs_per_level():
    knots = [
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)

    np.testing.assert_equal(T.afunc_level, {0: [0, 1, 2, 3]})


def test_active_funcs_per_level_refine():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    cells = {0: [0]}
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {0: 8, 1: 4})


def test_functions_to_deactivate_from_cells():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0]}
    new_cells = T.mesh.refine(marked_cells)
    marked_functions = T.functions_to_deactivate_from_cells(marked_cells)

    np.testing.assert_equal(marked_functions, {0: [0]})
