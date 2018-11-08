from typing import List

import numpy as np

from src.BSplines import generate_tensor_product_space, compute_knot_insertion_matrix, TensorProductSpace


class Space(object):

    def __init__(self, initial_degrees, initial_knots, dim):
        self.functions = [TensorProductSpace(initial_degrees, initial_knots, dim)]
        self.knots = [initial_knots]
        self.depth = 0
        self.active_function_indices = len(self.functions)
        self.deactivated_function_indices = []
        self.coefficient_matrices = []
        self.dim = dim
        self.degrees = initial_degrees
        self.function_to_child_map = {}

    def get_children_of_function(self, level: int, func_index: int) -> List[int]:
        """
        Given the index of a function at the given level, return the indices of the functions on level level+1 corresponding
        to the given function.
        :param level:
        :param func_index:
        :return:
        """

        assert level <= self.depth - 1

        return self.function_to_child_map[level][func_index]

    def get_parent_of_function(self, level: int, func_index: int) -> int:
        """
        Given the index of a function on the given level, return the index of its parent function at the previous level.
        :param level: 
        :param func_index: 
        :return:
        """

        assert 1 <= level

        parents = []
        for i in range(len(self.functions[level - 1])):
            if func_index in self.get_children_of_function(level - 1, i):
                parents.append(i)
        if len(parents) == 0:
            return -1
        return parents

    def add_level(self):

        old_knots = self.knots[-1]
        new_knots = []
        coefficient_matrices = []
        for degree, knot_vector in zip(self.degrees, old_knots):
            refined_knots = insert_midpoints(knot_vector, degree)
            new_knots.append(refined_knots)
            coefficient_matrices.append(compute_knot_insertion_matrix(degree, knot_vector, refined_knots))

        self.functions.append(generate_tensor_product_space(self.degrees, new_knots, self.dim))
        self.knots.append(new_knots)
        self.depth += 1

        # compute the tensor product coefficient matrix
        A = coefficient_matrices[0]
        for i in range(self.dim - 1):
            A = np.kron(A, coefficient_matrices[i + 1])
        self.coefficient_matrices.append(A)
        self._set_function_children()

    def _set_function_children(self):
        level = self.depth - 1
        self.function_to_child_map[level] = {}
        for i in range(len(self.functions[-2])):
            self.function_to_child_map[level][i] = np.flatnonzero(self.coefficient_matrices[-1][:, i])

    def set_active_functions(self, level, region):
        pass

    def set_deactivated_functions(self, level, region):
        pass


def insert_midpoints(knots, p):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot vector.
    :param knots: p + 1 regular knot vector to be refined
    :param p: spline degree
    :return: refined_knots
    """

    knots = np.array(knots, dtype=np.float64)
    midpoints = (knots[p:-p - 1] + knots[p + 1:-p]) / 2
    new_array = np.zeros(len(knots) + len(midpoints), dtype=np.float64)

    new_array[:p + 1] = knots[:p + 1]
    new_array[-p - 1:] = knots[-p - 1]
    new_array[p + 1:p + 2 * len(midpoints):2] = midpoints
    new_array[p + 2:p + 2 * len(midpoints) - 1:2] = knots[p + 1:-p - 1]

    return new_array
