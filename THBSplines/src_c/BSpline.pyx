
cimport numpy as np
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint allclose(double[:] x, double[:] y):
    """
    Returns true if the two memory views are equal up to tolerance and length
    :param x: memoryview 
    :param y: memoryview
    :return: True or false
    """

    cdef int n = x.shape[0], m = y.shape[0]
    if n != m:
        return False

    cdef Py_ssize_t i
    cdef double eps = 1.0e-14
    for i in range(n):
        if abs(x[i] - y[i]) > eps:
            return False
    return True


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int knot_index(double [:] knots, double x, int end):
    """
    Returns the index such that inserting x at this index yields a sorted array.
    :param knots: 
    :param x: 
    :return: 
    """

    cdef unsigned int n
    cdef Py_ssize_t i
    n = knots.shape[0]

    if end == 0:
        for i in range(n-1):
            if knots[i] <= x < knots[i+1]:
                return i
        return -1
    else:
        for i in range(n-1):
            if knots[i] < x <= knots[i+1]:
                return i
        return -1



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double evaluate_single_basis_function(double x, int degree, double[:] knots, int end):
    """
    Evaluates the single basis function defined over the local knot vector, using the recursion formula. 
    :param x: point of evaluation
    :param degree: degree of b-spline
    :param knots: knot vector of exactly degree + 2 knots
    :return: basis function evaluated at point
    """

    cdef int i = knot_index(knots, x, end)
    cdef int n = knots.shape[0]
    if i == -1:
        return 0.0

    if degree == 0:
        return 1


    cdef double left, right
    left = evaluate_single_basis_function(x, degree - 1, knots[:n-1], 0)
    right = evaluate_single_basis_function(x, degree - 1, knots[1:n], end)

    cdef double denom_left = knots[n-2] - knots[0]
    cdef double denom_right = knots[n-1] - knots[1]
    cdef double eps = 1.0e-14

    if denom_left < eps:
        left = 0
    else:
        left *= (x - knots[0]) / denom_left
    if denom_right < eps:
        right = 0
    else:
        right *= (knots[n-1] - x) / denom_right

    return left + right

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] evaluate_single_basis_function_vectorized(double [:] x, int degree, double[:] knots, int end):
    cdef int n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double[:] out_vector = result

    cdef Py_ssize_t i
    for i in range(n):
        out_vector[i] = evaluate_single_basis_function(x[i], degree, knots, end)
    return result

cpdef evaluate_single_basis_derivative(x, degree, knots, end, r):

    cdef int i = knot_index(knots, x, end)
    cdef int n = knots.shape[0]
    cdef double denom_left = knots[n-2] - knots[0]
    cdef double denom_right = knots[n-1] - knots[1]
    cdef double eps = 1.0e-14

    cdef double left, right

    if r == 1: # right hand side consists of evaluation only
        left = evaluate_single_basis_function(x, degree-1, knots[:n-1], 0)
        right = evaluate_single_basis_function(x, degree-1, knots[1:], end)

    else:
        left = evaluate_single_basis_derivative(x, degree-1, knots[:n-1], 0, r-1)
        right = evaluate_single_basis_derivative(x, degree-1, knots[1:], end, r-1)
    if denom_left < eps:
        left = 0
    else:
        left /= denom_left

    if denom_right < eps:
        right = 0
    else:
        right /= denom_right

    return degree * (left - right)


cdef evaluate_single_basis_derivative_vectorized(x, degree, knots, evaluate_end, r):
    cdef int n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(n):
        result[i] = evaluate_single_basis_derivative(x[i], degree, knots, evaluate_end, r)
    return result

cdef class BSpline:

    cdef public:
        int degree
        double[:] knots
        bint evaluate_end

    def __init__(self, degree, knots, evaluate_end=0):
        self.degree = degree
        self.knots = knots
        self.evaluate_end = evaluate_end

    def __call__(self, x):
        return self.evaluate(np.array(x, dtype=np.float64))

    cdef evaluate(BSpline self, double[:] x):
        return evaluate_single_basis_function_vectorized(x, self.degree, self.knots, self.evaluate_end)

    cpdef derivative(BSpline self, double[:] x, r):
        return evaluate_single_basis_derivative_vectorized(x, self.degree, self.knots, self.evaluate_end, r)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __eq__(BSpline self, BSpline other):
        """
        Checks for BSpline equality by comparing degrees and knot vectors.
        :param other:
        :return:
        """
        if self.degree != other.degree:
            return False
        return allclose(self.knots, other.knots)

cdef class TensorProductBSpline:
    cdef public:
        int[:] degrees
        int parametric_dimension
        int[:] end_evaluation
        list univariate_b_splines

    def __init__(self, degrees, knots, end_evaluation=None):
        self.degrees = degrees
        self.parametric_dimension = len(degrees)
        self.univariate_b_splines = []
        if end_evaluation is None:
            end_evaluation = np.zeros(self.parametric_dimension, dtype=np.intc)
        self.end_evaluation = end_evaluation

        cdef int n = knots.shape[0]
        cdef Py_ssize_t i
        for i in range(n):
            self.univariate_b_splines.append(BSpline(degrees[i], knots[i], self.end_evaluation[i]))

    def __call__(TensorProductBSpline self, x):

        x = np.array(x, dtype=np.float64).reshape(-1, self.parametric_dimension)
        result = self.evaluate(x)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef evaluate(self, double [:, :] x):
        cdef int n = x.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] out_vector = np.ones(n, dtype=np.float64)
        cdef Py_ssize_t i, j
        cdef np.ndarray[np.float64_t, ndim=2] temp_vector = np.zeros_like(x, dtype=np.float64)
        for j in range(self.parametric_dimension):
            temp_vector[:, j] = self.univariate_b_splines[j](x[:, j])
        for i in range(n):
            for j in range(self.parametric_dimension):
                out_vector[i] *= temp_vector[i, j]
        return out_vector

    def evaluate_gridded_data(self, x):
        n = x.shape[0]
        dim = x.shape[1]
        shape = tuple([n for _ in range(dim)])
        print(shape)
        z = np.zeros(shape = shape, dtype=np.float64)
        print(z.shape)
        for i, _ in np.ndenumerate(x):
            point = np.array([x[i[d], d] for d in range(dim)], dtype=np.float64)
            z[i] = self(point)
        return z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double integrate(double[:] bi_values, double[:] bj_values, double[:] weights, double area, int dim):

    cdef double I = 0
    cdef Py_ssize_t i
    cdef int n = bi_values.shape[0]
    for i in range(n):
        I += weights[i] * bi_values[i] * bj_values[i]

    I *= area / 2**dim

    return I
