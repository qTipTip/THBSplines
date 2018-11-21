import numpy as np

from THBSplinesOld2.THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplinesOld2.THBSplines.THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplinesOld2.THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

knots = [
    np.pad(np.linspace(1, 8, 30), mode='edge', pad_width=3)
]

d = [3]
dim = 1

S = TensorProductSpace(d, knots, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

LR = T


def sample_function(greville_points, func):
    y = []
    for p in greville_points:
        y.append(func(p))
    return y


def compute_knot_averages(t, p):
    n = len(t) - p - 1
    k = np.zeros(n)
    for i in range(n):
        k[i] = sum(t[i + 1: i + p + 1]) / float(p)

    return k


def assemble_interpolation_matrix(basis, interpolation_points):
    n = len(basis)
    m = len(interpolation_points)

    a = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            I = basis[j](interpolation_points[i])
            a[i, j] = I
    return a


def run_interpolation(knots, d):
    greville_points = compute_knot_averages(knots[0], d[0])

    basis = T.get_truncated_basis()
    a = assemble_interpolation_matrix(basis, greville_points)
    y = sample_function(greville_points, lambda x: np.sin(x))
    c = np.linalg.solve(a, y)

    f = lambda x: sum([C * b(x) for C, b in zip(c, basis)])

    X = np.linspace(1, 8, 100)
    Y = [f(x) for x in X]
    import matplotlib.pyplot as plt

    print("""
        The THB interpolation matrix has:
            condition number {}
            determinant      {}
    """.format(np.linalg.cond(a), np.linalg.det(a)))

    plt.scatter(greville_points, y)
    plt.plot(X, Y)
    plt.show()


def refine_thb():
    marked_entities = {0: {14, 15, 16, 17, 18, 19, 20}}
    T.refine(marked_entities, type='cells')

refine_thb()
run_interpolation(knots, d)