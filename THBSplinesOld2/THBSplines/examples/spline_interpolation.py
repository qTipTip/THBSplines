import numpy as np

from THBSplinesOld2.THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplinesOld2.THBSplines.THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplinesOld2.THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

knots = [
    np.pad(np.linspace(1, 8, 10), mode='edge', pad_width=3)
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


def get_greville_points(basis):
    return [function.greville_point for function in basis]


def run_interpolation():
    LR = TensorProductSpace([3], [get_univariate_tensor_product_knots(T)], dim)
    LR = HierarchicalSpace(HierarchicalMesh(LR.mesh), LR)

    basis_thb = T.get_truncated_basis(sort=True)
    basis_lr = LR.get_truncated_basis()
    greville_points_thb = get_greville_points(basis_thb)
    greville_points_lr = get_greville_points(basis_lr)
    a_thb = assemble_interpolation_matrix(basis_thb, greville_points_thb)
    a_lr = assemble_interpolation_matrix(basis_lr, greville_points_lr)


    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.spy(a_thb, markersize=1)
    plt.subplot(212)
    plt.spy(a_lr, markersize=1)
    plt.show()

    print("""
        The THB interpolation matrix has:
            condition number {}
            determinant      {}
    """.format(np.linalg.cond(a_thb), np.linalg.det(a_thb)))

    print("""
        The LR interpolation matrix has:
            condition number {}
            determinant      {}
        """.format(np.linalg.cond(a_lr), np.linalg.det(a_lr)))
    X = np.linspace(1, 8, 1000)
    plt.subplot(211)
    for b in basis_thb:
        yy = [b(x) for x in X]
        plt.plot(X, yy)
        plt.plot(greville_points_thb, np.zeros_like(greville_points_thb), '*')
    plt.subplot(212)
    for b in basis_lr:
        yy = [b(x) for x in X]
        plt.plot(X, yy)
        plt.plot(greville_points_lr, np.zeros_like(greville_points_lr), '*')
    plt.show()


    exact = lambda x: np.sin(x)
    y_thb = sample_function(greville_points_thb, exact)
    c_thb = np.linalg.solve(a_thb, y_thb)

    y_lr = sample_function(greville_points_lr, exact)
    c_lr = np.linalg.solve(a_lr, y_lr)

    f_thb = lambda x: sum([C * b(x) for C, b in zip(c_thb, basis_thb)])
    f_lr = lambda x: sum([C * b(x) for C, b in zip(c_lr, basis_lr)])

    Y_thb = [f_thb(x) for x in X]
    Y_lr = [f_lr(x) for x in X]






    plt.plot(X, exact(X), '--')
    plt.scatter(greville_points_thb, y_thb)
    plt.plot(X, Y_thb)
    plt.show()


def refine_thb(marked_entities):
    T.refine(marked_entities, type='cells')


def get_univariate_tensor_product_knots(T: HierarchicalSpace):
    """
    Given a univariate hierarchical space, return the full knot vector, for use in msb-interpolation.
    :return:
    """

    cells = []
    for level in range(T.mesh.number_of_levels):
        for cell in T.mesh.mesh_per_level[level].cells[list(T.mesh.active_elements_per_level[level])]:
            cells.append(cell)
    cells = np.pad(np.unique(np.array(cells).flatten()), mode='edge', pad_width=3)
    return cells


cells = {}
for i in range(3):
    cells[i] = T.refine_in_rectangle(np.array([[2 + i, 7 - i]]), i)
    refine_thb(cells)
    run_interpolation()
