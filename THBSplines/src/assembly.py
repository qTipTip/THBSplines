import numpy as np
import scipy.integrate
from tqdm import tqdm


def hierarchical_mass_matrix(T):
    mesh = T.mesh

    n = T.nfuncs
    M = np.zeros((n, n))

    ndofs_u = 0
    ndofs_v = 0
    C = T.create_subdivision_matrix('full')
    for level in range(mesh.nlevels):
        ndofs_u += T.nfuncs_level[level]
        ndofs_v += T.nfuncs_level[level]

        if mesh.nel_per_level[level] > 0:
            M_local = local_mass_matrix(T, level)

            dofs_u = range(ndofs_u)
            dofs_v = range(ndofs_v)

            ix = np.ix_(dofs_u, dofs_v)
            M[ix] += C[level].T @ M_local @ C[level]

    return M


def translate_points(points, cell, weights):
    """
    Translates the gauss-quadrature points to the cell
    :param points:
    :param cell:
    :return:
    """
    n = len(points)
    dim = cell.shape[0]
    quad_points = np.zeros((dim, n))
    quad_weights = np.zeros((dim, n))
    quad_weights[:] = weights

    for i in range(dim):
        for j in range(n):
            quad_points[i, j] = 0.5 * (points[j] + 1) * (cell[i, 1] - cell[i, 0]) + cell[i, 0]
    weights = np.prod(np.stack(np.meshgrid(*quad_weights), -1).reshape(-1, dim), axis=1)
    points = np.stack(np.meshgrid(*quad_points), -1).reshape(-1, dim)
    area_cell = np.prod(np.diff(cell[:]))

    return points, weights, area_cell

def local_mass_matrix(T, level):
    active_cells = T.mesh.meshes[level].cells[T.mesh.aelem_level[level]]

    ndofs_u = T.spaces[level].nfuncs
    ndofs_v = T.spaces[level].nfuncs

    M = np.zeros((ndofs_u, ndofs_v))

    points, weights = np.polynomial.legendre.leggauss(T.spaces[level].degrees[0] + 1)
    for cell in tqdm(active_cells):
        qp, qw, area = translate_points(points, cell, weights)
        active_basis_functions = T.spaces[level].get_functions_on_rectangle(cell)
        for i in active_basis_functions:
            bi = T.spaces[level].basis[i]
            for j in active_basis_functions:
                bj = T.spaces[level].basis[j]

                val = integrate(bi, bj, qp, qw, area)
                M[i, j] += val

    return M


def integrate(bi, bj, points, weights, area):
    values_i = bi(points)
    values_j = bj(points)
    I = 0
    for i in range(len(points)):
        I += weights[i] * values_i[i] * values_j[i]

    dim = points.shape[1]
    I *= area / 2**dim

    return I

def integrate_smart(bi, bj, cell):
    dim = cell.shape[0]

    if dim == 1:
        return scipy.integrate.quad(lambda x: bi(x) * bj(x), cell[0, 0], cell[0, 1])[0]
    elif dim == 2:
        val, info = scipy.integrate.dblquad(lambda y, x: bi(np.array([x, y])) * bj(np.array([x, y])), cell[0, 0], cell[0, 1],
                                   lambda x: cell[1, 0], lambda x: cell[1, 1])
