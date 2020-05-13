import numpy as np
import scipy.integrate
import scipy.sparse as sp
from THBSplines.src_c.BSpline import integrate as cintegrate, integrate_grad as cintegrate_grad
from tqdm import tqdm


def hierarchical_mass_matrix(T, order=None, integration_region=None, mode='reduced'):
    """
    Computes the mass matrix for the hierarchical spline space T.

    :param T: hierarchical space
    :param order: Integration order
    :param integration_region: If None, integrate over the whole domain, else restrict to relevant cells
    :param mode: whether to use reduced or full projection matrices. Only reduced currently supported.
    :return: sparse mass matrix
    """

    mesh = T.mesh

    n = T.nfuncs
    M = sp.lil_matrix((n, n), dtype=np.float64)

    ndofs_u = 0
    ndofs_v = 0
    C = T.create_subdivision_matrix(mode)
    for level in range(mesh.nlevels):
        if integration_region is None:
            element_indices = None
        else:
            element_indices = T.refine_in_rectangle(integration_region, level)

        ndofs_u += T.nfuncs_level[level]
        ndofs_v += T.nfuncs_level[level]

        if mesh.nel_per_level[level] > 0:
            M_local = local_mass_matrix(T, level, order, element_indices= element_indices)

            dofs_u = range(ndofs_u)
            dofs_v = range(ndofs_v)

            ix = np.ix_(dofs_u, dofs_v)
            M[ix] += C[level].T @ M_local @ C[level]

    return M


def hierarchical_stiffness_matrix(T, order=None, integration_region = None, mode='reduced'):
    """
    Computes the stiffness matrix for the hierarchical spline space T.

    :param T: hierarchical space
    :param order: Integration order
    :param integration_region: If None, integrate over the whole domain, else restrict to relevant cells
    :param mode: whether to use reduced or full projection matrices. Only reduced currently supported.
    :return: sparse stiffness matrix
    """
    mesh = T.mesh

    n = T.nfuncs
    M = sp.lil_matrix((n, n), dtype=np.float64)

    ndofs_u = 0
    ndofs_v = 0
    C = T.create_subdivision_matrix(mode)
    for level in range(mesh.nlevels):
        if integration_region is None:
            element_indices = None
        else:
            element_indices = T.refine_in_rectangle(integration_region, level)
        ndofs_u += T.nfuncs_level[level]
        ndofs_v += T.nfuncs_level[level]

        if mesh.nel_per_level[level] > 0:
            M_local = local_stiffness_matrix(T, level, order, element_indices=element_indices)

            dofs_u = range(ndofs_u)
            dofs_v = range(ndofs_v)

            ix = np.ix_(dofs_u, dofs_v)
            M[ix] += C[level].T @ M_local @ C[level]

    return M


def translate_points(points, cell, weights):
    """
    Translates the gauss-quadrature points and weights to the cell

    :param points: quadrature points
    :param cell: cell to translate to 
    :param weights: quadrature weights
    :return: translated quadrature points
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


def local_mass_matrix(T, level, order=None, element_indices=None):
    """
    Computes the mass matrix on a given level

    :param T: HierarchicalSpace object
    :param level: hierarchical level
    :param order: integration order
    :param element_indices: elements to integrate over. If None, all elements on this level will be integrated over.
    :return: local mass matrix
    """
    if element_indices is None:
        element_indices = range(T.mesh.meshes[level].nelems)
    active_cells_i = np.intersect1d(T.mesh.aelem_level[level], element_indices)
    active_cells = T.mesh.meshes[level].cells[active_cells_i]
    ndofs_u = T.spaces[level].nfuncs
    ndofs_v = T.spaces[level].nfuncs

    M = sp.lil_matrix((ndofs_u, ndofs_v), dtype=np.float64)

    if order is None:
        order = T.spaces[level].degrees[0] + 1

    points, weights = np.polynomial.legendre.leggauss(order)

    for cell in tqdm(active_cells, desc=f"level = {level}"):
        qp, qw, area = translate_points(points, cell, weights)
        dim = qp.shape[1]
        active_basis_functions = T.spaces[level].get_functions_on_rectangle(cell)
        for glob_ind_i, i in enumerate(active_basis_functions):
            bi = T.spaces[level].construct_B_spline(i)
            bi_values = bi(qp)
            for glob_ind_j, j in enumerate(active_basis_functions[glob_ind_i:]):
                bj = T.spaces[level].construct_B_spline(j)
                bj_values = bj(qp)
                val = cintegrate(bi_values, bj_values, qw, area, dim)
                M[i, j] += val
                if i == j:
                    continue
                M[j, i] += val

    return M


def local_stiffness_matrix(T, level, order=None, element_indices=None):
    """
    Computes the stiffness matrix on a given level

    :param T: HierarchicalSpace object
    :param level: hierarchical level
    :param order: integration order
    :param element_indices: elements to integrate over. If None, all elements on this level will be integrated over.
    :return: local stiffness matrix
    """
    if element_indices is None:
        element_indices = range(T.mesh.meshes[level].nelems)
    active_cells_i = np.intersect1d(T.mesh.aelem_level[level], element_indices)
    active_cells = T.mesh.meshes[level].cells[active_cells_i]
    ndofs_u = T.spaces[level].nfuncs
    ndofs_v = T.spaces[level].nfuncs

    M = sp.lil_matrix((ndofs_u, ndofs_v), dtype=np.float64)

    if order is None:
        order = T.spaces[level].degrees[0] + 1

    points, weights = np.polynomial.legendre.leggauss(order)
    for cell in tqdm(active_cells, desc=f"level = {level}"):
        qp, qw, area = translate_points(points, cell, weights)
        dim = qp.shape[1]
        active_basis_functions = T.spaces[level].get_functions_on_rectangle(cell)
        for glob_ind_i, i in enumerate(active_basis_functions):
            bi = T.spaces[level].construct_B_spline(i)
            bi_values = bi.grad(qp)
            for glob_ind_j, j in enumerate(active_basis_functions[glob_ind_i:]):
                bj = T.spaces[level].construct_B_spline(j)
                bj_values = bj.grad(qp)
                val = cintegrate_grad(bi_values, bj_values, qw, area, dim)
                M[i, j] += val
                if i == j:
                    continue
                M[j, i] += val

    return M


def integrate(bi_values, bj_values, weights, area, dim):
    I = 0
    for i in range(len(bi_values)):
        I += weights[i] * bi_values[i] * bj_values[i]

    I *= area / 2 ** dim

    return I


def integrate_smart(bi, bj, cell):
    dim = cell.shape[0]

    if dim == 1:
        return scipy.integrate.quad(lambda x: bi(x) * bj(x), cell[0, 0], cell[0, 1])[0]
    elif dim == 2:
        val, info = scipy.integrate.dblquad(lambda y, x: bi(np.array([x, y])) * bj(np.array([x, y])), cell[0, 0],
                                            cell[0, 1],
                                            lambda x: cell[1, 0], lambda x: cell[1, 1])
