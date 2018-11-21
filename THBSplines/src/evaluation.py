import numpy as np
import scipy.sparse as sp
from src.hierarchical_space import HierarchicalSpace
from src.refinement import refine


def create_subdivision_matrix(hspace: HierarchicalSpace, mode='reduced') -> dict:
    """
    Returns hspace.nlevels-1 matrices used for representing coarse B-splines in terms of the finer B-splines.
    :param hspace: HierarchicalSpace containing the needed information
    :return: a dictionary mapping
    """

    mesh = hspace.mesh

    C = {}
    C[0] = sp.identity(hspace.spaces[0].nfuncs, format='lil')
    C[0] = C[0][:, hspace.afunc_level[0]]


    if mode == 'reduced':
        func_on_active_elements = hspace.spaces[0].get_basis_functions(mesh.aelem_level[0])
        func_on_deact_elements = hspace.spaces[0].get_basis_functions(mesh.delem_level[0])
        func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)

        for level in range(1, hspace.nlevels):
            # I = sp.identity(hspace.spaces[level].nfuncs, format='lil')
            # I = I[:, hspace.afunc_level[level]]

            I_row_idx = hspace.afunc_level[level]
            I_col_idx = list(range(hspace.nfuncs_level[level]))

            data = np.ones(len(I_col_idx))
            I = sp.coo_matrix((data, (I_row_idx, I_col_idx)))
            aux = sp.lil_matrix(hspace.get_basis_conversion_matrix(level - 1))[:, func_on_deact_elements]

            func_on_active_elements = hspace.spaces[level].get_basis_functions(mesh.aelem_level[level])
            func_on_deact_elements = hspace.spaces[level].get_basis_functions(mesh.delem_level[level])
            func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)
            C[level] = sp.hstack([aux @ C[level - 1], I])
        return C
    else:
        for level in range(1, hspace.nlevels):
            I = sp.identity(hspace.spaces[level].nfuncs, format='lil')
            aux = sp.lil_matrix(hspace.get_basis_conversion_matrix(level-1))
            C[level] = sp.hstack([aux @ C[level - 1], I[:, hspace.afunc_level[level]]])
        return C



if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
    ]
    d = 1
    degrees = [1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0]}
    T = refine(T, marked_cells)
    C = create_subdivision_matrix(T, mode='full')
    N = 30
    x = np.linspace(0, 3, N)
    z = np.zeros(N)

    B = T.spaces[-1].basis

    c = C[T.nlevels-1]
    c = c.toarray()

    for i in range(T.nfuncs):
        u = np.zeros(T.nfuncs)
        u[i] = 1
        u_fine = c @ u


        f = T.spaces[T.nlevels - 1].construct_function(u_fine)

        for i in range(N):
            for j in range(N):
                z[i, j] += f(np.array([x[i], y[j]]))


    import matplotlib.pyplot as plt
    fig = T.mesh.plot_cells(return_fig=True)
    plt.contourf(x, y, z)
    plt.show()
