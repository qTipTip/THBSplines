import numpy as np
import scipy.sparse as sp
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def create_subdivision_matrix(hspace: HierarchicalSpace) -> dict:
    """
    Returns hspace.nlevels-1 matrices used for representing coarse B-splines in terms of the finer B-splines.
    :param hspace: HierarchicalSpace containing the needed information
    :return: a dictionary mapping
    """

    mesh = hspace.mesh

    C = {}
    C[0] = sp.identity(hspace.spaces[0].nfuncs, format='lil')
    C[0] = C[0][:, hspace.afunc_level[0]]

    for level in range(1, hspace.nlevels):
        I = sp.identity(hspace.spaces[level].nfuncs, format='lil')
        aux = sp.lil_matrix(hspace.get_basis_conversion_matrix(level-1))
        C[level] = sp.hstack([aux @ C[level - 1], I[:, hspace.afunc_level[level]]])
    return C



if __name__ == '__main__':
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3],
        [0, 0, 0, 1, 2, 3, 3, 3]
    ]
    d = 2
    degrees = [2, 2]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3]}
    T = refine(T, marked_cells)
    C = create_subdivision_matrix(T)
    N = 30
    x = np.linspace(0, 3, N)
    y = np.linspace(0, 3, N)
    z = np.zeros((N, N))

    X, Y = np.meshgrid(x, y)
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
                z[i, j] = f(np.array([x[i], y[j]]))


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = T.mesh.plot_cells(return_fig=True)
        axs = Axes3D(fig)
        axs.plot_surface(X, Y, z, cmap='viridis')
        plt.show()
