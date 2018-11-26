from src.hierarchical_space import HierarchicalSpace
import logging


def refine(hspace: HierarchicalSpace, marked_entities, type='cells') -> HierarchicalSpace:
    logging.info("""
    Refining hierarchical space
    ===========================
    space.nlevels = {}
    hmesh.nlevels = {}
    """.format(hspace.nlevels, hspace.mesh.nlevels))
    marked_cells = marked_entities
    new_cells = hspace.mesh.refine(marked_cells)
    marked_functions = hspace.functions_to_deactivate_from_cells(marked_cells)
    hspace.refine(marked_functions, new_cells)

    logging.info("""
    After refinement, the space has
    ===============================
    space.nfuncs = {}
    space.nelems = {}
    """.format(hspace.nfuncs, hspace.mesh.nel))
    return hspace


if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3, 4, 5, 6]}
    T = refine(T, marked_cells)
    marked_cells = {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 3, 4, 5, 6]}
    T = refine(T, marked_cells)
    T.mesh.plot_cells()
    print(T)