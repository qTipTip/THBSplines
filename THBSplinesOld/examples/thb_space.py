from THBSplinesOld.src.THBSpace import THBSpace

knots = [
    [0, 0, 0, 1, 2, 3, 3, 3],
]

d = [2]
dim = 1

T = THBSpace(d, knots, dim)


#T.refine([], type='cells')
T.plot_hierarchical_basis()
T.refine([[], [0, 1, 2, 3]], type='cells')
T.plot_hierarchical_basis()