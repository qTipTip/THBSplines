from THBSplinesOld.src.Mesh import Mesh

knots = [
    [0, 0, 1, 2, 3, 3],
    [0, 0, 1, 2, 3, 3]
]

d = [1, 1]
dim = 2

M = Mesh(knots, d, dim)

M.add_level()
print(M.cell_to_children_map)
print(M.get_children_of_cell(0, 2))
print(M.get_parent_of_cell(1, 4))