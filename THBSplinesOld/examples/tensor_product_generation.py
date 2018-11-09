import numpy as np

from THBSplinesOld.src.Space import Space

knots = [
    [0, 0, 1, 2, 3, 4, 4],
    [0, 0, 1, 2, 3, 4, 4]
]

d = [1, 1]
dim = 2

S0 = Space(d, knots, dim)

N = 100
x = np.linspace(0, 3, N)
y = np.linspace(0, 4, N)

z = np.zeros((N, N))

X, Y = np.meshgrid(x, y)
S0.add_level()

print(S0.coefficient_matrices[0].shape)
print(len(S0.functions[0]))
print(len(S0.functions[1]))
for level in S0.function_to_child_map:
    print(S0.function_to_child_map[level])
print(S0.get_children_of_function(0, 2))
print(S0.get_parent_of_function(1, 3))
print(S0.function_to_child_map)
