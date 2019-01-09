import THBSplines as thb
import numpy as np

NUMBER_OF_SMALLEST_EIGVALS_TO_COMPUTE = 20
NUMBER_OF_REFINEMENTS = 4
GRID_SIZE = 11
PADDING = False
FOLDER = 'DataExampleExtraRefinement'
EPS = np.spacing(1)

degre = [3, 3]
knots, step = np.linspace(0, 1, GRID_SIZE, dtype=np.float64, retstep=True)
left = np.arange(knots[0] - degre[0] * step, knots[0] - step, step)
right = np.array([knots[-1] + (i + 1) * step for i in range(degre[0])])
knots = np.concatenate((left, knots, right))
if PADDING:
    knots = np.pad(knots, pad_width=degre[0], mode='edge')
knots = np.array([knots, knots])
DOMAIN = np.array([[0, 1 + EPS], [0, 1 + EPS]])

T = thb.HierarchicalSpace(degrees=degre, dim=2, knots=knots)

filename = FOLDER + '/thb_large_ghost/'
filetype = '.pdf'

# order of the gaussian quadrature rule. Order of four integrates cubic splines exactly over each element.
order = 4

# Define the regions marked for hierarchical refinement
refinement_regions = [
    [],
    [[0.3, 0.7000001], [0.3, 0.7000001]],
    [[0.4, 0.6000001], [0.4, 0.6000001]],
    [[0.45, 0.55000001], [0.45, 0.55000001]],
    [[0.475, 0.525000001], [0.475, 0.525000001]],
    [[0.4875, 0.5125000001], [0.4875, 0.5125000001]]
]

# Lists for storing the finite element matrices and condition numbers
mass_matrices = []
mass_cond = []
stiffness_matrices = []
stiffness_cond = []  # using smallest eig val
stiffness_cond_avg = []  # using avg of k smallest eig vals
ndofs = []
min_vals = []
max_vals = []
top = 0.7
bot = 0.3
width = step
full_width = 0.4

cells = {}

for refinement_level in range(NUMBER_OF_REFINEMENTS):
    if refinement_level != 0:
        region = [[bot - EPS, top + EPS], [bot - EPS, top + EPS]]
        cells[refinement_level - 1] = T.refine_in_rectangle(np.array(region, dtype=np.float64),
                                                            refinement_level - 1)
        T = thb.refine(T, cells)
        bot += 0.1 / 2 ** (refinement_level - 1)
        top -= 0.1 / 2 ** (refinement_level - 1)

    M = thb.hierarchical_mass_matrix(T, order=order, integration_region=DOMAIN)
    print('Hello', 1 - M.nnz / M.size)
    #A = thb.hierarchical_stiffness_matrix(T, order=order - 1, integration_region=DOMAIN)
