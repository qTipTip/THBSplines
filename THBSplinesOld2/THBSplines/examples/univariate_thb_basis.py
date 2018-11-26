import matplotlib.pyplot as plt
import numpy as np

from THBSplinesOld2.THBSplines.THBSplines.HierarchicalMesh import HierarchicalMesh
from THBSplinesOld2.THBSplines.THBSplines.HierarchicalSpace import HierarchicalSpace
from THBSplinesOld2.THBSplines.THBSplines.TensorProductSpace import TensorProductSpace

knots_old = [
    [1, 2, 3, 4, 5, 6, 7, 8]
]

d = [3]
dim = 1

S = TensorProductSpace(d, knots_old, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

marked_cells = [{3, 4, 5, 6}]
T.refine(marked_cells)

B1 = T.get_truncated_basis()

x = np.linspace(0, 9, 400)



ax1 = plt.subplot(2, 1, 1)
#plt.scatter(knots_old[0], np.zeros_like(knots_old[0]), s=100, zorder=-1, c='grey')
plt.plot(knots_old[0], np.zeros_like(knots_old[0]), marker='|', color='black', zorder=100, markersize=8, markevery=range(len(knots_old[0])))
ax1.set_title('THB-splines')
for b in B1:
    y = [b(X) for X in x]
    plt.plot(x, y, 'black')
plt.setp(ax1.get_xticklabels(), visible=False)
knots = [
    [1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
]
d = [3]
dim = 1

S = TensorProductSpace(d, knots, dim)
H = HierarchicalMesh(S.mesh)
T = HierarchicalSpace(H, S)

B2 = T.get_truncated_basis()

x = np.linspace(0, 9, 400)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.set_title('LR-splines')


for b in B2:
    y = [b(X) for X in x]
    plt.plot(x, y, 'black')

only_new_knots = np.setdiff1d(knots[0], knots_old[0])
plt.plot(knots_old[0], np.zeros_like(knots_old[0]), marker='|', color='black', zorder=100, markersize=8, markevery=range(len(knots_old[0])))
plt.plot(only_new_knots, np.zeros_like(only_new_knots), marker='|', color='black', zorder=100, markersize=4, markevery=range(len(only_new_knots)))
plt.tight_layout()
plt.savefig('thb_vs_lr_univariate_cubic.pdf')

plt.clf()

for i in range(3):
    ax1 = plt.subplot(3, 1, i+1)
    yy = [B2[i](X) for X in x]
    y = [B1[i](X) for X in x]
    plt.plot(x, y, '--', label='THB', c='black')
    plt.plot(x, yy, label='LR', c='black',)
    plt.ylim(0, 0.75)
    if i != 2:
        plt.setp(ax1.get_xticklabels(), visible=False)
    plt.scatter(knots_old[0], np.zeros_like(knots_old[0]), s=100, zorder=-1, c='grey')
    plt.scatter(knots[0], np.zeros_like(knots[0]), s=50, zorder=-1, c='grey')
    plt.legend()
plt.tight_layout()
plt.savefig('thb_vs_lr_single_same_plot{}.pdf'.format(i))
plt.clf()