import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects
from itertools import product

from plt_grids_with_stencils import plot_axes

fig = plt.figure(figsize=(5, 5), layout="constrained")
ax = fig.add_axes([0.0, 0, 1, 1])
ax.margins(0.2, 0.2)


offsets = list(product([-1, 0, 1], repeat=2)) + [(-2, -2), (-2, 2), (2, 2), (2, -2)]
#offsets = list(product([-1, 0, 1], repeat=2)) + [(-2, 0), (2, 0), (0, 2), (0, -2)]

plot_axes(ax, offsets, None, [-2, -1, 0, 1, 2], with_coefs=False, with_offsets=False)
t = ax.text(
    0.5, 0.95, "findiff",
    path_effects=[path_effects.withSimplePatchShadow()],
    transform=fig.transFigure,
    fontsize=65,
    fontstyle="italic",
    fontweight="medium",
    fontfamily="serif",
    color="C0",
    horizontalalignment="center",
    verticalalignment="top"
)

t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
t.set_path_effects([
    path_effects.PathPatchEffect(
        offset=(4, -4), hatch='.......', facecolor='#A0A0A0'),
    path_effects.PathPatchEffect(
        edgecolor='white', linewidth=1.1, facecolor='C0')])

fig.savefig("findiff_logo.png")

plt.show()