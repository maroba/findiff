import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def label(ax, xy, names, font_size=12):
    text = ""
    for i, off in enumerate(xy):
        sign = "-" if off < 0 else "+"
        if off == 0:
            sign = ""
        value = abs(off) if abs(off) > 0 else ""
        text += "%s%s%s" % (names[i], sign, value)
        if i != len(xy) - 1:
            text += ","

    y = xy[1] - 0.3
    ax.text(xy[0], y, text, ha="center", family='sans-serif', size=font_size)


def plot_stencil(ax, stencil, xrange=None, yrange=None, label=label, font_size=12):
    stencil = np.array(stencil)

    if xrange is None:
        xrange = get_range(stencil, 0)
    if yrange is None:
        yrange = get_range(stencil, 1)

    for s in stencil:
        ax.add_patch(Circle(s, 0.1))
        label(ax, s, "ij", font_size)

    ax.set_aspect('equal')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_axis_off()
    ax.grid(visible=True, axis='both')


def plot_num_grid(ax, xrange, yrange):
    for x in range(*xrange):
        for y in range(*yrange):
            ax.add_patch(Circle((x, y), 0.1, facecolor='#DDDDDD'))


def get_range(stc, axis):
    ax_min = min(stc[:, axis])
    ax_max = max(stc[:, axis])
    return (ax_min - 1, ax_max + 1)

#stc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

from itertools import product
stc = list(product([-2, -1, 0, 1, 2], repeat=2))

fig = plt.figure()
ax = fig.add_subplot(111)

plot_num_grid(ax, (-3, 4), (-3, 4))
plot_stencil(ax, stc, font_size=8)
plt.show()
