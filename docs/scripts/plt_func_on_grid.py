import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


import numpy as np

plt.rcParams.update({"font.size": 18, "font.family":"serif",
                     "mathtext.fontset": "dejavuserif"})

xr = (0.7, 2)

x_fine = np.linspace(*xr, 100)
x = np.linspace(*xr, 7)

f_fine = np.sin(x_fine)
f = np.sin(x)

fig, ax = plt.subplots(layout="constrained")

ax.set_xlabel("")
ax.set_ylabel("$f\,(x)$")
labels = [r"$x_{i%+d}$" % (i-3) for i in range(len(x))]
labels[3] = r"$x_i$"

ax.set_xticks(ticks=x, labels=labels)
ax.set_yticks(ticks=[], labels=[])

ax.set_ylim(0.6, 1.02)

for xi, fi in zip(x, f):
    line = Line2D((xi, xi), (0, fi), color="grey", linestyle="--")
    ax.add_line(line)

ax.plot(x_fine, f_fine, "-")
ax.plot(x, f, 'o')

dx = x[1] - x[0]
arrow = FancyArrowPatch((x[1], f[1] - 0.1), (x[2], f[1] - 0.1), arrowstyle='<->', mutation_scale=20)
ax.add_patch(arrow)
ax.text(x[1] + dx/2, f[1] - 0.1, r"$\Delta x$", horizontalalignment="center", verticalalignment="bottom")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_linestyle("->")


#ax.plot(ax.xaxis.get_view_interval()[1], f[0], ">k", clip_on=False)
#ax.plot(0, 1, "^k", clip_on=False)

plt.savefig("func_on_grid.png")

plt.show()

