import matplotlib.pyplot as plt

from plt_grids_with_stencils import *

plt.rcParams.update({'font.size': 20, "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(8, 8))
plot_axes(ax, stencil=[(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)], coeffs=[1, -4, 1, 1, 1], markersize=40)
fig.savefig("laplace2d.png")

fig, ax = plt.subplots(figsize=(8, 8))
plot_axes(ax, stencil=[(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)], coeffs=[-2, 0.5, 0.5, 0.5, 0.5], markersize=40)
fig.savefig("laplace2d-x.png")
