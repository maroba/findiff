from findiff import FinDiff
from plt_grids_with_stencils import *

laplace = FinDiff(0, 1, 2) + FinDiff(1, 1, 2)

plt.rcParams.update({'font.size': 20, "mathtext.fontset": "cm"})
stencil_set = laplace.stencil((5, 5))

stencil = stencil_set.data[('C', 'C')]
fig, ax = plt.subplots(figsize=(8, 8))
plot_axes(ax, stencil=list(stencil.keys()), grid_kernel=[0], coeffs=list(stencil.values()), markersize=40)
fig.savefig("stencil_laplace2d_center.png")

stencil = stencil_set.data[('L', 'C')]
fig, ax = plt.subplots(figsize=(8, 8))
plot_axes(ax, stencil=list(stencil.keys()), grid_kernel=[0], coeffs=list(stencil.values()), markersize=40)
fig.savefig("stencil_laplace2d_border.png")

stencil = stencil_set.data[('L', 'L')]
fig, ax = plt.subplots(figsize=(8, 8))
plot_axes(ax, stencil=list(stencil.keys()), grid_kernel=[0], coeffs=list(stencil.values()), markersize=40)
fig.savefig("stencil_laplace2d_corner.png")

print(stencil_set)
