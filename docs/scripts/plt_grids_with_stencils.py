import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from itertools import product


def switch_off_xy_axis(ax):
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    for direction in ['bottom', 'top', 'left', 'right']:
        ax.spines[direction].set_visible(False)


def plot_grid(ax, offsets, color, markersize=30):
    offsets = np.array(offsets)
    ax.plot(offsets[:, 0], offsets[:, 1], 'o', color=color, markersize=markersize)


def plot_offsets(ax, offsets, vshift=-0.15):
    for x, y in offsets:
        ax.text(x, y + vshift, "$ " + str((x, y)) + " $", verticalalignment="top", horizontalalignment="center")


def plot_coefficients(ax, offsets, coefs):
    if not coefs:
        return
    for (x, y), c in zip(offsets, coefs):
        ax.text(
            x, y, c,
            color="white",
            verticalalignment="center", horizontalalignment="center"
        )


def tex(s):
    return "$ " + str(s) + " $"


def plot_axes(ax, stencil, coeffs, grid_kernel=[-1, 0, 1], with_coefs=True, with_offsets=True, markersize=30):
    switch_off_xy_axis(ax)
    plot_grid(ax, list(product(grid_kernel, repeat=2)), color="#C0C0C0", markersize=markersize)
    plot_grid(ax, stencil, color="C0", markersize=markersize)
    if with_offsets:
        plot_offsets(ax, stencil)
    if with_coefs:
        plot_coefficients(ax, stencil, coeffs)


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 20, "mathtext.fontset": "cm"})

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_axes([0, 0.5, 0.5, 0.5])
    ax.margins(0.4, 0.4)
    plot_axes(ax, stencil=[(-1, 0), (0, 0), (1, 0)], coeffs=[1, -2, 1])

    ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
    ax.margins(0.4, 0.4)
    plot_axes(ax, stencil=[(0, -1), (0, 0), (0, 1)], coeffs=[1, -2, 1])

    ax = fig.add_axes([0.25, 0, 0.5, 0.5])
    ax.margins(0.4, 0.4)
    plot_axes(ax, stencil=[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)], coeffs=[1, -4, 1, 1, 1])

    fig.text(0.48, 0.72, "+", fontsize=34)
    fig.text(0.48, 0.5, "=", fontsize=34)

    fig.text(0.25, 0.95, r"$\frac{\partial^2}{\partial x^2}$", fontsize=30, horizontalalignment="center")
    fig.text(0.75, 0.95, r"$\frac{\partial^2}{\partial y^2}$", fontsize=30, horizontalalignment="center")

    fig.text(0.8, 0.25, r"$\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$",
             fontsize=30, horizontalalignment="center", verticalalignment="center")

    fig.savefig("composite_stencil.png")

    plt.show()
