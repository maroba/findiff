import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_stencil(ax, grid, stencil, labels, stubs=[-0.5, 0.5], coefs=None, no_x=False):

    markersize = 16
    downshift = -0.01

    if coefs:
        markersize = 24
        downshift = -0.02


    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.set_ylim(-0.1, 0.1)

    for direction in ['bottom', 'top', 'left', 'right']:
        ax.spines[direction].set_visible(False)

    ax.add_line(mpl.lines.Line2D([min(grid) + stubs[0], max(grid)+ stubs[1]], [0, 0], color='black'))
    ax.plot(grid, [0] * len(grid), 'o', color='#C0C0C0', markersize=markersize)
    ax.plot(stencil, [0] * len(stencil), 'o', color='C0', markersize=markersize)

    if coefs:
        for x, coef in zip(stencil, coefs):
            ax.text(x, 0, "$%s$" % coef, fontsize="x-small",
                    horizontalalignment="center", verticalalignment="center", color="white")

    for x, label in zip(grid, labels):
        if no_x:
            txt = "$ " + label + " $"
        else:
            txt = "$x_{%s}$" % label
        txt = ax.text(x, downshift, txt, clip_on=True, verticalalignment="top", horizontalalignment="center")
        if no_x:
            txt.set_fontsize("x-small")
            txt.set_y(1.5*downshift)

plt.rcParams.update({'font.size': 20, "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(10, 2), layout='constrained')
plot_stencil(ax, list(range(-3, 4)), list(range(-1, 2)), labels=["k-3", "k-2", "k-1", "k", "k+1", "k+2", "k+3"])
fig.savefig("stencil_1d_center.png")

fig, ax = plt.subplots(figsize=(10, 2), layout='constrained')
plot_stencil(ax, list(range(8)), list(range(4)), labels=["0", "1", "2", "3", "4", "5", "6", "7"], stubs=[0, 0.5])
fig.savefig("stencil_1d_forward.png")

fig, ax = plt.subplots(figsize=(10, 2))
plot_stencil(ax, list(range(-3, 4)), list(range(-1, 2)), labels=["k-3", "k-2", "k-1", "k", "k+1", "k+2", "k+3"],
             coefs=[1, -2, 1]
             )
fig.savefig("stencil_1d_center_with_weights.png")

fig, ax = plt.subplots(figsize=(5, 2))
plot_stencil(ax, list(range(-1, 2)), list(range(-1, 2)), labels=["-1", "0", "1"],
             coefs=[1, -2, 1], no_x=True
             )
fig.savefig("stencil_1d_center_compact.png")

plt.show()