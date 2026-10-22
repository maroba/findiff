import matplotlib.pyplot as plt
import numpy as np
from findiff import CompactScheme, Diff


def get_started_plot_1():
    x = np.linspace(-np.pi, np.pi, 31)
    dx = x[1] - x[0]
    f = np.sin(x)
    d_dx = Diff(0, dx)
    df_dx = d_dx(f)

    plt.xlabel('x')
    plt.ylabel('f, df_dx')
    plt.plot(x, f, '-o', label='f=sin(x)')
    plt.plot(x, df_dx, '-o', label='df_dx' )
    plt.grid()
    plt.legend()

    plt.savefig('get_started_plot_1.png')


def compact_modified_wavenumber():
    """Modified wavenumber of explicit, maximal-order compact and spectral schemes."""
    w = np.linspace(0, np.pi, 400)
    schemes = [
        ("explicit, 7 points (6th order)", CompactScheme._solve_jointly(1, 0, 3)),
        ("compact=5, 7 points (10th order)", CompactScheme._solve_jointly(1, 2, 3)),
        ("compact=5, 7 points, spectral (4th order)", CompactScheme.spectral(1, 5, 7, acc=4)),
    ]

    plt.figure(figsize=(6, 4.5))
    plt.plot(w, w, "k--", label="exact")
    for label, scheme in schemes:
        plt.plot(w, scheme.modified_wavenumber(w), label=label)
    plt.xlabel("wavenumber $w = kh$")
    plt.ylabel("modified wavenumber $w'$")
    plt.xlim(0, np.pi)
    plt.ylim(0, np.pi)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/compact_modified_wavenumber.png", dpi=120)
    plt.close()


if __name__ == '__main__':
    get_started_plot_1()
    compact_modified_wavenumber()
