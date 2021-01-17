import matplotlib.pyplot as plt
import numpy as np
from findiff import FinDiff


def get_started_plot_1():
    x = np.linspace(-np.pi, np.pi, 31)
    dx = x[1] - x[0]
    f = np.sin(x)
    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

    plt.xlabel('x')
    plt.ylabel('f, df_dx')
    plt.plot(x, f, '-o', label='f=sin(x)')
    plt.plot(x, df_dx, '-o', label='df_dx' )
    plt.grid()
    plt.legend()

    plt.savefig('get_started_plot_1.png')


if __name__ == '__main__':
    get_started_plot_1()
