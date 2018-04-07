import numpy as np
from findiff import *
from timeit import timeit

nx, ny, nz = 100, 100, 100


def stat_timeit(block, number=10):
    times = []
    for _ in range(number):
        times.append(timeit(block, number=1))
    times = np.array(times)
    return np.mean(times), np.min(times), np.max(times), np.std(times)


def non_uniform():
    x = np.r_[np.linspace(0,3, nx/2, endpoint=False), np.linspace(3, 10, nx/2)]
    y = np.r_[np.linspace(0,3, nx/2, endpoint=False), np.linspace(3, 10, nx/2)]
    z = np.r_[np.linspace(0,3, nx/2, endpoint=False), np.linspace(3, 10, nx/2)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.exp(-X ** 2 - Y ** 2 - Z ** 2)

    d_dx = FinDiff((0,), coords=[x, y, z])
    d_dx(f)


def uniform():

    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    z = np.linspace(0, 10, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.exp(-X ** 2 - Y ** 2 - Z ** 2)

    d_dx = FinDiff((0, x[1]-x[0]), acc=2)
    d_dx(f)


print(stat_timeit(non_uniform, 20))
print(stat_timeit(uniform, 20))