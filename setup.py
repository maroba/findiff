#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import re

from setuptools import setup, find_packages

name = 'findiff'


def get_version():
    file_name = os.path.join(name, '__init__.py')
    with open(file_name) as init_file:
        content = init_file.readlines()
    for line in content:
        match = re.match('^ *__version__ *= *[\'"]([^\'"]+)', line)
        if match:
            return match.group(1)
    raise Exception('Could not parse version string.')


setup(
    name=name,
    version=get_version(),
    description='A Python package for finite difference derivatives in any number of dimensions.',
    long_description="""A Python package for finite difference derivatives in any number of dimensions.

    Features:

        * Differentiate arrays of any number of dimensions along any axis
        * Partial derivatives of any desired order
        * Accuracy order can be specified
        * Accurate treatment of grid boundary
        * Includes standard operators from vector calculus like gradient, divergence and curl
        * Can handle uniform and non-uniform grids
        * Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
        * Fully vectorized for speed
        * Calculate raw finite difference coefficients for any order and accuracy for uniform and non-uniform grids
        * _New in version 0.7:_ Generate matrix representations of arbitrary linear differential operators
        * _New in version 0.8:_ Solve partial differential equations with Dirichlet or Neumann boundary conditions
        * _New in version 0.9:_ Generate differential operators for generic stencils

    """,

    license='MIT',
    url='https://github.com/maroba/findiff',

    author='Matthias Baer',
    author_email='matthias.r.baer@googlemail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=['finite-differences', 'numerical-derivatives', 'scientific-computing'],
    packages=find_packages(exclude=("tests",)),
    package_dir={name: name},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'sympy'],
    setup_requires=["pytest-runner"],
    python_requires=">=3.6",
    tests_require=["pytest"],
    platforms=['ALL'],
)
