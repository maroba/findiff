from setuptools import setup

setup(
    name='findiff',
    version='0.8.1',
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

    """,

    license='MIT',
    url='https://github.com/maroba/findiff',

    author='Matthias Baer',  # Optional
    author_email='mrbaer@t-online.de',  # Optional

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['finite-differences',  'numerical-derivatives', 'scientific-computing'],  # Optional
    packages=['findiff'],
    install_requires=['numpy', 'scipy'],  # Optional

)
