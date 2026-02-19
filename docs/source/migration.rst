====================================
Migrating from FinDiff to Diff
====================================

Starting with version 0.11 the recommended API uses the ``Diff`` class.
The old ``FinDiff`` class still works but emits a ``DeprecationWarning``
and will be removed in a future release.

This page maps old patterns to their new equivalents.

.. contents:: On this page
   :local:


Basic Derivatives
-----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old (``FinDiff``)
     - New (``Diff``)
   * - ``FinDiff(0, dx)``
     - ``Diff(0, dx)``
   * - ``FinDiff(0, dx, 2)``
     - ``Diff(0, dx) ** 2``
   * - ``FinDiff(1, dy, 3)``
     - ``Diff(1, dy) ** 3``


Mixed Partial Derivatives
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old (``FinDiff``)
     - New (``Diff``)
   * - ``FinDiff((0, dx), (1, dy))``
     - ``Diff(0, dx) * Diff(1, dy)``
   * - ``FinDiff((0, dx, 2), (1, dy))``
     - ``Diff(0, dx)**2 * Diff(1, dy)``


Accuracy Control
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old (``FinDiff``)
     - New (``Diff``)
   * - ``FinDiff(0, dx, acc=4)``
     - ``Diff(0, dx, acc=4)``
   * - ``FinDiff(0, dx, 2, acc=6)``
     - ``Diff(0, dx, acc=6) ** 2``


Coefficient and Identity
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old
     - New
   * - ``Coef(x) * FinDiff(0, dx)``
     - ``x * Diff(0, dx)``
   * - ``Id()``
     - ``Identity()``


Operator Arithmetic
-------------------

Both old and new APIs support the same arithmetic. The main difference is
how you express higher-order and mixed derivatives:

**Old style:**

.. code:: python

    from findiff import FinDiff, Coef, Id

    L = Coef(2) * FinDiff(0, dx, 2) + Coef(3) * FinDiff(1, dy, 2)

**New style:**

.. code:: python

    from findiff import Diff

    L = 2 * Diff(0, dx)**2 + 3 * Diff(1, dy)**2


Lazy Grid Setting
-----------------

With ``Diff`` you can define operators without specifying the grid
and set it later:

.. code:: python

    L = Diff(0)**2 + Diff(1)**2
    L.set_grid({0: dx, 1: dy})
    result = L(f)

This was not possible with ``FinDiff``.


Matrix Representation
---------------------

Both APIs use the same ``matrix()`` method:

.. code:: python

    # Old
    mat = FinDiff(0, dx, 2).matrix(shape)

    # New
    mat = (Diff(0, dx) ** 2).matrix(shape)


Suppressing the Deprecation Warning
--------------------------------------

If you need to keep using ``FinDiff`` temporarily, suppress the warning:

.. code:: python

    import warnings
    warnings.filterwarnings('ignore', message='FinDiff is deprecated')
