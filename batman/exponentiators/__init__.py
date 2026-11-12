"""Tools for matrix exponentiation.

Two layers, as everywhere in this package: a matrix exponential, which
approximates ``exp(A t) v`` and knows nothing about burnup
(:class:`MatrixExponential`), and an exponentiator, which turns a step's decay
and reaction matrices and flux into that matrix (:class:`Exponentiator`).

The solver of choice is :data:`ShiftedCram48`, ``ConstantFlux(StructuralCram())``:
:class:`StructuralCram` (see :mod:`batman.exponentiators.cram.structural`) is
CRAM-48 in 4 substeps that reuse their factorizations, block triangular
ordering with factorizations only of the strongly connected blocks (none at
all for decay), no row pivoting, and shift levels that make every entry
accurate relative to its own value (about 1e-13), not only relative to the
largest entry as plain CRAM is. :class:`ConstantFlux` forms the matrix of a
step at constant flux.

:class:`StructuralCram` takes other settings, e.g.
``ConstantFlux(StructuralCram(24, substeps=1, shifted=False))`` for plain
CRAM-24, about twice as fast and as accurate for every nuclide that holds more
than about 1e-12 of the atoms. :data:`IPFCram16` and :data:`IPFCram48` are the previous
solvers (a SciPy ``spsolve`` per pole and step), kept for comparison.

Examples
--------
>>> import scipy.sparse as sp
>>> import numpy as np
>>> d = sp.csr_matrix(np.array([[-np.log(2), 0.], [0., -np.log(2)]]))
>>> r = sp.csr_matrix(np.zeros((2,2)))
>>> f = lambda x: 0.
>>> n0 = np.ones(2)
>>> dt = 1
>>> exp: IPFCramSolver = IPFCram16
>>> bool(np.isclose(exp(d, r, f, n0, dt)[0], 0.5))
True

"""

from batman.exponentiators.constant_flux import ConstantFlux as ConstantFlux
from batman.exponentiators.cram import *
from batman.exponentiators.protocol import (
    Exponentiator as Exponentiator,
    FluxFunc as FluxFunc,
    MatrixExponential as MatrixExponential,
)


IPFCram16 = IPFCramSolver(*cram16_coefficients())
IPFCram48 = IPFCramSolver(*cram48_coefficients())
ShiftedCram48 = ConstantFlux(StructuralCram(order=48, substeps=4, shifted=True))
Magnus = CF3Magnus2IPFCRAM(*cram48_coefficients())
