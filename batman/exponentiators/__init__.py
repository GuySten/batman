"""Tools for matrix exponentiation.

The solver of choice is :data:`ShiftedCram48`
(see :mod:`batman.exponentiators.cram.structural`):
CRAM-48 in 4 substeps that reuse their factorizations, block triangular
ordering of the nuclides with factorizations only of the strongly connected
blocks (none at all for decay), no row pivoting, and shift levels that make
every nuclide accurate relative to its own value (about 1e-13), not only
relative to the largest populations as plain CRAM is.

:class:`CramExponentiator` gives other settings, e.g.
``CramExponentiator(24, substeps=1, shifted=False)`` for plain CRAM-24, about
twice as fast and as accurate for every nuclide that holds more than about
1e-12 of the atoms. :data:`IPFCram16` and :data:`IPFCram48` are the previous
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

from batman.exponentiators.cram import *
from batman.exponentiators.protocol import Exponentiator as Exponentiator, FluxFunc as FluxFunc


IPFCram16 = IPFCramSolver(*cram16_coefficients())
IPFCram48 = IPFCramSolver(*cram48_coefficients())
ShiftedCram48 = CramExponentiator(order=48, substeps=4, shifted=True)
Magnus = CF3Magnus2IPFCRAM(*cram48_coefficients())
