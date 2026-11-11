"""Chebyshev Rational Approximation Method module

Implements two different forms of CRAM.

Taken almost as is from OpenMC, according to its MIT license, given here as
required.

Copyright (c) 2011-2026 Massachusetts Institute of Technology, UChicago Argonne
LLC, and OpenMC contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from batman.exponentiators.protocol import FluxFunc
from batman.units import Second

__all__ = ["IPFCramSolver", '_ipf_cram']


def _ipf_cram(m: sp.csr_matrix, v: np.ndarray, *,
              alpha0: float, alpha: Iterable[complex], theta: Iterable[complex],
              solver: Callable, **kwargs):
    ident = sp.eye(m.shape[0], format='csr')
    for alph, thet in zip(alpha, theta):
        v += 2 * np.real(alph * solver(m - thet * ident, v, **kwargs))
    return v * alpha0


class IPFCramSolver:
    r"""CRAM depletion expo that uses incomplete partial factorization

    Provides a :meth:`__call__` that utilizes an incomplete
    partial factorization (IPF) for the Chebyshev Rational Approximation
    Method (CRAM), as described in the following paper:
    M. Pusa, "Higher-Order Chebyshev Rational Approximation Method and
    Application to Burnup Equations", https://doi.org/10.13182/NSE15-26,
    Nucl. Sci. Eng., 182:3, 297-318.

    Attributes
    ----------
    alpha : numpy.ndarray
        Complex residues of poles :attr:`theta` in the incomplete partial
        factorization. Denoted as :math:`\tilde{\alpha}`
    theta : numpy.ndarray
        Complex poles :math:`\theta` of the rational approximation
    alpha0 : float
        Limit of the approximation at infinity

    """

    def __init__(self,
                 alpha: Sequence[complex],
                 theta: Sequence[complex],
                 alpha0: float,
                 solver: Callable = sla.spsolve,
                 **kwargs):
        """Initialization according to CRAM level.

        Parameters
        ----------
        alpha : numpy.ndarray
            Complex residues of poles used in the factorization. Must be a
            vector with even number of items.
        theta : numpy.ndarray
            Complex poles. Must have an equal size as ``alpha``.
        alpha0 : float
            Limit of the approximation at infinity
        solver: Callable
            A function that solves Ax=b for some square matrix A and some vector b.
            This function can require additional keyword parameters.

            .. danger::
                According to Maria Pusa [1], iterative solvers such as BiCGStab and
                GMRES are not suitable for burnup matrices. A check has shown that a full
                decay problem with ~1000 isotopes gives a garbage solution with BiCGStab.
                Therefore, one should choose their solver carefully.
                [1] Maria Pusa & Jaakko Leppänen (2013) Solving Linear Systems
                with Sparse Gaussian Elimination in the Chebyshev Rational
                Approximation Method, Nuclear Science and Engineering, 175:3, 250-258,
                DOI: 10.13182/NSE12-52

        **kwargs : Keyword arguments to send to the solver.

        """
        assert len(alpha) == len(theta)
        assert not (len(alpha) & 1)  # Checks for even length
        self.alpha = alpha
        self.theta = theta
        self.alpha0 = alpha0
        self.solver = solver
        self.kw = kwargs

    def __call__(self, d: sp.csr_matrix, r: sp.csr_matrix,
                 flux: FluxFunc, n0: np.ndarray, dt: Second):
        """Solve depletion equations using IPF CRAM

        Parameters
        ----------
        d,r : scipy.sparse.csr_matrix
            Sparse transmutation matrices ``A[j, i]`` desribing rates at
            which isotope ``i`` transmutes to isotope ``j``. d is the decay
            transmutation matrix and r is the induced transmutation matrix.
        flux : Function that returns the flux level at different times.
        n0 : numpy.ndarray
            Initial compositions, typically given in number of atoms in some
            material or an atom density
        dt : float
            Time [s] of the specific interval to be solved

        Returns
        -------
        numpy.ndarray
            Final compositions after ``dt``

        """
        m = sp.csr_matrix(dt * (d + flux(0.) * r), dtype=np.float64)
        y = np.array(n0, dtype=np.float64)
        return _ipf_cram(m, y,
                         alpha0=self.alpha0, alpha=self.alpha, theta=self.theta,
                         solver=self.solver, **self.kw)
