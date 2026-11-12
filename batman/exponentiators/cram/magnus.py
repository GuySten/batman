"""Commutator free Magnus method based exponentiator.
When solving dN/dt = A(t)*N(t) for a matrix A(t), A may not commute with itself
at different times. This is because A(t) = λ + σ(t) = λ + φ(t)σ. However, we
can use Lie algebra math to find approximations to the solution of this ODE.
Sometimes we can also write these approximations using commutator-free
expressions, which is valuable in terms of computational resources.

One such way would be to use the method described in:
Yun Cai, Xingjie Peng, Qing Li, Lin Du, Lingfang Yang, "Solving point burnup
equations by Magnus method", Nuclear Engineering and Technology, 51, p.949-953,
2019.

These authors describe a commutator free magnus method based on CRAM which is
correct up to 5th order in t. This version uses 2 CRAM operations per time step.

"""
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp

from batman.exponentiators.protocol import FluxFunc
from batman.units import Second

from .ipfcram import IPFCramSolver, _ipf_cram

__all__ = ['CF3Magnus2IPFCRAM']


def _weighted_mean(a: Iterable[float], f: Iterable[float]) -> float:
    return sum(va * vf for va, vf in zip(a, f)) / sum(a)


class CF3Magnus2IPFCRAM(IPFCramSolver):
    """A CRAM exponentiator that is based on the commutator free Magnus method

    """

    def __call__(self, d: sp.csr_matrix, r: sp.csr_matrix,
                 flux: FluxFunc, n0: np.ndarray, dt: Second) -> np.ndarray:
        """Use ME3/CF3 magnus method from Yun Cai et al (see above) to
        approximate the solution of dn/dt = (d + flux(t)*r) * n.

        Parameters
        ----------
        d - Decay sparse matrix λ
        r - Reaction rate sparse matrix σ
        flux - Scalar flux as a function of time φ(t)
        Together they represent A(t) = λ + φ(t)σ
        n0 - The initial density of each isotope as a ndarray.
        dt - Size of time step, in seconds.

        Returns
        -------
        A vector of mixtures at the end of the time step.

        """
        h1, h2 = 0.5 + np.sqrt(3.) / 6, 0.5 - np.sqrt(3.) / 6
        f1, f2 = flux(h1 * dt), flux(h2 * dt)
        a1, a2 = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
        bdiff = _weighted_mean((a2, a1), (f1, f2))
        bsame = _weighted_mean((a1, a2), (f1, f2))
        m = sp.csr_matrix((dt / 2) * (d + bdiff * r), dtype=np.float64)
        y = _ipf_cram(m, np.array(n0, dtype=np.float64),
                      alpha0=self.alpha0, alpha=self.alpha, theta=self.theta,
                      solver=self.solver, **self.kw)
        m = sp.csr_matrix((dt / 2) * (d + bsame * r), dtype=np.float64)
        return _ipf_cram(m, y,
                         alpha0=self.alpha0, alpha=self.alpha, theta=self.theta,
                         solver=self.solver, **self.kw)
