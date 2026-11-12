"""An exponentiator for steps over which the flux is constant, built on any
matrix exponential.

"""
import numpy as np
from scipy.sparse import csr_matrix

from batman.exponentiators.protocol import FluxFunc, MatrixExponential
from batman.units import Second

__all__ = ["ConstantFlux"]


class ConstantFlux:
    """Solve ``dn/dt = (d + flux * r) n`` over a step in which the flux is
    constant, with a :class:`MatrixExponential` for ``exp(A dt) n0``.

    This is the one place that turns batman's decay and reaction matrices
    into the matrix ``A = d + flux(0) * r`` of the step. The flux is sampled
    at the start of the step because it is taken to be constant over it:
    batman's predictors pass a constant flux function to the exponentiator
    (``lambda t: norm``), so for them the sample is exact. For a flux that
    varies within a step, use a scheme that samples it inside the step, such
    as the Magnus exponentiator. When the flux is 0, ``r`` is left out
    altogether, so the matrix keeps the pattern of ``d`` (for decay data,
    one without cycles).

    Parameters
    ----------
    expm - The matrix exponential, called as ``expm(A, n0, dt)``.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from batman.exponentiators import StructuralCram
    >>> d = sp.csr_matrix(np.array([[-np.log(2), 0.], [np.log(2), 0.]]))
    >>> r = sp.csr_matrix((2, 2))
    >>> step = ConstantFlux(StructuralCram())
    >>> [round(float(v), 12) for v in step(d, r, lambda t: 0., np.array([1., 0.]), 1.)]
    [0.5, 0.5]

    """

    def __init__(self, expm: MatrixExponential):
        self.expm = expm

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expm!r})"

    def __call__(self, d: csr_matrix, r: csr_matrix, flux: FluxFunc,
                 n0: np.ndarray, dt: Second) -> np.ndarray:
        """Solve the depletion equations over one step.

        Parameters
        ----------
        d,r - Sparse decay and induced transmutation matrices, ``A[j, i]``
              the rate at which isotope ``i`` turns into isotope ``j``.
        flux - Flux level as a function of time; constant over the step.
        n0 - Initial compositions.
        dt - Length of the step, in seconds.

        Returns
        -------
        The compositions after ``dt``.

        """
        phi = flux(0.)
        return self.expm(d + phi * r if phi else d, n0, dt)
