"""Batman exponentiator for the block-structured, shifted CRAM of
:mod:`batman.exponentiators.cram.system`.

"""
from typing import Optional

import numpy as np
import scipy.sparse as sp

from batman.exponentiators.protocol import FluxFunc
from batman.units import Second

from .pusa import CRAM_ORDERS
from .system import StructureCache, _canonical, cram_apply, shift_width

__all__ = ["CramExponentiator"]


class CramExponentiator:
    """Solve ``dn/dt = (d + flux * r) n`` over a step with block-structured
    CRAM. The flux is taken constant over the step, at ``flux(0)``.

    The default (``order=48, substeps=4, shifted=True``) is shifted CRAM-48:
    every nuclide is computed to about 1e-13 relative to its own
    value, down to 1e-280 of the atoms, with no negative values and no
    nuclide appearing where nothing can make it. Plain CRAM, including
    :class:`IPFCramSolver`, is only accurate relative to the bulk: its error is
    about ``alpha0`` times the largest population, in every nuclide.

    The pattern of every matrix is analyzed once (block ordering, levels) and
    kept in a small cache, so steps of one component reuse it. Decay steps
    (``flux(0) == 0``, or ``r`` empty) need no factorization at all.

    Parameters
    ----------
    order - CRAM order, one of 4, 8, ..., 48. From 24 on, plain CRAM is at
            the round-off of double precision for the bulk; 48 is needed
            for the activity of nuclides that decay away within a step.
    substeps - Substeps per call, sharing the factorizations.
    shifted - Whether to solve again, with a shift, the nuclides whose
              ancestors all decay fast (their accuracy relative to their
              own value).
    width - Width of the shift levels; None picks 20 for CRAM-48 and 10 below.
    cache_size - Number of matrix patterns whose analysis is kept.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> d = sp.csr_matrix(np.array([[-np.log(2), 0.], [np.log(2), 0.]]))
    >>> r = sp.csr_matrix((2, 2))
    >>> expo = CramExponentiator()
    >>> [round(float(v), 12) for v in expo(d, r, lambda t: 0., np.array([1., 0.]), 1.)]
    [0.5, 0.5]

    """

    def __init__(self, order: int = 48, substeps: int = 4, shifted: bool = True,
                 width: Optional[float] = None, cache_size: int = 8):
        if order not in CRAM_ORDERS:
            raise ValueError(f"CRAM order must be one of {CRAM_ORDERS}, not {order}")
        if int(substeps) < 1:
            raise ValueError(f"substeps must be at least 1, not {substeps}")
        self.order = int(order)
        self.substeps = int(substeps)
        self.shifted = bool(shifted)
        self.width = (0.0 if not shifted
                      else shift_width(order) if width is None else float(width))
        self._structures = StructureCache(cache_size)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(order={self.order}, substeps={self.substeps}, "
                f"shifted={self.shifted})")

    def apply(self, a, n0: np.ndarray, dt: Second) -> np.ndarray:
        """Return ``exp(a dt) n0`` for a burnup matrix ``a``."""
        a = _canonical(a)
        return cram_apply(a, n0, dt, order=self.order, substeps=self.substeps,
                          width=self.width, structure=self._structures.get(a))

    def __call__(self, d: sp.csr_matrix, r: sp.csr_matrix, flux: FluxFunc,
                 n0: np.ndarray, dt: Second) -> np.ndarray:
        """Solve the depletion equations over one step.

        Parameters
        ----------
        d,r - Sparse decay and induced transmutation matrices, ``A[j, i]``
              the rate at which isotope ``i`` turns into isotope ``j``.
        flux - Flux level as a function of time; used at the step's start.
        n0 - Initial compositions.
        dt - Length of the step, in seconds.

        Returns
        -------
        The compositions after ``dt``.

        """
        phi = flux(0.)
        a = d + phi * r if phi else d
        return self.apply(a, n0, dt)
