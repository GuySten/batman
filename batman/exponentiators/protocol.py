"""Exponentiator agreed upon protocol

"""
from collections.abc import Callable
from typing import Protocol

from numpy import ndarray
from scipy.sparse import csr_matrix

from batman.units import Second

FluxFunc = Callable[[Second], float]


class Exponentiator(Protocol):
    """A Protocol for exponentiators. Basically a function object with an agreed
    upon signature.

    """

    def __call__(self, d: csr_matrix, r: csr_matrix, flux: FluxFunc,
                 n0: ndarray, dt: Second) -> ndarray:
        raise NotImplementedError


class MatrixExponential(Protocol):
    """A Protocol for the action of a matrix exponential, ``exp(a t) v``, with
    no knowledge of where the matrix comes from. An :class:`Exponentiator`
    can be built on one, as :class:`ConstantFlux` does.

    """

    def __call__(self, a: csr_matrix, v: ndarray, t: float) -> ndarray:
        raise NotImplementedError
