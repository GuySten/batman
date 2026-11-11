"""Common tools for the batman expo

"""
from collections.abc import Sequence
from contextlib import contextmanager
from logging import captureWarnings

import numpy as np
from coremaker.protocols.mixture import Mixture
from isotopes import ZAID

from batman.models import DecayModel, ReactionModel
from batman.units import Volume


def mixture_to_nd(mixture: Mixture, isos: Sequence[ZAID], *,
                  dtype: str) -> np.ndarray:
    """Get a numpy array for number densities given a mixture and isotopes to
    use

    Parameters
    ----------
    mixture - Mixture to turn
    isos - Isotopes to use.
    dtype - Numpy datatype string definition

    """
    return np.fromiter((mixture.isotopes.get(iso, 0.) for iso in isos),
                       dtype=dtype,
                       count=len(isos))


DepletionData = tuple[Sequence[ZAID], DecayModel, ReactionModel]
RunData = tuple[DepletionData, Mixture, Volume]


def append_doc_of(f):
    """Append to this function the documentation for another.

    Parameters
    ----------
    f - function to add documentation to.

    """
    def _deco(fun):
        fun.__doc__ += f.__doc__
        return fun
    return _deco


@contextmanager
def capture_warnings():
    """A context manager within which warnings turn into logging.warning
    logs.

    """
    captureWarnings(True)
    yield
    captureWarnings(False)
