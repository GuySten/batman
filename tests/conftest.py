"""Fixtures used in multiple test modules

"""
import logging

import numpy as np
import pytest
from coremaker.protocols.mixture import Mixture

slow = pytest.mark.slow
regression = pytest.mark.regression

logging.basicConfig(format='%(filename)s - %(lineno)d - %(asctime)s - '
                           '%(levelname)s: %(message)s')


def allclose(first: Mixture, second: Mixture, *, comparator=np.isclose,
             verbose=False,
             **kwargs) -> bool:
    """Tests for float-based near equality, which is probably better...

    Parameters
    ----------
    first, second - Mixtures to compare
    comparator - Comparing callable used on the values
    verbose - Flag for verbosally printing what isn't the same
    kwargs - Keyword-Arguments for the comparator callable

    """

    extras = set(second.keys()) - set(first.keys())
    if not verbose:
        return (all(comparator(v, second.get(key), **kwargs)
                    for key, v in first.items())
                and all(comparator(second[key], first.get(key), **kwargs)
                        for key in extras))
    for key in set(first.keys()) | set(second.keys()):
        if not comparator(first.get(key), second.get(key), **kwargs):
            print(f'Mixtures differ on {key}, with values {first.get(key)}, '
                  f'{second.get(key)} correspondingly')
            return False
    return True
