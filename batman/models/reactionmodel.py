"""Reaction matrix tools"""

import operator
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache, reduce

import numpy as np
import scipy.sparse as spr
from isotopes import ZAID
from multipledispatch import dispatch
from reactions import ReactionRate, ReactionType

from batman.units import EV_TO_MJ, MWPerCM3, PerCmBarnArray


class FissionMat:
    """Object that acts like a fission reaction yield matrix.

    Why does this object exist, you ask? Let me tell you a story.
    There once was a man named... Eshed. Yes. Really. A terrible name, I know.
    He loved using matrix objects for things that are matrices. Oh, the fun he
    had with all his reaction matrices! A branching reaction never caused him to
    do anything special, he just added all those branching entries in and he was
    happy as a clam.

    But then came a terrible realization! His matrices differed in between
    locations in space, so that they had to be stored separately, and had to be
    transmitted in full whenever he wanted to tell his lovely worker process
    friends about his matrices! And his matrices were so large, they sank the
    Ethernet boats they were riding on, and were so big they wouldn't fit in
    memory, because there were so many of them! Eshed did love his matrices,
    after all, and could never have enough matrices!

    "But most of those matrices are just repeating fission yield vectors!",
    Eshed yelled. Alas, there was no response, since matrices can't speak.
    Seeing no other way, Eshed went to work. He separated out the repeating
    vector structures, and he cached them on construction, so that if he ever
    wanted to create a vector he already had, he will just get a pointer to his
    old vector.
    Now he still had all his lovely matrices, all tens of thousands of them, and they were
    much lighter without all those pesky branching vectors! His vectors were
    indeed pretty large, as they filled almost the entire isotope column, but
    there were only about 20 of them, which would not sink an Ethernet boat!

    "But I do miss my wonderful full matrices...", Eshed thought to himself,
    longingly. "I should store the information required to make these vectors
    into proper metrices all in one place. Then, whenever I want to have my
    matrix, I can just turn each vector into a sparse matrix, and add them
    all up."

    And so was the first FissionMat object born. Not quite a vector, not yet
    a matrix. More of a promise, really, of matrices to come.

    """

    def __init__(self, fission_vec: np.ndarray, rate: float, index: int):
        """
        Parameters
        ----------
        fission_vec - Fission yield array, already ordered by isotopes
        rate - Fission reaction rate, which is a magnitude for the vector above.
        index - Index of the fissile isotope that causes this in the ordered
                array.

        """
        self.fission_vec = fission_vec
        self.index = index
        self.rate = rate

    @dispatch(float)
    def __mul__(self, other: float) -> "FissionMat":
        return FissionMat(self.fission_vec, self.rate * other, self.index)

    @dispatch(object)
    def __mul__(self, other: object):  # noqa: F811
        raise TypeError(f"Cannot multiply FissionMat by {type(other)}")

    def to_csr(self) -> spr.csr_matrix:
        """Turn this matrix to csr format"""
        n = len(self.fission_vec)
        m = spr.dok_matrix((n, n), dtype=self.fission_vec.dtype)
        m[self.index, :] = self.rate * self.fission_vec
        return m.T.tocsr()


@lru_cache(maxsize=500)
def _arrgen(reaction: ReactionType, isos: Sequence[ZAID]) -> np.ndarray:
    arr = np.zeros(len(isos))
    isod = {iso: i for i, iso in enumerate(isos)}
    for reaction_specific, br in reaction.branches():
        if reaction_specific.target in isod:
            arr[isod[reaction_specific.target]] = br
    return arr


def fiss_arr_gen(isos: Sequence[ZAID], reactions: Sequence[ReactionType]) -> dict[ReactionType, np.ndarray]:
    """Generate a branch vector for each branching reaction.

    Parameters
    ----------
    isos - Possible target isotopes, assume ordered correctly.
    reactions - Initializing reactions

    """
    return {reac: _arrgen(reac, isos) for reac in reactions if reac.branching}


class ReactionModel:
    """This acts as if it was a sparse matrix, but it's actually a very
    strange object. The entire purpose of the object is to reuse the fission
    branching data in a way that won't copy too much data when there are many fuel regions.
    So this is basically a wild memory trick. The main use of this object is to add it
    to another sparse matrix, at which point it magically transforms to a
    csr_matrix.

    """

    def __init__(
        self,
        *,
        isos: Sequence[ZAID],
        reactions: Iterable[ReactionRate],
        dtype="float64",
        accumulate: frozenset[ZAID],
        branch_factory: Callable[..., np.ndarray] = _arrgen,
    ):
        """
        Parameters
        ----------
        isos - Isotopes in this model. If needed, you can assume it is sorted.
        reactions - The reaction rates for this model, filtered to isos.
        dtype - NumPy type string.
        accumulate - Isotopes that will never die, but will generate others.
        branch_factory - Factory function that provides branching vectors.
        """
        n = len(isos)

        self.r: spr.spmatrix = spr.dok_matrix((n, n), dtype=dtype)
        self.fiss = []
        self.energy_model = np.zeros(n, dtype=dtype)
        self.prod_model = np.zeros(n, dtype=dtype)
        self.dtype = dtype
        for reac_rate in reactions:
            try:
                parent = isos.index(reac_rate.parent)
            except ValueError as e:
                raise ValueError(
                    f"{reac_rate.parent} from reaction {reac_rate.typus} not in isotope list: {isos}"
                ) from e
            self.r[parent, parent] -= reac_rate.mean if reac_rate.parent not in accumulate else 0.0
            self.prod_model[parent] += reac_rate.mean * reac_rate.nu
            self.energy_model[parent] += EV_TO_MJ * reac_rate.mean * reac_rate.energy
            if reac_rate.branching:
                fiss_vec = branch_factory(reac_rate.reaction, tuple(isos))
                self.fiss.append(FissionMat(fiss_vec, reac_rate.mean, parent))
            else:
                try:
                    target = isos.index(reac_rate.target)
                    self.r[target, parent] += reac_rate.mean
                except ValueError:
                    pass
        self.r: spr.csr_matrix = self.r.tocsr()

    @property
    def mat(self) -> spr.csr_matrix:
        """Return the reaction matrix for this model."""
        return self.r + self.fissions()

    def fissions(self) -> spr.csr_matrix:
        """Return the fission yield matrix"""
        return reduce(operator.add, map(FissionMat.to_csr, self.fiss), spr.csr_matrix(self.r.shape))

    def energy(self, nd: PerCmBarnArray, norm: float) -> MWPerCM3:
        """Returns the energy emission given some material number densities.

        Parameters
        ----------
        nd - Number densities vector, in 1/cm-barn
        norm - Normalization for this reaction model

        """
        return float(norm * np.dot(self.energy_model, nd))

    def production(self, nd: PerCmBarnArray) -> float:
        """Return the unnormalized neutron production rate

        Parameters
        ----------
        nd - Number densities vector, in 1/cm-barn.

        """
        return float(np.dot(self.prod_model, nd))

    @property
    def absorption_model(self) -> np.ndarray:
        """Vector for the total annihilation rate for each parent isotope."""
        # TODO: correct this for reaction that create neutrons like n2n

        return -self.r.diagonal()

    def absorption(self, nd: PerCmBarnArray) -> float:
        """Return the unnormalized neutron absorption rate

        Parameters
        ----------
        nd - Number densities vector, in 1/cm-barn.

        """
        return float(np.dot(self.absorption_model, nd))
