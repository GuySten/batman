"""Distributed version of the common inputs. This is used so that they will
have similar API but with a distributed workload in input generation.

The only real thing to note here is DistEasyData. Everything else are just
helper functions for its construction. Constructing a DistEasyData is complex,
since you need to create a Dask Bag while minimizing the cost. Bags are not
designed for this use case, they're mostly designed for parsing files as an
interim step to create a Dask Array/Frame. However, since we have 25000 or so
tasks to map over, having a collection of delayed values we can do functional
programming over is quite nice. So we go through the trouble of making this Bag.

"""
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from typing import Any, Self, TypeVar

import numpy as np
from coremaker.protocols.mixture import Mixture
from isotopes import ZAID
from more_itertools import divide
from reactions import ReactionRate, ReactionType
from toolz import unique

from batman.graphs import DecayGraph, GraphFilter
from batman.models import DepletionModel, ReactionModel, depletion_model, fiss_arr_gen

from .inputs import ComponentData, EasyData, FiniteIterable, InputData
from .utils import RunData, append_doc_of

__all__ = ["DistEasyData"]

modlogger = logging.getLogger('batman.expo.input_dist')

T = TypeVar('T')

try:
    from dask import delayed
    from dask.bag import Bag, from_delayed

    _no_dask = False
except ImportError:
    Bag = Any
    _no_dask = True


class DistEasyData(EasyData):
    """Distributed version of the EasyData object

    """

    def __init__(self, bag: Bag):
        self.bag: Bag = bag
        if _no_dask:
            raise ImportError("Dask support is required for a DistEasyData")

    def __deepcopy__(self, memo: dict):
        return type(self)(self.bag)

    @classmethod
    def from_input(cls, data: InputData, *,
                   dtype: str = 'float64',
                   partitions: int = 1) -> Self:
        """Make Distributed EasyData from a local InputData.

        Parameters
        ----------
        data - Input data to use
        partitions - Partitions in the resulting EasyData bag. Aim for
                     relatively high numbers compared to the cpu count.
                     For example, see test_dist_fullcore.py.
                     A more thorough investigation in needed to optimize this.
        dtype - NumPy data type to use in numerical pieces

        """

        # First, make sure you do the least amount of work necessary
        depargs = tuple(unique(_get_depargs(x) for x in data))
        modlogger.info("No. Unique depargs: %d", len(depargs))
        depdict = {i: depargs.index(_get_depargs(comp)) for i, comp in
                   enumerate(data)}

        def expand_bag(*values):
            """Expand a short solution range into the full Bag shape"""
            return from_delayed_sequence(
                delayed(partial(_expand, length=len(data), mp=depdict),
                        pure=True)(values),
                n=partitions)

        make_data_bag = partial(_from_sequence,
                                n=partitions)
        # Lazyily make the minimal amount of depletion models needed
        tiny_models = [delayed(_depletion, nout=3, pure=True)(args, dtype=dtype)
                       for args in depargs]
        isos, decmodels, rg_filters = zip(*tiny_models)
        reacts = tuple(args[1] for args in depargs)
        # Lazily make the fission vectors
        fiss_vecs = [delayed(fiss_arr_gen, pure=True)(isolist, reaclist)
                     for isolist, reaclist in zip(isos, reacts)]
        # Broadcast the isotope lists to the full bag
        isos = expand_bag(*isos)
        rr = make_data_bag(data.reaction_models)
        accum = make_data_bag(data.accumulates)
        # Lazily make the reaction models
        reacmodels = rr.map(_reacmodel,
                            rfilter=expand_bag(*rg_filters),
                            isotopes=isos,
                            accumulate=accum,
                            fiss_vecs=expand_bag(*fiss_vecs),
                            dtype=dtype)
        # Broadcast the DecayModels to the full bag
        decmodels = expand_bag(*decmodels)
        mixtures = make_data_bag(data.mixtures)
        vols = make_data_bag(data.volumes)
        bag = _zip(_zip(isos, decmodels, reacmodels), mixtures, vols)
        return cls(bag.persist())

    # noinspection PyProtocol
    @property
    def mixtures(self) -> Sequence[Mixture]:
        """API for generating a sequence of mixtures out of the current data.

        """
        return tuple(self.bag.pluck(1))

    @mixtures.setter
    def mixtures(self, mixs: Sequence[Mixture]) -> None:
        mixs = _from_sequence(mixs, n=self.bag.npartitions)
        self.bag = self.bag.map(_update_single_mixture, mixs).persist()

    def __iter__(self) -> Iterator[RunData]:
        yield from self.bag.flatten().compute()

    def __len__(self) -> int:
        return self.bag.count().compute()

    def map(self, func: Callable[[RunData], T]) -> Bag:
        """The map method of the underlying bag.

        """
        return self.bag.map(func)

    @append_doc_of(EasyData.map_reduce)
    def map_reduce(self, fmap: Callable[[RunData], T],
                   freduce: Callable[[T, T], T], *,
                   initial: T) -> T:
        """Docstring taken from the Protocol class: EasyData.map_reduce:
        """
        return self.map(fmap).fold(freduce, initial=initial).compute()


def _zip(fbag: Bag, *bags: Bag) -> Bag:
    """Zip a bunch of bags together to create a bag of tuples.

    Parameters
    ----------
    fbag - First bag to use
    bags - The rest of the bags

    Returns
    -------
    A lazily calculated bag that is the zipping of these bags.

    """
    return fbag.map(lambda *x: tuple(x), *bags)


ModelInput = tuple[DecayGraph, Iterable[ReactionType], GraphFilter,
frozenset[ZAID]]


def _depletion(comp: ModelInput, *,
               dtype: str) -> DepletionModel:
    dg, rs, filt, accum = comp
    return depletion_model(dg, frozenset(rs), filt, dtype=dtype,
                           accumulate=accum)


def _reacmodel(rr: Iterable[ReactionRate], *,
               rfilter: Callable[[ReactionRate], bool],
               isotopes: Sequence[ZAID],
               accumulate: frozenset[ZAID],
               fiss_vecs: dict[ReactionType, np.ndarray],
               dtype: str) -> ReactionModel:
    reacts = filter(rfilter, rr)
    return ReactionModel(
        isos=isotopes,
        reactions=reacts,
        dtype=dtype,
        accumulate=accumulate,
        branch_factory=lambda x, _: fiss_vecs[x])


def from_delayed_sequence(seq: Sequence[T], n: int) -> Bag:
    """Make a bag out of a finite iterable, and assume that you can't
    preemptively call len(seq).

    Parameters
    ----------
    seq - Finite iterable to bag
    n - Number of partitions to use

    """
    return from_delayed(list(delayed(divide, pure=True, nout=n)(n, seq)))


def _from_sequence(seq: FiniteIterable[T], n: int) -> Bag:
    """Make a bag out of a finite iterable, and assume you can call len(seq).

    Parameters
    ----------
    seq - Finite iterable to bag
    n - Number of partitions to use

    """
    return from_delayed([delayed(part) for part in divide(n, seq)])


def _get_depargs(comp: ComponentData):
    dg, rr, filt, _, _, accum = comp
    return dg, frozenset(v.reaction for v in rr), filt, accum


def _expand(shortit: Sequence[T],
            length: int,
            mp: dict[int, int]) -> list[T]:
    return [shortit[mp[i]] for i in range(length)]


def _update_single_mixture(data: RunData, mixture: Mixture):
    dep, _, vol = data
    return dep, mixture, vol
