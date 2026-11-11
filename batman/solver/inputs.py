"""Tools used to create input for the batman package solvers.

"""

import logging
from abc import abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence, Sized
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, reduce
from typing import Protocol, Self, TypeVar

from coremaker.protocols.mixture import Mixture
from isotopes import ZAID
from reactions import ReactionRate

from batman.graphs import DecayGraph, GraphFilter
from batman.models import ReactionModel, depletion_model
from batman.units import EV, EV_TO_MJ, MJ, MW, Volume

from .power_normalization import add_two, calc_norm, single_power_produced
from .utils import DepletionData, RunData, append_doc_of

ModelData = tuple[DecayGraph, Iterable[ReactionRate], GraphFilter,
                  frozenset[ZAID]]
ComponentData = tuple[DecayGraph, Iterable[ReactionRate], GraphFilter,
                      Mixture, Volume, frozenset[ZAID]]
BurnData = Iterable[ModelData]

modlogger = logging.getLogger('batman.expo.input')
T = TypeVar('T')


class FiniteIterable(Iterable[T], Sized, Protocol):
    """An iterable that is guaranteed to end at some point

    """


@dataclass(init=True, frozen=False)
class InputData:
    """Data outer API for external modules.

    All the parameters should be sequences of the same length, which will
    correspond to the sequence of spatial burnup regions to solve for. Thus,
    all these sequences should be ordered according to those spatial regions.

    In the beginning, this object was without form, and void. Then a researcher
    said "Let there be Graphs!", and there were graphs. Decay graphs and
    reaction graphs. Even filters.
    But everything changed when the fire nation attacked.
    Graphs were too slow to build, communicate and so on, and they clogged up
    the system. A less convenient but lighter approach was necessary.
    And so previous reaction graphs turned into iterables of reaction rates,
    the lightest RAMP object that holds the required data.

    So you should think about InputData this way:
    To burn things up you need to know the way things decay in each spatial
    location, and what the reaction rates are. So please give us the "graphs".
    Then you also need to know how to filter all that data to just the stuff that
    matters to you. That's a GraphFilter, so you should supply one of those.
    In order to deplete a material you must have a material, so obviously supply
    one of those, and since that material is given as number density, you should
    also let someone know the volume of said material, so you can normalize
    things correctly.
    Then there is "accumulate". "Accumulate" is an immutable set of isotopes
    that you want to force to never be removed from the system. Why would you
    want that? That's up to you. It's very useful for analytic solutions in
    tests, and I don't ever expect anyone to actually use this, but it's there
    if you want to. Please don't say I didn't warn you.

    """
    decay_models: Sequence[DecayGraph]
    reaction_models: Sequence[Iterable[ReactionRate]]
    filters: Sequence[GraphFilter]
    mixtures: Sequence[Mixture]
    volumes: Sequence[Volume]
    accumulates: Sequence[frozenset[ZAID]] = None

    def __post_init__(self):
        self.accumulates = (self.accumulates
                            or [frozenset() for _ in range(len(self))])
        try:
            assert (len(self.decay_models) == len(self.reaction_models)
                    == len(self.filters) == len(self.mixtures)
                    == len(self.volumes) == len(self.accumulates))
        except AssertionError as e:
            raise ValueError(
                "Length of all sequences in the input data must "
                "be identical.:" +
                '\n'.join(
                    [f"{title}:{len(seq)}"
                     for title, seq in zip(
                        ['decay', 'reaction', 'filter',
                         'mixture', 'volume', 'accumulates'],
                        [self.decay_models, self.reaction_models,
                         self.filters, self.mixtures, self.volumes,
                         self.accumulates])]
                    )
                ) from e

    def __iter__(self) -> Iterator[ComponentData]:
        yield from zip(self.decay_models,
                       self.reaction_models,
                       self.filters,
                       self.mixtures,
                       self.volumes,
                       self.accumulates)

    def to_burndata(self):
        """Make the sub-info BurnData out of this one.

        """
        return zip(self.decay_models, self.reaction_models,
                   self.filters, self.accumulates)

    def __len__(self):
        return len(self.mixtures)


Td = TypeVar('Td', bound="EasyData")


class EasyData(Protocol):
    """Protocol for how data for the depletion algorithms should look like.

    """

    mixtures: Sequence[Mixture]

    @classmethod
    @abstractmethod
    def from_input(cls: type[Td], data: InputData, *, dtype: str = 'float64') \
            -> Td:
        """Make EasyData from InputData

        Parameters
        ----------
        data - InputData, see InputData
        dtype - NumPy type string

        """
        raise NotImplementedError

    def normalize(self, p: MW, *, decay_power_allowed: bool) -> float:
        """Normalize the reaction models to reach a specific power

        Parameters
        ----------
        p - Power to reach.
        decay_power_allowed - Flag for whether it's ok to have 0 desired power
                              but still have decay power going on.

        Returns
        -------
        The normalization factor used for re-normalization.

        """
        decp, reacp = self.powers(norm=1.)
        norm = self.calc_norm(p, decp, reacp,
                              decay_power_allowed=decay_power_allowed)
        return norm

    # noinspection PyMethodMayBeStatic
    def calc_norm(self, p: MW, decp: MW, reacp: MW,
                  decay_power_allowed: bool) -> float:
        """Method used to calculate the normalization factor needed to achieve
        power p

        Parameters
        ----------
        p - Power to obtain, in MW
        decp - Decay power at the moment
        reacp - Reaction power at the moment
        decay_power_allowed - Flag for whether non-0 power is ok for desired 0
                              power.

        """
        return calc_norm(p, decp, reacp,
                         decay_power_allowed=decay_power_allowed)

    def power(self, norm: float) -> MW:
        """Get the power at the current normalization and state

        Parameters
        ----------
        norm - Normalization factor for the reaction models.

        """
        return sum(self.powers(norm=norm))

    def powers(self, norm: float) -> tuple[MW, MW]:
        """Calculate the decay and reaction powers at the current state and
        normalization.

        Parameters
        ----------
        norm - Normalization factor for the reaction models.
        """
        return self.map_reduce(partial(single_power_produced, norm=norm),
                               add_two,
                               initial=(0., 0.))

    @abstractmethod
    def __iter__(self) -> Generator[RunData, None, None]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def map(self, func: Callable[[RunData], T]) -> Iterable[T]:
        """Map a function over this object.

        Parameters
        ----------
        func - function to map with.

        Returns
        -------

        An iterable over the mapped results.

        """
        raise NotImplementedError

    @abstractmethod
    def map_reduce(self, fmap: Callable[[RunData], T],
                   freduce: Callable[[T, T], T], *,
                   initial: T) -> T:
        """Map a function over this object and then reduce it.

        Parameters
        ----------
        fmap - Function to map the rundata with.
        freduce - Binary operation to reduce with to one value.

        Returns
        -------
        The post-reduce of the mapped function.

        """
        raise NotImplementedError


class SerialEasyData:
    """Ease-of-use data object for run. Acts like Iterable[RunData] but
    easily updatable and copyable as the run goes along.

    """

    def __init__(self, *, datagen: Iterable[RunData] = (),
                 _models=None, _mixtures=None, _volumes=None):
        given = (_models, _mixtures, _volumes)
        self.models, self.mixtures, self.volumes = (
            tuple(tuple(i) for i in given) if all(given)
            else map(tuple, zip(*tuple(datagen)))
            )

    def __deepcopy__(self, memo: dict):
        mixtures = (deepcopy(mixture) for mixture in self.mixtures)
        return type(self)(_models=self.models, _mixtures=mixtures,
                          _volumes=self.volumes)

    @classmethod
    def from_input(cls, data: InputData, *, dtype: str = 'float64') \
            -> Self:
        """Make SerialEasyData from InputData

        Parameters
        ----------
        data - InputData, see InputData
        dtype - NumPy type string

        """
        return cls(datagen=make_run_data(data, dtype=dtype))

    normalize = EasyData.normalize
    power = EasyData.power
    calc_norm = EasyData.calc_norm
    powers = EasyData.powers

    def __iter__(self) -> Iterator[RunData]:
        yield from zip(self.models, self.mixtures, self.volumes)

    def __len__(self) -> int:
        return len(self.mixtures)

    def tot_energy(self, iso: ZAID, iso_ener: EV) -> MJ:
        """Return the total energy given in a specific isotope

        Parameters
        ----------
        iso - Isotope to look for
        iso_ener = Energy of fission of one of a single isotope of this type

        """

        content = sum(mixture.get(iso) * volume for _, mixture, volume in self)
        return content * EV_TO_MJ * iso_ener

    def map(self, func: Callable[[RunData], T]) -> Generator[T, None, None]:
        """Map a function over the RunData.

        Parameters
        ----------
        func - Function to transform the RunData with

        Yields
        ------
        The results of the given function over the RunData.

        """

        yield from map(func, iter(self))

    @append_doc_of(EasyData.map_reduce)
    def map_reduce(self, fmap: Callable[[RunData], T],
                   freduce: Callable[[T, T], T], *,
                   initial: T) -> T:
        """Docstring taken from the Protocol class: EasyData.map_reduce:
        """
        return reduce(freduce, self.map(fmap), initial)


def make_run_data(data: InputData, *,
                  dtype: str = 'float64') -> Generator[RunData, None, None]:
    """Create the data used to run stuff in this package from user input

    Parameters
    ----------
    data - InputData object.
    dtype - NumPy type string.

    """
    yield from zip(make_depletion_models(data.to_burndata(), dtype=dtype),
                   data.mixtures,
                   data.volumes)


def make_depletion_models(data: BurnData, *,
                          dtype: str = 'float64') -> \
        Generator[DepletionData, None, None]:
    """Generator for depletion models given the initial data.

    Parameters
    ----------
    data - Original BurnData used in this calculation.
    dtype - NumPy data type used in this calculation.

    """

    for dg, rr, filt, accumulate in data:
        isos, decmat, rg_filter = depletion_model(
            dg,
            frozenset(r.reaction for r in rr),
            filt,
            dtype=dtype,
            accumulate=accumulate)
        reacts = filter(rg_filter, rr)
        reacmat = ReactionModel(isos=isos, reactions=reacts, dtype=dtype,
                                accumulate=accumulate)
        yield isos, decmat, reacmat
