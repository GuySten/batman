"""Decay graph specifics. This module defines decay processes, and makes
graphs out of them.

"""

from functools import wraps
from pathlib import Path
from typing import Any, Hashable, Type, TypeVar, Union

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import endf
import networkx as nx
from isotopes import ZAID, Isotope
from ramp_core.serializable import Serializable

from batman.graphs.decay_data import DecayProcess, SPFData, halflife_to_rate, parse_decay_processes
from batman.units import Second

DECAY = "radioactive decay"

EdgeData = tuple[Hashable, Any]  # A (key, value) pair from a dictionary
EdgeRep = tuple[ZAID, ZAID, Hashable, frozenset[EdgeData]]


# This type represents a DecayGraph edge in a frozen representation. The two
# ZAIDs are the parent and daughter nuclides, then comes the edge key, followed
# by a frozenset representation of the data on the edge.


def _invalidates_cache(f):
    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        self.invalidate_cache()
        return f(self, *args, **kwargs)

    return _wrapper


class DecayGraph(nx.MultiDiGraph):
    """A decay reaction graph."""

    ser_identifier = "DecayGraph"

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self._rep_cache = None
        self._hash: Union[int, None] = None

    def invalidate_cache(self) -> None:
        """Invalidate cached quantities. Do this whenever you modify the graph"""
        self._rep_cache = None
        self._hash = None

    def cache(self) -> None:
        """Force calculation of caches in this object."""
        _ = self._rep()
        _ = hash(self)

    add_node = _invalidates_cache(nx.MultiDiGraph.add_node)
    add_edge = _invalidates_cache(nx.MultiDiGraph.add_edge)

    def add_decay_edge(
        self, u_of_edge, v_of_edge, key: Hashable = None, *, halflife: Second = None, fraction: float = 1.0, **attr
    ) -> Hashable:
        """See nx.MultiDiGraph.add_edge for details

        Parameters
        ----------
        u_of_edge - source isotope
        v_of_edge - target isotope
        key - Hashable key to use for this edge. If not given, the return
              value is the key used, which is the lowest unused positive
              integer.
        halflife - Decay process half-life [sec].
        fraction - Decay process yield out of all radioactive decay processes.
        _strict - Flag for whether to be strict about values making sense.
                  The reason this is defaulted to false is to allow networkx
                  algorithms that don't check data but do add edges to work.
        attr - Any other information, passed on as is. The keyword 'rate' is
               overwritten, so do not supply it yourself.

        Raises
        ------
        ZeroDevisionError if the decay halflife is unset or set to 0

        Returns
        -------
        Key used for this decay process.

        """
        return super().add_edge(
            u_of_edge,
            v_of_edge,
            key=key,
            fraction=fraction,
            rate=halflife_to_rate(halflife, branching=fraction),
            **attr,
        )

    def add_edge_from_process(self, process: DecayProcess, *, ignore_zero: bool = False) -> None:
        """Generate an edge in the graph from a DecayProcess object.

        Parameters
        ----------
        process - The decay process used to generate the edge.
        ignore_zero - Flag to ignore decay paths with 0 halflife (yes, these
                      exist in endf7.1. See, for example, W182->Hf178)

        Raises
        ------
        ZeroDivisionError if ignore_zero is False and the halflife is 0.

        """

        try:
            for target, branching in process.target_branching.items():
                self.add_decay_edge(
                    process.parent,
                    target,
                    key=",".join(process.mode),
                    halflife=process.halflife,
                    fraction=process.fraction * branching,
                )
            if (process.parent, process.parent, (DECAY,)) not in self.edges:
                self.add_decay_edge(process.parent, process.parent, key=DECAY, halflife=process.halflife, fraction=-1.0)
        except ZeroDivisionError:
            if not ignore_zero:
                raise

    def _rep(self) -> tuple[frozenset[ZAID], frozenset[EdgeRep]]:
        """A frozen representation of this graph, using a set of nodes and a set
        of edge representations. Since this is relatively expensive to calculate
        we chose to cache this. The cache has to be easily invalidated, so we
        used a private variable rather than LRU cache.

        Further tests are required to check if this specific cache saves time,
        since its only merit is for when decay graphs are compared using the
        __eq__ operator.

        Returns
        -------

        A frozenset of nodes and a frozenset of edges (parent, child, key and
        data). Immutalibility is important to ensure that objects can be
        easily compared.

        """
        if self._rep_cache is None:
            edges = frozenset((s, d, k, frozenset(dat.items())) for s, d, k, dat in self.edges(data=True, keys=True))
            self._rep_cache = frozenset(self.nodes), edges
        return self._rep_cache

    def serialize(self) -> tuple[str, dict[str, Any]]:
        nodes = [int(node) for node in self.nodes]
        edges = [(int(u), int(v), key, dat) for u, v, key, dat in self.edges(data=True, keys=True)]
        return self.ser_identifier, {"nodes": nodes, "edges": edges}

    @classmethod
    def deserialize(cls: Self, d, *, supported: dict[str, Type[Serializable]]) -> Self:
        g = cls()
        for zaid in d["nodes"]:
            iso = Isotope.from_int_with_fallback(zaid)
            g.add_node(iso)
        for u, v, key, dat in d["edges"]:
            u, v = (Isotope.from_int_with_fallback(x) for x in (u, v))
            g.add_edge(u, v, key=key, **dat)
        return g

    def __hash__(self) -> int:
        """Hashing a massive graph that includes many nodes and edges is a very
        slow process, apparently. This is a wonderful case for caching, so this
        is cached. The cache is written with a private variable to make it
        easier to invalidate compared to an LRU cache.

        """
        if self._hash is None:
            self._hash = hash(self._rep())
        return self._hash

    def __eq__(self, other: "DecayGraph") -> bool:
        try:
            return self._rep() == other._rep()
        except AttributeError:
            return super().__eq__(other)


def parse_decay_graph(*file_or_material: Union[endf.Material, Path, str],
                      spf_db: SPFData | None = None) -> DecayGraph:
    """Decay graph of ENDF-6 decay files or materials.

    Parameters
    ----------
    file_or_material - ENDF materials, or paths of ENDF-6 decay files (each
                       may hold many materials).
    spf_db - Spontaneous fission yields; without them, spontaneous fission
             removes the parent but produces nothing.

    """
    processes = parse_decay_processes(*file_or_material, spf_db=spf_db)
    g = DecayGraph()
    for process in processes:
        g.add_edge_from_process(process, ignore_zero=True)
    g.cache()
    return g
