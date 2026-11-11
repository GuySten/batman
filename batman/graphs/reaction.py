"""Reaction graph specifics. These differ from decay graphs both in the data
held in nodes and edges, and in that fact that they cannot be auto-generated,
and must be introduced externally, so their API should probably be nicer.

"""
from collections.abc import Hashable, Iterable
from typing import NoReturn

import networkx as nx
from reactions import ReactionRate

from batman.units import CMBarnPerSecond

NGAMMA = r'(n,$\gamma$)'
FISSION = '(n,f)'


class ReactionGraph(nx.MultiDiGraph):
    """A nuclear reaction graph of processes between isotopes.

    """

    def add_edge(self, u_of_edge, v_of_edge, key: Hashable = None,
                 rate: CMBarnPerSecond = None, _strict: bool = False,
                 **attr) -> Hashable:
        """See nx.MultiDiGraph.add_edge for details

        u_of_edge - source isotope
        v_of_edge - target isotope
        key - Hashable key to use for this edge. If not given, the return
              value is the key used, which is the lowest unused positive
              integer.
        rate - Reaction rate. This is not necessarily normalized in any way.
               This is given in cm-barn/s. This should be normalized such that
               the extracted total energy is the required power in the core.
        _strict - Flag for whether to be strict about values making sense.
                  The reason this is defaulted to false is to allow networkx
                  algorithms that don't check data but do add edges to work.
        attr - Any other information, passed on as is

        Raises
        ------
        TypeError if the decay rate is unset.

        Returns
        -------
        The key used for the new edge.

        """

        if not attr and not _strict and rate is None:
            return super().add_edge(u_of_edge, v_of_edge, key=key)
        if rate is None:
            raise TypeError("Must set the reaction rate properly. None is "
                            "unsupported.")
        attr['rate'] = rate
        return super().add_edge(u_of_edge, v_of_edge, key=key, **attr)

    def add_edge_from_result(self, result: ReactionRate) -> NoReturn:
        """Generate an edge in the graph from a Result object.

        Parameters
        ----------
        result - Common protocol for a reaction rate result object.

        """
        self.add_edge(result.parent, result.target,
                      key=result.typus + str(result.target),
                      energy=result.energy, rate=result.rate)
        self.add_edge(result.parent, result.parent,
                      key=result.typus + str(result.target),
                      energy=result.energy, rate=-result.rate)

    def add_edges_from_results(self,
                               results: Iterable[ReactionRate]) -> NoReturn:
        """Outer API for adding edges using an iterable of results.

        Parameters
        ----------
        results - Iterable of the common reaction rate protocol objects.

        """
        for result in results:
            self.add_edge_from_result(result)

    def renormalize(self, factor: float) -> 'ReactionGraph':
        """Used to multiply all the reaction rates by a single factor.

        This is a commonly used utility, so it was given its own method.

        Parameters
        ----------
        factor - Numeric factor to multiply by.

        """

        for _, _, d in self.edges.data():
            d['rate'] *= factor
        return self
