"""Graph filters that are used to generate leaner, meaner graphs."""

import logging
from collections.abc import Callable, Hashable
from typing import Any

import networkx as nx
from toolz import identity

from batman.graphs.decay_data import SPF
from batman.graphs.types import BatmanGraph

__all__ = [
    "GraphFilter",
    "blacklist_filter",
    "descendents_subgraph",
    "exclude_spf_filter",
    "predecessors_subgraph",
    "subsets_connection_subgraph",
    "whitelist_filter",
]

modlogger = logging.getLogger("batman.filter")


class GraphFilter:
    """Object that can filter a graph according to a required specification.

    Make sure that any arguments and keyword arguments are hashable if you
    want this to hash well.

    """

    def __init__(self, func: Callable[[BatmanGraph, Any], BatmanGraph] = identity, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._filter = func

    def __call__(self, g: BatmanGraph):
        return self._filter(g, *self.args, **self.kwargs)

    def __hash__(self):
        try:
            fhash = hash(self._filter)
            fhash = self._filter
        except TypeError:
            fhash = 0
        return hash((fhash, self.args, frozenset(self.kwargs.items())))

    def __eq__(self, other: "GraphFilter"):
        try:
            return self._filter is other._filter and self.args == other.args and self.kwargs == other.kwargs
        except AttributeError:
            return super().__eq__(other)


def descendents_subgraph(g: BatmanGraph, s: set[Hashable]) -> BatmanGraph:
    """Returns the subgraph of nodes that are reachable from s, including s.

    Somehow, this is not an algorithm in networkx. They do have the descendents
    algorithm, but they explicitly remove the source from the resulting set.
    This is stupid, so I had to make my own version of it.
    Also, their static typing is off. They ask for a list when all they need
    is an iterator.

    Parameters
    ----------
    g - Parent graph to take nodes and edges from.
    s - Set of source nodes

    Returns
    -------

    DiGraph of nodes the are reachable from the node set, including themselves.

    """

    def _descendents(source):
        if not g.has_node(source):
            return frozenset()
        return frozenset(nx.dfs_preorder_nodes(g, source=source))

    # noinspection PyTypeChecker
    return nx.compose_all(g.subgraph(_descendents(source)) for source in s)


def predecessors_subgraph(g: BatmanGraph, s: set[Hashable]) -> BatmanGraph:
    """Returns the subgraph of nodes that can reach s, including s.

    Uses graph reversal and descendents_subgraph.

    Parameters
    ----------
    g - Parent graph to take nodes and edges from.
    s - Set of destination nodes

    Returns
    -------

    The subgraph we are interested in.

    """
    return descendents_subgraph(g.reverse(False), s).reverse(False)


def subsets_connection_subgraph(g: BatmanGraph, src: set[Hashable], dest: set[Hashable]) -> BatmanGraph:
    """Returns the subgraph of nodes that can be reached from src and can reach
    dest.

    Parameters
    ----------
    g - Parent graph to filter.
    src - Subset of source nodes.
    dest - Subset of destination nodes.

    Returns
    -------

    nx.DiGraph which is a subgraph of g.

    """

    return predecessors_subgraph(descendents_subgraph(g, src), dest)


def blacklist_filter(g: BatmanGraph, s: set[Hashable]) -> BatmanGraph:
    """Return the subgraph that is the same but excludes the nodes in s

    Parameters
    ----------
    g - Graph to filer.
    s - Set of nodes to exclude

    Returns
    -------
    nx.DiGraph that is a subgraph of g and does not contain s.

    """
    return g.subgraph(set(g.nodes) - s)


def whitelist_filter(g: BatmanGraph, s: set[Hashable]) -> BatmanGraph:
    """Return the subgraph that only has isotopes in the specific set.

    Parameters
    ----------
    g - Graph to filter from
    s - Set of whitelisted_isotopes

    Returns
    -------

    Whitelisted graph.

    """

    return g.subgraph(s)


def exclude_spf_filter(g: BatmanGraph) -> BatmanGraph:
    """Filter that cuts off any spontaneous fission processes.

    Parameters
    ----------
    g - Graph to filter

    Returns
    -------
    Subgraph of g that has no SPF edges in it.

    """
    return g.edge_subgraph((u, v, k) for u, v, k in g.edges(default=(), keys=True) if SPF not in k)
