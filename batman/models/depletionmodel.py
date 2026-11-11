"""Model creation tools

"""
from collections.abc import Callable, Sequence
from functools import lru_cache

import networkx as nx
from isotopes import ZAID
from reactions import ReactionRate, ReactionType

from batman.graphs.decay import DecayGraph
from batman.graphs.filters import GraphFilter
from batman.graphs.reaction import ReactionGraph

from .decaymodel import DecayModel

DepletionModel = tuple[Sequence[ZAID],
                       DecayModel,
                       Callable[[ReactionRate], bool]]


@lru_cache(maxsize=5)
def depletion_model(dg: DecayGraph,
                    reacts: frozenset[ReactionType],
                    _filter: GraphFilter, *,
                    dtype: str = 'float64',
                    accumulate: frozenset[ZAID] = frozenset()
                    ) -> DepletionModel:
    """Generate the depletion model for some composition's data. This model
    includes which isotopes are to burn (and at what order), their decay
    matrix and the filtered reaction graph.

    Parameters
    ----------
    dg - Decay graph, unfiltered.
    reacts - Reaction rates from the transport calculation for this component.
    _filter - User defined filter for the resulting giant graph.
    dtype - NumPy type for vectors etc.
    accumulate - Isotopes that are to be held constant.

    Returns
    -------
    A 3-tuple. The first value is a sequence of isotopes, the second is the
    decay matrix to use and the third is a filter for reaction rates, so we
    can filter out reaction rates that have no effect on the resulting model.

    """

    rg = ReactionGraph()
    fake_rates = (ReactionRate('moo', reaction_specific, 1., 0.)
                  for reaction in reacts
                  for reaction_specific, _ in reaction.branches())
    rg.add_edges_from_results(fake_rates)
    g = _filter(nx.compose(dg, rg))

    isos = tuple(sorted(g.nodes))

    dg: DecayGraph = dg.subgraph(isos)
    decmat = DecayModel(dg, isos, dtype=dtype, accumulate=accumulate)

    rg: ReactionGraph = rg.subgraph(isos)
    edges = frozenset((parent, typus) for parent, _, typus in
                      rg.edges(keys=True))

    def _reacfilter(rr: ReactionRate) -> bool:
        return any((r.parent, r.typus + str(r.target)) in edges for r in
                   rr.expand())

    return isos, decmat, _reacfilter
