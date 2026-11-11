"""Tests for graphs"""

import json
from math import isclose

import isotopes
import networkx as nx
import numpy as np
import pytest
from ramp_core import RampJSONDecoder, RampJSONEncoder
from reactions import Fission, NGamma, ProtoReaction, Reaction, ReactionRate
from reactions import jsonable as reac_jsonable

from batman.graphs.decay import DecayGraph
from batman.graphs.decay_data import ALPHA, BETA_M, SPF, DecayProcess
from batman.graphs.filters import (
    blacklist_filter,
    descendents_subgraph,
    predecessors_subgraph,
    subsets_connection_subgraph,
    whitelist_filter,
)
from batman.graphs.matgen import graph_to_sparse_matrix
from batman.graphs.reaction import ReactionGraph


@pytest.fixture
def decay_graph() -> DecayGraph:
    """Generates a decay graph used in many cases.

    Returns
    -------
    A sample decay graph
    """

    dec = DecayGraph()
    processes = (
        DecayProcess(*arg)
        for arg in (
            (isotopes.I135, isotopes.Xe135, 6.57 * 3600, (BETA_M,)),
            (isotopes.Xe135, isotopes.Cs135, 9.14 * 3600, (BETA_M,)),
            (isotopes.Cs135, isotopes.Ba135, 2.3e6 * 365.25 * 24 * 3600, (BETA_M,)),
            (isotopes.U239, isotopes.Np239, 23.45 * 60, (BETA_M,)),
            (isotopes.Np239, isotopes.Pu239, 2.356 * 24 * 3600, (BETA_M,)),
            (isotopes.U235, isotopes.Th231, 7.04e8 * 365.25 * 24 * 3600, (ALPHA,)),
            (isotopes.U238, isotopes.Th234, 4.468e9 * 365.25 * 24 * 3600, (ALPHA,), 1.0 - 1e-8 - 1e-11),
            (isotopes.U238, isotopes.I135, 4.468e9 * 365.25 * 24 * 3600, (SPF,), 1e-8),
            (isotopes.U238, isotopes.Xe135, 4.468e9 * 365.25 * 24 * 3600, (SPF,), 1e-11),
        )
    )
    for process in processes:
        dec.add_edge_from_process(process)
    return dec


@pytest.fixture
def multiple_reaction_graph() -> ReactionGraph:
    """Generates a reaction graph used in many cases.

    Returns
    -------

    A sample reaction graph

    """

    g = ReactionGraph()
    make_r = Reaction.from_reaction
    make_br = ProtoReaction.from_reaction
    reactions = (
        make_r(isotopes.Xe135, NGamma),
        make_r(isotopes.Hf174, NGamma),
        make_r(isotopes.U235, NGamma),
        make_r(isotopes.U236, NGamma),
        make_r(isotopes.U237, NGamma),
        make_r(isotopes.U238, NGamma),
        make_br(isotopes.U235, Fission, branching={isotopes.I135: 2.4, isotopes.Xe135: 0.13}),
        make_br(isotopes.Pu239, Fission, branching={isotopes.I135: 2.1, isotopes.Xe135: 1e-6}),
    )
    rreactions = [(r, b) for pr in reactions for r, b in pr.branches()]
    prates = (3.3, 2.2, 5.0, 0.01, 1e-4, 1.5, 1.0, 1.0, 1.0, 1.0)
    reactions = [r for r, _ in rreactions]
    rates = [b * r for (_, b), r in zip(rreactions, prates)]
    processes = (ReactionRate("moo", reaction, rate, 0.0) for reaction, rate in zip(reactions, rates))
    g.add_edges_from_results(processes)
    return g


jsonable = reac_jsonable + [DecayGraph]
RampJSONDecoder.supported = {c.ser_identifier: c for c in jsonable}


def test_ser_deser_for_graph(decay_graph):
    g = decay_graph
    s = json.dumps(g, cls=RampJSONEncoder)
    try:
        v = json.loads(s, cls=RampJSONDecoder)
    except (RuntimeError, TypeError):
        print(s)
        raise
    assert g == v, (g, v, s)


def test_descendents_subgraph_has_the_right_nodes_for_an_example(decay_graph: DecayGraph):
    sources = {isotopes.I135, isotopes.U238}
    subg = descendents_subgraph(decay_graph, sources)
    assert set(subg.nodes) == {
        isotopes.I135,
        isotopes.Xe135,
        isotopes.Cs135,
        isotopes.Ba135,
        isotopes.Th234,
        isotopes.U238,
    }


def test_predecessors_subgraph_gets_right_nodes_and_specific_edge_for_example(decay_graph: DecayGraph):
    interest = {isotopes.Xe135}
    subg = predecessors_subgraph(decay_graph, interest)
    assert set(subg.nodes) == {isotopes.Xe135, isotopes.I135, isotopes.U238}
    assert (isotopes.I135, isotopes.Xe135, BETA_M) in subg.edges


def test_subset_connection_filter_has_the_right_nodes_for_example(decay_graph: DecayGraph):
    interest = {isotopes.Xe135}
    src = {isotopes.U238, isotopes.U235}
    subg = subsets_connection_subgraph(decay_graph, src, interest)
    assert set(subg.nodes) == {isotopes.U238, isotopes.I135, isotopes.Xe135}


def test_reaction_graph_has_an_edge_with_correct_rate(multiple_reaction_graph: ReactionGraph):
    # noinspection PyPep8Naming
    U235a = Reaction.from_reaction(isotopes.U235, NGamma)
    assert isclose(
        multiple_reaction_graph[isotopes.U235][isotopes.U235][U235a.typus + str(isotopes.U236)]["rate"], -5.0
    )


@pytest.fixture
def combined_graph(decay_graph, multiple_reaction_graph) -> nx.MultiDiGraph:
    """A combined reaction and decay graph."""
    return nx.compose(decay_graph, multiple_reaction_graph)


@pytest.fixture
def filtered_combined(combined_graph) -> nx.MultiDiGraph:
    """A filtered down combined graph for testing."""

    return subsets_connection_subgraph(
        combined_graph,
        src={isotopes.U235, isotopes.U238},
        dest={isotopes.U235, isotopes.U238, isotopes.Pu239, isotopes.Xe135},
    )


@pytest.fixture
def bare_filter(combined_graph) -> nx.MultiDiGraph:
    """Graph, filtered down to U235->I135->Xe135"""
    ng = blacklist_filter(combined_graph, {isotopes.U238})
    return subsets_connection_subgraph(ng, src={isotopes.U235}, dest={isotopes.Xe135})


def test_combination_of_graphs_has_the_right_nodes_and_edges_for_example(
    combined_graph, decay_graph, multiple_reaction_graph
):
    assert set(combined_graph.nodes) == (set(decay_graph.nodes) | set(multiple_reaction_graph.nodes))
    assert set(combined_graph.edges) == (set(decay_graph.edges) | set(multiple_reaction_graph.edges))


def test_filtered_has_the_right_nodes_and_edges_for_example(filtered_combined):
    assert set(filtered_combined.nodes) == {
        isotopes.U235,
        isotopes.U236,
        isotopes.U237,
        isotopes.U238,
        isotopes.U239,
        isotopes.Pu239,
        isotopes.Np239,
        isotopes.I135,
        isotopes.Xe135,
    }
    assert {
        (isotopes.U235, isotopes.U236),
        (isotopes.U236, isotopes.U237),
        (isotopes.U237, isotopes.U238),
        (isotopes.U238, isotopes.U239),
        (isotopes.U239, isotopes.Np239),
        (isotopes.Np239, isotopes.Pu239),
        (isotopes.U235, isotopes.I135),
        (isotopes.U235, isotopes.Xe135),
        (isotopes.U238, isotopes.I135),
        (isotopes.U238, isotopes.Xe135),
        (isotopes.Pu239, isotopes.I135),
        (isotopes.Pu239, isotopes.Xe135),
        (isotopes.I135, isotopes.Xe135),
    } | {(n, n) for n in filtered_combined.nodes} == set(filtered_combined.edges(keys=False))


def test_bare_filter_has_right_nodes_and_edges_for_example(bare_filter):
    assert set(bare_filter.nodes) == {isotopes.U235, isotopes.I135, isotopes.Xe135}
    assert set(bare_filter.edges(keys=False)) == {
        (isotopes.U235, isotopes.I135),
        (isotopes.I135, isotopes.Xe135),
        (isotopes.U235, isotopes.Xe135),
        (isotopes.U235, isotopes.U235),
        (isotopes.I135, isotopes.I135),
        (isotopes.Xe135, isotopes.Xe135),
    }


def test_whitelist_filter_on_combined_graph_example_is_the_bare_example(combined_graph, bare_filter):
    s = {isotopes.I135, isotopes.Xe135, isotopes.U235}
    g = whitelist_filter(combined_graph, s)
    assert set(g.nodes) == set(bare_filter.nodes)
    assert set(g.edges) == set(bare_filter.edges)


def test_to_matrix_on_bare_filter_has_correct_values_calculated_by_hand(bare_filter: nx.MultiDiGraph):
    """Test turning a graph into a matrix."""

    m = graph_to_sparse_matrix(bare_filter, (isotopes.U235, isotopes.I135, isotopes.Xe135))
    assert np.allclose(m[0, :].toarray(), (-7.53, 0, 0)), tuple(bare_filter.edges(keys=True))
    assert isclose(m[1, 0], 2.4)
    assert isclose(m[2, 0], 0.13)
    assert all(m[i, i] < 0 for i in range(3))
