"""Tools to generate the numerical derivative matrix.

Currently solely based on decay and reaction graphs.

"""
from collections.abc import Hashable, Iterable

import networkx as nx
from isotopes import ZAID
from scipy.sparse import csr_matrix, dok_matrix


def graph_to_sparse_matrix(g: nx.MultiDiGraph,
                           order: Iterable[Hashable] | None = None,
                           dtype: str | None = None,
                           accumulate: frozenset[ZAID] = frozenset()) -> \
        csr_matrix:
    """Generate a sparse matrix from the decay and reaction graph.
    Assumes the graph has all the per-second constants normalized to the
    current power etc, in the correct units.

    Parameters
    ----------
    g - Directed graph of reaction and decay processes between isotopes.
    order - The order of the equations, by isotope. If an isotope does not
            appear in the order but does appear in g, no equation for it is
            generated.
    dtype - Valid NumPy dtype to initialize the matrix. None means NumPy
            default.
    accumulate - Set of nodes to not ever decrease. Used for debugging, but
                 also for cases where some isotope is modeled constant but it
                 does generate other things.

    Returns
    -------

    A sparse matrix whose ij component is the reaction rate that turns isotope
    i into isotope j, per isotope i nucleus.

    """

    order = tuple(order) if order else tuple(g.nodes)
    if g.number_of_nodes() and g.number_of_edges():  # networkx fails otherwise
        g_reversed = g.reverse(copy=True)
        g_reversed.add_nodes_from(order-g_reversed.nodes)
        m = dok_matrix(nx.convert_matrix.to_scipy_sparse_array(g_reversed,
                                                     nodelist=order,
                                                     dtype=dtype,
                                                     weight='rate',
                                                     format='dok'))
    else:
        n = len(order)
        m = dok_matrix((n, n))
    if accumulate:
        for iso in accumulate:
            index = order.index(iso)
            m[index, index] = 0.
    m = m.tocsr(copy=False)
    return m
