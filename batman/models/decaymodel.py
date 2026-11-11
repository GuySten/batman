"""Decay matrix data.

"""
from collections.abc import Sequence

import networkx as nx
import numpy as np
from isotopes import ZAID

from batman.graphs import DecayGraph
from batman.graphs.matgen import graph_to_sparse_matrix
from batman.units import EV_TO_MJ, MWPerCM3, PerCmBarnArray


class DecayModel:
    """Decay matrix model. Basically a matrix and decay energy wrapper

    """

    def __init__(self, dg: DecayGraph, isos: Sequence[ZAID], *,
                 dtype: str = 'float64',
                 accumulate: frozenset[ZAID] = frozenset()):

        n = len(isos)
        self.mat = graph_to_sparse_matrix(dg, isos,
                                          dtype=dtype,
                                          accumulate=accumulate)
        energy = {iso: -rate * e * EV_TO_MJ
                  for (iso, _, e), (_, _, rate) in
                  zip(nx.selfloop_edges(dg, data='energy', default=0.),
                      nx.selfloop_edges(dg, data='rate', default=0.))
                  }
        self.energy_model = np.fromiter((energy.get(iso, 0.) for iso in isos),
                                        dtype=dtype,
                                        count=n)

    def energy(self, nd: PerCmBarnArray) -> MWPerCM3:
        """Returns the energy emission given some material number densities.

        Parameters
        ----------
        nd - Number densities vector

        """
        return float(np.dot(self.energy_model, nd))
