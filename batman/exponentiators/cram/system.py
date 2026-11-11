"""Block-structured CRAM for burnup matrices, with substeps and shift levels.

This module solves ``n(t) = exp(A t) n0`` for a burnup matrix ``A``
(``A[i, j]`` the rate at which nuclide ``j`` turns into nuclide ``i``) with the
incomplete partial fraction (IPF) form of CRAM, and differs from
:class:`IPFCramSolver` in the following ways.

* **Block triangular ordering.** The nuclides are ordered by the strongly
  connected components (blocks) of the transmutation graph, parents first, so
  ``A`` is block lower triangular. Blocks are grouped by their depth in the
  graph of blocks (longest path from a block without parents), and blocks of
  one depth do not feed each other. Each depth is solved in turn by block
  forward substitution, so only the diagonal blocks are ever factorized and
  fill-in stays inside them (as in KLU). Decay matrices have no cycles: every
  block is a single nuclide, and a CRAM step needs no factorization at all.
* **No row pivoting.** The diagonal blocks are factorized by SuperLU with a
  symmetric fill-reducing ordering and diagonal pivots, as recommended by
  M. Pusa and J. Leppanen, Nucl. Sci. Eng. 175 (2013) 250. Every nuclide keeps
  its own equation as pivot, so a nuclide that nothing feeds stays exactly 0.
* **Substeps with reused factors.** A step of length ``dt`` is taken as
  ``substeps`` steps of ``dt / substeps`` with the same factorizations
  (A. Isotalo and M. Pusa, Nucl. Sci. Eng. 183 (2016) 65).
* **Shift levels (shifted CRAM).** CRAM leaves an absolute error of about
  ``alpha0`` times the largest initial population in every nuclide. For a
  nuclide whose ancestors all decay fast, that is far larger than its own
  value. Such nuclides are solved again, with their ancestors, as
  ``exp(-s t) exp((A + s) t)`` with a shift ``s`` close to their slowest decay
  rate, which makes the error relative to their own scale. With CRAM-48 and 4
  substeps, every nuclide is then computed to about 1e-13 relative to its own
  value.

"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sla

from .pusa import CRAM_ORDERS, cram_coefficients

__all__ = ["BurnupStructure", "cram_apply", "shift_width"]

# Diagonal pivots: a symmetric ordering and a pivot threshold of 0, so SuperLU
# never swaps rows.
_SPLU_KW = dict(permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                options=dict(SymmetricMode=True))
_SPLU_TRIANGULAR_KW = dict(permc_spec="NATURAL", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))

# exp(-814) is far below the smallest double (about 4.9e-324 = exp(-744.4)),
# with room for the number of nuclides.
_UNDERFLOW = 814.0


def shift_width(order: int) -> float:
    """Width of the shift levels of shifted CRAM, in units of decay over a
    substep (mu h): 20 for CRAM-48 and 10 for lower orders.

    A nuclide at the top of its level is solved with a residual decay of up
    to exp(-width) over a substep, so the error of CRAM relative to it is
    about alpha0 * exp(width): 1e-38 for CRAM-48 and 5e-12 for CRAM-16.

    """
    return 20.0 if int(order) >= 48 else 10.0


def _canonical(a) -> sp.csr_matrix:
    """A float64 CSR copy with sorted indices, no duplicates and no explicit
    zeros, so that the pattern says which transmutations exist."""
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return a


def _pattern_key(a: sp.csr_matrix) -> tuple:
    return a.shape, a.indptr.tobytes(), a.indices.tobytes()


@dataclass(frozen=True)
class _Levels:
    """Solve groups of a block lower triangular matrix: rows ``ptr[k]`` to
    ``ptr[k+1]`` form group ``k``, which only earlier groups feed;
    ``cyclic[k]`` tells whether it holds a block of more than one nuclide
    (else its matrix is lower triangular)."""
    ptr: np.ndarray
    cyclic: np.ndarray


class BurnupStructure:
    """Symbolic analysis of one burnup matrix pattern.

    Computed once per pattern and reused for every matrix with that pattern
    (every time step and flux level of one component).

    Attributes
    ----------
    n - Number of nuclides.
    perm - New (elimination) order to original index: ``perm[k]`` is the
           original index of the k-th nuclide.
    n_blocks - Number of strongly connected components.
    block_ptr - Boundaries of the blocks in the new order.
    acyclic - True when every block is a single nuclide (a decay matrix).

    """

    def __init__(self, a: sp.csr_matrix):
        a = _canonical(a)
        n = a.shape[0]
        if a.shape != (n, n):
            raise ValueError(f"Burnup matrix must be square, not {a.shape}")
        self.n = n
        coo = a.tocoo()
        off = coo.row != coo.col
        rows, cols = coo.row[off], coo.col[off]

        n_blocks, labels = csgraph.connected_components(
            a, directed=True, connection="strong")
        sizes = np.bincount(labels, minlength=n_blocks)

        # Graph of the blocks: an entry A[i, j] makes the block of j a parent
        # of the block of i.
        child, parent = labels[rows], labels[cols]
        cross = child != parent
        code = np.unique(child[cross].astype(np.int64) * n_blocks
                         + parent[cross])
        b_child, b_parent = code // n_blocks, code % n_blocks
        depth = _longest_path_depth(n_blocks, b_parent, b_child)

        # New order: by depth, then block, then original index.
        perm = np.lexsort((np.arange(n), labels, depth[labels]))
        self.perm = perm
        self.iperm = np.empty(n, dtype=np.intp)
        self.iperm[perm] = np.arange(n)

        # Renumber the blocks in the new order (parents get lower numbers).
        old_labels = labels[perm]
        starts = np.flatnonzero(np.r_[True, old_labels[1:] != old_labels[:-1]])
        self.n_blocks = len(starts)
        self.block_ptr = np.r_[starts, n].astype(np.intp)
        new_of_old = np.empty(n_blocks, dtype=np.intp)
        new_of_old[old_labels[starts]] = np.arange(self.n_blocks)
        self.nuc_block = new_of_old[old_labels]
        self.block_depth = depth[old_labels[starts]]
        self.nuc_depth = self.block_depth[self.nuc_block]
        self.nuc_cyclic = sizes[old_labels] > 1
        self.acyclic = not self.nuc_cyclic.any()

        # Permuted pattern, and where each of its entries comes from in A.
        marker = sp.csr_matrix((np.arange(1, a.nnz + 1, dtype=np.float64),
                                a.indices, a.indptr), shape=a.shape)
        p = marker[perm][:, perm].tocsr()
        p.sort_indices()
        self._p_indptr, self._p_indices = p.indptr, p.indices
        self._data_index = p.data.astype(np.int64) - 1
        p_rows = np.repeat(np.arange(n), np.diff(p.indptr))
        self._in_block = self.nuc_block[p_rows] == self.nuc_block[p.indices]
        self._in_block_cols = p.indices[self._in_block]

        self.levels = _levels_of(self.nuc_depth, self.nuc_cyclic)

        # Block edges in the new numbering, grouped by the depth of the child
        # (parents always lie at a smaller depth).
        e_child, e_parent = new_of_old[b_child], new_of_old[b_parent]
        order = np.argsort(self.block_depth[e_child], kind="stable")
        self._e_child, self._e_parent = e_child[order], e_parent[order]
        e_depth = self.block_depth[self._e_child]
        n_depths = int(self.block_depth.max()) + 1 if self.n_blocks else 0
        self._e_ptr = np.searchsorted(e_depth, np.arange(n_depths + 1))

    def permuted(self, a: sp.csr_matrix) -> sp.csr_matrix:
        """``A`` in the new order. ``a`` must be canonical (see
        :func:`_canonical`) with the pattern of this structure."""
        return sp.csr_matrix((a.data[self._data_index], self._p_indices,
                              self._p_indptr), shape=a.shape)

    def block_rates(self, p: sp.csr_matrix) -> np.ndarray:
        """A lower bound on the slowest decay rate of every block: its
        smallest column leak (Gershgorin). Exact for single nuclides."""
        colsum = np.bincount(self._in_block_cols,
                             weights=p.data[self._in_block],
                             minlength=self.n)
        return np.minimum.reduceat(-colsum, self.block_ptr[:-1])

    def edges_at_depth(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        """(parent, child) block pairs whose child is at depth ``d``."""
        lo, hi = self._e_ptr[d], self._e_ptr[d + 1]
        return self._e_parent[lo:hi], self._e_child[lo:hi]

    @property
    def n_depths(self) -> int:
        return len(self._e_ptr) - 1


def _longest_path_depth(n: int, parent: np.ndarray,
                        child: np.ndarray) -> np.ndarray:
    """Depth of every node of a DAG: 0 for nodes without parents, else one
    more than the deepest parent (Kahn's algorithm, one front at a time)."""
    depth = np.zeros(n, dtype=np.intp)
    if not len(parent):
        return depth
    g = sp.csr_matrix((np.ones(len(parent), dtype=np.int8), (parent, child)),
                      shape=(n, n))
    indeg = np.bincount(child, minlength=n)
    front = np.flatnonzero(indeg == 0)
    level = 0
    while front.size:
        depth[front] = level
        children = g[front].indices
        indeg -= np.bincount(children, minlength=n)
        children = np.unique(children)
        front = children[indeg[children] == 0]
        level += 1
    return depth


def _levels_of(nuc_depth: np.ndarray, nuc_cyclic: np.ndarray) -> _Levels:
    """Solve groups of nuclides already sorted by depth.

    Every depth level that holds a block of more than one nuclide is a group
    of its own: its blocks do not feed each other, so its factors have no
    fill between them. Consecutive levels of single nuclides are merged into
    one group: their matrix is lower triangular, and its factors have no
    fill at all in the natural order. A decay matrix is then a single group,
    solved by one sparse triangular solve per pole.

    """
    n = len(nuc_depth)
    if not n:
        return _Levels(np.zeros(1, dtype=np.intp), np.zeros(0, dtype=bool))
    starts = np.flatnonzero(np.r_[True, nuc_depth[1:] != nuc_depth[:-1]])
    cyclic = np.logical_or.reduceat(nuc_cyclic, starts)
    # a level starts a new group unless it and the level before are acyclic
    new_group = np.r_[True, cyclic[1:] | cyclic[:-1]]
    ptr = np.r_[starts[new_group], n].astype(np.intp)
    return _Levels(ptr, cyclic[new_group])


class _Factors:
    """Everything a CRAM substep of ``m = h A`` needs, for every pole: the
    parts of ``m`` below the diagonal blocks of every level, and the inverse
    diagonals or LU factors of ``m - theta`` on the diagonal blocks."""

    def __init__(self, m: sp.csr_matrix, levels: _Levels, theta: np.ndarray):
        self.levels = levels
        self.n_poles = len(theta)
        diag = m.diagonal()
        self.off = []
        self.inv_diag = []
        self.lu = []
        for k in range(len(levels.cyclic)):
            lo, hi = levels.ptr[k], levels.ptr[k + 1]
            rows = m[lo:hi]
            off = rows[:, :lo] if lo else None
            self.off.append(off if off is not None and off.nnz else None)
            block = rows[:, lo:hi]
            if block.nnz > np.count_nonzero(block.diagonal()) or levels.cyclic[k]:
                # cyclic: a symmetric fill-reducing order inside the blocks;
                # triangular: the natural order, which has no fill
                kw = _SPLU_KW if levels.cyclic[k] else _SPLU_TRIANGULAR_KW
                block = block.tocsc().astype(np.complex128)
                eye = sp.eye(hi - lo, format="csc", dtype=np.complex128)
                self.lu.append([sla.splu(block - t * eye, **kw)
                                for t in theta])
                self.inv_diag.append(None)
            else:
                self.lu.append(None)
                self.inv_diag.append(
                    1.0 / (diag[lo:hi, None] - theta[None, :]))


def _cram_substeps(f: _Factors, y: np.ndarray, alpha: np.ndarray,
                   alpha0: float, substeps: int,
                   scale: float = 1.0) -> np.ndarray:
    """``substeps`` IPF CRAM steps of y with the factors ``f``; after every
    substep the result is multiplied by ``scale`` (in two halves, so that
    every factor stays a normal number)."""
    levels = f.levels
    n_poles = f.n_poles
    x = np.zeros((len(y), n_poles), dtype=np.complex128)
    xr = x.view(np.float64)  # (n, 2 poles), for real sparse products
    two_alpha = 2.0 * alpha
    half = math.sqrt(scale) if scale != 1.0 else 1.0
    y = np.array(y, dtype=np.float64)
    for _ in range(substeps):
        for k in range(len(levels.cyclic)):
            lo, hi = levels.ptr[k], levels.ptr[k + 1]
            yl = y[lo:hi]
            off = f.off[k]
            s = (off @ xr[:lo]).view(np.complex128) if off is not None else None
            inv = f.inv_diag[k]
            lus = f.lu[k]
            for j in range(n_poles):
                r = yl if s is None else yl - s[:, j]
                xj = r * inv[:, j] if lus is None else lus[j].solve(
                    r.astype(np.complex128))
                x[lo:hi, j] = xj
                yl = yl + (two_alpha[j] * xj).real
            y[lo:hi] = yl * alpha0
        if half != 1.0:
            y *= half
            y *= half
    return y


def _shift_levels(st: BurnupStructure, p: sp.csr_matrix, h: float,
                  substeps: int, y0: np.ndarray, y: np.ndarray,
                  alpha: np.ndarray, theta: np.ndarray, alpha0: float,
                  width: float) -> None:
    """Shifted CRAM: solve again, in place in ``y`` (new order), the nuclides
    whose ancestors all decay fast over a substep of length ``h``.

    The active nuclides are those that ``y0`` or their ancestors hold atoms
    in. ``mu`` of a block is the slowest decay rate of it and its active
    ancestors (by Gershgorin, the smallest column leak of a block). Blocks
    with ``mu h`` in ``[k width, (k+1) width)``, ``k >= 1``, form level ``k``;
    each level is solved again with its active ancestors, as
    ``exp(-s t) exp((A + s) t)`` with ``s = k width / h`` (at most the smallest
    removal rate of the subset), so that CRAM's error is relative to the
    level's own scale. Its values replace those of ``y``. Blocks whose ``mu h``
    exceeds the range of doubles are set to exactly 0.

    """
    top = float(np.max(np.abs(y0))) if len(y0) else 0.0
    if not (top > 0.0) or not math.isfinite(top):
        return
    cut = math.log(top) + _UNDERFLOW
    if cut < width:
        return

    nb = st.n_blocks
    rate = st.block_rates(p)
    active = np.zeros(nb, dtype=bool)
    active[st.nuc_block[y0 != 0.0]] = True
    mu = np.maximum(rate, 0.0)
    for d in range(1, st.n_depths):
        par, ch = st.edges_at_depth(d)
        if not len(par):
            continue
        live = active[par]
        par, ch = par[live], ch[live]
        active[ch] = True
        np.minimum.at(mu, ch, mu[par])

    mh = np.where(active, mu * h, 0.0)
    dead = active & (mh >= cut)
    if dead.any():
        y[dead[st.nuc_block]] = 0.0
    level = np.where(active & ~dead, np.floor(mh / width), 0.0).astype(np.int64)
    if level.max(initial=0) <= 0:
        return

    diag = p.diagonal()
    for lv in np.unique(level[level > 0])[::-1]:
        members = level == lv
        subset = members.copy()
        for d in range(int(st.block_depth[members].max()), 0, -1):
            par, ch = st.edges_at_depth(d)
            if not len(par):
                continue
            up = subset[ch] & active[par]
            subset[par[up]] = True
        sub = np.flatnonzero(subset[st.nuc_block])
        shift = min(lv * width / h, float(np.min(-diag[sub])))
        m = h * p[sub][:, sub] + sp.eye(len(sub), format="csr") * (h * shift)
        levels = _levels_of(st.nuc_depth[sub], st.nuc_cyclic[sub])
        f = _Factors(sp.csr_matrix(m), levels, theta)
        u = _cram_substeps(f, y0[sub], alpha, alpha0, substeps,
                           scale=math.exp(-shift * h))
        keep = members[st.nuc_block[sub]]
        y[sub[keep]] = u[keep]


def cram_apply(a, n0: np.ndarray, t: float, *, order: int = 48,
               substeps: int = 4, width: Optional[float] = None,
               structure: Optional[BurnupStructure] = None) -> np.ndarray:
    """Compute ``exp(A t) n0`` with block-structured CRAM.

    Parameters
    ----------
    a - Burnup matrix (any SciPy sparse format), ``A[i, j]`` the rate at which
        nuclide ``j`` turns into nuclide ``i``.
    n0 - Initial vector.
    t - Time step, in the units of the rates (non-negative).
    order - CRAM order, one of 4, 8, ..., 48.
    substeps - Number of substeps the factorizations are reused over.
    width - Width of the shift levels (shifted CRAM); 0 turns shifting off.
            None picks :func:`shift_width` of the order.
    structure - Symbolic analysis of the pattern of ``a``, if already known.

    Returns
    -------
    The vector after ``t``, in the original order.

    """
    if order not in CRAM_ORDERS:
        raise ValueError(f"CRAM order must be one of {CRAM_ORDERS}, not {order}")
    if int(substeps) < 1:
        raise ValueError(f"substeps must be at least 1, not {substeps}")
    if t < 0:
        raise ValueError(f"Time step must not be negative, not {t}")
    a = _canonical(a)
    n0 = np.asarray(n0, dtype=np.float64)
    if n0.shape != (a.shape[0],):
        raise ValueError(f"Vector of shape {n0.shape} does not fit a matrix "
                         f"of shape {a.shape}")
    if t == 0 or not len(n0):
        return n0.copy()
    if structure is None:
        structure = BurnupStructure(a)
    width = shift_width(order) if width is None else float(width)
    alpha, theta, alpha0 = cram_coefficients(order)
    st = structure
    p = st.permuted(a)
    h = t / int(substeps)
    y0 = n0[st.perm]
    f = _Factors(h * p, st.levels, theta)
    y = _cram_substeps(f, y0, alpha, alpha0, int(substeps))
    if width > 0:
        _shift_levels(st, p, h, int(substeps), y0, y, alpha, theta, alpha0,
                      width)
    out = np.empty_like(y)
    out[st.perm] = y
    return out


class StructureCache:
    """A small least-recently-used cache of :class:`BurnupStructure` objects,
    keyed by matrix pattern. Not pickled (it is rebuilt where it is used)."""

    def __init__(self, size: int = 8):
        self.size = int(size)
        self._items: OrderedDict = OrderedDict()

    def get(self, a: sp.csr_matrix) -> BurnupStructure:
        key = _pattern_key(a)
        try:
            self._items.move_to_end(key)
            return self._items[key]
        except KeyError:
            st = BurnupStructure(a)
            self._items[key] = st
            if len(self._items) > self.size:
                self._items.popitem(last=False)
            return st

    def __getstate__(self):
        return {"size": self.size}

    def __setstate__(self, state):
        self.size = state["size"]
        self._items = OrderedDict()
