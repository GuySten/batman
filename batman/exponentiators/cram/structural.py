"""CRAM for sparse matrices of production and removal rates, solved through
their sparsity structure.

Usage
-----
:class:`StructuralCram` (or the function :func:`cram_apply`) returns
``exp(A t) v`` for a sparse matrix ``A`` whose off-diagonal entries are
non-negative: ``A[i, j] >= 0`` for ``i != j`` is the rate at which compartment
``j`` produces compartment ``i``, and ``-A[j, j]`` is the rate at which ``j``
is removed. A column may hand on *more* than it loses, so the total need not
be conserved: fission produces about two fission products per fissioning
nucleus, and alpha decay both a daughter and He-4. The solver knows nothing
about where ``A`` comes from. In batman, ``A`` is a burnup matrix and the
compartments are nuclides; :class:`batman.exponentiators.ConstantFlux` forms
it from the decay and reaction matrices of a step.

Full accuracy needs one more property, which batman's matrices have: inside
every strongly connected block (see below), a compartment hands back to its
own block no more than it loses, so no block grows. By Gershgorin's theorem
on the columns, every eigenvalue of ``A`` then lies in the left half-plane,
where CRAM is accurate. The extra products of fission and alpha decay do not
break this, because they leave the parent's block: fission products never
turn back into actinides, nor He-4 into heavy nuclides. Production above 1
per removal between blocks is allowed anywhere.

Everything that depends only on *where* ``A`` has nonzero entries is computed
once per sparsity pattern (:class:`MatrixStructure`) and cached by pattern
(:func:`structure_of`), so matrices with the same pattern and new values
reuse it.

The problem
-----------
CRAM approximates ``exp(x)`` on the negative real axis by a rational function
with a uniform *absolute* error ``alpha0`` (2.3e-47 for CRAM-48). Applied to a
matrix, the error of ``exp(A t) v`` is about ``alpha0`` times the largest entry
of ``v``, in every compartment. That is negligible for the large entries of
the result, but not for an entry far below the largest one. A compartment fed
only by compartments that empty within the step holds a tiny fraction of the
initial contents (``exp(-rate t)``, which can be 1e-130 or less), and plain
CRAM gets it wrong by many orders of magnitude, or produces contents where
there should be none. In burnup this is a short-lived daughter of a nuclide
that decays away within the step;
``test_short_lived_daughter_is_accurate_relative_to_itself`` in
``tests/test_blockcram.py`` shows plain IPF CRAM-48 off by more than 1000x on
a two-compartment chain, and this module within 1e-12.

The strategy
------------
1. **IPF CRAM** (M. Pusa, Nucl. Sci. Eng. 182 (2016) 297, incomplete partial
   fraction form). With ``M = h A`` for a substep of length ``h``::

       exp(M) v  ~=  alpha0 * prod_j (I + 2 Re[alpha_j (M - theta_j I)^-1]) v

   over the ``order / 2`` poles ``theta_j`` with residues ``alpha_j``. The
   factors are applied one at a time, so each pole needs one linear solve
   ``(M - theta_j I) x = y`` (:func:`_cram_substeps`).
2. **Block structure.** The compartments are ordered by the strongly
   connected components (*blocks*) of the graph of ``A`` (an edge ``j -> i``
   for every ``A[i, j] != 0``), parents first, so ``M`` is block lower
   triangular. The solves are then a block forward substitution, and only the
   diagonal blocks are ever factorized. A matrix without cycles is triangular
   and needs no fill-producing factorization (:class:`MatrixStructure`,
   :func:`_solve_groups`).
3. **No row pivoting** (M. Pusa and J. Leppanen, Nucl. Sci. Eng. 175 (2013)
   250). Every compartment keeps its own equation as pivot, so a compartment
   that nothing feeds stays exactly 0 (:func:`_factor_poles`).
4. **Substeps** (A. Isotalo and M. Pusa, Nucl. Sci. Eng. 183 (2016) 65):
   ``exp(A t) = exp(A h)^s`` with ``h = t / s``, reusing the factorizations of
   step 1 for all ``s`` substeps, so each extra substep costs only its solves.
   CRAM is most accurate when entries change little within one step, and
   substeps limit the change per substep. Here that matters for the smallest
   entries, the ones the shift levels correct: in a decay step of irradiated
   fuel (a fuel-like test chain, against an exact solver), one step left
   entries below 1e-100 of the largest off by up to 7e4, and four substeps
   kept every entry within about 1e-13. Larger entries are at round-off
   either way.
5. **Shift levels**, which solve the problem above. For any number ``s``,
   ``exp(A t) = exp(-s t) exp((A + s I) t)``. For a compartment whose
   ancestors (the compartments that feed it, directly or not) all empty faster
   than ``s``, the shifted matrix decays slowly, so CRAM's absolute error on
   it is relative to the compartment's own scale, and the factor
   ``exp(-s t)`` is exact. :func:`_shift_levels` finds such compartments,
   groups them by how fast their ancestors empty, and solves each group again
   with its own shift. Only properties of ``A`` itself are needed: its
   pattern, and the signs of its off-diagonal entries.

With CRAM-48, 4 substeps and shift levels, every compartment is accurate to
about 1e-13 relative to its own value (``test_decay_chain_against_bateman``,
``test_high_shift_levels_do_not_underflow`` and the tests around them),
fission-like production included (``test_yield_two_daughter_accurate_relative_to_itself``).
If a block can grow (one of its columns hands back to it more than it loses),
the solver still works, as accurate as CRAM is for that block's eigenvalues,
and the block gets no shift.

Notation below: ``A`` is the matrix permuted into the structure's order,
``h`` the substep length, ``M = h A``, and ``theta_j``, ``alpha_j`` and
``alpha0`` the poles, residues and limit at infinity of the CRAM order.

"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse import csgraph

from .pusa import CRAM_ORDERS, CramOrder, cram_coefficients

__all__ = ["MatrixStructure", "StructuralCram", "cram_apply", "shift_width", "structure_of"]

# Diagonal pivots: a pivot threshold of 0 in symmetric mode, so SuperLU never
# swaps rows. The matrices come in an order that already keeps the fill small
# (see MatrixStructure), so SuperLU does not order them again.
_SPLU_KW = {"permc_spec": "NATURAL", "diag_pivot_thresh": 0.0,
            "options": {"SymmetricMode": True}}
# Fill-reducing ordering of the cyclic groups, computed once per pattern.
_ORDERING_KW = {"permc_spec": "MMD_AT_PLUS_A", "diag_pivot_thresh": 0.0,
                "options": {"SymmetricMode": True}}

# exp(-814) is far below the smallest double (about 4.9e-324 = exp(-744.4)),
# with room (about 70 e-folds) for the number of compartments and for
# production above 1 per removal along a chain.
_UNDERFLOW = 814.0


def shift_width(order: int) -> float:
    """Width of the shift levels of shifted CRAM, in units of decay over a
    substep (mu h): 20 for CRAM-48 and 10 for lower orders.

    A compartment at the top of its level is solved with a residual decay of
    up to exp(-width) over a substep, so the error of CRAM relative to it is
    about alpha0 * exp(width): 1e-38 for CRAM-48 and 5e-12 for CRAM-16.

    """
    return 20.0 if int(order) >= 48 else 10.0


def _canonical(a) -> sp.csr_matrix:
    """A float64 CSR copy with sorted indices, no duplicates and no explicit
    zeros, so that the pattern says which transfers exist."""
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return a


@dataclass(frozen=True)
class _SolveGroups:
    """Groups of rows of a block lower triangular matrix, solved one after
    the other: rows ``ptr[k]`` to ``ptr[k+1]`` form group ``k``, which only
    earlier groups feed. ``cyclic[k]`` tells whether the group holds a block
    of more than one compartment; if it does not, its own matrix is lower
    triangular."""
    ptr: np.ndarray
    cyclic: np.ndarray


class MatrixStructure:
    """Structural analysis of a sparse matrix: everything that depends only
    on which of its entries are nonzero, not on their values.

    It is computed once per pattern and reused for every matrix with that
    pattern (in batman, every time step and flux level of one component: the
    values change from step to step, but the transfers that exist do not).

    Attributes
    ----------
    n - Number of compartments (rows).
    perm - New (elimination) order to original index: ``perm[k]`` is the
           original index of the k-th compartment.
    groups - The solve groups (:class:`_SolveGroups`) in the new order.
    n_blocks - Number of strongly connected components (blocks).
    block_of - Block of every compartment (new order). Blocks are numbered so
               that parents come before their children.
    depth_of - Depth of the block of every compartment (new order).
    cyclic_of - Whether every compartment's block holds more than one
                compartment (new order).
    block_depth - Depth of every block in the graph of blocks: 0 for a block
                  without parents, else one more than its deepest parent.
    acyclic - True when every block is a single compartment.

    """

    def __init__(self, a: sp.csr_matrix):
        a = _canonical(a)
        n = a.shape[0]
        if a.shape != (n, n):
            raise ValueError(f"The matrix must be square, not {a.shape}")
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

        # Renumber the blocks in the new order (parents get lower numbers).
        old_labels = labels[perm]
        starts = np.flatnonzero(np.r_[True, old_labels[1:] != old_labels[:-1]])
        self.n_blocks = len(starts)
        new_of_old = np.empty(n_blocks, dtype=np.intp)
        new_of_old[old_labels[starts]] = np.arange(self.n_blocks)
        block_of = new_of_old[old_labels]
        self.block_depth = depth[old_labels[starts]]
        cyclic_of = sizes[old_labels] > 1
        self.groups = _solve_groups(self.block_depth[block_of], cyclic_of)

        # Inside every cyclic group, a fill-reducing order of its blocks (they
        # do not feed each other, so any order of the group is valid).
        local = np.arange(n)
        for k in np.flatnonzero(self.groups.cyclic):
            lo, hi = self.groups.ptr[k], self.groups.ptr[k + 1]
            local[lo:hi] = lo + _fill_reducing_order(a[perm[lo:hi]][:, perm[lo:hi]])
        perm = perm[local]
        self.perm = perm
        self.iperm = np.empty(n, dtype=np.intp)
        self.iperm[perm] = np.arange(n)
        self.block_of = block_of[local]
        self.depth_of = self.block_depth[self.block_of]
        self.cyclic_of = cyclic_of[local]
        self.acyclic = not self.cyclic_of.any()

        # Permuted pattern, and where each of its entries comes from in A.
        marker = sp.csr_matrix((np.arange(1, a.nnz + 1, dtype=np.float64),
                                a.indices, a.indptr), shape=a.shape)
        p = marker[perm][:, perm].tocsr()
        p.sort_indices()
        self._p_indptr, self._p_indices = p.indptr, p.indices
        self._data_index = p.data.astype(np.int64) - 1
        p_rows = np.repeat(np.arange(n), np.diff(p.indptr))
        self._in_block = self.block_of[p_rows] == self.block_of[p.indices]
        self._in_block_cols = p.indices[self._in_block]

        # Parent -> child links between blocks, in the new numbering, sorted
        # by the depth of the child (parents always lie at a smaller depth).
        e_child, e_parent = new_of_old[b_child], new_of_old[b_parent]
        order = np.argsort(self.block_depth[e_child], kind="stable")
        self._link_child, self._link_parent = e_child[order], e_parent[order]
        link_depth = self.block_depth[self._link_child]
        n_depths = int(self.block_depth.max()) + 1 if self.n_blocks else 0
        self._link_ptr = np.searchsorted(link_depth, np.arange(n_depths + 1))

    def permuted(self, a: sp.csr_matrix) -> sp.csr_matrix:
        """``A`` in the new order. ``a`` must be canonical (see
        :func:`_canonical`) with the pattern of this structure."""
        return sp.csr_matrix((a.data[self._data_index], self._p_indices,
                              self._p_indptr), shape=a.shape)

    def block_decay_bounds(self, p: sp.csr_matrix) -> np.ndarray:
        """A lower bound on how fast every block empties, for the shift levels.

        The *decay rate* of a block is the slowest rate at which its contents
        leave it when nothing feeds it from outside: the smallest decay rate
        of the modes of its diagonal block of ``A``. For a single compartment,
        that is its own removal rate. For a cyclic block, contents that move
        between its compartments have not left it, so each compartment's rate
        of leaving the block is its *leak*: the column of the block's part of
        ``-A``, summed (removal minus what goes back into the block). By
        Gershgorin's theorem on the columns, which needs only non-negative
        off-diagonal entries, no mode of the block decays slower than its
        smallest leak, which this returns (one value per block; exact for
        single compartments).

        Only each block's own columns enter, so what a column produces in
        *other* blocks does not matter, however much it is (the fission
        products of an actinide, for example). A block that can grow has a
        negative leak; the shift levels clip the bound at 0, so such a block
        gets no shift.

        Parameters
        ----------
        p - The matrix in this structure's order, as :meth:`permuted` returns it.

        """
        colsum = np.bincount(self._in_block_cols,
                             weights=p.data[self._in_block],
                             minlength=self.n)
        rate = np.full(self.n_blocks, np.inf)
        np.minimum.at(rate, self.block_of, -colsum)
        return rate

    def links_into_depth(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        """The parent -> child links between blocks whose child is at depth
        ``d``, for walking the block graph one depth at a time.

        Returns
        -------
        Two arrays of block numbers of the same length, ``parents`` and
        ``children``: ``parents[i]`` feeds ``children[i]``, and every child is
        at depth ``d``. Each link appears once. A child with several parents
        appears once per parent, and a parent with several children at depth
        ``d`` once per child. Parents can be at any smaller depth; a parent
        with no child at depth ``d`` does not appear. Both arrays are empty
        for ``d = 0``, whose blocks have no parents.

        """
        lo, hi = self._link_ptr[d], self._link_ptr[d + 1]
        return self._link_parent[lo:hi], self._link_child[lo:hi]

    @property
    def n_depths(self) -> int:
        """Number of depths in the graph of blocks"""
        return len(self._link_ptr) - 1


def structure_of(a: sp.csr_matrix) -> MatrixStructure:
    """The :class:`MatrixStructure` of a canonical matrix (see
    :func:`_canonical`), cached by its pattern: the most recent patterns are
    kept, so matrices with a pattern seen before are not analyzed again."""
    return _structure_of_pattern(a.shape[0], a.indptr.dtype.str, a.indptr.tobytes(),
                                 a.indices.dtype.str, a.indices.tobytes())


@functools.lru_cache(maxsize=8)
def _structure_of_pattern(n: int, indptr_dtype: str, indptr: bytes,
                          indices_dtype: str, indices: bytes) -> MatrixStructure:
    # The structure depends on the pattern only, which (unlike a sparse
    # matrix) is hashable as bytes, so the standard cache keys on it.
    indptr_array = np.frombuffer(indptr, dtype=indptr_dtype)
    indices_array = np.frombuffer(indices, dtype=indices_dtype)
    pattern = sp.csr_matrix((np.ones(len(indices_array)), indices_array, indptr_array),
                            shape=(n, n))
    return MatrixStructure(pattern)


def _fill_reducing_order(block: sp.csr_matrix) -> np.ndarray:
    """A symmetric fill-reducing order (new to old) of a square pattern,
    as SuperLU's minimum degree ordering on A + A^T chooses it (with the
    elimination tree postorder). Depends on the pattern only, so a
    diagonally dominant matrix with that pattern stands in for the values."""
    m = block.shape[0]
    pattern = sp.csr_matrix((np.ones(block.nnz), block.indices, block.indptr),
                            shape=block.shape)
    pattern.setdiag(0.0)
    pattern = pattern - sp.eye(m) * (m + 1.0)
    lu = sla.splu(pattern.tocsc(), **_ORDERING_KW)
    return np.argsort(lu.perm_c)


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


def _solve_groups(depth_of: np.ndarray, cyclic_of: np.ndarray) -> _SolveGroups:
    """Solve groups of compartments already sorted by depth.

    Every depth that holds a block of more than one compartment is a group of
    its own: its blocks do not feed each other, so its factors have no fill
    between them. Consecutive depths of single compartments are merged into
    one group: their matrix is lower triangular, and its factors have no fill
    at all in the natural order. A matrix without cycles is then a single
    group, solved by one sparse triangular solve per pole.

    """
    n = len(depth_of)
    if not n:
        return _SolveGroups(np.zeros(1, dtype=np.intp), np.zeros(0, dtype=bool))
    starts = np.flatnonzero(np.r_[True, depth_of[1:] != depth_of[:-1]])
    cyclic = np.logical_or.reduceat(cyclic_of, starts)
    # a depth starts a new group unless it and the depth before are acyclic
    new_group = np.r_[True, cyclic[1:] | cyclic[:-1]]
    ptr = np.r_[starts[new_group], n].astype(np.intp)
    return _SolveGroups(ptr, cyclic[new_group])


class _Factors:
    """Everything one CRAM substep of ``M = h A`` needs, prepared once and
    reused by every substep.

    Each pole ``theta_j`` of the IPF form needs one solve of
    ``(M - theta_j I) x = y`` (see the module notes). Solved by block forward
    substitution, group ``k`` (rows ``lo:hi``) needs two things:

    * ``off[k]``: the rows of ``M`` of the group, left of it (columns
      ``:lo``). It brings in what earlier groups feed into the group, and is
      the same for every pole (None when nothing feeds the group).
    * The solver of the group's own diagonal block ``M_kk - theta_j I``, for
      every pole: ``inv_diag[k]`` (an ``(hi - lo, poles)`` array of
      ``1 / (M_ii - theta_j)``) when that block is diagonal, otherwise
      ``lu[k]``, one SuperLU factorization per pole (:func:`_factor_poles`).

    """

    def __init__(self, m: sp.csr_matrix, groups: _SolveGroups, theta: np.ndarray):
        self.groups = groups
        self.n_poles = len(theta)
        diag = m.diagonal()
        self.off = []
        self.inv_diag = []
        self.lu = []
        for k in range(len(groups.cyclic)):
            lo, hi = groups.ptr[k], groups.ptr[k + 1]
            rows = m[lo:hi]
            off = rows[:, :lo] if lo else None
            self.off.append(off if off is not None and off.nnz else None)
            block = rows[:, lo:hi]
            if block.nnz > np.count_nonzero(block.diagonal()):
                self.lu.append(_factor_poles(block, theta))
                self.inv_diag.append(None)
            else:
                self.lu.append(None)
                self.inv_diag.append(
                    1.0 / (diag[lo:hi, None] - theta[None, :]))


def _factor_poles(block: sp.csr_matrix, theta: np.ndarray) -> list:
    """SuperLU factors of ``block - theta_j I`` for every pole ``theta_j``.

    ``block`` is the diagonal block ``M_kk`` of one solve group, and the
    matrices ``M_kk - theta_j I`` are the ones its IPF solves need (one per
    pole; see :class:`_Factors`). They are factorized without row pivoting
    and in the given order, which keeps fill small: lower triangular groups
    have none, and cyclic ones are already in their fill-reducing order. The
    pattern is built once; every pole only changes the diagonal.

    """
    m = block.shape[0]
    coo = block.tocoo()
    rows = np.r_[coo.row, np.arange(m)]
    cols = np.r_[coo.col, np.arange(m)]
    vals = np.r_[coo.data, np.zeros(m)].astype(np.complex128)
    base = sp.csc_matrix((vals, (rows, cols)), shape=(m, m))  # sums a + 0
    base.sort_indices()
    col_of = np.repeat(np.arange(m), np.diff(base.indptr))
    diag_pos = np.flatnonzero(base.indices == col_of)
    factors = []
    for t in theta:
        data = base.data.copy()
        data[diag_pos] -= t
        factors.append(sla.splu(
            sp.csc_matrix((data, base.indices, base.indptr), shape=(m, m)),
            **_SPLU_KW))
    return factors


def _cram_substeps(f: _Factors, y: np.ndarray, alpha: np.ndarray,
                   alpha0: float, substeps: int,
                   half: float | np.ndarray = 1.0) -> np.ndarray:
    """``substeps`` IPF CRAM steps of ``y`` with the factors ``f``: each one
    is ``y <- alpha0 * prod_j (I + 2 Re[alpha_j (M - theta_j I)^-1]) y``.

    The factors of the product are applied one pole at a time. Each needs
    ``x_j = (M - theta_j I)^-1 y_j``, where ``y_j`` is ``y`` after the
    previous poles, followed by ``y_{j+1} = y_j + 2 Re(alpha_j x_j)``. With
    ``M`` block lower triangular, ``x_j`` is found group by group: group
    ``k`` solves ``(M_kk - theta_j I) x_j[k] = y_j[k] - M_k,<k x_j[<k]``,
    using the parts of ``x_j`` the earlier groups have already found. The
    poles of one group only need ``y`` on that group, so each group runs
    through all the poles before the next group starts.

    ``half`` scales the result by ``half ** 2`` after every substep, either
    one number or one per entry. Only the shift levels use it: a level solved
    with the shifted matrix ``A + s I`` is scaled back by ``exp(-s h)`` per
    substep. That scale is applied as two halves because ``exp(-s h)`` alone
    can underflow to 0 when the entries it multiplies are large
    (``test_high_shift_levels_do_not_underflow``).

    """
    groups = f.groups
    n_poles = f.n_poles
    x = np.zeros((len(y), n_poles), dtype=np.complex128)
    xr = x.view(np.float64)  # (n, 2 poles), for real sparse products
    two_alpha = 2.0 * alpha
    scaled = not (np.isscalar(half) and half == 1.0)
    y = np.array(y, dtype=np.float64)
    for _ in range(substeps):
        for k in range(len(groups.cyclic)):
            lo, hi = groups.ptr[k], groups.ptr[k + 1]
            yk = y[lo:hi]
            off = f.off[k]
            fed = (off @ xr[:lo]).view(np.complex128) if off is not None else None
            inv = f.inv_diag[k]
            lus = f.lu[k]
            for j in range(n_poles):
                rhs = yk if fed is None else yk - fed[:, j]
                xj = rhs * inv[:, j] if lus is None else lus[j].solve(
                    rhs.astype(np.complex128))
                x[lo:hi, j] = xj
                yk = yk + (two_alpha[j] * xj).real
            y[lo:hi] = yk * alpha0
        if scaled:
            y *= half
            y *= half
    return y


@dataclass(frozen=True)
class _ShiftLevels:
    """The shift level of every block of a step, from :func:`_assign_shift_levels`.

    Why shift levels exist is in the module notes (strategy, item 5). How a
    block gets its level: ``mu`` of a block is the slowest decay rate of it
    and its active ancestors (:meth:`MatrixStructure.block_decay_bounds`,
    clipped at 0). A block with ``k width <= mu h < (k + 1) width``, ``k >= 1``,
    is in level ``k``. Level ``k`` is solved again, with its active
    ancestors, using the shift ``s = k width / h``, capped at the smallest
    removal rate in that subsystem so that no diagonal entry of ``A + s I``
    is positive.

    level - Shift level of every block: 0 for blocks that need no shift (and
            for inactive ones), k >= 1 otherwise.
    active - Blocks that are not empty at the start of the step, or that one
             of their ancestors feeds.
    decayed - Active blocks whose ancestors all decay by more than the range
              of doubles within a substep: their value is exactly 0.

    """
    level: np.ndarray
    active: np.ndarray
    decayed: np.ndarray


def _assign_shift_levels(st: MatrixStructure, p: sp.csr_matrix, h: float,
                         y0: np.ndarray, width: float) -> _ShiftLevels | None:
    """Assign every block of the step from ``y0`` its shift level.

    Returns None when there is nothing to correct: a zero vector (CRAM is
    exact on it), or entries that are not finite (they overflowed before this
    step, so its result is not finite anyway).

    """
    top = float(np.max(np.abs(y0))) if len(y0) else 0.0
    if not (top > 0.0) or not math.isfinite(top):
        return None
    # A block whose ancestors all decay by more than this over a substep
    # holds less than ``top exp(-cut)``, times the production above 1 per
    # removal along its chains: below the smallest double.
    cut = math.log(top) + _UNDERFLOW

    active = np.zeros(st.n_blocks, dtype=bool)
    active[st.block_of[y0 != 0.0]] = True
    mu = np.maximum(st.block_decay_bounds(p), 0.0)
    for d in range(1, st.n_depths):  # parents first, so their mu is final
        parents, children = st.links_into_depth(d)
        live = active[parents]
        parents, children = parents[live], children[live]
        active[children] = True
        np.minimum.at(mu, children, mu[parents])

    decay = np.where(active, mu * h, 0.0)
    decayed = active & (decay >= cut)
    level = np.where(active & ~decayed, np.floor(decay / width), 0.0).astype(np.int64)
    return _ShiftLevels(level, active, decayed)


@dataclass(frozen=True)
class _StackedLevels:
    """The shift levels of a step as one block diagonal system.

    Each level is a *copy*: the compartments of its blocks and of their active
    ancestors, with the level's shift. All copies are stacked into one matrix,
    ordered by depth (the copies do not feed each other), so they are solved
    together, as any other matrix, with one factorization per pole.

    m - The stacked matrix ``h (A + s_copy I)``, one diagonal block per copy.
    groups - Its solve groups.
    entry - Compartment (structure order) of every stacked row.
    half - ``exp(-s_copy h / 2)`` of every stacked row (see :func:`_cram_substeps`).
    result - The stacked rows that are members of their level (not just an
             ancestor brought along); their values are the shifted results.

    """
    m: sp.csr_matrix
    groups: _SolveGroups
    entry: np.ndarray
    half: np.ndarray
    result: np.ndarray


def _stack_shift_levels(st: MatrixStructure, p: sp.csr_matrix, h: float,
                        levels: _ShiftLevels, width: float) -> _StackedLevels:
    """Build the stacked system of the shift levels above 0."""
    level_numbers = np.unique(levels.level[levels.level > 0])

    # in_copy[b, c]: block b is in copy c (a member of level level_numbers[c],
    # or an active ancestor of one); marked from the deepest blocks up.
    in_copy = levels.level[:, None] == level_numbers[None, :]
    for d in range(int(st.block_depth[levels.level > 0].max()), 0, -1):
        parents, children = st.links_into_depth(d)
        reached = in_copy[children] & levels.active[parents, None]
        np.logical_or.at(in_copy, parents, reached)

    # stacked rows: (copy, compartment) pairs, by depth, then copy, then compartment
    copy, entry = np.nonzero(in_copy[st.block_of].T)
    order = np.lexsort((entry, copy, st.depth_of[entry]))
    copy, entry = copy[order], entry[order]
    size = len(entry)
    row_of = np.full((len(level_numbers), st.n), -1, dtype=np.int64)
    row_of[copy, entry] = np.arange(size)

    # the shift of each copy: its level, at most its smallest removal rate
    diag = p.diagonal()
    smallest_removal = np.full(len(level_numbers), np.inf)
    np.minimum.at(smallest_removal, copy, -diag[entry])
    shift = np.minimum(level_numbers * width / h, smallest_removal)

    # the entries of A inside every copy, then the shifts on the diagonal
    pc = p.tocoo()
    inside = in_copy[st.block_of[pc.row]] & in_copy[st.block_of[pc.col]]
    nz, nz_copy = np.nonzero(inside)
    rows = np.r_[row_of[nz_copy, pc.row[nz]], np.arange(size)]
    cols = np.r_[row_of[nz_copy, pc.col[nz]], np.arange(size)]
    vals = np.r_[h * pc.data[nz], h * shift[copy]]
    m = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))
    m.sort_indices()

    return _StackedLevels(
        m=m,
        groups=_solve_groups(st.depth_of[entry], st.cyclic_of[entry]),
        entry=entry,
        half=np.exp(-0.5 * shift * h)[copy],
        result=levels.level[st.block_of[entry]] == level_numbers[copy],
    )


def _shift_levels(st: MatrixStructure, p: sp.csr_matrix, h: float,
                  substeps: int, y0: np.ndarray, y: np.ndarray,
                  alpha: np.ndarray, theta: np.ndarray, alpha0: float,
                  width: float) -> None:
    """Shifted CRAM: solve again, in place in ``y`` (structure order), the
    compartments whose ancestors all empty fast over a substep of length
    ``h`` (see :class:`_ShiftLevels`), and set to exactly 0
    those whose value is below the range of doubles.

    ``test_short_lived_daughter_is_accurate_relative_to_itself`` shows a case
    where this is needed: without it the fed compartment is wrong by orders
    of magnitude, with it it is within 1e-12 of the exact value.

    """
    levels = _assign_shift_levels(st, p, h, y0, width)
    if levels is None:
        return
    if levels.decayed.any():
        y[levels.decayed[st.block_of]] = 0.0
    if not (levels.level > 0).any():
        return
    stacked = _stack_shift_levels(st, p, h, levels, width)
    factors = _Factors(stacked.m, stacked.groups, theta)
    u = _cram_substeps(factors, y0[stacked.entry], alpha, alpha0, substeps,
                       half=stacked.half)
    y[stacked.entry[stacked.result]] = u[stacked.result]


def cram_apply(a, v: np.ndarray, t: float, *, order: CramOrder | int = CramOrder.CRAM48,
               substeps: int = 4, width: float | None = None,
               structure: MatrixStructure | None = None) -> np.ndarray:
    """Compute ``exp(A t) v`` with block-structured CRAM.

    Parameters
    ----------
    a - Sparse square matrix ``A`` (any SciPy sparse format) with non-negative
        off-diagonal entries; for full accuracy, no block of it grows (see
        the module notes).
    v - Vector, one entry per row of ``A``.
    t - Time step, in the units of the rates in ``A``.
    order - CRAM order (:class:`CramOrder`, or one of the ints 4, 8, ..., 48).
    substeps - Number of substeps the factorizations are reused over.
    width - Width of the shift levels (shifted CRAM); 0 turns shifting off.
            None picks :func:`shift_width` of the order.
    structure - Structural analysis of the pattern of ``a``, if already known;
                otherwise the cached one of :func:`structure_of`.

    Returns
    -------
    The vector ``exp(A t) v``, in the original order.

    """
    alpha, theta, alpha0 = cram_coefficients(order)
    a = _canonical(a)
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (a.shape[0],):
        raise ValueError(f"The vector has shape {v.shape}, but the matrix has {a.shape[0]} "
                         f"rows: it needs one entry per row, shape ({a.shape[0]},)")
    if t == 0 or not len(v):
        return v.copy()
    st = structure_of(a) if structure is None else structure
    width = shift_width(order) if width is None else float(width)
    p = st.permuted(a)
    h = t / substeps
    y0 = v[st.perm]
    y = _cram_substeps(_Factors(h * p, st.groups, theta), y0, alpha, alpha0, substeps)
    if width > 0:
        _shift_levels(st, p, h, substeps, y0, y, alpha, theta, alpha0, width)
    out = np.empty_like(y)
    out[st.perm] = y
    return out


class StructuralCram:
    """``exp(A t) v`` for sparse matrices of production and removal rates, by
    block-structured, shifted CRAM (see the module notes). A
    :class:`MatrixExponential`.

    The default (``order=48, substeps=4, shifted=True``) is shifted CRAM-48:
    every entry of the result is computed to about 1e-13 relative to its own
    value, down to 1e-280 of the largest one, with no negative values and no
    entry appearing where nothing can feed it. Plain CRAM, including
    :class:`IPFCramSolver`, is only accurate relative to the largest entry: its
    error is about ``alpha0`` times the largest entry of ``v``, in every entry.

    Parameters
    ----------
    order - CRAM order (:class:`CramOrder`, or one of the ints 4, 8, ..., 48).
            From 24 on, plain CRAM is at the round-off of double precision
            for the largest entries; 48 is needed where large entries empty
            within a step and their rates weigh the result.
    substeps - Substeps per call, sharing the factorizations.
    shifted - Whether to solve again, with a shift, the entries fed only by
              compartments that empty fast (their accuracy relative to their
              own value).
    width - Width of the shift levels; None picks 20 for CRAM-48 and 10 below.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> a = sp.csr_matrix(np.array([[-np.log(2), 0.], [np.log(2), 0.]]))
    >>> [round(float(x), 12) for x in StructuralCram()(a, np.array([1., 0.]), 1.)]
    [0.5, 0.5]

    """

    def __init__(self, order: CramOrder | int = CramOrder.CRAM48, substeps: int = 4,
                 shifted: bool = True, width: float | None = None):
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, not {substeps}")
        try:
            self.order = CramOrder(order)
        except ValueError:
            raise ValueError(f"CRAM order must be one of {CRAM_ORDERS}, not {order!r}") from None
        self.substeps = substeps
        self.shifted = shifted
        self.width = (0.0 if not shifted
                      else shift_width(order) if width is None else float(width))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(order={int(self.order)}, substeps={self.substeps}, "
                f"shifted={self.shifted})")

    def __call__(self, a, v: np.ndarray, t: float) -> np.ndarray:
        """Return ``exp(a t) v``."""
        a = _canonical(a)
        return cram_apply(a, v, t, order=self.order, substeps=self.substeps,
                          width=self.width, structure=structure_of(a))
