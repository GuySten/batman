"""Tests of the block-structured, shifted CRAM exponentiator.

"""
import math
from decimal import Decimal, localcontext

import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
from hypothesis import given, settings

from batman.exponentiators import (
    CRAM_ORDERS,
    ConstantFlux,
    CramOrder,
    IPFCram48,
    ShiftedCram48,
    StructuralCram,
    cram_coefficients,
)
from batman.exponentiators.cram.structural import MatrixStructure, cram_apply, structure_of


def _ipf_scalar(order, x):
    alpha, theta, alpha0 = cram_coefficients(order)
    r = np.ones_like(x)
    for a, t in zip(alpha, theta):
        r = r * (1 + 2 * np.real(a / (x - t)))
    return alpha0 * r


@pytest.mark.parametrize("order", CRAM_ORDERS)
def test_every_order_reaches_its_equioscillation_level(order):
    x = -np.concatenate([np.linspace(0, 60, 6001), np.logspace(-4, 5, 4001)])
    err = np.max(np.abs(_ipf_scalar(order, x) - np.exp(x)))
    alpha0 = cram_coefficients(order)[2]
    assert err <= max(1.01 * alpha0, 1e-14)


def test_cram24_uses_the_corrected_pole():
    _, theta, _ = cram_coefficients(24)
    assert np.any(np.isclose(theta.real, -7.155311153664576, rtol=0, atol=1e-15))
    assert not np.any(np.isclose(theta.real, -7.153311153664576, rtol=0, atol=1e-15))


def test_unknown_order_is_rejected():
    with pytest.raises(ValueError, match="CRAM order must be one of"):
        cram_coefficients(10)
    with pytest.raises(ValueError, match="CRAM order must be one of"):
        StructuralCram(order=10)


def test_orders_as_enum_or_int():
    a = _chain([1.0, 1e-3, 0.0])
    n0 = np.array([1.0, 0.0, 0.0])
    assert StructuralCram(CramOrder.CRAM24).order == 24
    assert np.array_equal(cram_apply(a, n0, 5.0, order=CramOrder.CRAM24),
                          cram_apply(a, n0, 5.0, order=24))


def _chain(lams):
    """Linear decay chain 0 -> 1 -> ... with the given decay constants."""
    n = len(lams)
    a = sp.diags(-np.asarray(lams, dtype=float)).tolil()
    for i in range(n - 1):
        a[i + 1, i] = lams[i]
    return a.tocsr()


def _bateman(lams, t):
    """Atoms of every member of a linear chain from one atom of the first
    (distinct decay constants; the last may be 0): the Bateman solution,
    summed in 60-digit decimal arithmetic, where its terms cancel."""
    with localcontext() as ctx:
        ctx.prec = 60
        lams = [Decimal(float(lam)) for lam in lams]
        t = Decimal(float(t))
        out = []
        for k in range(len(lams)):
            prod = Decimal(1)
            for lam in lams[:k]:
                prod *= lam
            total = Decimal(0)
            for i in range(k + 1):
                denom = Decimal(1)
                for j in range(k + 1):
                    if j != i:
                        denom *= lams[j] - lams[i]
                total += (-lams[i] * t).exp() / denom
            out.append(float(prod * total))
    return np.array(out)


def test_short_lived_daughter_is_accurate_relative_to_itself():
    """A daughter of a fast parent, far below alpha0 of the bulk: plain CRAM
    gets it wrong by many orders of magnitude, shifted CRAM to round-off."""
    lams = [10.0, 1.0]
    a = _chain(lams + [0.0])
    n0 = np.array([1.0, 0.0, 0.0])
    t = 300.0
    exact = _bateman(lams, t)  # about [0, 5e-131]
    new = ShiftedCram48(a, sp.csr_matrix(a.shape), lambda _t: 0., n0, t)
    old = IPFCram48(a, sp.csr_matrix(a.shape), lambda _t: 0., n0, t)
    assert new[0] == 0.0  # exp(-3000) is below the smallest double
    assert new[1] == pytest.approx(exact[1], rel=1e-12)
    assert new[2] == pytest.approx(1.0, rel=1e-14)
    assert abs(old[1] - exact[1]) > 1e3 * exact[1]  # plain CRAM: bulk-relative


def test_yield_two_daughter_accurate_relative_to_itself():
    """Production above 1 per removal, as in fission (about two products per
    fission): the parent's column sums to more than 0, and its fast decay
    still puts the daughter in a shift level. The daughter is exactly twice
    the Bateman solution, to round-off relative to itself, with or without
    substeps; without the shift, a single CRAM step is far off."""
    lam_parent, lam_daughter = 10.0, 1.0
    a = sp.csr_matrix(np.array([[-lam_parent, 0.0, 0.0],
                                [2.0 * lam_parent, -lam_daughter, 0.0],
                                [0.0, lam_daughter, 0.0]]))
    assert a.sum(axis=0)[0, 0] > 0  # hands on twice what it loses
    t = 300.0
    exact = 2.0 * _bateman([lam_parent, lam_daughter], t)  # about [0, 1e-130]
    n0 = np.array([1.0, 0.0, 0.0])
    for expm in (StructuralCram(), StructuralCram(substeps=1)):
        got = expm(a, n0, t)
        assert got[0] == 0.0
        assert got[1] == pytest.approx(exact[1], rel=1e-12)
        assert got[2] == pytest.approx(2.0, rel=1e-14)
    plain = StructuralCram(substeps=1, shifted=False)(a, n0, t)
    assert abs(plain[1] - exact[1]) > 1e3 * exact[1]


def test_high_shift_levels_do_not_underflow():
    """A nuclide that decays by exp(-760) over one substep, from 1e24 atoms,
    still holds about 1e-306 atoms. Its shift scale exp(-760) underflows as
    one number, so it is applied in two halves of exp(-380)."""
    lam = 76.0
    a = _chain([lam, 0.0])
    got = StructuralCram(48, substeps=1)(a, np.array([1e24, 0.0]), 10.0)
    assert math.exp(-760.0) == 0.0  # the scale as one number
    assert got[0] == pytest.approx(1e24 * float(Decimal(-760).exp()), rel=1e-12)
    assert got[1] == pytest.approx(1e24, rel=1e-14)


@pytest.mark.parametrize("t", [1.0, 3600.0, 86400.0, 3e7, 3e9])
def test_decay_chain_against_bateman(t):
    lams = [np.log(2) / h for h in (1e2, 3e4, 8e5, 1e8, 4e9)]
    a = _chain(lams + [0.0])
    n0 = np.zeros(len(lams) + 1)
    n0[0] = 1.0
    exact = _bateman(lams + [0.0], t)
    got = cram_apply(a, n0, t)
    big = exact > 1e-290
    assert np.allclose(got[big], exact[big], rtol=1e-11, atol=0)


def _random_burnup(n, seed, density=0.08, flux=True):
    """Random stiff burnup matrix: positive off-diagonals, every column leaks
    at least what it feeds (rates over ten decades)."""
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=density, random_state=rng, format="lil")
    a.setdiag(0)
    a = a.tocsr()
    a.eliminate_zeros()
    if not flux:
        a = sp.tril(a, -1).tocsr()
    a.data = 10.0 ** rng.uniform(-6, 4, a.nnz)
    leak = np.asarray(a.sum(axis=0)).ravel() * (1 + rng.uniform(0, 1, n))
    leak[rng.random(n) < 0.1] += 10.0 ** rng.uniform(-6, 4)
    return (a - sp.diags(leak)).tocsr()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("flux", [True, False])
def test_against_dense_expm(seed, flux):
    a = _random_burnup(40, seed, flux=flux)
    rng = np.random.default_rng(seed)
    n0 = np.where(rng.random(40) < 0.5, 10.0 ** rng.uniform(-6, 0, 40), 0.0)
    for t in (1e-3, 1.0, 1e3):
        ref = la.expm(t * a.toarray()) @ n0
        got = cram_apply(a, n0, t)
        big = np.abs(ref) > 1e-9 * np.abs(ref).sum()
        assert np.allclose(got[big], ref[big], rtol=1e-9, atol=0)
        assert np.all(got >= 0)


@pytest.mark.parametrize("seed", range(3))
def test_fission_like_columns_against_dense_expm(seed):
    """A cyclic heavy part whose members also fission into a tail of
    products, two products per fission: their columns hand on more than they
    lose, but no block grows, so the result agrees with dense expm."""
    rng = np.random.default_rng(seed)
    heavy, products = 15, 25
    n = heavy + products
    a = sp.lil_matrix(_random_burnup(heavy, seed, density=0.2))
    a.resize((n, n))
    tail = _random_burnup(products, seed + 100, flux=False)  # decays only
    a[heavy:, heavy:] = tail
    for j in range(heavy):
        sigma = 10.0 ** rng.uniform(-3, 1)
        yields = 2.0 * rng.dirichlet(np.ones(products))
        a[heavy:, j] = (sigma * yields)[:, None]
        a[j, j] -= sigma
    a = a.tocsr()
    assert np.max(np.asarray(a.sum(axis=0)).ravel()[:heavy]) > 0
    n0 = np.r_[10.0 ** rng.uniform(-3, 0, heavy), np.zeros(products)]
    for t in (1e-2, 1.0, 1e2):
        ref = la.expm(t * a.toarray()) @ n0
        got = StructuralCram()(a, n0, t)
        big = np.abs(ref) > 1e-9 * np.abs(ref).sum()
        assert np.allclose(got[big], ref[big], rtol=1e-9, atol=0)
        assert np.all(got >= 0)


def test_growing_block_is_solved_without_a_shift():
    """A cyclic block that grows (one of its columns hands back more than it
    loses) has an eigenvalue above 0: the shift bound is clipped at 0, so the
    block is solved by plain CRAM, as accurately as CRAM is there."""
    a = sp.csr_matrix(np.array([[-1.0, 2.0, 0.0],
                                [1.0, -1.0, 0.0],
                                [0.0, 0.5, -0.1]]))
    assert np.max(np.linalg.eigvals(a.toarray()).real) > 0
    n0 = np.array([1.0, 0.0, 0.0])
    for t in (0.5, 3.0):
        ref = la.expm(t * a.toarray()) @ n0
        shifted = StructuralCram()(a, n0, t)
        plain = StructuralCram(shifted=False)(a, n0, t)
        assert np.array_equal(shifted, plain)
        assert np.allclose(shifted, ref, rtol=1e-6, atol=0)


def test_unreached_nuclides_stay_exactly_zero():
    """Nothing feeds nuclides 0..4 from the atoms present: no pivoting means
    they come out as exactly 0, cyclic system or not."""
    a = _random_burnup(30, 7).tolil()
    a[:5, 5:] = 0.0
    a = a.tocsr()
    n0 = np.r_[np.zeros(5), np.ones(25)]
    for expm in (StructuralCram(), StructuralCram(24, substeps=1, shifted=False)):
        out = expm(a, n0, 10.0)
        assert np.all(out[:5] == 0.0)


def test_structure_is_block_lower_triangular():
    a = _random_burnup(60, 3, density=0.03)
    st_ = MatrixStructure(a)
    p = st_.permuted(sp.csr_matrix(a)).tocoo()
    blocks = st_.block_of
    assert np.all(blocks[p.row] >= blocks[p.col])
    assert sorted(st_.perm) == list(range(60))


def test_decay_matrix_is_acyclic():
    assert MatrixStructure(_random_burnup(50, 1, flux=False)).acyclic
    assert not MatrixStructure(_random_burnup(50, 1, flux=True)).acyclic


def test_flux_and_protocol():
    d = _chain([1e-3, 0.0])
    r = sp.csr_matrix(np.array([[-2e-3, 0.0], [2e-3, 0.0]]))
    n0 = np.array([1.0, 0.0])
    got = ShiftedCram48(d, r, lambda _t: 0.5, n0, 100.0)
    lam = 1e-3 + 0.5 * 2e-3
    assert got[0] == pytest.approx(np.exp(-lam * 100.0), rel=1e-13)
    assert got.sum() == pytest.approx(1.0, rel=1e-14)
    assert np.array_equal(ShiftedCram48(d, r, lambda _t: 1., n0, 0.0), n0)


@pytest.mark.parametrize("seed", range(3))
def test_constant_flux_is_the_exponential_of_the_step_matrix(seed):
    """ConstantFlux(expm) gives exp((d + flux(0) r) dt) n0: the step's matrix
    at the flux of its start, which a flux that varies within the step does
    not change."""
    a = _random_burnup(30, seed)
    d = sp.csr_matrix(sp.tril(a, -1) + sp.diags(a.diagonal()))  # decay-like
    r = sp.csr_matrix(sp.triu(a, 1)) - sp.diags(np.asarray(sp.triu(a, 1).sum(axis=0)).ravel())
    n0 = np.random.default_rng(seed).random(30)
    step = ConstantFlux(StructuralCram())
    for phi, dt in ((0.0, 10.0), (1e-2, 10.0), (1.0, 0.1)):
        ref = la.expm(dt * (d + phi * r).toarray()) @ n0
        for flux in (lambda _t, phi=phi: phi, lambda t, phi=phi: phi + 5.0 * t):
            got = step(d, r, flux, n0, dt)
            big = np.abs(ref) > 1e-9 * np.abs(ref).sum()
            assert np.allclose(got[big], ref[big], rtol=1e-9, atol=0)


def test_wrong_vector_shape_says_what_fits():
    with pytest.raises(ValueError, match=r"vector has shape \(3,\).*shape \(2,\)"):
        cram_apply(_chain([1.0, 0.0]), np.ones(3), 1.0)


def test_structure_is_shared_by_matrices_with_one_pattern():
    a = _random_burnup(30, 4)
    b = a.copy()
    b.data *= 7.0
    assert structure_of(sp.csr_matrix(a)) is structure_of(sp.csr_matrix(b))


non_zero = (st.floats(min_value=1e-3, max_value=1.)
            | st.floats(max_value=-1e-3, min_value=-1.))
vec = st.tuples(non_zero, non_zero)


@settings(max_examples=100, deadline=None)
@given(m=st.lists(vec, min_size=2, max_size=2, unique=True), v=vec,
       t=st.floats(min_value=0.5, max_value=3.))
def test_any_small_matrix_against_expm(m, v, t):
    """Not a burnup matrix: positive eigenvalues, as in test_exp.py."""
    m = np.array(m)
    got = StructuralCram()(sp.csr_matrix(m), np.array(v), t)
    assert np.allclose(got, la.expm(t * m) @ np.array(v), rtol=1e-3)
