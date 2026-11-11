"""These are simple, analytic cases that should give us the required data."""

from collections.abc import Iterable
from functools import partial
from math import exp, inf, isclose
from os.path import dirname
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import pytest
from coremaker.materials.mixture import Mixture
from hypothesis import given, seed, settings
from isotopes import I135, U235, Xe135
from reactions import Fission, NGamma, ProtoReaction, Reaction, ReactionRate
from scipy.constants import eV
from scipy.integrate import solve_ivp

from batman.graphs import DecayGraph, GraphFilter
from batman.graphs.decay import DECAY
from batman.graphs.decay_data import DecayProcess, parse_decay_processes
from batman.graphs.filters import whitelist_filter
from batman.integrator import predictor
from batman.solver import Configuration, InputData, timestep_constant_power
from batman.solver import SerialEasyData as EasyData
from batman.solver.time_est import max_step_initial_correct_predictor as tguess
from batman.units import CM3, EV_TO_MJ, MW, Second

from .conftest import allclose, slow

U235FISS_ENER = 200e6  # eV
Day = float
test_data_dir = Path(dirname(__file__)) / "test_data"

data_missing = not all((test_data_dir / fname).exists() for fname in ["dec-053_I_135.endf", "dec-054_Xe_135.endf"])


# noinspection PyPep8Naming
@pytest.fixture(scope="module")
def fission_reaction() -> Iterable[ReactionRate]:
    """Mock reaction rate for fission."""
    U235nf = Reaction.from_reaction(U235, Fission, energy=U235FISS_ENER, target=I135)
    return (ReactionRate("moo", U235nf, 1e-3, 0),)


@pytest.fixture(scope="module")
def iodine_decay() -> DecayGraph:
    """Iodine-135's decay graph."""
    dec = DecayGraph()
    processes = parse_decay_processes(test_data_dir / "dec-053_I_135.endf", test_data_dir / "dec-054_Xe_135.endf")
    for process in processes:
        (target,) = process.targets
        if Xe135 in {process.parent, target}:
            process = DecayProcess.from_other(process, fraction=1.0)
        dec.add_edge_from_process(process)
    return dec


density_range = st.just(0.0) | st.floats(min_value=1e-6, max_value=1e-1)
time_range_seconds = st.floats(min_value=1.0, max_value=30.0 * 24.0 * 3600.0)
reac_rates = st.just(0.0) | st.floats(min_value=1e-9, max_value=1e-2)
power_range = st.just(0.0) | st.floats(min_value=1.0, max_value=20.0)


@pytest.mark.skipif(
    data_missing,
    reason=f"Missing decay data. Use the download script under {test_data_dir} to obtain the missing files",
)
@settings(max_examples=100)
@given(t=time_range_seconds)
def test_analytic_decay_is_exponential_down(iodine_decay: DecayGraph, t: Second):
    _filter = GraphFilter(whitelist_filter, frozenset({I135}))
    mixture = Mixture({I135: 1.0}, 20.0)
    rate = iodine_decay[I135][I135][DECAY]["rate"]
    expected_res = Mixture({I135: exp(rate * t)}, 20.0)
    data = InputData([iodine_decay], [[]], [_filter], [mixture], [1.0])
    easy = EasyData.from_input(data)
    config = Configuration(p_rtol=1.0, p_atol=0, integrator=predictor, time_guesser=tguess, threshold=1e-18)
    newden, _result = timestep_constant_power(easy, 0.0, t, config=config)
    assert allclose(newden[0], expected_res)


# noinspection PyPep8Naming
@settings(max_examples=100)
@given(t=time_range_seconds, P=power_range)
def test_analytic_power_constant_burn_causes_linear_drop_in_fuel(
    fission_reaction: Iterable[ReactionRate], t: Second, P: MW
):
    acc = 1e-1
    max_step = 2
    _filter = GraphFilter()
    volume = 0.05 * 60 * 6 * 21 * 28.0
    rate = P / (U235FISS_ENER * eV * 1e18) / volume
    u0 = 2e-3
    mixture = Mixture({U235: u0}, 20.0)
    _densities = {U235: u0 - rate * t, I135: rate * t}
    expected_res = Mixture({k: v for k, v in _densities.items() if v > 1e-18}, 20.0)
    data = InputData([DecayGraph()], [fission_reaction], [_filter], [mixture], [volume])
    easy = EasyData.from_input(data)
    isos, _, r = easy.models[0]
    assert r.energy_model[isos.index(U235)] > 0.0, r.energy_model
    config = Configuration(p_rtol=acc, p_atol=P * acc, integrator=predictor, time_guesser=tguess, threshold=1e-18)
    if max_step == 1:
        assert config.first_guess(easy, t=t, p=P) == t
    (newden,), result = timestep_constant_power(easy, P, t, config=config)
    assert result.time.total_seconds() == pytest.approx(t, rel=1e-4)
    assert allclose(newden, expected_res, comparator=isclose, rel_tol=1e-6, abs_tol=1e-5)
    assert result.steps <= max_step


# noinspection PyPep8Naming
def _iodine_den(Ieq, I0, decay_I, t) -> float:
    return Ieq - (Ieq - I0) * exp(-decay_I * t) if t else I0


# noinspection PyPep8Naming
@pytest.mark.skipif(
    data_missing,
    reason=f"Missing decay data. Use the download script under {test_data_dir} to obtain the missing files",
)
@settings(max_examples=100)
@given(I0=density_range, t=st.floats(min_value=1.0, max_value=30 * 24 * 3600.0), rateI=reac_rates)
def test_I135_analytic_solution(iodine_decay, I0, rateI, t):
    decay_I = -iodine_decay[I135][I135][DECAY]["rate"]
    Ieq = rateI / decay_I
    analytic_I = _iodine_den(Ieq, I0, decay_I, t)

    def _deriv_iodine(_, y):
        return rateI - decay_I * y

    sol = solve_ivp(fun=_deriv_iodine, t_span=(0, t), y0=np.array([I0]), rtol=1e-8, atol=1e-14)
    # noinspection PyUnresolvedReferences
    assert sol.status == 0
    # noinspection PyUnresolvedReferences
    assert sol.y[-1][-1] == pytest.approx(analytic_I, rel=1e-7, abs=1e-13)


# noinspection PyPep8Naming
@pytest.mark.skipif(
    data_missing,
    reason=f"Missing decay data. Use the download script under {test_data_dir} to obtain the missing files",
)
@settings(max_examples=300)
@given(I0=density_range, t=time_range_seconds, rateI=reac_rates)
def test_follows_analytic_solution_for_U235_iodine_system(iodine_decay, I0, t, rateI):
    _filter = GraphFilter(whitelist_filter, frozenset({I135, U235}))
    mixture = Mixture({I135: I0, U235: 1.0}, 20.0)
    decay_rate = -iodine_decay[I135][I135][DECAY]["rate"]
    Iinf = rateI / decay_rate
    assert Iinf >= 0
    U235nf = ProtoReaction.from_reaction(U235, Fission, branching={I135: 1.0}, energy=0.0)
    reactions = (ReactionRate("moo", U235nf, rateI, 0.0),)
    expected_res = Mixture({I135: _iodine_den(Iinf, I0, decay_rate, t), U235: 1.0}, 20.0)
    data = InputData([iodine_decay], [reactions], [_filter], [mixture], [1.0], accumulates=[frozenset({U235})])
    easy: EasyData = EasyData.from_input(data)
    easy.calc_norm = lambda *a, **ka: 1.0
    config = Configuration(p_rtol=1.0, p_atol=0.0, integrator=predictor, time_guesser=tguess, threshold=1e-18)
    newden, result = timestep_constant_power(easy, 0, t, config=config)
    assert allclose(newden[0], expected_res)
    assert result.steps <= 2


# noinspection PyPep8Naming
def _uranium_den(U0, rateF, t) -> float:
    return U0 - rateF * t


# noinspection PyPep8Naming
@pytest.mark.skipif(
    data_missing,
    reason=f"Missing decay data. Use the download script under {test_data_dir} to obtain the missing files",
)
@settings(max_examples=300)
@given(I0=density_range, X0=density_range, t=time_range_seconds, rateI=reac_rates, rateX=reac_rates, sigX=reac_rates)
def test_follows_analytic_solution_for_constant_U235_iodine_xenon_system(iodine_decay, I0, X0, t, rateI, rateX, sigX):
    _filter = GraphFilter(whitelist_filter, frozenset({I135, U235, Xe135}))
    mixture = Mixture({I135: I0, Xe135: X0, U235: 1.0}, 20.0)
    decay_I = -iodine_decay[I135][I135][DECAY]["rate"]
    decay_X = -iodine_decay[Xe135][Xe135][DECAY]["rate"] + sigX
    cX = rateX
    cI = rateI
    expected_res_x = (
        X0 * exp(-decay_X * t)
        + ((cX + cI) / decay_X) * (1 - exp(-decay_X * t))
        + (((cI - I0 * decay_I) / (decay_X - decay_I)) * (exp(-decay_X * t) - exp(-decay_I * t)))
    )
    Ieq = cI / decay_I
    expected_res_i = _iodine_den(Ieq, I0, decay_I, t)
    bI, bX = (rateI / (rateI + rateX), rateX / (rateI + rateX)) if (rateI + rateX) > 0.0 else (0.0, 0.0)
    U235nf = ProtoReaction.from_reaction(U235, Fission, branching={I135: bI, Xe135: bX}, energy=0.0)
    Xe135a = Reaction.from_reaction(Xe135, NGamma)
    reactions = tuple(
        ReactionRate("moo", reac, mean, 0.0) for reac, mean in zip([U235nf, Xe135a], [rateI + rateX, sigX])
    )
    data = InputData([iodine_decay], [reactions], [_filter], [mixture], [1.0], accumulates=[frozenset({U235})])
    easy = EasyData.from_input(data)
    easy.calc_norm = lambda *a, **ka: 1.0
    config = Configuration(p_rtol=1.0, p_atol=0.0, integrator=predictor, time_guesser=tguess, threshold=1e-18)
    newden, result = timestep_constant_power(easy, 0.0, t, config=config)
    res_mix = newden[0]
    approx = partial(pytest.approx, rel=1e-5, abs=1e-14)
    assert res_mix.get(U235) == pytest.approx(1.0, rel=1e-10)
    assert res_mix.get(I135) == approx(expected_res_i), (
        f"resI={res_mix.get(I135)}, expected={expected_res_i}\n",
        res_mix,
    )
    assert res_mix.get(Xe135) == approx(expected_res_x), (
        f"resX={res_mix.get(Xe135)}, expected={expected_res_x}\n",
        res_mix,
    )
    assert result.steps <= 2


# noinspection PyPep8Naming
def numerically_solve_xe_ODE(
    *,
    U0: float,
    I0: float,
    X0: float,
    rateI: float,
    rateXcreate: float,
    rateF: float,
    rateXabs: float,
    decay_X: float,
    decay_I: float,
    dt: float,
    rtol: float = 1e-8,
    atol: float = 1e-14,
) -> float:
    """Solve the ODE for Xe135 in a true U235->I135->Xe135, U235->Xe system.

    Parameters
    ----------
    U0 - Initial U235 density, in 1/barn-cm.
    I0 - Initial I135 density, in 1/barn-cm.
    X0 - Initial Xe135 density, in 1/barn-cm.
    rateXcreate - Rate of Xe135 creation, in 1/barn-cm-s.
    rateF - Rate of U235 fission, in 1/barn-cm-s.
    rateXabs - Rate of Xe135 neutron absorption, in 1/barn-cm-s
    decay_X - Decay constant of Xe135, in 1/s
    decay_I - Decay constant of I135, in 1/s
    dt - Time step size, in s.

    Returns
    -------
    The Xe135 density at the end of the time step.

    """

    Ieq = rateI / decay_I
    iodine_density = partial(_iodine_den, Ieq=Ieq, I0=I0, decay_I=decay_I)
    uranium_density = partial(_uranium_den, U0=U0, rateF=rateF)

    def _xe_absorption(t) -> float:
        if not uranium_density(t=t) or not rateF:
            return 0.0
        res = rateXabs * U0 / uranium_density(t=t)
        return res

    def _xe_deriv(t: float, xe: np.ndarray) -> np.ndarray:
        return decay_I * iodine_density(t=t) + rateXcreate - (decay_X + _xe_absorption(t=t)) * xe

    sol = solve_ivp(fun=_xe_deriv, t_span=(0.0, dt), y0=np.array([X0]), rtol=rtol, atol=atol)
    # noinspection PyUnresolvedReferences
    assert sol.status == 0
    # noinspection PyUnresolvedReferences
    return sol.y[-1][-1] if dt else X0


def _fission_rate(p: MW, vol: CM3):
    return p / (U235FISS_ENER * EV_TO_MJ * vol)


def _max_burntime(uden: float, p: float, vol: float) -> float:
    return 0.8 * uden / _fission_rate(p, vol) / 3600 / 24 if p else inf


uden_range = st.shared(st.just(0.0) | st.floats(min_value=1e-4, max_value=1e-2), key="fuel")
challenge_times = st.floats(min_value=1.0, max_value=30.0)
densities = st.tuples(density_range, density_range, uden_range).filter(lambda x: sum(x) > 0)
pow_st = st.shared(uden_range.map(lambda x: 20.0 if x else 0.0), key="pow")
vol_st = st.just(60.0**3)
time_st = st.tuples(uden_range, pow_st, vol_st, challenge_times).map(lambda x: min(x[-1], _max_burntime(*x[:-1])))
pow_reac_rates = st.tuples(reac_rates, pow_st).map(lambda x: x[0] if x[1] else 0.0)


# noinspection PyPep8Naming
@pytest.mark.xfail(reason="Our current implementation isn't fully constant in power")
@pytest.mark.skipif(
    data_missing,
    reason=f"Missing decay data. Use the download script under {test_data_dir} to obtain the missing files",
)
@slow
@seed(22821844992351348578153625617434270913)
@settings(max_examples=1000, deadline=2e3)
@given(
    density0=densities, days=time_st, p=pow_st, vol=vol_st, rateI=pow_reac_rates, rateX=pow_reac_rates, sigX=reac_rates
)
def test_follows_analytic_solution_for_U235_iodine_xenon_system(
    iodine_decay, density0, p, vol, days, rateI, rateX, sigX
):
    I0, X0, U0 = density0
    t = days * 24 * 3600
    rateF = _fission_rate(p, vol)
    _filter = GraphFilter(whitelist_filter, frozenset({I135, U235, Xe135}))
    mixture = Mixture({I135: I0, Xe135: X0, U235: U0}, 20.0)
    decay_I = -iodine_decay[I135][I135][(DECAY,)]["rate"]
    decay_X = -iodine_decay[Xe135][Xe135][(DECAY,)]["rate"]
    bI, bX = (rateI / rateF, rateX / rateF) if rateF > 0.0 else (0.0, 0.0)
    U235nf = ProtoReaction.from_reaction(U235, Fission, branching={I135: bI, Xe135: bX}, energy=U235FISS_ENER)
    Xe135a = Reaction.from_reaction(Xe135, NGamma)
    reactions = tuple(
        ReactionRate("moo", reac, mean, 0.0) for reac, mean in zip([U235nf, Xe135a], [rateF / U0 if U0 else 0.0, sigX])
    )
    expected_res_x = numerically_solve_xe_ODE(
        U0=U0,
        I0=I0,
        X0=X0,
        rateI=rateI,
        rateXcreate=rateX,
        rateXabs=sigX,
        rateF=rateF,
        decay_X=decay_X,
        decay_I=decay_I,
        dt=t,
    )
    expected_res_i = _iodine_den(Ieq=rateI / decay_I, I0=I0, decay_I=decay_I, t=t)
    expected_res_u = _uranium_den(U0, rateF, t)
    data = InputData([iodine_decay], [reactions], [_filter], [mixture], [vol])
    easy = EasyData.from_input(data)
    config = Configuration(p_rtol=1e-2, p_atol=1e-10, integrator=predictor, time_guesser=tguess, threshold=1e-18)
    newden, _result = timestep_constant_power(
        easy,
        p,
        t,
        config=config,
    )
    res_mix = newden[0]
    approx_simple = partial(pytest.approx, rel=1e-4, abs=1e-13)
    approx_hard = partial(pytest.approx, rel=5e-3 if p else 1e-6, abs=1e-13)
    assert res_mix.get(U235) == approx_simple(expected_res_u), (
        f"resU={res_mix.get(U235)}, expected={expected_res_u}\n",
        res_mix,
    )
    assert res_mix.get(I135) == approx_simple(expected_res_i), (
        f"resI={res_mix.get(I135)}, expected={expected_res_i}\n",
        res_mix,
    )
    assert res_mix.get(Xe135) == approx_hard(expected_res_x), (
        f"resX={res_mix.get(Xe135)}, expected={expected_res_x}\n",
        res_mix,
    )
