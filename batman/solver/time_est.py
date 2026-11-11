"""Tools to estimate the length of a timestep.

"""
from functools import partial
from logging import getLogger
from math import inf, isclose, log
from warnings import warn

import numpy as np

from batman.units import MW, Second

from .inputs import EasyData
from .power_normalization import add_two
from .utils import RunData, mixture_to_nd

__all__ = ['deriv_p', 'halfstep', 'max_step_initial_correct_predictor']


logger = getLogger('batman.time_est')


def _deriv_p_single(data: RunData, *, norm: float) -> tuple[float, float]:
    """Calculate the derivative in power generation for a single RunData.

    Parameters
    ----------
    data - The rundata in question
    norm - The normalization factor at the moment.

    Returns
    ------

    If the normalization is proper, this has units of MW/s
    Returns (DecayPower derivative, ReactionPower derivative)

    """
    (isos, decm, reacm), mix, vol = data
    nd = mixture_to_nd(mixture=mix, isos=isos, dtype=reacm.dtype)
    dndt = (decm.mat + (norm * reacm.mat)) @ nd
    return decm.energy(dndt) * vol, reacm.energy(dndt, norm=norm) * vol


def deriv_p(data: EasyData, *, norm: float) -> float:
    r"""Calculate the time derivative in power generation for the full model.

    .. math::
        \frac{dP}{dt} = \text{energy model} \dot \frac{dN}{dt} = \text{energy model} \dot (\lambda + R) \dot N

    Parameters
    ----------
    data - EasyData to use.
    norm - Normalization factor to use.

    Returns
    -------

    The time derivative of power generation, in MW/s

    """

    return sum(data.map_reduce(partial(_deriv_p_single, norm=norm),
                               add_two,
                               initial=(0., 0.)
                               )
               )


def _allowed_predictor_step(p0: MW, dpdt: float, *,
                            rtol: float, atol: MW,
                            nudge: float):
    if isclose(dpdt, 0.):
        return inf
    if rtol > 1. or atol > p0:
        return inf
    rate = dpdt / p0
    tol = (atol * nudge / p0, rtol * nudge)
    return max(log(1.+np.sign(rate)*a) / rate for a in tol)


def max_step_initial_correct_predictor(
        data: EasyData, t: Second, p0: MW, *,
        atol: MW,
        rtol: float,
        too_small: Second = 0.,
        nudge: float = 1.-1e-2,
        norm: float | None = None) -> Second:
    """Yields an initial guess using the power derivative.
    This method assumes that the power starts at the correct power level that
    the user desires, and then the flux remains constant. For a non-breeding
    core with 1 type of fuel this causes an exponentially decaying power level
    within the calculated step.
    This can be used to estimate how long such a step can be without the
    momentary power level dropping under a given tolerance.
    More complicated cases where the power doesn't tend to decay exponentially during
    the relevant step are not supported by this function, and you may have to look elsewhere.
    The solution for two fuels or a breeding fuel has multiple exponentials.
    If your time step is small enough, maybe one of these is dominant, but your mileage may
    vary. It's best to write your own function in that case.

    A simple greedy algorithm was found to be undesirable. A desired step of 5
    days with a possible step of 4.99999 days would cause a very small step
    afterward. This small step would change the flux level (renormalization),
    and thus have a large Xe135 derivative. This would throw dk/dt off significantly,
    which breaks other things. That's why if the desired step is between 1 and 2
    tolerable steps, we want to divide it up into 2 equal parts.

    The too_small parameter sets a threshold for warning that we still end up
    with a small timestep that can throw us off, but it only warns and changes
    nothing else.

    Parameters
    ----------
    data - Batman solving data.
    t - Desired step to take, in seconds.
    p0 - Power to operate at
    atol - Absolute tolerance in the power to operate at.
    rtol - Relative tolerance in the power to operate at.
    too_small - Timescale under which rapid Xe change makes dk/dt unstable.
    norm - Normalization factor to use. Defaults to match p0.
    nudge - A float, very close to but smaller than 1., that is here to ensure
            we don't go overboard. The reason it isn't 1. is that this estimate
            is so good it causes the power to be about 1e-5 off the desired
            margin. Sometimes this is 1e-5 under it, though, and that causes
            2 additional calculations. It would be better if we just avoided
            this problem by taking an additional margin over the desired one.

    Returns
    -------
    A time step estimate that would be within at least one of the tolerances.

    """
    norm = (norm if norm is not None
            else data.normalize(p0, decay_power_allowed=True))
    dpdt = deriv_p(data, norm=norm)
    power_allowed_step = _allowed_predictor_step(p0, dpdt,
                                                 rtol=rtol, atol=atol,
                                                 nudge=nudge)
    res = t if t < power_allowed_step else min(power_allowed_step, t/2)
    if res < too_small:
        warn("Time step is considered too small for accurate dk/dt"
             "information due to Xe135 rapid change: "
             f"{res:.3e} < {too_small:.3e} seconds")
    return res


def halfstep(*_, t: Second, **_kw) -> Second:
    """Method used to guess a correct timestep if the previous one failed.

    Default implementation is just to half the guess.

    Parameters
    ----------
    t - Current timestep guess.

    Returns
    -------
    A corrected step if the previous step was bad.

    """
    return t / 2
