"""Simple forward predictor.

"""
from collections.abc import Sequence
from functools import partial
from math import exp

from coremaker.protocols.mixture import Mixture

from batman.exponentiators import Exponentiator
from batman.solver.inputs import EasyData
from batman.solver.time_est import deriv_p
from batman.units import Second


def predictor(data: EasyData, t: Second, norm: float, *,
              stepper, expo: Exponentiator,
              threshold: float) -> Sequence[Mixture]:
    """Predictor that assumes the flux is at the level normalized at the start.

    Parameters
    ----------
    data - Data to step forward.
    t - Time to step forward, in Seconds.
    norm - Norm for the flux normalization to use directly.
    stepper - Stepping method to use that handles the RunData.
    expo - Exponentiator that does the exp(At) when needed.
    threshold - Threshold under which isotopes are dropped.

    Returns
    -------
    A sequence of post-step mixtures.

    """
    return tuple(data.map(partial(stepper, norm=norm, t=t, expo=expo,
                                  threshold=threshold)))


def energy_conserving_predictor(
        data: EasyData, t: Second, norm: float, *,
        stepper, expo: Exponentiator,
        threshold: float) -> Sequence[Mixture]:
    r"""Predictor that tries to conserve the energy emitted within the time step
    even though the power could decay as fuel is burnt if the flux is kept constant.
    This requires the predictor to lie about the initial flux level, and for the power
    to decay exponentially in time, which may not be your case!

    .. math::
        p(t) = p0*exp(-a*t)

    .. math::
        dp/dt(0) = -p0*a  ==>  a = -(dp/dt(0)) / p0

    .. math::
        \int_{0}^{t}{A*p(t)dt} = p0*t

    .. math::
        A = p0*t / ((p0/a)*(1-exp(a*t)))  \implies  A = a*t / (1-exp(a*t))

    So we can call the regular predictor with norm=A*norm, once we have A.

    Parameters
    ----------
    data - Data to step forward.
    t - Time step size in Seconds.
    norm - Normalization factor that would yield the desired power.
    stepper - Stepping method to use that handles the RunData etc.
    expo - Exponentiation method that gives exp(At)v for A,v,t.
    threshold - Density threshold under which to drop an isotope.

    Returns
    -------
    A sequence of post-step mixtures.

    """

    try:
        assert norm > 0.
        dpdt = deriv_p(data, norm=norm)
        p0 = data.power(norm=norm)
        a = -dpdt / p0
        fac = a*t / (1.-exp(-a*t))
    except (AssertionError, ZeroDivisionError):
        fac = 1.
    return predictor(data, t, norm=fac*norm,
                     stepper=stepper, expo=expo,
                     threshold=threshold)
