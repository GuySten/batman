"""Tools for reaching a desired k by burnup.

"""
import logging
from collections.abc import Callable, Container, Sequence
from copy import deepcopy
from math import inf

import numpy as np
from coremaker.protocols.mixture import Mixture

from batman.units import MW, Second

from .inputs import EasyData
from .k_est import calculate_loss_factor, deriv_k
from .solve import BurnResult, Configuration, timestep_constant_power

modlogger = logging.getLogger('batman.solve.reach_k')


def _taylor1_est(x0: float, y0: float, y: float, dydx: float):
    """Estimate the x value for which y will have a target value using a local
    derivative and value.

    Parameters
    ----------
    x0 - Local x value
    y0 - Local y value
    y - Desired y value
    dydx - Local derivative of y with respect to x

    Returns
    -------

    The linear gradient descent estimate for the right x

    """
    return x0 + ((y-y0)/dydx)


def _time_guess(*, x0: Second, y0: float, y: float, dydx: float,
                minx: Second, maxx: Second,
                disallowed: Container[Second],
                fallback: Second = None) -> Second:
    xdiff = min(_taylor1_est(x0=x0, y0=y0, y=y,
                             dydx=dydx) - minx,
                maxx - minx)
    if minx + xdiff in disallowed or xdiff < 0:
        if fallback is not None and fallback not in disallowed:
            return fallback
        else:
            assert dydx < 0, dydx
            print(f"{x0=}, {y0=}, {y=}, {dydx=}, {minx=}, {maxx=}")
            raise ValueError(f"Got an illegal time step of {xdiff:.0f} "
                             f"with no legal fallback: {fallback}")
    else:
        return xdiff


def step_desired_k_at_power(data: EasyData, *,
                            p: MW,
                            k0: float, k: float, k_tolerance: float,
                            config: Configuration,
                            maxt: Second = inf,
                            loss_fac: float | None = None,
                            guess: Second | None = None,
                            ) -> tuple[Sequence[Mixture], BurnResult]:
    """Timestep a system at constant power until its k-eigenvalue reaches a
    desired value.
    Assumes k is monotonically decreasing, which may not be your case if you have
    a lot of burnable poison, a Thorium reactor, etc.
    In that case, we suggest you build your own method to estimate your step size.

    The math here is that we assume that during a time step, the derivative of k
    follows the math in :func:`batman.solver.k_est.deriv_k`. We then use a first order
    taylor estimate to guess how large the timestep should be.

    Parameters
    ----------
    data - System data to work with
    p - Power to deplete at
    k0 - Initial state k-eigenvalue, used for calibration
    k - End state desired k-eigenvalue
    k_tolerance - Acceptable absolute error in between the estimated k of the
                  final state and the desired k.
    maxt - Maximal timestep to take, after which the system returns regardless
    loss_fac - The unmodeled death probability.
    guess - initial guess for k search.
    config - Configuration object, defines how the algorithms are performed

    Returns
    -------
    A tuple of the final state mixtures, the estimated eigenvalue at that state
    and the total time in seconds this depletion step was done at.

    """
    if k0 < k - k_tolerance:
        raise ValueError(f"Initial {k0=} must be greater than desired {k=}")
    loss_fac = loss_fac or calculate_loss_factor(data, k0)

    def _at_desired_value(_k) -> bool:
        return abs(_k - k) <= k_tolerance

    t = 0.
    guess = min(guess, maxt) if guess else guess
    tguess = (guess or
              _time_guess(x0=0., y0=k0, y=k,
                          dydx=deriv_k(data, loss_factor=loss_fac, p=p),
                          minx=0, maxx=maxt, disallowed=frozenset())
              )
    info = BurnResult.empty(k=k0)
    while not (_at_desired_value(info.k) or np.isclose(t, maxt)):
        data, ninfo, t, tguess = _try_to_converge(
            data, t, tguess,
            p=p, loss_fac=loss_fac, config=config, maxt=maxt, ktarget=k,
            too_low=lambda x: x < k - k_tolerance)
        info = info + ninfo

    modlogger.info("Maximal allowed step reached"
                   if np.isclose(t, maxt)
                   else "k within tolerance")
    k_dot = info.dkdt or deriv_k(data, loss_factor=loss_fac, p=p)
    info = BurnResult(k=info.k, time=info.time,
                      dkdt=k_dot, steps=info.steps, last_norm=info.last_norm)
    return data.mixtures, info


def _try_to_converge(data: EasyData, t: Second, tguess: Second, *,
                     p: MW,
                     loss_fac: float,
                     config: Configuration,
                     maxt: Second,
                     ktarget: float,
                     too_low: Callable[[float], bool]) -> \
        tuple[EasyData, BurnResult, Second, Second]:
    newdata = deepcopy(data)
    newdata.mixtures, info = timestep_constant_power(data, p, tguess,
                                                     loss_fac=loss_fac,
                                                     config=config)
    cur_k, k_dot = info.k, info.dkdt
    data = data if too_low(cur_k) else newdata
    t = t if too_low(cur_k) else tguess
    tguess = _time_guess(x0=tguess, y0=cur_k, y=ktarget,
                         dydx=k_dot,
                         minx=t, maxx=maxt,
                         fallback=tguess/2,
                         disallowed=frozenset())
    return data, info, t, tguess
