"""High level methods for solving the Bateman equation

"""
import logging
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from math import isclose
from typing import TypeVar

import numpy as np
from coremaker.materials.mixture import Mixture
from isotopes import ZAID
from scipy.constants import day

from batman.exponentiators import Exponentiator, FluxFunc
from batman.exponentiators import ShiftedCram48 as Cram
from batman.units import MW, PCM, PCMPerSecond, PerSecond, Second

from .inputs import EasyData
from .k_est import calculate_loss_factor, deriv_k, estimate_k
from .time_est import halfstep
from .utils import RunData, capture_warnings, mixture_to_nd

DenArray = np.ndarray
IsoData = dict[ZAID, tuple[MW, PerSecond]]
PowerGuess = Callable[[Second, MW, MW, MW], Second]

__all__ = [
    'BurnResult',
    'Configuration',
    'depstep_single',
    'timestep_constant_flux',
    'timestep_constant_power',
]

modlogger = logging.getLogger('batman.solve')


def depstep_single(data: RunData, t: Second, norm: FluxFunc, *,
                   expo: Exponentiator,
                   threshold: float) -> Mixture:
    """Perform a depletion step for a single component's RunData.

    Parameters
    ----------
    data - RunData for a single component
    t - Time to deplete for.
    norm - Normalization factor for the reaction models.
    expo - Exponentiator function for the models.
    threshold - Minimal density under which isotopes are thrown out the window.

    Returns
    -------
    The mixture after the depletion step.

    """

    (isos, decmodel, reacmodel), mixture, _ = data
    nd = mixture_to_nd(mixture=mixture, isos=isos, dtype=decmodel.mat.dtype)
    sol = expo(decmodel.mat, reacmodel.mat, norm, nd, t)
    isos = mixture.isotopes | {k: v for k, v in zip(isos, sol)}
    return Mixture({k: v for k, v in isos.items() if v > threshold},
                   mixture.temperature,
                   mixture.sab)


EasyType = TypeVar('EasyType', bound=EasyData)


class Configuration:
    """Configuration for how to perform the burnup steps on serial data.

    Parameters
    ----------
    p_rtol - Relative tolerance in the end of step power to desired power.
    p_atol - Absolute tolerance in the same thing.
    decay_power_allowed - A flag for whether non-0 power is allowed when the
                          user's desired power is 0. This only matters if the
                          decay heat is modeled in the decay processes.
    integrator - Integrator used to step the time forward. I.e predictor,
                 predictor/corrector and the like.
    expo - Tool used to calculate the exponential matrix application on a vector
    time_guesser - Tool for guessing how large a time step one can take without
                   violating the tolerances. It's a function that returns a time
                   step in seconds, but its signature cannot be type hinted in
                   current Python (3.8).
    threshold - Density threshold under which we throw away an isotope. This
                ensures we don't lengthen our vectors for meaningless
                quantities.
    guess_correction - Strategy used to find a better step if the time step
                       taken was too large to keep the power level as desired.

    """

    def __init__(self, *, p_rtol: float, p_atol: MW,
                 decay_power_allowed: bool = True,
                 integrator: Callable,
                 expo: Exponentiator = Cram,
                 time_guesser: Callable,
                 threshold: float,
                 guess_correction: Callable = halfstep):
        self.p_rtol = p_rtol
        self.p_atol = p_atol
        self.decay_power_allowed = decay_power_allowed
        self.expo = expo
        self.threshold = threshold
        self.integrator = integrator
        self.time_guesser = time_guesser
        self.guess_correction = guess_correction

    def depstepper(self, data: EasyData,
                   t: Second, norm: FluxFunc) -> Sequence[Mixture]:
        """Do a single burnup step at a given flux level.
        """
        return self.integrator(data, t, norm,
                               stepper=depstep_single,
                               expo=self.expo,
                               threshold=self.threshold)

    def first_guess(self,
                    data: EasyData,
                    t: Second,
                    p: MW,
                    norm: float | None = None) -> Second:
        """Make an initial guess for the timestep to take given the data at
        hand.

        Parameters
        ----------
        data - Easydata to work with
        t - Desired timestep
        p - Power to deplete at
        norm - Normalization factor to use.

        """
        with capture_warnings():
            return self.time_guesser(data, t, p,
                                     atol=self.p_atol,
                                     rtol=self.p_rtol,
                                     norm=norm)


def timestep_constant_flux(data: EasyData, norm: float, t: Second, *,
                           config: Configuration) -> Sequence[Mixture]:
    """Do a burnup step, such that the flux is constant within the timestep.

    Parameters
    ----------
    data - Burnup data to use. See EasyData for details.
    norm - Normalization factor to use in the reaction models.
    t - Time to burn, in seconds.
    config - Configuration object for how to run this.

    """
    return config.depstepper(data, t, norm=lambda x: norm)


@dataclass(frozen=True)
class BurnResult:
    """Data about how the burnup step went.

    Steps is the number of constant flux depletion steps it required.
    Time is the time in seconds the system progressed.
    _last_norm is the last normalization factor used. This is important
    whenever we want to get a derivative at the end of a time step. If we were
    to renormalize the flux, the fast-changing isotopes would try to jump
    exponentially towards their new equilibrium, which would mess up the
    derivative estimate.

    """
    steps: int
    time: timedelta
    last_norm: float
    k: float | None = None
    dkdt: float | None = None

    @classmethod
    def empty(cls, **kwargs):
        """Return an empty result

        """
        return cls(0, timedelta(0.), 1., **kwargs)

    @property
    def rho(self) -> PCM | None:
        """Reactivity in PCM"""
        return self.k and 1e5*(1. - 1./self.k)

    @property
    def drhodt(self) -> PCMPerSecond | None:
        """Reactivity change rate, in PCM per second.
        Returns None if k or dkdt are unknown.

        dr/dt = dr/dk * dk/dt = (1e5/k**2) * dk/dt

        """
        try:
            return 1e5 * self.dkdt / (self.k**2)
        except TypeError:
            return None

    def __add__(self, other: "BurnResult") -> "BurnResult":
        return BurnResult(self.steps + other.steps, self.time + other.time,
                          other.last_norm, other.k, other.dkdt)

    def __radd__(self, other: "BurnResult") -> "BurnResult":
        return BurnResult(self.steps + other.steps, self.time + other.time,
                          self.last_norm, self.k, self.dkdt)


def _step_constant_power(data: EasyData, p: MW, tstep: Second, *,
                         config: Configuration,
                         loss_fac: float | None,
                         ) -> tuple[Sequence[Mixture], BurnResult]:
    """Do one burnup step, such that the total power is constant within the
    timestep.

    Parameters
    ----------
    data - Burnup data to use. See EasyData for details
    p - Power to operate at in MW.
    tstep - Time to burn, in seconds.
    config - Configuration object for how to run this algorithm.
    loss_fac - Unmodeled losses to production ratio. Used for k-info
    """

    norm = data.normalize(p, decay_power_allowed=config.decay_power_allowed)
    cpy = deepcopy(data)
    cpy.mixtures = config.depstepper(data, tstep, norm=lambda x: norm)
    decp, reacp = cpy.powers(norm)
    kest = loss_fac and estimate_k(cpy, loss_factor=loss_fac)
    dkdt = loss_fac and deriv_k(cpy, loss_factor=loss_fac, p=p, _norm=norm)
    if (isclose(decp + reacp, p,
                rel_tol=config.p_rtol,
                abs_tol=config.p_atol)
            or p == reacp == 0.):
        modlogger.info(
            f"The power was {decp + reacp:.8e}, which was deemed "
            f"close enough to {p:.8e}")
        return cpy.mixtures, BurnResult(1, timedelta(seconds=tstep), norm,
                                        k=kest, dkdt=dkdt)
    else:
        del cpy
        modlogger.info(f"Time step failed. Was {tstep / day:.3f} days. "
                       f"Power was {decp + reacp:.8e} instead of {p:.8e}")
        small_step = config.guess_correction(t=tstep, p=p, decp=decp,
                                             reacp=reacp)
        modlogger.info(f"Trying a smaller step of {small_step / day:.3f} "
                       "days.")
        mixtures, result = _step_constant_power(data, p=p,
                                                tstep=small_step,
                                                config=config,
                                                loss_fac=loss_fac)
        return (mixtures,
                BurnResult(1, timedelta(0.), norm, k=kest, dkdt=dkdt) + result)


def timestep_constant_power(data: EasyData, p: MW, t: Second, *,
                            config: Configuration,
                            loss_fac: float | None = None,
                            k0: float | None = None,
                            ) -> tuple[Sequence[Mixture], BurnResult]:
    """Perform burnup steps until you get to the desired burnup (power*time)
    while maintaining constant power within each step.

    Parameters
    ----------
    data - Burnup data to use. See EasyData for details
    p - Power to operate at in MW.
    t - Time to burn, in seconds.
    config - Configuration object for how to run this algorithm.
    loss_fac - Ratio of unmodeled losses to production. Used for k-info.
    k0 - Initial k before the step begins. This can be used for k-info if
         loss_fac is not available.

    Returns
    -------
    The newly updated mixtures and additional information.

    """

    onestep = config.first_guess(data=data, t=t, p=p)
    loss_fac = loss_fac or (k0 and calculate_loss_factor(data, k0))
    modlogger.info(f'Trying a step of {onestep / day:.1f} Days '
                   f'out of {t / day:.1f}')
    next_data = deepcopy(data)
    next_data.mixtures, initial_result = _step_constant_power(data, p=p,
                                                              tstep=onestep,
                                                              config=config,
                                                              loss_fac=loss_fac)
    tstep = initial_result.time.total_seconds()
    if isclose(tstep, t, rel_tol=1e-4, abs_tol=1e-10):
        return next_data.mixtures, initial_result
    else:
        mixtures, result = timestep_constant_power(next_data, p=p, t=t-tstep,
                                                   config=config,
                                                   loss_fac=loss_fac)
        return mixtures, initial_result + result
