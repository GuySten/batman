"""Tools for estimating k and its derivative at a given burnup using specific
data

"""
import operator
from functools import partial

import numpy as np

from batman.units import MW

from .inputs import EasyData
from .utils import RunData, mixture_to_nd

__all__ = [
    'absorption',
    'calculate_loss_factor',
    'calculate_losses',
    'deriv_k',
    'estimate_k',
    'production',
]


def estimate_k(data: EasyData, loss_factor: float) -> float:
    """Estimate the k-eigenvalue given the current data and the external
    unnormalized losses term.

    This follows the following equation:

    k = sum(nusigf*N)/(sum(abs*N)+L)

    Parameters
    ----------
    data - EasyData for the current state of the system.
    loss_fac - Unmodeled losses over production

    Returns
    -------
    k-eigenvalue estimate

    """

    prod = production(data)
    return prod / (absorption(data) + loss_factor*prod)


def production(data: EasyData) -> float:
    """Calculate the total neutron production rate.

    Parameters
    ----------
    data - Batman solving data.

    """
    return data.map_reduce(partial(_get_from_single, attribute='production'),
                           operator.add,
                           initial=0.)


def absorption(data: EasyData) -> float:
    """Calculate the total neutron absorption rate in data

    Parameters
    ----------
    data - Batman solving data.

    """
    return data.map_reduce(partial(_get_from_single, attribute='absorption'),
                           operator.add,
                           initial=0.)


def _get_from_single(data: RunData, attribute: str) -> float:
    (isos, _, reacmodel), mixture, vol = data
    nd = mixture_to_nd(mixture, isos, dtype=reacmodel.dtype)
    return vol*getattr(reacmodel, attribute)(nd=nd)


def calculate_losses(data: EasyData, k: float) -> float:
    """Calculate the losses term given k

    Losses are calculated as (sum(nusigf * N) - k*sum(abs * N))/k

    Parameters
    ----------
    data - System data at this time.
    k - Pre-calculated k-eigenvalue.

    """

    return (production(data) - (k * absorption(data))) / k


def calculate_loss_factor(data: EasyData, k: float) -> float:
    """Calculate the loss factor from knowing the initial k and data.

    Parameters
    ----------
    data - Initial state data.
    k - k eigenvalue for the system that is represented by the data.

    Returns
    -------
    The fraction of unmodeled losses in data given this k value.

    """

    return _calculate_loss_factor(prod0=production(data),
                                  loss0=calculate_losses(data, k))


def _calculate_loss_factor(prod0: float, loss0: float):
    """Calculate the unmodeled losses to production ratio,
    a.k.a the unmodeled death probability

    In this scheme we assume the unmodeled death probability is constant.

    Parameters
    ----------
    prod0 - Initial production rate
    loss0 - Initial unmodeled loss rate

    Returns
    -------
    The unmodeled loss to production ratio.

    """
    return loss0/prod0


def deriv_k(data: EasyData, *, loss_factor: float, p: MW,
            _norm: float = 0.0) -> float:
    r"""Estimate the time derivative of the eigenvalue with the current data.

    Derivative follows:

    .. math::
        \frac{dk}{dt}=\frac{dk}{dN} \dot \frac{dN}{dt} = \frac{dk}{dN} \dot AN

    .. math::
        \left(\frac{dk}{dN}\right)_i = \frac{prod_i(totabs-prod \dot loss)
        - prod \dot abs_i} {totabs^2}

    This derivation assumes that

    .. math::
        k = \frac{\sum_i\nu\sigma_{f,i}N_i}{\sum_i\sigma_{c,i}N_i+L\sum_i\nu\sigma_{f,i}N_i}

    where L doesn't change during the timestep, and doesn't depend on the values of N during the timestep.

    Your mileage may vary, but we found this to be true in most scenarios.

    Parameters
    ----------
    data - Current batman data.
    loss_fac - unmodeled losses to production factor.

    """
    norm = _norm or data.normalize(p=p, decay_power_allowed=False)
    totprod = production(data)
    totabs = absorption(data) + loss_factor*totprod
    _deriv = partial(_deriv_k_single, norm=norm, absorb=totabs, prod=totprod,
                     loss_factor=loss_factor)
    return data.map_reduce(_deriv, operator.add, initial=0.)


def _deriv_k_single(data: RunData, *,
                    norm: float, absorb: float, prod: float,
                    loss_factor: float) -> float:
    (isos, decm, reacm), mix, vol = data
    nd = mixture_to_nd(mixture=mix, isos=isos, dtype=reacm.dtype)
    dkdn = (vol*((absorb-prod*loss_factor)*reacm.prod_model
                 - prod*reacm.absorption_model)
            / (absorb**2))
    dndt = (decm.mat + (norm*reacm.mat)) @ nd
    return float(np.dot(dkdn, dndt))
