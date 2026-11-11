"""Tools for the calculation of power and renormalization

"""

from batman.units import MW

from .utils import RunData, mixture_to_nd


def add_two(x: tuple[MW, MW], y: tuple[MW, MW]) -> tuple[MW, MW]:
    """Element-wise add tuples of decay-energy/reaction-energy.

    Parameters
    ----------
    x - First tuple
    y - Second tuple

    """
    return x[0] + y[0], x[1] + y[1]


def single_power_produced(compdata: RunData, norm: float) -> tuple[MW, MW]:
    """Calculate the power produced in a single component

    Parameters
    ----------
    compdata - Data about the component. See RunData.
    norm - Normalization factor for the reaction models.

    """

    (isos, decmodel, reacmodel), mixture, volume = compdata
    nd = mixture_to_nd(mixture=mixture, isos=isos, dtype=decmodel.mat.dtype)
    return decmodel.energy(nd) * volume, reacmodel.energy(nd, norm=norm) * volume


def calc_norm(p: MW, decp: MW, reacp: MW, *,
              decay_power_allowed: bool) -> float:
    """Calculate the normalization for the reaction part given the current
    breakdown and desired power.

    Parameters
    ----------
    p - Desired power, MW.
    decp - Decay power, MW
    reacp - Induced reaction based power, MW
    decay_power_allowed - Flag for whether it's ok to have non-0 current power
                          but 0 desired power.

    Returns
    -------

    Normalization to the reaction matrix to achieve desired p.

    Raises
    ------
    ZeroDivionError if the desired power is positive, the reaction power is 0
    and decay_power_allowed is False.

    ValueError if the desired power is positive and the decay power is
    greater than that power.

    """
    if p and not reacp and not decay_power_allowed:
        raise ZeroDivisionError(f"Cannot reach desired power {p} since "
                                f"pre-normalized reaction power was {reacp}")
    elif p and decp > p:
        raise ValueError(f"Decay power ({decp}) greater than non-zero-desired "
                         f"power: {p}")
    else:
        pleft = (p - decp) if p else 0.
        return pleft / reacp if reacp else 0.
