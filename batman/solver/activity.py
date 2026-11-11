"""module to calculate activities"""

from isotopes import ZAID
from toolz import merge_with

from batman.solver.inputs import EasyData
from batman.solver.utils import RunData
from batman.units import Bq


def _get_single_activities(data: RunData) -> dict[ZAID, Bq]:
    """Calculate activity in Becquerel per unstable isotope"""
    dep, mixture, vol = data
    zaids, decay_model, _ = dep
    return {zaid: -dec * nd * vol * 1e24 for i, zaid in enumerate(zaids) if
            (dec := decay_model.mat[i, i]) and (nd := mixture.get(zaid))}


def activities(data: EasyData) -> dict[ZAID, Bq]:
    """Calculate activity in Becquerel per unstable isotope"""
    return merge_with(sum, *data.map(_get_single_activities))
