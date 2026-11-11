"""Radioactive decay data from ENDF-6 files (MF=8, MT=457).

The files are read with the ``endf`` package (https://github.com/paulromano/endf-python),
and every decay mode of every radioactive nuclide becomes a :class:`DecayProcess`:
the parent, its products with their multiplicities, the half-life, the sequence
of decay modes and the branching ratio.

Examples
--------
>>> p = DecayProcess(ZAID(53, 135, 0), ZAID(54, 135, 0), 6.57 * 3600, (BETA_M,))
>>> p.targets == frozenset({ZAID(54, 135, 0)})
True
>>> round(p.rate * 1e5, 4)
2.9306

"""

import itertools as it
import logging
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from os import PathLike
from typing import Any

import endf
from isotopes import H1, ZAID, He4

from batman.units import Second

__all__ = [
    "ALPHA",
    "BETA_M",
    "BETA_P",
    "ISOMERIC_TRANSITION",
    "NEUTRON",
    "PROTON",
    "SPF",
    "DecayProcess",
    "SPFData",
    "decay_modes",
    "decay_processes",
    "halflife_to_rate",
    "parse_decay_processes",
]

modlogger = logging.getLogger("batman.decay_data")

# Names of the decay modes. They are the keys of the decay graph's edges, so
# they must stay the same for serialized graphs to keep their meaning.
BETA_M = r"$\beta^-$"
BETA_P = r"$\beta^+$/e.c."
ALPHA = r"$\alpha$"
NEUTRON = r"neutron emission"
SPF = r"spontaneous fission"
PROTON = "proton emission"
ISOMERIC_TRANSITION = "isomeric transition"

# The digits of the ENDF RTYP of a decay mode (ENDF-6 manual, MF=8 MT=457)
_RTYP_DECAY_MODES = {
    1: BETA_M,
    2: BETA_P,
    3: ISOMERIC_TRANSITION,
    4: ALPHA,
    5: NEUTRON,
    6: SPF,
    7: PROTON,
}
# (change in Z, change in A) of the decaying nucleus in each mode
_DECAY_MODES_CHANGES = {
    BETA_M: (1, 0),
    BETA_P: (-1, 0),
    ISOMERIC_TRANSITION: (0, 0),
    ALPHA: (-2, -4),
    NEUTRON: (0, -1),
    PROTON: (-1, -1),
}
# Light particles produced by a mode, counted as products
_DECAY_MODES_PRODUCTS = {
    PROTON: H1,
    ALPHA: He4,
}

#: Spontaneous fission products: for a fissioning nuclide, (product, yield,
#: yield uncertainty) triplets
SPFData = Mapping[ZAID, Sequence[tuple[ZAID, float, float]]]

EndfSource = endf.Material | PathLike | str


def halflife_to_rate(halflife: Second, branching: float = 1.0) -> float:
    """Decay rate [1/s] of a decay mode with a half-life and a branching ratio.

    Raises
    ------
    ZeroDivisionError if the half-life is 0.

    """
    return log(2.0) * branching / halflife


@dataclass(init=True, frozen=True, unsafe_hash=True)
class DecayProcess:
    """One radioactive decay mode of a nuclide.

    Parameters
    ----------
    parent - The decaying nuclide.
    _target - The products: a single nuclide, or (nuclide, multiplicity) pairs.
    halflife - Half-life of the parent [s].
    mode - The successive decay modes, e.g. ``(BETA_M, NEUTRON)``.
    fraction - Branching ratio of this mode.
    fraction_err - Uncertainty of the branching ratio.
    energy - Total decay energy of the mode [eV].
    energy_err - Uncertainty of the decay energy [eV].
    spectra - Radiation spectra of the mode, if any.

    """

    parent: ZAID
    _target: tuple[tuple[ZAID, float], ...] | ZAID
    halflife: Second
    mode: Sequence[str]
    fraction: float = 1.0
    fraction_err: float = 0.0
    energy: float = 0.0
    energy_err: float = 0.0
    spectra: dict[Any, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def target_branching(self) -> dict[ZAID, float]:
        """Products and their multiplicities"""
        if isinstance(self._target, ZAID):
            return {self._target: 1.0}
        return dict(self._target)

    @property
    def targets(self) -> frozenset[ZAID]:
        """Nuclides the decay produces"""
        return frozenset(self.target_branching)

    @property
    def rate(self) -> float:
        """Decay rate of this mode [1/s] per parent nucleus"""
        return halflife_to_rate(self.halflife, branching=self.fraction)

    @classmethod
    def from_other(cls, other: "DecayProcess", **kwargs) -> "DecayProcess":
        """A copy of ``other`` with the given fields changed (for example, to
        redefine a branching ratio)."""
        for key in ("parent", "_target", "halflife", "mode", "fraction", "fraction_err",
                    "energy", "energy_err", "spectra"):
            kwargs.setdefault(key, getattr(other, key))
        return cls(**kwargs)


def decay_modes(rtyp: float) -> tuple[str, ...]:
    """The successive decay modes of an ENDF RTYP value: one digit per mode,
    e.g. 1.5 is beta- followed by neutron emission.

    Raises
    ------
    ValueError for a digit that is not a decay mode (0, 8, 9).

    """
    digits = f"{rtyp:.6f}".rstrip("0").replace(".", "")
    try:
        return tuple(_RTYP_DECAY_MODES[int(d)] for d in digits)
    except KeyError:
        raise ValueError(f"RTYP {rtyp} is not a sequence of decay modes") from None


def _products(parent: ZAID, modes: Sequence[str], daughter_state: int,
              spf_db: SPFData | None) -> tuple[tuple[ZAID, float], ...]:
    """Products of one decay mode and their multiplicities. Spontaneous
    fission ends the sequence; its products come from ``spf_db`` and are left
    out if it has none for the fissioning nuclide."""
    counter: Counter = Counter()
    iso: ZAID | None = parent
    for mode in modes:
        if product := _DECAY_MODES_PRODUCTS.get(mode):
            counter[product] += 1
        if changes := _DECAY_MODES_CHANGES.get(mode):
            iso = ZAID(iso.Z + changes[0], iso.A + changes[1], daughter_state)
        if mode == SPF:
            for target, yield_, _ in (spf_db or {}).get(iso, ()):
                counter[target] += yield_
            if not spf_db or iso not in spf_db:
                modlogger.debug("No spontaneous fission yields for %s: its products are left out", iso)
            iso = None
            break
    if iso is not None:
        counter[iso] += 1
    return tuple(counter.items())


def decay_processes(material: endf.Material, spf_db: SPFData | None = None) -> list[DecayProcess]:
    """Decay processes of one ENDF material (empty for stable nuclides).

    Parameters
    ----------
    material - An ENDF-6 material with radioactive decay data (MF=8, MT=457).
    spf_db - Spontaneous fission yields; without them, spontaneous fission
             removes the parent but produces nothing.

    """
    data = material.section_data[8, 457]
    z, a = divmod(round(data["ZA"]), 1000)
    parent = ZAID(z, a, int(data["LISO"]))
    if data["NST"] == 1:
        return []
    halflife = float(data["T1/2"][0])
    processes = []
    for m in data["modes"]:
        modes = decay_modes(m["RTYP"])
        processes.append(DecayProcess(
            parent,
            _products(parent, modes, int(m["RFS"]), spf_db),
            halflife,
            modes,
            fraction=float(m["BR"][0]),
            fraction_err=float(m["BR"][1]),
            energy=float(m["Q"][0]),
            energy_err=float(m["Q"][1]),
        ))
    return processes


def _materials(source: EndfSource) -> Iterable[endf.Material]:
    if isinstance(source, endf.Material):
        return (source,)
    # endf warns about every section it does not parse (covariances, for
    # example); only MF=8, MT=457 is read here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"MF=\d+, MT=\d+ ignored", category=UserWarning)
        return endf.get_materials(source)


def parse_decay_processes(*sources: EndfSource, spf_db: SPFData | None = None) -> list[DecayProcess]:
    """Decay processes of ENDF-6 decay files or materials.

    Parameters
    ----------
    sources - ENDF materials, or paths of ENDF-6 files (each may hold many
              materials, as a whole decay sublibrary in one file does).
    spf_db - Spontaneous fission yields; see :func:`decay_processes`.

    """
    return list(it.chain.from_iterable(
        decay_processes(material, spf_db=spf_db)
        for source in sources
        for material in _materials(source)
        if (8, 457) in material.section_data
    ))
