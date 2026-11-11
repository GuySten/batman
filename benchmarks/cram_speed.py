"""Speed (and agreement) of batman's exponentiators on a fuel-like chain.

No nuclear data is needed: the chain is synthetic but shaped like a real one.
Nuclides lie on a band of the (Z, A) chart around the valley of stability,
with half-lives from stable down to 1e-20 s away from it. Decays are beta-,
beta+, alpha and beta-delayed neutron emission. The reactions are (n,gamma),
(n,2n), (n,p), (n,alpha), and fission of every actinide into several hundred
fission products. Under a flux, almost every nuclide is in one strongly
connected block, as in a real fuel chain.

Usage::

    python benchmarks/cram_speed.py            # about 3000 nuclides
    python benchmarks/cram_speed.py --small    # about 1000 nuclides

Every exponentiator is called as batman calls it,
``expo(d, r, flux, n0, dt)``, on the same matrices and vectors. The table
reports the time of one call and, for the old solver, the largest
difference from the new default relative to the total atoms (both are at
the round-off of double precision for the bulk; the new one is also
accurate relative to every nuclide's own value).

"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from batman.exponentiators import CramExponentiator, IPFCram16, IPFCram48, ShiftedCram48

DAY = 86400.0
YEAR = 365.25 * DAY


def make_chain(z_max: int = 98, width: int = 15, seed: int = 1):
    """Return (d, r, index): the decay matrix [1/s], the reaction matrix per
    unit flux [cm2], and a dict (Z, A) -> nuclide index."""
    rng = np.random.default_rng(seed)
    def center(z):
        return z * (2.0 + 0.006 * z)  # mass number of the valley of stability

    nuclides = []
    for z in range(1, z_max + 1):
        c = round(center(z))
        for a in range(max(z, c - width), c + width + 1):
            nuclides.append((z, a, a - c))
    index = {(z, a): i for i, (z, a, _) in enumerate(nuclides)}
    n = len(nuclides)

    rows, cols, vals = [], [], []

    def add(i, target, rate):
        j = index.get(target)
        if j is not None and rate > 0:
            rows.append(j)
            cols.append(i)
            vals.append(rate)

    # Every isobar decays towards its own stable charge, so decay has no cycles.
    z_grid = np.arange(1, z_max + 1)
    lam = np.zeros(n)
    for i, (z, a, _) in enumerate(nuclides):
        dz = z - int(z_grid[np.argmin(np.abs(center(z_grid) - a))])
        if dz == 0 and (z < 84 or rng.random() < 0.5):
            continue  # stable (or, for heavy nuclides, alpha only)
        log_t = 17.0 - 5.0 * abs(dz) + rng.normal(0.0, 1.5)
        lam[i] = np.log(2.0) / 10.0 ** np.clip(log_t, -20.0, 18.0)
        branches = {}
        if dz < 0:
            branches[(z + 1, a)] = 1.0
            if dz < -2:
                branches[(z + 1, a - 1)] = 0.05
        elif dz > 0:
            branches[(z - 1, a)] = 1.0
        if z >= 84:
            branches[(z - 2, a - 4)] = rng.uniform(0.1, 10.0)
        total = sum(branches.values())
        for target, b in branches.items():
            add(i, target, lam[i] * b / total)
    d = (sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
         - sp.diags(lam)).tocsr()

    rows, cols, vals = [], [], []
    removal = np.zeros(n)
    fission_products = [i for i, (z, _, dist) in enumerate(nuclides)
                        if 30 <= z <= 66 and abs(dist) <= 10]
    for i, (z, a, _) in enumerate(nuclides):
        xs = {(z, a + 1): 10.0 ** rng.uniform(-25.0, -21.0),
              (z, a - 1): 10.0 ** rng.uniform(-27.0, -25.0),
              (z - 1, a): 10.0 ** rng.uniform(-28.0, -26.0),
              (z - 2, a - 3): 10.0 ** rng.uniform(-28.0, -26.0)}
        for target, sigma in xs.items():
            add(i, target, sigma)
            removal[i] += sigma
        if z >= 90:
            sigma_f = 10.0 ** rng.uniform(-24.0, -21.5)
            products = rng.choice(fission_products,
                                  size=min(600, len(fission_products)),
                                  replace=False)
            yields = rng.dirichlet(np.full(len(products), 0.3)) * 2.0
            for j, y in zip(products, yields):
                rows.append(j)
                cols.append(i)
                vals.append(sigma_f * y)
            removal[i] += sigma_f
    r = (sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
         - sp.diags(removal)).tocsr()
    return d, r, index


def timed(expo, d, r, flux, n0, dt, repeat=1):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = expo(d, r, lambda _t: flux, n0, dt)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--small", action="store_true",
                        help="about 1000 nuclides instead of 3000")
    args = parser.parse_args()
    d, r, index = make_chain(z_max=98, width=5 if args.small else 15)
    n = d.shape[0]
    print(f"{n} nuclides, {(d + r).nnz} matrix entries under flux, "
          f"{d.nnz} in decay")

    fuel = np.zeros(n)
    fuel[index[(92, 235)]] = 0.04
    fuel[index[(92, 238)]] = 0.96
    fuel[index[(8, 16)]] = 2.0
    flux = 3e14

    new_default = ShiftedCram48
    candidates = {
        "old IPFCram48 (spsolve)": IPFCram48,
        "old IPFCram16 (spsolve)": IPFCram16,
        "new ShiftedCram48 (default)": new_default,
        "new CRAM-48, no substeps/shift": CramExponentiator(48, substeps=1, shifted=False),
        "new CRAM-24, no substeps/shift": CramExponentiator(24, substeps=1, shifted=False),
    }
    # warm the pattern caches: batman reuses one exponentiator for many steps
    for expo in candidates.values():
        if isinstance(expo, CramExponentiator):
            expo(d, r, lambda _t: flux, fuel, DAY)
            expo(d, r, lambda _t: 0.0, fuel, DAY)

    burnt, _ = timed(new_default, d, r, flux, fuel, 360 * DAY)
    burnt = np.maximum(burnt, 0.0)
    cases = [("irradiate 1 d", fuel, flux, DAY),
             ("irradiate 30 d", fuel, flux, 30 * DAY),
             ("decay 1 s", burnt, 0.0, 1.0),
             ("decay 1 d", burnt, 0.0, DAY),
             ("decay 1 y", burnt, 0.0, YEAR)]

    header = f"{'case':16s}" + "".join(f"{name:>34s}" for name in candidates)
    print(header)
    for label, n0, phi, dt in cases:
        ref, _ = timed(new_default, d, r, phi, n0, dt, repeat=2)
        line = f"{label:16s}"
        for expo in candidates.values():
            out, sec = timed(expo, d, r, phi, n0, dt, repeat=2)
            diff = np.max(np.abs(out - ref)) / np.sum(np.abs(ref))
            line += f"{sec * 1e3:>18.1f} ms (diff {diff:7.1e})"
        print(line)


if __name__ == "__main__":
    main()
