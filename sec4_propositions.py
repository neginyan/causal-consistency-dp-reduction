"""Sec. IV -- causal consistency of the Diosi-Penrose time scale.

Paper: Sec. IV C-E, Propositions 1 and 2, Eqs. (23)-(29), Table II.

Part A: the switch-on deficit of the branch difference vanishes
---------------------------------------------------------------
Sec. IV B retarded only the saturated self-energy. If the field of the whole
branch difference Delta rho = rho_+ - rho_- is switched on at t = 0 and every
pair of mass elements is retarded by its own light travel time, Fubini gives

    int_0^inf [E - E(t)] dt = (G/c) * ( int Delta rho d^3x )^2 = 0 ,

because both branches carry the same mass m. The cross (interaction) term is
small, ~ G m^2 / xi, but arrives late, after ~ xi/c, so its deficit is again
G m^2 / c and cancels the self terms. A deficit of 2 G m^2 / c survives only
for sigma_eff/c << T << xi/c, i.e. when the branches are causally
disconnected; in the laboratory regime cT >> xi it vanishes. The switch-on
deficit therefore cannot be the origin of Delta S. This agrees with the
statement that the Newtonian constraint field carries no which-path
information (Danielson, Satishchandran and Wald 2022).

Part B: a profile-independent identity and the Heisenberg cut
-------------------------------------------------------------
Let p(r) be the normalised distribution of pair separations inside one
branch, E_self = G m^2 <1/r>_p its self-energy, and

    <t> = (1/c) int r (p/r) dr / int (p/r) dr

the light travel time between mass elements weighted by their contribution to
the self-energy (the mean retardation delay of Sec. IV B). Then, for every
regular profile,

    E_self <t> = G m^2 / c                                   (Proposition 1)

In the saturated regime xi >> sigma_eff, Delta E_G = 2 E_self, and with the
Diosi-Penrose time tau_DP = hbar / Delta E_G,

    tau_DP / <t> = (m_H / m)^2 ,   m_H = sqrt(hbar c / 2G) = M_P / sqrt 2   (Proposition 2)

independently of sigma_eff and of the profile. Above m_H the Diosi-Penrose
reduction would be faster than the time the driving self-energy needs to be
established causally across the branch; the instantaneous Newtonian
description is not self-consistent there. That m_H marks a physical cut is
the remaining hypothesis.

For finite separations Delta E_G(xi) < 2 E_self and the threshold rises,
m_H(xi) = m_H sqrt(2 E_self / Delta E_G(xi)); for the Gaussian profile
Delta E_G(xi) / Delta E_G^sat = 1 - (sqrt(pi)/2) erf(x)/x with x = xi/sigma_eff.

Units: G = c = m = sigma_eff = 1 unless stated; SI where stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from model import Report, TAU_GAUSS

# --- SI constants (CODATA 2018) --------------------------------------------
G_SI, C_SI, HBAR_SI = 6.67430e-11, 2.99792458e8, 1.054571817e-34
M_H_SI = np.sqrt(HBAR_SI * C_SI / (2 * G_SI))          # kg

# --- pair-separation distributions, each normalised: int p dr = 1 -----------
PROFILES = {
    "Gaussian": (lambda r: 4 * np.pi * r**2 * np.exp(-r**2) / np.pi**1.5, np.inf),
    "Plummer": (lambda r: 3 * r**2 * (1 + r**2) ** -2.5, np.inf),
    "exponential": (lambda r: r**2 * np.exp(-r) / 2, np.inf),
    "top-hat": (lambda r: 3 * r**2 / 8 * (r < 2), 2.0),
}


def inv_r(p, R=np.inf) -> float:
    """<1/r>_p, i.e. E_self / (G m^2)."""
    return quad(lambda r: p(r) / r, 0, R, limit=400)[0]


def mean_delay(p, R=np.inf) -> float:
    """<t> = <r/c> weighted by p(r)/r."""
    return quad(p, 0, R, limit=400)[0] / inv_r(p, R)


def fubini_deficit(p, R=np.inf) -> float:
    """int_0^inf dt int_{r > ct} p(r)/r dr, computed as a genuine double integral."""
    inner = lambda t: quad(lambda r: p(r) / r, t, R, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
    cuts = [0, 1, 5, 20, 100, R] if R == np.inf else [0, R]
    return sum(quad(inner, a, b, limit=400, epsabs=1e-14, epsrel=1e-12)[0] for a, b in zip(cuts, cuts[1:]))


def p_pair(r: float, a: float) -> float:
    """Radial density of |x| for a unit-width 3D Gaussian centred at distance a."""
    if a == 0:
        return PROFILES["Gaussian"][0](r)
    return 4 * np.pi * r**2 * np.exp(-(r - a) ** 2) * (-np.expm1(-4 * r * a)) / (4 * r * a) / np.pi**1.5


def pair_deficit(a: float, T: float) -> float:
    """int_0^T [E - E(t)] dt / (G m^2) for one pair class (self: a = 0, cross: a = xi)."""
    top = a + 12
    inner = lambda t: quad(lambda r: p_pair(r, a) / r, t, top, points=[a] if a > t else None, limit=400)[0]
    return quad(inner, 0, T, points=[a] if 0 < a < T else None, limit=400)[0]


def branch_difference_deficit(xi: float, T: float) -> float:
    """Deficit of Delta E_G = 2 (E_self - E_int) with both terms retarded, in G m^2 / c."""
    return 2 * (pair_deficit(0, T) - pair_deficit(xi, T))


def saturation_fraction(x: float) -> float:
    """Delta E_G(xi) / Delta E_G^sat for the Gaussian profile, x = xi / sigma_eff."""
    return 1 - np.sqrt(np.pi) / 2 * erf(x) / x


def run() -> Report:
    rep = Report("Sec. IV C-E  Propositions 1-2, finite separation, vanishing deficit")

    # ---------------- Part A ------------------------------------------------
    xi = 20.0
    late = [branch_difference_deficit(xi, T) for T in (2 * xi, xi + 12)]
    rep.check("(A1) with the interaction term also retarded, the branch-difference deficit vanishes",
              max(abs(v) for v in late) < 1e-6,
              f"xi = 20 sigma_eff, T = 40 and 32 sigma_eff/c: deficit = {late[0]:.1e}, {late[1]:.1e} G m^2/c")

    rows = [(T, branch_difference_deficit(xi, T)) for T in (3.0, 10.0, 40.0)]
    rep.check("(A2) 2 G m^2/c survives only while the branches are causally disconnected (T << xi/c)",
              rows[0][1] > 1.6 and abs(rows[1][1] - 1) < 1e-3 and abs(rows[2][1]) < 1e-6,
              "deficit at T = 3, 10, 40 sigma_eff/c (xi = 20): " + ", ".join(f"{v:.4f}" for _, v in rows))

    cross = [pair_deficit(a, a + 12) for a in (5.0, 20.0)]
    rep.check("(A3) the cross term's deficit is G m^2/c for every xi: small energy x long delay",
              np.allclose(cross, 1.0, atol=1e-6),
              f"cross-term deficit for xi = 5, 20: {cross[0]:.8f}, {cross[1]:.8f} G m^2/c")

    # ---------------- Part B ------------------------------------------------
    prods = {n: inv_r(p, R) * mean_delay(p, R) for n, (p, R) in PROFILES.items()}
    rep.check("(B1) Proposition 1: E_self <t> = G m^2 / c for every profile",
              np.allclose(list(prods.values()), 1.0, rtol=1e-10),
              ", ".join(f"{n} {v:.10f}" for n, v in prods.items()))

    delays = {n: mean_delay(p, R) for n, (p, R) in PROFILES.items()}
    rep.check("(B2) ...although <t> itself is profile dependent (the dependence cancels in the product)",
              abs(delays["Gaussian"] - TAU_GAUSS) < 1e-10 and max(delays.values()) / min(delays.values()) > 2,
              ", ".join(f"{n} {v:.6f}" for n, v in delays.items()) + "  (sigma_eff/c)")

    fub = {n: fubini_deficit(p, R) for n, (p, R) in PROFILES.items() if n != "top-hat"}
    fub["top-hat"] = quad(lambda t: quad(lambda r: PROFILES["top-hat"][0](r) / r, t, 2)[0], 0, 2)[0]
    rep.check("(B3) the Fubini route: int dt int_{r>ct} p/r dr = int p dr / c = 1/c, as a double integral",
              np.allclose(list(fub.values()), 1.0, rtol=1e-8),
              ", ".join(f"{n} {v:.10f}" for n, v in fub.items()))

    worst = 0.0
    for n, (p, R) in PROFILES.items():
        E, t = inv_r(p, R), mean_delay(p, R)
        for m in (0.3, 1 / np.sqrt(2), 2.0):                    # hbar = 1 -> m_H^2 = 1/2
            worst = max(worst, abs((1 / (2 * m * m * E)) / t / (0.5 / m**2) - 1))
    rep.check("(B4) Proposition 2: tau_DP / <t> = (m_H/m)^2 for every profile and mass",
              worst < 1e-10, f"max relative deviation over 4 profiles x 3 masses = {worst:.1e}")

    ratios = []
    for s in (1e-9, 1e-6, 1e-3):                                 # sigma_eff in metres
        E = G_SI * M_H_SI**2 * 2 / (np.sqrt(np.pi) * s)          # Gaussian E_self
        ratios.append((HBAR_SI / (2 * E)) / (TAU_GAUSS * s / C_SI))
    rep.check("(B5) in SI units tau_DP = <t> exactly at m_H = 15.39 ug, for any sigma_eff",
              np.allclose(ratios, 1.0, rtol=1e-12) and abs(M_H_SI * 1e9 - 15.39) < 0.01,
              f"m_H = {M_H_SI * 1e9:.4f} ug; tau_DP/<t> at sigma_eff = 1 nm, 1 um, 1 mm: "
              + ", ".join(f"{r:.12f}" for r in ratios))

    x_list = (0.5, 2.0, 10.0, 1e3)
    num = [2 * (inv_r(PROFILES["Gaussian"][0]) - quad(lambda r, a=x: p_pair(r, a) / r, 0, x + 12,
                                                        points=[x], limit=400)[0]) for x in x_list[:3]]
    closed = [saturation_fraction(x) * 4 / np.sqrt(np.pi) for x in x_list[:3]]
    rep.check("(B6) finite separation: Delta E_G(xi)/Delta E_G^sat = 1 - (sqrt(pi)/2) erf(x)/x",
              np.allclose(num, closed, rtol=1e-8),
              "numerical vs closed form at x = 0.5, 2, 10: " + ", ".join(f"{a:.6f}/{b:.6f}" for a, b in zip(num, closed)))

    x_small = 1e-3
    mH_xi = 1 / np.sqrt(saturation_fraction(x_small))
    rep.check("(B7) small separations are far from the cut: m_H(xi) = m_H sqrt(3) sigma_eff/xi for xi << sigma_eff",
              abs(mH_xi / (np.sqrt(3) / x_small) - 1) < 1e-3,
              f"xi/sigma_eff = 1e-3 (illustrative): m_H(xi) = {mH_xi:.1f} m_H = {mH_xi * M_H_SI * 1e6:.1f} mg")
    closed = {"Gaussian": (2 / np.sqrt(np.pi), np.sqrt(np.pi) / 2), "Plummer": (1.0, 1.0),
              "exponential": (0.5, 2.0), "top-hat": (3 / (2 * 2.0), 2 * 2.0 / 3)}      # R = 2
    dev = max(max(abs(inv_r(p, R) / closed[n][0] - 1), abs(mean_delay(p, R) / closed[n][1] - 1))
              for n, (p, R) in PROFILES.items())
    rep.check("(B8) Table II closed forms: <1/r> and c<t> = 2/(sqrt(pi) s), (sqrt(pi)/2) s; 1/b, b; 1/(2l), 2l; 3/(2R), 2R/3",
              dev < 1e-8, f"max relative deviation over the four profiles = {dev:.1e}")

    unw = [inv_r(p, R) * quad(lambda r, p=p: r * p(r), 0, R, limit=400)[0] for n, (p, R) in PROFILES.items()]
    rep.check("(B9) Sec. IV C: the unweighted mean separation gives a profile-dependent product",
              np.allclose(unw, [4 / np.pi, 2.0, 1.5, 1.125], rtol=1e-8),
              "E_self <r>_p / (G m^2 / c) = " + ", ".join(f"{v:.3f}" for v in unw) + "  (paper: 1.273, 2.000, 1.500, 1.125)")
    return rep


if __name__ == "__main__":
    run().print()
