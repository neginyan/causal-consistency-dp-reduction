"""Sec. V -- what happens at m_H: two hypotheses and their predictions.

Paper: Sec. V, Eqs. (30)-(35), Table III and Fig. 1.

Sec. IV proved that m_H = M_P/sqrt 2 is the mass at which the Diosi-Penrose
time equals the causal self-crossing time <t> of a branch, for every profile
and every sigma_eff. What happens there depends on one physical question:
does the gravitational branch phase keep accumulating over times longer than
<t>?

H1  (accumulation; Diosi-Penrose plus causality). The phase exponent is
        Gamma(T) = (Delta E_G / hbar) int_0^T F(t) dt ,
    with F(t) the fraction of the self-energy that has arrived. Coherence is
    lost when Gamma = 1. Using int_0^T F dt = int_0^T (T - r/c) (p/r) dr / int (p/r) dr
    (Fubini), one finds
      m << m_H :  tau_D = <t> [1 + (m_H/m)^2]          (DP delayed by <t>)
      m >> m_H :  tau_D = (6/a)^(1/3) (m/m_H)^(-2/3)    (units sigma_eff/c, c = 1)
    where a = lim_{r->0} p(r)/r^2 is the central pair density. The exponent
    -2/3 is the same for every profile that is smooth at the centre; the
    crossover sits at m_H for every profile. For m > m_H, tau_D < 2 <t>, so
    the branches are still causally disconnected when coherence is lost and
    the self-term-only calculation is exact (the Sec. IV E cancellation has not
    started).

H2  (no carry-over; the carrier re-samples every <t>). The phase per interval
    is Delta E_G <t> / hbar = (m/m_H)^2 exactly (Proposition 2 of Sec. IV).
      strong form   : nothing accumulates -> no gravitational decoherence for
                      m < m_H at any time; loss within one interval for m > m_H.
                      This is the Heisenberg cut.
      random-walk form: kicks of size (m/m_H)^2 with random sign accumulate as
                      |<e^{i Phi}>| = cos((m/m_H)^2)^N, so decoherence is only
                      delayed: tau_D ~ 2 <t> (m_H/m)^4.

Common prediction. Under H1 and both forms of H2, a saturated superposition
(xi >> sigma_eff) with m >= m_H loses coherence within about 2 <t>, i.e. a few
sigma_eff/c. One observation of such a superposition surviving much longer
falsifies the whole framework. Below m_H the hypotheses predict m^-2, m^-4, or
no gravitational decoherence, and can be told apart by the mass dependence.

Units: c = sigma_eff = 1, times in units of <t> where stated; SI where stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from model import Report
from sec4_propositions import PROFILES, inv_r, mean_delay, p_pair, M_H_SI, C_SI

CENTRAL_A = {"Gaussian": 4 / np.sqrt(np.pi), "Plummer": 3.0, "exponential": 0.5, "top-hat": 3 / 8}


def arrived_action(p, R, T: float) -> float:
    """int_0^T F(t) dt, with F the arrived fraction of the self-energy (Fubini form)."""
    top = min(T, R)
    cuts = [0.0] + [c for c in (1.0, 5.0, 20.0, 100.0, 1e3, 1e4) if c < top] + [top]
    f = lambda r: (T - r) * p(r) / r
    return sum(quad(f, a, b, limit=400, epsabs=1e-15)[0] for a, b in zip(cuts, cuts[1:])) / inv_r(p, R)


def tau_H1(name: str, mu: float) -> float:
    """H1 decoherence time (units sigma_eff/c) for m/m_H = mu: Gamma(tau) = 1."""
    p, R = PROFILES[name]
    tdp = mean_delay(p, R) / mu**2            # tau_DP = <t> (m_H/m)^2   (Sec. IV, Prop. 2)
    return brentq(lambda T: arrived_action(p, R, T) - tdp, 1e-9, tdp + 60 * mean_delay(p, R), xtol=1e-14)


def tau_H2_random_walk(mu: float) -> float:
    """H2 random-walk form: number of intervals N with cos(mu^2)^N = 1/e (units <t>)."""
    return -1.0 / np.log(np.cos(mu**2))


def slope(f, mu: float, h: float = 1.01) -> float:
    return (np.log(f(mu * h)) - np.log(f(mu / h))) / (2 * np.log(h))


def run() -> Report:
    rep = Report("Sec. V       What happens at m_H: H1, H2 and their predictions")
    names = list(PROFILES)

    # ---------------- H1 --------------------------------------------------
    worst = 0.0
    for n in names:
        t = mean_delay(*PROFILES[n])
        for mu in (0.05, 0.1):
            worst = max(worst, abs(tau_H1(n, mu) / (t * (1 + mu**-2)) - 1))
    rep.check("(a) H1, m << m_H: tau_D = <t> [1 + (m_H/m)^2] for every profile (DP delayed by <t>)",
              worst < 1e-6, f"max relative deviation, 4 profiles x m/m_H = 0.05, 0.1: {worst:.1e}")

    lo = [slope(lambda m, n=n: tau_H1(n, m), 0.02) for n in names]
    rep.check("(b) H1 below m_H: d ln tau_D / d ln m -> -2 (Diosi-Penrose law)",
              np.allclose(lo, -2, atol=0.01), ", ".join(f"{n} {s:.4f}" for n, s in zip(names, lo)))

    big = 1e3
    dev = {m: [abs(tau_H1(n, m) / ((6 / CENTRAL_A[n]) ** (1 / 3) * m ** (-2 / 3)) - 1) for n in names] for m in (1e3, 1e5)}
    rep.check("(c) H1, m >> m_H: tau_D -> (6/a)^(1/3) (m/m_H)^(-2/3), a = central pair density",
              max(dev[1e5]) < 1e-3 and all(b < 1e-10 or b < a / 10 for a, b in zip(dev[1e3], dev[1e5])),
              "deviation at m/m_H = 1e3 -> 1e5 (exponential: O(tau_D) correction, ~100^(2/3) smaller): " + ", ".join(f"{n} {a:.0e}->{b:.0e}" for n, a, b in zip(names, dev[1e3], dev[1e5])))

    hi = [slope(lambda m, n=n: tau_H1(n, m), big) for n in names]
    rep.check("(d) H1 above m_H: the exponent -2/3 is the same for every profile smooth at the centre",
              np.allclose(hi, -2 / 3, atol=0.005), ", ".join(f"{n} {s:.4f}" for n, s in zip(names, hi)))

    at1 = [tau_H1(n, 1.0) / mean_delay(*PROFILES[n]) for n in names]
    mid = [slope(lambda m, n=n: tau_H1(n, m), 1.0) for n in names]
    rep.check("(e) the crossover sits at m_H for every profile: tau_D(m_H) ~ 2<t>, slope between -2 and -2/3",
              all(1.7 < v < 2.05 for v in at1) and all(-2 < s < -2 / 3 for s in mid),
              "tau_D(m_H)/<t>: " + ", ".join(f"{v:.3f}" for v in at1) + " | slope: " + ", ".join(f"{s:.2f}" for s in mid))

    # self-term-only is exact above m_H: the other branch has not been felt yet
    xi, T = 10.0, tau_H1("Gaussian", 1.0)
    cross = quad(lambda r: (T - r) * p_pair(r, xi) / r, 0, T, limit=400)[0]
    rep.check("(f) H1 above m_H: tau_D < xi/c, so the cross term has not arrived (Sec. IV E does not apply)",
              cross < 1e-20 and T < xi, f"xi = 10 sigma_eff, tau_D(m_H) = {T:.3f}: arrived cross action = {cross:.1e}")

    # ---------------- H2 --------------------------------------------------
    per = [2 * m * m * inv_r(*PROFILES[n]) * mean_delay(*PROFILES[n]) for n in names for m in (0.5, 1 / np.sqrt(2), 1.0)]
    expect = [2 * m * m for n in names for m in (0.5, 1 / np.sqrt(2), 1.0)]      # hbar = 1, m_H^2 = 1/2
    rep.check("(g) H2: phase per interval Delta E_G <t>/hbar = (m/m_H)^2 for every profile -> strong form is a cut",
              np.allclose(per, expect, rtol=1e-10),
              "m/m_H = 0.71, 1, 1.41 -> phase per interval 0.5, 1, 2 (all 4 profiles)")

    rw = [tau_H2_random_walk(m) / (2 * m**-4) for m in (0.05, 0.1)]
    rw_slope = slope(tau_H2_random_walk, 0.05)
    rep.check("(h) H2 random-walk form: tau_D -> 2 <t> (m_H/m)^4, a distinct m^-4 law",
              np.allclose(rw, 1, rtol=0.01) and abs(rw_slope + 4) < 0.01,
              f"ratio to 2(m_H/m)^4 at 0.05, 0.1: {rw[0]:.5f}, {rw[1]:.5f}; slope {rw_slope:.4f}")

    # ---------------- common prediction -------------------------------------
    mus = (1.0, 1.5, 3.0, 10.0)
    h1 = max(tau_H1(n, m) / mean_delay(*PROFILES[n]) for n in names for m in mus)
    h2rw = max(tau_H2_random_walk(m) if m * m < np.pi / 2 else 1.0 for m in mus)
    rep.check("(i) common, falsifiable prediction: for m >= m_H every hypothesis gives tau_D <= 2 <t>",
              h1 <= 2.0 + 1e-9 and h2rw <= 2.0,
              f"max tau_D/<t> for m/m_H in [1, 10]: H1 {h1:.3f}, H2 strong <= 1, H2 random walk {h2rw:.3f}")

    sigma = 1e-6                                                     # 1 um, illustrative
    t_si = np.sqrt(np.pi) / 2 * sigma / C_SI
    mu = 1e-3
    h1_si = t_si * (1 + mu**-2)
    rw_si = tau_H2_random_walk(mu) * t_si
    rep.check("(j) SI illustration (sigma_eff = 1 um, m = m_H/1000 = 15 ng): the hypotheses differ by orders of magnitude",
              h1_si < 1e-8 and rw_si > 1e-3,
              f"<t> = {t_si * 1e15:.2f} fs; H1: {h1_si * 1e9:.2f} ns; H2 random walk: {rw_si * 1e3:.1f} ms; H2 strong: never "
              f"(m_H = {M_H_SI * 1e9:.2f} ug)")
    g = tau_H1("Gaussian", 1e5) / mean_delay(*PROFILES["Gaussian"]) * (1e5) ** (2 / 3)
    at_mH = [round(tau_H1(n, 1.0) / mean_delay(*PROFILES[n]), 2) for n in names]
    rep.check("(k) paper numbers: Gaussian coefficient (12/pi)^(1/3) = 1.563; tau_D(m_H)/<t> = 1.99, 1.88, 1.94, 2.00",
              abs(g - (12 / np.pi) ** (1 / 3)) < 1e-4 and round((12 / np.pi) ** (1 / 3), 3) == 1.563
              and at_mH == [1.99, 1.88, 1.94, 2.0],
              f"tau_D (m/m_H)^(2/3)/<t> at m/m_H = 1e5: {g:.5f}; tau_D(m_H)/<t> = {at_mH}")
    return rep


if __name__ == "__main__":
    run().print()
