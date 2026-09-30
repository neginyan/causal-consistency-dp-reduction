"""Sec. IV B -- the causal self-crossing time <t> from the retarded Green function.

Paper: Sec. IV B, Eqs. (19)-(20), and the Gaussian and Plummer rows of Table II.

The pair distribution P(r) acts for an instant. The retarded potential at the
centre is phi(t) = int d^3x P(x) delta(t - |x|/c) / (4 pi |x|), which vanishes
for t < 0 because of the support of the retarded Green function. Its time
integral is the static Newtonian potential, and its mean delay

    <t> = int t phi(t) dt / int phi(t) dt

does not depend on the normalisation of P. It equals the light travel time
r/c averaged with the self-energy weight P(r)/r. For the Gaussian profile
<t> = (sqrt(pi)/2) sigma_eff / c; for the Plummer profile <t> = b / c.

Checks (i) and (g) are supplementary comparisons: the mode average of Sec. II A
gives 1/(2 pi) of the Gaussian <t>, and Delta E_G^sat <t> = 2 G m^2 / c.

Units: sigma_eff = c = G = m = 1 unless stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from model import Report, TAU_GAUSS


def P_gauss(r: float, s: float = 1.0) -> float:
    """Gaussian pair distribution exp(-r^2/s^2), normalised to 1."""
    return np.exp(-r**2 / s**2) / (np.pi**1.5 * s**3)


def phi_ret(t: float, P=P_gauss) -> float:
    """Retarded potential at the centre of an instantaneous source (c = 1)."""
    return t * P(t) if t > 0 else 0.0


def phi_ret_direct(t: float, eps: float, P=P_gauss) -> float:
    """Same quantity from the retarded integral with a narrow pulse of width eps."""
    f = lambda r: r * P(r) * np.exp(-(t - r) ** 2 / (2 * eps**2)) / np.sqrt(2 * np.pi * eps**2)
    return quad(f, 0, 12, points=[t - 5 * eps, t, t + 5 * eps], limit=4000, epsabs=1e-14)[0]


def mean_delay(P=P_gauss) -> float:
    num = quad(lambda t: t * phi_ret(t, P), 0, np.inf, limit=400)[0]
    den = quad(lambda t: phi_ret(t, P), 0, np.inf, limit=400)[0]
    return num / den


def run() -> Report:
    rep = Report("Sec. IV B    Causal self-crossing time from the retarded Green function")

    ts = (0.3, 0.8, 1.5)
    direct = [phi_ret_direct(t, 1e-4) for t in ts]
    rep.check("(a) retarded potential of an instantaneous Gaussian source: phi(t) = t P(t), zero for t < 0",
              np.allclose(direct, [phi_ret(t) for t in ts], rtol=1e-4),
              "direct retarded integral vs t P(t) at t = 0.3, 0.8, 1.5 sigma_eff/c")

    static = quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / (4 * np.pi * r), 0, np.inf)[0]
    rep.check("(b) int phi dt equals the static Newtonian potential (Newtonian limit recovered)",
              abs(quad(phi_ret, 0, np.inf)[0] / static - 1) < 1e-12,
              f"static potential = {static:.10f}")

    md = mean_delay()
    rep.check("(c) mean retardation delay <t> = (sqrt(pi)/2) sigma_eff/c for the Gaussian profile",
              abs(md / TAU_GAUSS - 1) < 1e-12, f"<t> = {md:.12f}, sqrt(pi)/2 = {TAU_GAUSS:.12f}")

    scaled = mean_delay(lambda r: 7.3 * P_gauss(r))
    rep.check("(d) independent of the normalisation of the source (no peak normalisation needed)",
              abs(scaled / md - 1) < 1e-12, f"<t> with P -> 7.3 P: {scaled:.12f}")

    widths = (0.01, 1.0, 50.0)
    ratios = [mean_delay(lambda r, s=s: P_gauss(r, s)) / s for s in widths]
    rep.check("(e) <t> scales exactly as sigma_eff / c",
              np.allclose(ratios, TAU_GAUSS, rtol=1e-10),
              "<t>/(sigma_eff/c) for sigma_eff = 0.01, 1, 50: " + ", ".join(f"{r:.10f}" for r in ratios))

    # pair-separation form: <r/c> weighted by P(r)/r over d^3r
    pair = (quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / r * r, 0, np.inf)[0]
            / quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / r, 0, np.inf)[0])
    rep.check("(f) same value as the light travel time r/c averaged with the self-energy weight P(r)/r",
              abs(pair / md - 1) < 1e-12, f"<r/c>_(P/r) = {pair:.12f}")

    dE_sat = 4 / np.sqrt(np.pi)                    # Delta E_G^sat, G = m = sigma_eff = 1
    rep.check("(g) Delta E_G^sat * <t> = 2 G m^2 / c for the Gaussian profile",
              abs(dE_sat * md - 2) < 1e-12, f"Delta E_sat * <t> = {dE_sat * md:.12f}")

    rho_plummer = lambda r: (1 + r**2) ** -2.5
    md_pl = mean_delay(rho_plummer)
    rep.check("(h) profile dependence of <t> alone: the Plummer profile gives <t> = b/c",
              abs(md_pl - 1) < 1e-10, f"<t>_Plummer = {md_pl:.10f} b/c")

    tau_static_small = 1 / (4 * np.sqrt(np.pi))     # Sec. II A, small separations
    rep.check("(i) Sec. II A (mean inverse mode frequency) gives exactly 1/(2 pi) of this value",
              abs(tau_static_small / md * 2 * np.pi - 1) < 1e-12,
              f"ratio = {tau_static_small / md:.6f}, 1/(2 pi) = {1 / (2 * np.pi):.6f}  (supplementary comparison)")
    return rep


if __name__ == "__main__":
    run().print()
