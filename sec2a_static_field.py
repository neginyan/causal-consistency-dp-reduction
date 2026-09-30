"""Sec. II A -- static, eternal superposition in the scalar-field toy model.

Paper: Sec. II A, Eqs. (6)-(8).

Checks
------
(a) the field model reproduces the closed form of Delta E_G(xi);
(b) the emergent tau_eff is proportional to sigma_eff / c (not assumed);
(c) small separations: tau_eff -> sigma_eff / (4 sqrt(pi) c), i.e. 1/(2 pi) of the Gaussian <t>
    (supplementary: the mode average differs from the causal self-crossing time);
(d) large separations: I1 = ln(xi/sigma_eff) + gamma_E/2 + ln 2 - 1, no saturation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, spherical_jn

from model import (I0, I0_numeric, I1, I1_asymptotic, Report, TAU_GAUSS,
                   cut_mass_ratio, tau_static)


def run() -> Report:
    rep = Report("Sec. II A    Static superposition: field overlap and logarithmic growth")

    xs = [0.05, 0.4, 1.0, 3.0, 10.0]
    num = np.array([4 / np.pi * I0_numeric(x) for x in xs])
    paper = np.array([2 * (2 / np.sqrt(np.pi) - erf(x) / x) for x in xs])
    rel = np.max(abs(num / paper - 1))
    rep.check("(a) field-theory Delta E_G equals its closed form (4Gm^2/sqrt(pi) sigma_eff)[1 - (sqrt(pi)/2) erf(x)/x]",
              rel < 1e-8, f"max rel. deviation = {rel:.1e}")

    def tau_with_sigma(xi, sigma):
        f0 = lambda k: (1 - spherical_jn(0, k * xi)) * np.exp(-k**2 * sigma**2 / 4)
        f1 = lambda k: f0(k) / k if k > 0 else 0.0
        up = 14 / sigma
        pts = [j * np.pi / xi for j in range(1, 100) if j * np.pi / xi < up]
        return quad(f1, 0, up, points=pts, limit=4000)[0] / (4 * quad(f0, 0, up, points=pts, limit=4000)[0])
    ratios = [tau_with_sigma(3.0 * s, s) / s for s in (0.01, 1.0, 100.0)]
    rep.check("(b) emergent tau_eff is proportional to sigma_eff / c",
              np.ptp(ratios) / np.mean(ratios) < 1e-6,
              "tau_eff/(sigma_eff/c) at xi = 3 sigma_eff for sigma_eff = 0.01, 1, 100: "
              + ", ".join(f"{r:.6f}" for r in ratios))

    small = tau_static(1e-2)
    rep.check("(c) small separations: tau_eff -> sigma_eff / (4 sqrt(pi) c) = (Gaussian <t>) / (2 pi)",
              abs(small * 4 * np.sqrt(np.pi) - 1) < 1e-3,
              f"tau_eff = {small:.6f} sigma_eff/c;  ratio to Gaussian <t> = {small / TAU_GAUSS:.4f} "
              f"(1/2pi = {1 / (2 * np.pi):.4f})")

    devs = [I1(x) - I1_asymptotic(x) for x in (100.0, 1000.0)]
    rep.check("(d) large separations: I1 = ln(xi/sigma_eff) + gamma_E/2 + ln 2 - 1 (no saturation)",
              max(abs(d) for d in devs) < 1e-3,
              f"deviations at xi/sigma_eff = 100, 1000: {devs[0]:.1e}, {devs[1]:.1e}")
    return rep


def table() -> None:
    print("\nSec. II A (supplementary): static mode average vs the Gaussian <t> = 0.8862 sigma_eff/c")
    print(f"{'xi/sigma_eff':>13} {'tau_eff/(sigma/c)':>18} {'tau_eff/<t>_Gauss':>18} {'m_cut/m_H':>10}")
    for x in (0.01, 1.0, 10.0, 1e2, 1e3, 1e4, 1e6):
        i1 = I1(x) if x <= 1e3 else I1_asymptotic(x)
        te = i1 / (4 * I0(x))
        mr = f"{cut_mass_ratio(i1):10.3f}" if x >= 10 else f"{'-':>10}"
        print(f"{x:13.3g} {te:18.4f} {te / TAU_GAUSS:18.4f} {mr}")


if __name__ == "__main__":
    run().print()
    table()
