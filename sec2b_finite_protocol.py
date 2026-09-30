"""Sec. II B -- finite-time superposition in the scalar-field toy model.

Paper: Sec. II B and Sec. III, Eqs. (9)-(17) and Table I.

The source difference exists for a time T (sudden switch-on at t = 0 and
switch-off at t = T). Each field mode picks up the factor 4 sin^2(ckT/2).
Two regimes appear:

* xi >> cT: the time T replaces the separation as infrared cut-off,
      J -> 2 ln(cT/sigma_eff) + gamma_E + 2 ln 2 ,
  so that
      tau_eff / (sigma_eff/c) -> [ln(cT/sigma_eff) + gamma_E/2 + ln 2] / (2 sqrt(pi)).
* cT >> xi (every laboratory experiment: T >= 1 us means cT >= 300 m):
      J -> 2 I1(xi/sigma_eff) ,
  i.e. the static static result, doubled by the switching. The logarithmic
  dependence on xi/sigma_eff remains.

Checks
------
(a) J does not depend on xi once xi >> cT (T acts as infrared cut-off);
(b) dJ_far/dy = 4 D(y) (Dawson function), exactly;
(c) J_far = 2 ln y + gamma_E + 2 ln 2 + O(1/y^2) for y >> 1;
(d) tau_eff = sqrt(pi)/2 sigma_eff/c (the Gaussian <t>) requires
    cT/sigma_eff = 8.70 (exact); the asymptotic form gives the approximation
    exp(pi - gamma_E/2)/2 = 8.67. Both are obtained by solving backwards,
    not derived;
(e) laboratory regime cT >> xi: J = 2 I1(xi/sigma_eff);
(f) short times: J_far -> 2 y^2.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import dawsn

from model import (EULER_GAMMA, I0, I1, J, J_far, J_far_asymptotic, Report,
                   TAU_GAUSS, cut_mass_ratio)

Y_STAR = np.exp(np.pi - EULER_GAMMA / 2) / 2


def run() -> Report:
    rep = Report("Sec. II B-III Finite protocol: Dawson asymptotics, Table I, Eq. (17)")

    vals = [J(x, 10.0) for x in (1e3, 1e4, 1e5)]
    rep.check("(a) xi >> cT: J is independent of the separation (T is the infrared cut-off)",
              np.ptp(vals) / np.mean(vals) < 1e-4,
              "J(xi, cT = 10 sigma_eff) for xi/sigma_eff = 1e3, 1e4, 1e5: "
              + ", ".join(f"{v:.5f}" for v in vals))

    ys = (0.5, 2.0, 7.0)
    deriv = [quad(lambda k: 2 * np.exp(-k**2 / 4) * np.sin(k * y), 0, 20, limit=2000)[0] for y in ys]
    rep.check("(b) dJ_far/dy = 4 D(y) (Dawson function)",
              np.allclose(deriv, [4 * dawsn(y) for y in ys], rtol=1e-8),
              ", ".join(f"y={y}: {d:.6f} vs {4 * dawsn(y):.6f}" for y, d in zip(ys, deriv)))

    devs = [J_far(y) - J_far_asymptotic(y) for y in (1e2, 1e3, 1e4)]
    rep.check("(c) J_far = 2 ln(cT/sigma_eff) + gamma_E + 2 ln 2 + O(sigma_eff^2/c^2T^2)",
              all(abs(d) * y**2 < 1 for d, y in zip(devs, (1e2, 1e3, 1e4))),
              "deviations at y = 1e2, 1e3, 1e4: " + ", ".join(f"{d:.1e}" for d in devs))

    y_num = brentq(lambda y: J_far(y) / (4 * np.sqrt(np.pi)) - TAU_GAUSS, 3, 30, xtol=1e-12)
    rep.check("(d) tau_eff = (sqrt(pi)/2) sigma_eff/c requires cT/sigma_eff = 8.70 (protocol tuning, Sec. III B)",
              abs(y_num - 8.70) < 0.01 and abs(y_num / Y_STAR - 1) < 5e-3,
              f"exact root {y_num:.4f}; asymptotic approximation exp(pi - gamma_E/2)/2 = {Y_STAR:.4f}")

    pairs = ((10.0, 1e3), (100.0, 1e4))
    lab = [(J(x, y), 2 * I1(x)) for x, y in pairs]
    rep.check("(e) laboratory regime cT >> xi: J = 2 I1(xi/sigma_eff) (static result, doubled)",
              all(abs(a / b - 1) < 1e-4 for a, b in lab),
              "; ".join(f"xi={x:g}, cT={y:g}: J={a:.4f}, 2 I1={b:.4f}" for (x, y), (a, b) in zip(pairs, lab)))

    small = [J_far(y) / (2 * y**2) for y in (1e-2, 3e-2)]
    rep.check("(f) short times: J_far -> 2 (cT/sigma_eff)^2",
              all(abs(s - 1) < 1e-3 for s in small),
              "J_far/(2 y^2) at y = 0.01, 0.03: " + ", ".join(f"{s:.5f}" for s in small))
    table_I = {0.25: (0.1224, 0.0173), 0.5: (0.4610, 0.0651), 1.0: (1.4789, 0.2088), 2.0: (3.1847, 0.4496),
               3.0: (4.0992, 0.5787), 10.0: (6.5635, 0.9266), 30.0: (8.7644, 1.2373), 100.0: (11.1638, 1.5760)}
    i0 = I0(1000.0)
    ok_tab = all(abs(J(1000.0, y) - a) < 6e-5 and abs(J(1000.0, y) / (4 * i0) - b) < 6e-5 for y, (a, b) in table_I.items())
    eq17 = [round(J(x, 10.0), 5) for x in (1e3, 1e4, 1e6)]
    jinf = (round(float(J_far(100.0)), 4), round(float(J_far_asymptotic(100.0)), 4))
    lab4 = [round(J(10.0, 1e3), 4), round(J(100.0, 1e4), 4)]
    rep.check("(g) paper numbers: Table I, Eq. (17), J_inf(100) = 11.1738 vs 11.1739, lab values 4.5737, 9.1739",
              ok_tab and eq17 == [6.56354, 6.56364, 6.56364] and jinf == (11.1738, 11.1739) and lab4 == [4.5737, 9.1739],
              f"Table I reproduced to 4 decimals: {ok_tab}; Eq. (17): {eq17}; J_inf(100): {jinf}; lab: {lab4}")
    return rep


def table() -> None:
    print("\nSec. II B, regime xi >> cT:  tau_eff/(sigma_eff/c)")
    print(f"{'cT/sigma_eff':>13} {'exact':>9} {'asymptote':>10}")
    for y in (3.0, 10.0, 30.0, 100.0, 1000.0, 1e4):
        print(f"{y:13g} {J_far(y) / (4 * np.sqrt(np.pi)):9.4f} "
              f"{J_far_asymptotic(y) / (4 * np.sqrt(np.pi)):10.4f}")
    y_num = brentq(lambda y: J_far(y) / (4 * np.sqrt(np.pi)) - TAU_GAUSS, 3, 30, xtol=1e-10)
    print(f"(Gaussian <t> = {TAU_GAUSS:.4f} is reached at cT/sigma_eff = {y_num:.3f}; "
          f"asymptotic approximation {Y_STAR:.3f})")

    print("\nSec. II B, laboratory regime cT >> xi:  implied cut mass")
    print(f"{'xi/sigma_eff':>13} {'J = 2 I1':>9} {'m_cut/m_H':>10}")
    for x in (10.0, 100.0, 1000.0):
        j = 2 * I1(x)
        print(f"{x:13g} {j:9.3f} {cut_mass_ratio(j):10.3f}")


if __name__ == "__main__":
    run().print()
    table()
