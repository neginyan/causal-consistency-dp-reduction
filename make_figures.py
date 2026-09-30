"""Figures for the README (written to ./figures)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from model import I0, I1, I1_asymptotic, J_far, J_far_asymptotic, TAU_GAUSS
from supp_retarded_gaussian import C
from sec4b_self_crossing_time import P_gauss, phi_ret
from sec5_what_happens_at_mH import tau_H1, tau_H2_random_walk
from sec4_propositions import PROFILES, branch_difference_deficit, inv_r, mean_delay

OUT = pathlib.Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False,
})


def fig_static():
    x = np.logspace(-2, 3, 120)
    tau = np.array([I1(v) / (4 * I0(v)) for v in x])
    xb = np.logspace(1.5, 6, 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, tau, color=BLUE, label="Emergent $\\tau_{\\rm eff}$, static superposition (Sec. II A)")
    ax.plot(xb, I1_asymptotic(xb) / (4 * np.sqrt(np.pi)), color=BLUE, linestyle=":", linewidth=1.5,
            label="Asymptote $[\\ln(\\xi/\\sigma_{\\rm eff})+\\gamma_E/2+\\ln 2-1]/4\\sqrt{\\pi}$")
    ax.axhline(TAU_GAUSS, color=ORANGE, linestyle="--", label="Gaussian $\\langle t\\rangle = \\sqrt{\\pi}/2$")
    ax.set_xscale("log")
    ax.set_xlabel("Branch separation $\\xi$ / $\\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\tau_{\\rm eff}$ in units of $\\sigma_{\\rm eff}/c$")
    ax.set_title("Sec. II A: the $\\sigma_{\\rm eff}/c$ scale emerges; the coefficient grows with $\\xi$",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "supp_fig1_static_overlap.png", dpi=160)
    plt.close(fig)


def fig_finite_time():
    y = np.logspace(-1, 4, 120)
    tau = np.array([J_far(v) for v in y]) / (4 * np.sqrt(np.pi))
    yb = np.logspace(0.5, 4, 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(y, tau, color=BLUE, label="Emergent $\\tau_{\\rm eff}$ for $\\xi \\gg cT$ (Sec. II B)")
    ax.plot(yb, J_far_asymptotic(yb) / (4 * np.sqrt(np.pi)), color=BLUE, linestyle=":", linewidth=1.5,
            label="Asymptote $[\\ln(cT/\\sigma_{\\rm eff})+\\gamma_E/2+\\ln 2]/2\\sqrt{\\pi}$")
    ax.axhline(TAU_GAUSS, color=ORANGE, linestyle="--", label="Gaussian $\\langle t\\rangle = \\sqrt{\\pi}/2$")
    ax.axvline(8.70, color=INK2, linewidth=1, linestyle=":")
    ax.text(10, 0.05, "$cT = 8.70\\,\\sigma_{\\rm eff}$\n(solved backwards)", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Duration $cT$ / $\\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\tau_{\\rm eff}$ in units of $\\sigma_{\\rm eff}/c$")
    ax.set_title("Sec. II B: a finite duration replaces $\\xi$ as the cut-off (only if $cT < \\xi$)",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sec2b_finite_protocol.png", dpi=160)
    plt.close(fig)


def fig_retarded():
    u = np.linspace(0.05, 2.0, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(u, [C(v) for v in u], color=BLUE, label="$C(u) = 4u/(1+1/4u^2)$, retarded Gaussian model")
    ax.axhline(2, color=ORANGE, linestyle="--", label="Paper: $\\Delta S = 2Gm^2/c$")
    for uu, cc, txt in ((0.5, C(0.5), "half-line width:  C = 1"), (0.7328, 2.0, "C = 2 at u = 0.733")):
        ax.plot([uu], [cc], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(txt, (uu, cc), xytext=(uu + 0.12, cc - 0.55), fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("$u = c\\,\\delta_t / \\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\Delta S$ in units of $Gm^2/c$")
    ax.set_title("Retarded model: the coefficient depends on the time profile", loc="left",
                 fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "supp_fig2_retarded_gaussian.png", dpi=160)
    plt.close(fig)


def fig_mean_retardation():
    t = np.linspace(0, 3.2, 400)
    phi = np.array([phi_ret(v) for v in t])
    phi /= phi.max()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(t, phi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(t, phi, color=BLUE, label="Retarded potential of an instantaneous Gaussian source")
    ax.axvline(TAU_GAUSS, color=ORANGE, linestyle="--",
               label="Mean delay $\\langle t\\rangle = (\\sqrt{\\pi}/2)\\,\\sigma_{\\rm eff}/c$ (Gaussian)")
    ax.set_xlabel("Time $t$ after the source acts, in units of $\\sigma_{\\rm eff}/c$")
    ax.set_ylabel("Potential at the centre (normalised)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Sec. IV B: causal by construction, no truncation, no peak normalisation",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sec4b_self_crossing_time.png", dpi=160)
    plt.close(fig)


def fig_action_lag():
    T = np.linspace(0, 3.5, 400)
    S0 = T
    SR = T - TAU_GAUSS + np.sqrt(np.pi) / 2 * (1 - erf(T))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(T, S0, color=INK2, linestyle=":", label="Instantaneous reference $S_0 = \\Delta E\\,T$")
    ax.plot(T, SR, color=BLUE, label="Retarded action $S_R(T) = \\Delta E\\int_0^T F\\,dt$")
    ax.plot(T, T - TAU_GAUSS, color=BLUE, linestyle="--", lw=1.2, alpha=0.6,
            label="Asymptote $\\Delta E\\,(T - \\langle t\\rangle)$")
    x0 = 3.0
    ax.annotate("", xy=(x0, x0 - TAU_GAUSS), xytext=(x0, x0),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
    ax.text(x0 + 0.08, x0 - TAU_GAUSS / 2, "$\\Delta E\\,\\langle t\\rangle$\n(lag)",
            color=ORANGE, va="center", fontsize=10)
    ax.set_xlabel("Hold time $T$, in units of $\\sigma_{\\rm eff}/c$")
    ax.set_ylabel("Action, in units of $\\Delta E\\,\\sigma_{\\rm eff}/c$")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 3.6)
    ax.set_title("Sec. IV B: the retarded action lags by $\\langle t\\rangle$ but still grows with $T$",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sec4b_arrival_lag.png", dpi=160)
    plt.close(fig)


def fig_causal_consistency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xi = 20.0
    T = np.linspace(0.05, 34, 90)
    d = [branch_difference_deficit(xi, v) for v in T]
    ax1.plot(T, d, color=BLUE, label="Both terms retarded")
    ax1.axhline(2, color=INK2, linestyle=":", lw=1.4, label="Self term only (Sec. IV B): $2Gm^2/c$")
    ax1.axvline(xi, color=ORANGE, linestyle="--", lw=1.4, label="$T = \\xi/c$")
    ax1.set_xlabel("Hold time $T$, in units of $\\sigma_{\\rm eff}/c$  ($\\xi = 20\\,\\sigma_{\\rm eff}$)")
    ax1.set_ylabel("Switch-on deficit, in units of $Gm^2/c$")
    ax1.set_ylim(-0.1, 2.35)
    ax1.set_title("A: the deficit vanishes once $cT > \\xi$", loc="left", fontweight="bold", fontsize=11.5)
    ax1.legend(loc="upper right", fontsize=9)

    mu = np.logspace(-0.6, 0.6, 50)
    styles = {"Gaussian": "-", "Plummer": "--", "exponential": "-.", "top-hat": ":"}
    for n, (p, R) in PROFILES.items():
        E, t = inv_r(p, R), mean_delay(p, R)
        ax2.plot(mu, [1 / (2 * (m / np.sqrt(2)) ** 2 * E) / t for m in mu],
                 linestyle=styles[n], color=BLUE if n == "Gaussian" else INK2, lw=2.2 if n == "Gaussian" else 1.6,
                 label=f"{n} ($\\langle t\\rangle$ = {t:.3f} $\\sigma_{{\\rm eff}}/c$)")
    ax2.axhline(1, color=ORANGE, lw=1.2)
    ax2.axvline(1, color=ORANGE, lw=1.2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Mass $m / m_H$")
    ax2.set_ylabel("$\\tau_{\\rm DP} / \\langle t\\rangle$")
    ax2.set_title("B: $\\tau_{\\rm DP}/\\langle t\\rangle = (m_H/m)^2$ for every profile",
                  loc="left", fontweight="bold", fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sec4_propositions.png", dpi=160)
    plt.close(fig)


def fig_what_happens():
    t_g = mean_delay(*PROFILES["Gaussian"])
    mu = np.logspace(-2, 2, 90)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(1, 100, color=ORANGE, alpha=0.08, lw=0)
    ax.axhline(2, color=ORANGE, lw=1.0, linestyle=(0, (4, 3)))
    ax.text(10, 3.2, "Common prediction: $\\tau_D \\leq 2\\langle t\\rangle$ for $m \\geq m_H$",
            color=ORANGE, fontsize=9.5, ha="center")
    ax.plot(mu, [tau_H1("Gaussian", m) / t_g for m in mu], color=BLUE,
            label="H1: accumulation (DP + causality), $\\propto m^{-2} \\to m^{-2/3}$")
    mrw = mu[mu**2 < 1.5]
    ax.plot(mrw, [tau_H2_random_walk(m) for m in mrw], color=INK2, linestyle="--",
            label="H2, random-walk form: $\\propto m^{-4}$")
    ax.plot([1, 100], [1, 1], color=INK, lw=2.4, label="H2, strong form: cut at $m_H$")
    ax.annotate("", xy=(0.97, 3e5), xytext=(0.97, 1.6),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2.0))
    ax.text(0.8, 1.5e4, "no gravitational\ndecoherence\n(H2 strong)", ha="right", fontsize=9.5, color=INK)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e2); ax.set_ylim(0.05, 1e8)
    ax.set_xlabel("Mass $m / m_H$   ($m_H = M_P/\\sqrt{2} \\approx 15.4\\,\\mu$g)")
    ax.set_ylabel("Decoherence time $\\tau_D / \\langle t\\rangle$")
    ax.set_title("Sec. V: what happens at $m_H$ under H1 and H2",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sec5_what_happens_at_mH.png", dpi=160)
    plt.close(fig)


def fig_paper():
    """Fig. 1 of the paper (single column, Computer Modern)."""
    rc = {"font.size": 8.5, "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
          "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "axes.grid": False,
          "axes.linewidth": 0.7, "lines.linewidth": 1.5, "pdf.fonttype": 42, "legend.frameon": False,
          "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
          "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black", "text.color": "black"}
    with plt.rc_context(rc):
        BLUE_, INK_, GREY = "#2a78d6", "#111111", "#555555"
        BLUE, INK = BLUE_, INK_
        YLO, YHI = 0.05, 1e11
        frac = (8 - np.log10(YLO)) / (np.log10(YHI) - np.log10(YLO))      # data band ends at 1e8
        mu = np.logspace(-2, 2, 120)
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.axvspan(1, 100, ymax=frac, color=ORANGE, alpha=0.08, lw=0)
        ax.axvline(1, ymax=frac, color=GREY, lw=0.5)
        ax.axhline(2, color=ORANGE, lw=0.8, ls=(0, (4, 3)))
        ax.text(9, 3.4, r"$\tau_D \leq 2\langle t\rangle$ (all)", color=ORANGE, fontsize=8, ha="center")
        for n, ls, lw in (("Gaussian", "-", 1.5), ("Plummer", "--", 1.0)):
            t = mean_delay(*PROFILES[n])
            ax.plot(mu, [tau_H1(n, m) / t for m in mu], color=BLUE, ls=ls, lw=lw, label=f"H1 ({n})")
        mrw = mu[mu**2 < 1.5]
        ax.plot(mrw, [tau_H2_random_walk(m) for m in mrw], color=GREY, ls="-.", label="H2, random walk")
        ax.plot([1, 100], [1, 1], color=INK, lw=2.0, label="H2, strong")
        ax.annotate("", xy=(0.95, 1e5), xytext=(0.95, 1.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        ax.text(1.25, 4e4, "H2 strong: no\ngravitational\ndecoherence\nbelow $m_H$", ha="left", va="top", fontsize=8, color=INK)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 1e2); ax.set_ylim(YLO, YHI)
        ax.set_yticks([1e0, 1e2, 1e4, 1e6, 1e8])
        ax.set_xlabel(r"$m/m_H$"); ax.set_ylabel(r"$\tau_D/\langle t\rangle$")
        ax.legend(loc="upper center", fontsize=7.5, handlelength=2.2, ncol=2, columnspacing=1.2)
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / "paper_fig1_decoherence_time.pdf")
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_static()
    fig_finite_time()
    fig_retarded()
    fig_mean_retardation()
    fig_action_lag()
    fig_causal_consistency()
    fig_what_happens()
    fig_paper()
    print(f"figures written to {OUT}")
