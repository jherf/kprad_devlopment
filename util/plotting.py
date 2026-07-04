# plotting.py
"""
Summary and diagnostic plots for the kprad simulation.

Each routine accepts an optional ``qt`` (quench-time dict from
util.compute_quench_times) and ``summary`` (free-form dict) so the
figure carries the key derived quantities next to the traces.
"""

import matplotlib.pyplot as plt

import numpy as np

_SPECIES_COLOR = {
    "H": "C0",
    "He": "C4",
    "C": "C1",
    "Ne": "C2",
    "Ar": "C3",
    "W": "C5",
}
_PRAD_COLOR = {
    "H": "tab:red",
    "He": "tab:olive",
    "C": "tab:blue",
    "Ne": "tab:green",
    "Ar": "tab:purple",
    "W": "tab:brown",
}


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _decorate_quench(ax, tV, qt, show_legend_marker=True):
    """Overlay TQ window (yellow) and CQ window (light red) shading."""
    if qt is None:
        return
    i4, i5 = qt.get("i4"), qt.get("i5")
    i1, i2 = qt.get("i1"), qt.get("i2")
    if i4 is not None and i5 is not None and i5 > i4:
        ax.axvspan(
            tV[i4],
            tV[i5],
            color="gold",
            alpha=0.15,
            label="TQ" if show_legend_marker else None,
        )
    if i1 is not None and i2 is not None and i2 > i1:
        ax.axvspan(
            tV[i1],
            tV[i2],
            color="tab:red",
            alpha=0.10,
            label="CQ" if show_legend_marker else None,
        )


def _mark_first_light(ax, tfirst):
    if tfirst is not None:
        ax.axvline(tfirst, color="k", linestyle="--", linewidth=0.8, alpha=0.6)


def _fmt_val(v, fmt=".2f"):
    return (
        "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)
    )


# --------------------------------------------------------------------------- #
def plot_deposition(tgrid, Ndot, savepath=None):
    """SPI deposition pulse and its cumulative integral."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    axes[0].plot(tgrid, Ndot)
    axes[0].set(
        xlabel="time [ms]",
        ylabel=r"$\dot{N}$ [$10^{20}$ atoms / ms]",
        title="Deposition rate",
    )
    cum = np.concatenate([[0.0], np.cumsum(Ndot[1:] * np.diff(tgrid))])
    axes[1].plot(tgrid, cum)
    axes[1].set(xlabel="time [ms]", ylabel=r"atoms delivered [$10^{20}$]")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120)
    return fig


# --------------------------------------------------------------------------- #
def plot_main_results(
    res,
    qt=None,
    summary=None,
    Prad=None,
    injection=None,
    savepath=None,
):
    """
    Three-row, three-column simulation summary.

    Layout
    ------
    (0,0) Ip / Iw           (0,1) electron density        (0,2) Te / Ti
    (1,0) Zbar / Zeff       (1,1) thermal energy           (1,2) per-species ion density
    (2,0) radiated power    (2,1) injected particles        (2,2) summary text

    Parameters
    ----------
    res       : dict from util.postprocess
    qt        : dict from util.compute_quench_times  (optional)
    summary   : free-form dict of scalar diagnostics  (optional)
                Expected keys: Ip0, Te0_keV, tfirst, Pradtot,
                               Ephi_CQ, densecrit
    Prad      : dict {symbol: P[GW]}  from util.compute_radiated_power (optional)
    injection : dict {symbol: {'rate', 'cumulative'}} from util.compute_injection (optional)
    savepath  : str path for saving (optional)
    """
    tV = res["tV"]
    tfirst = (summary or {}).get("tfirst")
    qt = qt or {}

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.5))

    # ── (0,0)  Ip / Iw ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(tV, res["Ip"], "C0", linewidth=1.8, label=r"$I_p$")
    ax.plot(tV, res["Iw"], "C2", linewidth=1.2, label=r"$I_w$")
    _decorate_quench(ax, tV, qt, show_legend_marker=True)
    _mark_first_light(ax, tfirst)
    ax.set(xlabel="time [ms]", ylabel="I [MA]", title="Current")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (0,1)  ne ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogy(tV, res["dense"], "C3", linewidth=1.8, label=r"$n_e$ free")
    ax.semilogy(
        tV,
        res["densetot"],
        "C3",
        linewidth=1.0,
        linestyle=":",
        label=r"$n_{e,tot}$ (free + bound/2)",
    )
    _decorate_quench(ax, tV, qt, show_legend_marker=False)
    _mark_first_light(ax, tfirst)
    ax.set(xlabel="time [ms]", ylabel=r"$n_e$ [cm$^{-3}$]", title="Electron density")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── (0,2)  Te / Ti ───────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.semilogy(tV, res["Te"], "C0", linewidth=1.8, label=r"$T_e$")
    ax.semilogy(tV, res["Ti"], "C2", linewidth=1.2, label=r"$T_i$")
    _decorate_quench(ax, tV, qt, show_legend_marker=False)
    _mark_first_light(ax, tfirst)
    ax.set(xlabel="time [ms]", ylabel=r"$T_{e,i}$ [eV]", title="Temperature")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── (1,0)  Zbar / Zeff ───────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(tV, res["Zbar"], "C4", linewidth=1.8, label=r"$\langle Z \rangle$")
    if "Zeff" in res:
        ax.plot(tV, res["Zeff"], "C1", linewidth=1.2, label=r"$Z_\mathrm{eff}$")
    _decorate_quench(ax, tV, qt, show_legend_marker=False)
    _mark_first_light(ax, tfirst)
    ax.set(
        xlabel="time [ms]",
        ylabel=r"$\langle Z \rangle,\; Z_\mathrm{eff}$",
        title="Mean ionization",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (1,1)  thermal energy ─────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(tV, res["Wthe"], "C0", linewidth=1.8, label=r"$W_{th,e}$")
    ax.plot(tV, res["Wthi"], "C2", linewidth=1.2, label=r"$W_{th,i}$")
    ax.plot(
        tV, res["Wthe"] + res["Wthi"], "k--", linewidth=1.0, label=r"$W_{th}$ total"
    )
    _decorate_quench(ax, tV, qt, show_legend_marker=False)
    _mark_first_light(ax, tfirst)
    ax.set(xlabel="time [ms]", ylabel=r"$W_{th}$ [MJ]", title="Thermal energy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (1,2)  per-species total ion density ──────────────────────────────────
    ax = axes[1, 2]
    for key, arr in res.items():
        if not key.startswith("dens_"):
            continue
        sym = key[5:]
        ax.semilogy(
            tV,
            np.maximum(arr.sum(axis=1), 1e10),
            color=_SPECIES_COLOR.get(sym, "C7"),
            linewidth=1.5,
            label=sym,
        )
    _decorate_quench(ax, tV, qt, show_legend_marker=False)
    _mark_first_light(ax, tfirst)
    ax.set(
        xlabel="time [ms]",
        ylabel=r"$n_\mathrm{ions,tot}$ [cm$^{-3}$]",
        title="Per-species ion density",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── (2,0)  radiated power ─────────────────────────────────────────────────
    ax = axes[2, 0]
    if Prad:
        total_prad = np.zeros(len(tV))
        for sym, p in Prad.items():
            total_prad += p
            ax.semilogy(
                tV,
                np.maximum(p, 1e-10),
                color=_PRAD_COLOR.get(sym, "tab:gray"),
                linewidth=1.5,
                label=sym,
            )
        ax.semilogy(
            tV, np.maximum(total_prad, 1e-10), "k--", linewidth=1.2, label="total"
        )
        # i1, i2 = qt.get("i1"), qt.get("i2")
        # if i1 is not None:
        #    ax.axvline(tV[i1], color="k", linestyle=":", alpha=0.6)
        # if i2 is not None:
        #    ax.axvline(tV[i2], color="k", linestyle=":", alpha=0.6)
        _mark_first_light(ax, tfirst)
        _decorate_quench(ax, tV, qt, show_legend_marker=False)
    else:
        ax.text(
            0.5,
            0.5,
            "Prad not computed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    ax.set(
        xlabel="time [ms]",
        ylabel=r"$P_\mathrm{rad}$ [GW]",
        title="Radiated power",
        ylim=(1e-4, None),
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── (2,1)  injected particles ────────────────────────────────────────────
    ax = axes[2, 1]
    if injection:
        ax2 = ax.twinx()
        ax2.set_ylabel(r"Rate [$10^{20}$ atoms/ms]", fontsize=9, color="gray")
        ax2.tick_params(axis="y", colors="gray", labelsize=8)
        for sym, data in injection.items():
            color = _SPECIES_COLOR.get(sym, "C7")
            # Rate as a light shaded area (right axis)
            ax2.fill_between(tV, data["rate"], alpha=0.18, color=color)
            ax2.plot(tV, data["rate"], color=color, linewidth=0.7, alpha=0.5)
            # Cumulative on left axis
            total = float(data["cumulative"][-1])
            ax.plot(
                tV,
                data["cumulative"],
                linewidth=2.0,
                color=color,
                label=f"{sym}  ({total:.2f}×10²⁰)",
            )
        _mark_first_light(ax, tfirst)
        _decorate_quench(ax, tV, qt, show_legend_marker=False)
        ax.set(
            xlabel="time [ms]",
            ylabel=r"Delivered [$10^{20}$ atoms]",
            title="Injected particles vs time",
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Injection data not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax.set(xlabel="time [ms]", title="Injected particles vs time")

    # ── (2,2)  summary text panel ─────────────────────────────────────────────
    ax = axes[2, 2]
    ax.axis("off")
    s = summary or {}
    tTQ = qt.get("tTQ", float("nan"))
    tCQ = qt.get("tCQ", float("nan"))
    Pradtot = s.get("Pradtot")
    lines = [
        ("Simulation Summary", True),
        ("─" * 24, False),
        (f"t_TQ          = {_fmt_val(tTQ)} ms", False),
        (f"t_CQ          = {_fmt_val(tCQ)} ms", False),
        (
            f"E_rad         = {(_fmt_val(Pradtot*1e3, '.0f') if Pradtot is not None else 'N/A')} kJ",
            False,
        ),
        ("─" * 24, False),
        (f"I_p,0         = {_fmt_val(s.get('Ip0'))} MA", False),
        (f"T_e,0         = {_fmt_val(s.get('Te0_keV'))} keV", False),
        (f"first light   = {_fmt_val(s.get('tfirst'))} ms", False),
        (f"E_φ (mid-CQ)  = {_fmt_val(s.get('Ephi_CQ'))} V/m", False),
        (f"n_e,crit      = {_fmt_val(s.get('densecrit'), '.2e')} cm⁻³", False),
    ]
    text = "\n".join(t for t, _ in lines)
    ax.text(
        0.08,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5dc", alpha=0.7),
    )

    # ── footer ───────────────────────────────────────────────────────────────
    fig.suptitle("kprad simulation summary", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0.0, 1, 0.996])  # type: ignore
    if savepath:
        fig.savefig(savepath, dpi=120)
    return fig
