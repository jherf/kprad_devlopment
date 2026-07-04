#!/usr/bin/env python3
# compare_rates.py
"""
Compare CRETIN non-LTE rate coefficients (CretinRates, atomic_cretin.py) against
ADAS/Aurora coronal rate coefficients (AuroraRates, atomic.py).

Produces a 2x3 figure:

    top row    : ionization | recombination | radiation   vs  electron temperature
    bottom row : ionization | recombination | radiation   vs  electron density

Each panel overlays every charge state (color = charge state, 0 = neutral),
with CRETIN drawn as solid lines and ADAS/Aurora as dashed lines.

Both back-ends expose ``all_rates`` that return 0-based arrays of length
Zmax+1 (index i = charge state i), so the two are directly index-aligned:

    CretinRates.all_rates(symbol, ne, Te, Ta) -> (Sion, Srec, Prad)
    AuroraRates.all_rates(symbol, ne, Te)     -> (rion, rrec, rrad)

Units (per the class docstrings):
    ionization, recombination : cm^3 / s
    radiation                 : eV cm^3 / s

Note: CRETIN coefficients depend on the column density Ta (and are
nLTE / ne-dependent); the ADAS coefficients do not depend on Ta. So the
ADAS curves are identical for any Ta, and the CRETIN curves move with it.
Confirm the radiation normalization conventions of the two codes match for
your use before drawing quantitative conclusions from the radiation panels.

Run the script next to atomic_cretin.py and atomic.py (or put them on
PYTHONPATH). Example:

    python compare_rates.py --element Ne --Ta 1e12
    python compare_rates.py --element C --Ta 1e14 --save C_compare.png
"""

import argparse


import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from kprad.util.atomic_cretin import CretinRates
from kprad.util.atomic_adas import AuroraRates


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def cretin_sweep(rates, sym, ne_arr, Te_arr, Ta):
    """Evaluate CretinRates.all_rates point-by-point along a sweep.

    ne_arr and Te_arr are equal-length 1-D arrays. Returns three
    (Zmax+1, N) arrays: ionization, recombination, radiation.
    """
    ne_arr = np.atleast_1d(np.asarray(ne_arr, float))
    Te_arr = np.atleast_1d(np.asarray(Te_arr, float))
    n = len(ne_arr)
    nZ = rates.Zmax[sym] + 1

    Sion = np.zeros((nZ, n))
    Srec = np.zeros((nZ, n))
    Prad = np.zeros((nZ, n))
    for i in range(n):
        s, r, p = rates.all_rates(sym, float(ne_arr[i]), float(Te_arr[i]), Ta)
        Sion[:, i] = s
        Srec[:, i] = r
        Prad[:, i] = p
    return Sion, Srec, Prad


def aurora_sweep(rates, sym, ne_arr, Te_arr):
    """Evaluate AuroraRates.all_rates over a sweep (vectorized).

    Returns three (Zmax+1, N) arrays.
    """
    ne_arr = np.atleast_1d(np.asarray(ne_arr, float))
    Te_arr = np.atleast_1d(np.asarray(Te_arr, float))
    rion, rrec, rrad = rates.all_rates(sym, ne_arr, Te_arr, 0.0)
    # Guarantee 2-D (Zmax+1, N) even if the back-end collapsed a length-1 sweep.
    rion = np.atleast_2d(rion)
    rrec = np.atleast_2d(rrec)
    rrad = np.atleast_2d(rrad)
    return rion, rrec, rrad


def _mask(a, floor):
    """Replace non-finite and <= floor values with NaN so log plots stay clean."""
    a = np.array(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    a[a <= floor] = np.nan
    return a


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_panel(ax, x, cret, aur, charges, colors, floor, title, ylabel, xlabel):
    cret = _mask(cret, floor)
    aur = _mask(aur, floor)
    for z in charges:
        ax.plot(x, cret[z], "-", color=colors[z], lw=1.4)
        ax.plot(x, aur[z], "--", color=colors[z], lw=1.1, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)


def make_figure(cret, aur, sym, Ta, ne_fixed, Te_fixed, Te_arr, ne_arr, charges, floor):
    # --- evaluate Te sweep (fixed ne) -------------------------------------
    Sc_ion_T, Sc_rec_T, Sc_rad_T = cretin_sweep(
        cret, sym, np.full_like(Te_arr, ne_fixed), Te_arr, Ta
    )
    Sa_ion_T, Sa_rec_T, Sa_rad_T = aurora_sweep(
        aur, sym, np.full_like(Te_arr, ne_fixed), Te_arr
    )

    # --- evaluate ne sweep (fixed Te) -------------------------------------
    Sc_ion_n, Sc_rec_n, Sc_rad_n = cretin_sweep(
        cret, sym, ne_arr, np.full_like(ne_arr, Te_fixed), Ta
    )
    Sa_ion_n, Sa_rec_n, Sa_rad_n = aurora_sweep(
        aur, sym, ne_arr, np.full_like(ne_arr, Te_fixed)
    )

    nZ = min(Sc_ion_T.shape[0], Sa_ion_T.shape[0])
    if charges is None:
        charges = list(range(nZ))
    else:
        charges = [z for z in charges if 0 <= z < nZ]

    cmap = mpl.cm.viridis  # type: ignore
    norm = mpl.colors.Normalize(vmin=0, vmax=max(nZ - 1, 1))  # type: ignore
    colors = [cmap(norm(z)) for z in range(nZ)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ion_lbl = r"$\langle\sigma v\rangle$  [cm$^3$/s]"
    rad_lbl = r"radiation  [eV cm$^3$/s]"

    # ---- top row: vs Te --------------------------------------------------
    _plot_panel(
        axes[0, 0],
        Te_arr,
        Sc_ion_T,
        Sa_ion_T,
        charges,
        colors,
        floor,
        "Ionization",
        ion_lbl,
        r"$T_e$ [eV]",
    )
    _plot_panel(
        axes[0, 1],
        Te_arr,
        Sc_rec_T,
        Sa_rec_T,
        charges,
        colors,
        floor,
        "Recombination",
        None,
        r"$T_e$ [eV]",
    )
    _plot_panel(
        axes[0, 2],
        Te_arr,
        Sc_rad_T,
        Sa_rad_T,
        charges,
        colors,
        floor,
        "Radiation",
        rad_lbl,
        r"$T_e$ [eV]",
    )

    # ---- bottom row: vs ne ----------------------------------------------
    _plot_panel(
        axes[1, 0],
        ne_arr,
        Sc_ion_n,
        Sa_ion_n,
        charges,
        colors,
        floor,
        "Ionization",
        ion_lbl,
        r"$n_e$ [cm$^{-3}$]",
    )
    _plot_panel(
        axes[1, 1],
        ne_arr,
        Sc_rec_n,
        Sa_rec_n,
        charges,
        colors,
        floor,
        "Recombination",
        None,
        r"$n_e$ [cm$^{-3}$]",
    )
    _plot_panel(
        axes[1, 2],
        ne_arr,
        Sc_rad_n,
        Sa_rad_n,
        charges,
        colors,
        floor,
        "Radiation",
        rad_lbl,
        r"$n_e$ [cm$^{-3}$]",
    )

    # ---- annotate the fixed parameter per row ----------------------------
    axes[0, 0].text(
        0.02,
        0.04,
        rf"$n_e = {ne_fixed:.1e}$ cm$^{{-3}}$",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    axes[1, 0].text(
        0.02,
        0.04,
        rf"$T_e = {Te_fixed:.3g}$ eV",
        transform=axes[1, 0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )

    # ---- colorbar for charge state --------------------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.015)
    cbar.set_label("charge state Z  (0 = neutral)")

    # ---- line-style legend (placed in the empty corner of the Te radiation
    #      panel so it never collides with panel titles) -------------------
    style_handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.6, label="CRETIN (nLTE)"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label="ADAS / Aurora"),
    ]
    axes[0, 2].legend(
        handles=style_handles, loc="lower left", fontsize=9, framealpha=0.9
    )

    fig.suptitle(
        rf"{sym}: CRETIN vs ADAS rate coefficients   "
        rf"($T_a = {Ta:.1e}$ cm$^{{-2}}$)",
        fontsize=14,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--element", default="Ne", help="element symbol (default: Ne)")
    p.add_argument(
        "--Ta",
        type=float,
        default=1e12,
        help="element column density [cm^-2] for CRETIN (default: 1e12)",
    )
    p.add_argument(
        "--ne",
        type=float,
        default=1e14,
        help="fixed electron density [cm^-3] for the Te sweep (default: 1e14)",
    )
    p.add_argument(
        "--Te",
        type=float,
        default=100.0,
        help="fixed electron temperature [eV] for the ne sweep (default: 100)",
    )
    p.add_argument(
        "--Te-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("TEMIN", "TEMAX"),
        help="Te sweep bounds [eV] (default: CRETIN fit Te limits)",
    )
    p.add_argument(
        "--ne-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("NEMIN", "NEMAX"),
        help="ne sweep bounds [cm^-3] (default: CRETIN fit ne limits)",
    )
    p.add_argument(
        "--n-points", type=int, default=80, help="points per sweep (default: 80)"
    )
    p.add_argument(
        "--charges",
        type=int,
        nargs="+",
        default=None,
        help="subset of 0-based charge states to plot (default: all)",
    )
    p.add_argument(
        "--floor",
        type=float,
        default=1e-25,
        help="mask rate values <= this on the log plots (default: 1e-25)",
    )
    p.add_argument("--save", default=None, help="path to save the figure")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="do not call plt.show() (useful with --save)",
    )
    args = p.parse_args()

    sym = args.element

    print(f"Loading CretinRates({[sym]}) ...")
    cret = CretinRates([sym])
    print(f"Loading AuroraRates({[sym]}) ...")
    aur = AuroraRates([sym])

    csym = cret._check_element(sym)

    # Default the sweep ranges to the CRETIN fit limits so we stay in range
    # (CRETIN clamps outside its limits; ADAS extrapolates with its fill value).
    fit = cret._interp[csym]
    Te_lo, Te_hi = args.Te_range if args.Te_range else fit["Telim"]
    ne_lo, ne_hi = args.ne_range if args.ne_range else fit["nelim"]

    Te_arr = np.logspace(np.log10(Te_lo), np.log10(Te_hi), args.n_points)
    ne_arr = np.logspace(np.log10(ne_lo), np.log10(ne_hi), args.n_points)

    print(f"  Ta        = {args.Ta:.3e} cm^-2")
    print(f"  Te sweep  : {Te_lo:.3e} -> {Te_hi:.3e} eV   (fixed ne = {args.ne:.3e})")
    print(
        f"  ne sweep  : {ne_lo:.3e} -> {ne_hi:.3e} cm^-3 (fixed Te = {args.Te:.3g} eV)"
    )

    fig = make_figure(
        cret,
        aur,
        csym,
        args.Ta,
        args.ne,
        args.Te,
        Te_arr,
        ne_arr,
        args.charges,
        args.floor,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
