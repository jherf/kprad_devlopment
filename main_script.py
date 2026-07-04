# main_script.py
"""
Main runner to test things as I make them classes / defintions


IMPORTANT NOTE! One will need to manually download the adf11 file for Ne:
https://open.adas.ac.uk/detail/adf11/scd89/scd89_ne.dat

Put it in the aurora directory where it stores data, and rename it to scd96_ne.dat
Do a pip show aurora to see where it is stored, mine was here:
/opt/miniconda3/lib/python3.13/site-packages/aurora/adas_data/adf11

The scd96.dat file has some wrong ionization coefficients.

Note on H vs H2/D2: the config's ``initial.species`` keys are atomic and are
used directly against the layout, so dense0/densi0 are unaffected by the
D2 -> H aliasing. The alias only applies inside the rates classes and the
injector Torr-L conversion (where molecular species deliver 2 atoms each).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid


from kprad.util.physics import log_lambda_ei
from kprad.util.constants import _MU0, _EE
from kprad.util.solver import fkprad
from kprad.util.atomic_adas import AuroraRates
from kprad.util.atomic_cretin import CretinRates
from kprad.util.plotting import plot_main_results
from kprad.util.config import config_loader, build_initial_state, build_injectors
from kprad.util.layout import SolverLayout
from kprad.util.postprocess import (
    postprocess,
    compute_energy_balance,
    compute_injection,
    compute_pellet_injection,
    compute_quench_times,
    compute_radiated_power,
    save_results_h5,
)
from kprad.util.profile import uniform_grid

# Resolution order: CLI argument > $KPRAD_CONFIG > this fallback path.
DEFAULT_CONFIG_PATH = "/Users/plh/Documents/python_code/kprad/configs/206990_CSP.yaml"
DEFAULT_CONFIG_PATH = (
    "/Users/plh/Documents/python_code/kprad/configs/206990_PELLET.yaml"
)


def main(config_path: str | None = None, show: bool = True) -> dict:
    """
    Run one kprad simulation.

    Parameters
    ----------
    config_path : path to a YAML config file.  Falls back to ``DEFAULT_CONFIG_PATH``
                  or the first command-line argument.
    show        : call plt.show() at the end.

    Returns
    -------
    dict with all post-processed results (res + qt + Prad + injection + summary).
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------
    # ---- Load config
    # ------------------------------------------------------------------
    config = config_loader(config_path)
    if config is None:
        raise RuntimeError(f"Could not load config: {config_path}")

    species_densities = config["initial"]["species"]
    Te0_eV = config["initial"]["Te"] * 1e3
    Ti0_eV = config["initial"]["Ti"] * 1e3
    Ip0 = config["initial"]["Ip"]
    li = config["initial"]["li"]
    a = config["initial"]["a"]
    Ap = config["initial"]["Ap"]
    Vp = config["initial"]["Vp"]

    Rmaj = config["tokamak"]["Rmaj"]
    Rw = config["tokamak"]["Rw"]
    dw = config["tokamak"]["dw"]
    Cw = config["tokamak"]["Cw"]
    etaw = config["tokamak"]["etaw"]

    sim = config["simulation"]
    trange = sim["trange"]
    nt = sim.get("nt")  # output time points (was unused)
    Nr = int(sim.get("Nr", 1))  # radial cells; 1 = 0-D model
    t_eval = np.linspace(trange[0], trange[1], nt) if nt else None
    rtol = sim.get("rtol", 1e-2 if Nr == 1 else 1e-5)

    # ------------------------------------------------------------------
    # ---- Atomic rates, layout, and grid
    # ------------------------------------------------------------------
    elements = config["elements"]
    atom_ = config["simulation"]["atomic_backend"]
    print(f"→ Loading rates for {elements}")
    if atom_.upper() == "ADAS":
        rates = AuroraRates(elements=elements)
    elif atom_.upper() == "CRETIN":
        rates = CretinRates(elements=elements)
    else:
        raise RuntimeError(f"Please enter a valid Atomic database backend")
    print(f"→ Loaded rate coefficients from {atom_}")

    # ------------------------------------------------------------------
    # ---- Injectors (before layout: stateful ones claim aux slots)
    # ------------------------------------------------------------------
    injectors = build_injectors(config, Te0_eV=Te0_eV)
    print(f"→ Sources: {injectors}")

    n_aux = 0
    for inj in injectors:
        ns = getattr(inj, "n_state", 0)
        if ns:
            inj.aux_slice = slice(n_aux, n_aux + ns)
            n_aux += ns

    ftimes = [inj.first_light_time() for inj in injectors]
    tfirst = min((t for t in ftimes if t is not None), default=None)
    if tfirst is not None:
        print(f"→ First light at t = {tfirst:.3f} ms")

    layout = SolverLayout(rates, Nr=Nr, n_aux=n_aux)
    grid = uniform_grid(Vp, Nr)
    print(f"→ {layout}")

    # ------------------------------------------------------------------
    # ---- Derived scalars
    # ------------------------------------------------------------------
    tauw = 1e3 * _MU0 * dw * Cw * (np.log(8 * Rmaj / Rw) - 2) / (2 * np.pi * etaw)
    alphaL = 2 * (np.log(8 * Rmaj / Rw) - 2) / li

    # ------------------------------------------------------------------
    # ---- Initial state
    # ------------------------------------------------------------------
    solV0, dense0_prof, densi0_prof = build_initial_state(
        layout=layout,
        species_densities=species_densities,
        Te0_eV=Te0_eV,
        Ti0_eV=Ti0_eV,
        Ip0_MA=Ip0,
        Iw0_MA=0.0,  # Add the wall current later
        injectors=injectors,
    )

    dense0 = float(np.mean(dense0_prof))  # reference density [cm^-3]

    # Initial wall current from the pre-quench Spitzer estimate (MATLAB eta0)
    eta0 = 5.24e-5 * log_lambda_ei(dense0, Te0_eV, 2.0) * 2.0 / Te0_eV**1.5
    solV0[layout.IW] = eta0 * dw * Cw * Ip0 / (etaw * Ap)

    # ------------------------------------------------------------------
    # ---- ODE params
    # ------------------------------------------------------------------
    params = dict(
        rateStruct=rates,
        tauw=tauw,
        alphaL=alphaL,
        Vp=Vp,
        Rmaj=Rmaj,
        Rmin=a,
        Ap=Ap,
        li=li,
        layout=layout,
        grid=grid,
    )

    # ------------------------------------------------------------------
    # ---- Integrate
    # ------------------------------------------------------------------
    # Define tolerances for each item in solV0
    atol = np.empty(layout.size)
    atol[[layout.IP, layout.IW]] = 1e-3  # MA
    atol[layout.field_slice("We")] = 1e-3 / Vp  # MJ/m^3
    atol[layout.field_slice("Wi")] = 1e-3 / Vp
    for s in layout.elements:
        atol[layout.species_slice(s)] = 1e-3 * dense0  # cm^-3
    if layout.n_aux:
        atol[layout.aux_slice()] = 1e-4  # pellet inventory [1e20 molecules]

    print("→ Solving ODE …")
    sol = solve_ivp(
        fkprad,
        t_span=trange,
        y0=solV0,
        t_eval=t_eval,
        args=(injectors, params),
        method="BDF",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        print(f"  ODE warning: {sol.message}")
    print("→ Done.")

    tV = sol.t

    # ------------------------------------------------------------------
    # ---- Post-process
    # ------------------------------------------------------------------
    res = postprocess(tV, sol.y, layout, grid["Vcell"])
    qt = compute_quench_times(res, Ip0, Te0_eV, tfirst)  # type: ignore    tfirst may be None
    Prad = compute_radiated_power(res, params)
    injection = compute_injection(injectors, tV)
    # Stateful sources (pellets) are reconstructed from the aux trace and
    # merged, so the "injected particles" panel includes them.
    for sym, d in compute_pellet_injection(res, injectors).items():
        if sym in injection:
            injection[sym]["rate"] = injection[sym]["rate"] + d["rate"]
            injection[sym]["cumulative"] = (
                injection[sym]["cumulative"] + d["cumulative"]
            )
        else:
            injection[sym] = d
    Pradtot = float(trapezoid(sum(Prad.values()), tV))
    print(f"Total radiated energy = {Pradtot*1e3:.1f} kJ")

    # ------------------------------------------------------------------
    # ---- Global energy balance
    # ------------------------------------------------------------------
    # ebal = compute_energy_balance(res, sol.y, injectors, params)

    # ------------------------------------------------------------------
    # ---- E-field and critical density at mid-CQ
    # ------------------------------------------------------------------
    Ephi_CQ = densecrit = None
    if qt is not None:
        if qt["i3"] is not None:
            dIpdt = np.gradient(res["Ip"], tV, edge_order=2)
            Ephi = (1e9 * _MU0 * li / (4 * np.pi)) * (
                -dIpdt + alphaL * res["Iw"] / tauw
            )
            Ephi_CQ = float(Ephi[qt["i3"]])
            densecrit = 8e14 * Ephi_CQ
            print(
                f"Mid-CQ E_φ = {Ephi_CQ:.2f} V/m   "
                f"n_e,crit = {densecrit:.2e} /cm³   "
                f"n_e,tot = {res['densetot'][qt['i3']]:.2e} /cm³"
            )

    # ------------------------------------------------------------------
    # ---- Plot
    # ------------------------------------------------------------------
    summary = dict(
        Ip0=Ip0,
        Te0_keV=config["initial"]["Te"],
        tfirst=tfirst,
        Pradtot=Pradtot,
        Ephi_CQ=Ephi_CQ,
        densecrit=densecrit,
    )

    if show:
        plot_main_results(
            res,
            qt=qt,
            summary=summary,
            Prad=Prad,
            injection=injection,
            savepath=sim.get("plot_path"),  # optional config key; None = don't save
        )
        # plot_energy_balance(ebal, qt=qt)

        plt.show()

    res.update(Prad=Prad, Pradtot=Pradtot, injection=injection, **qt)  # type: ignore

    res.update(summary)
    res.update(dense0=dense0, ode_success=bool(sol.success))
    # res.update(energy_balance=ebal)

    # ------------------------------------------------------------------
    # ---- Save results to HDF5
    # ------------------------------------------------------------------
    # Enabled by default; set simulation.save_h5: false to skip. Destination
    # resolves via simulation.output_file / output_dir, else constants.OUTPUT_DIR.
    if sim.get("save_h5", True):
        try:
            save_results_h5(
                res,
                path=sim.get("output_file"),
                config=config,
                output_dir=sim.get("output_dir"),
            )
        except Exception as exc:  # never let a save failure lose the run
            print(f"  [!] could not save HDF5 output: {exc}")

    return res


if __name__ == "__main__":
    res = main()
