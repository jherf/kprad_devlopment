"""
ODE right-hand side for the kprad 0-D thermal-quench / current-quench model.

The state vector is managed by ``layout.SolverLayout``:
    solV = [Ip, Iw, We[0..Nr-1], Wi[0..Nr-1],
            n_H[Z=0..Zmax_H][0..Nr-1], n_C[…], …]

Units are absolute: energy densities We, Wi in MJ/m³; currents Ip, Iw in MA;
densities in cm⁻³; time t in ms.

``params`` dict keys
--------------------
rateStruct: AuroraRates or CretinRates
tauw      : float   wall time constant [ms]
alphaL    : float   ratio of external to internal inductance
Vp        : float   plasma volume [m³] (informational; RHS uses grid Vcell)
Rmaj      : float   major radius [m]
Rmin      : float   minor radius [m]
Ap        : float   plasma cross-section [m²]
li        : float   self-inductance parameter
layout    : SolverLayout
grid      : dict with ``Vcell`` (Nr,) cell volumes [m³]
"""

import numpy as np
from kprad.util.constants import (
    _MP_OVER_ME,
    _MIN_NE,
    _MIN_TE,
    _AMU,
    _EE,
    _W_FLOOR_MJ,
)
from kprad.util.physics import log_lambda_ei, eta_parallel

def fkprad(t: float, solV: np.ndarray, injectors: list, params: dict) -> np.ndarray:
    """Compute d(solV)/dt

    Parameters
    ----------
    t         : current time [ms]
    solV      : state vector of length ``layout.size``
    injectors : list of Injector instances (GaussianSPI, WallSputter, …)
    params    : dict — see module docstring for required keys
    """

    solV = np.asarray(solV)
    p = params

    rateStruct = p["rateStruct"]
    tauw = p["tauw"]
    alphaL = p["alphaL"]
    Rmaj = p["Rmaj"]
    Rmin = p["Rmin"]
    Ap = p["Ap"]
    li = p["li"]
    layout = p["layout"]
    Vcell = np.asarray(p["grid"]["Vcell"], dtype=float)  # (Nr,) [m^3]

    Nr = layout.Nr
    Vp = float(Vcell.sum())

    # --------------------------------------------------------------------------
    # ---- Unpack the state 
    # --------------------------------------------------------------------------
    Ip = solV[layout.IP]
    Iw = solV[layout.IW]
    We = layout.field(solV, "We") # (Nr,) Units MJ / m^3
    Wi = layout.field(solV, "Wi")
    dens = {sym: layout.species(solV, sym) for sym in layout.elements}
    # dens[sym] : (Zmax+1, Nr) [cm^-3]


    # --------------------------------------------------------------------------
    # ---- Per-cell electron / ion densities, Zeff, temperatures
    # --------------------------------------------------------------------------
    dense = np.zeros(Nr)   # free-electron density [cm^-3]
    A_zeff = np.zeros(Nr)
    densi = np.zeros(Nr)   # total ion+neutral density [cm^-3]
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        Zs = np.arange(1, Zmax + 1)[:, None]          # (Zmax, 1)
        ions = dens[sym][1:, :]                        # (Zmax, Nr)
        dense += np.sum(ions * Zs, axis=0)
        A_zeff += np.sum(ions * Zs**2, axis=0)
        densi += np.sum(dens[sym], axis=0)

    # Floors: protect the divisions below (Zeff, Te, Ti, rate_ei) against a
    # deep-recombination transient driving dense/densi -> 0, which would
    # hand the implicit solver a NaN/Inf with no diagnostic.
    dense = np.maximum(dense, _MIN_NE)
    densi = np.maximum(densi, _MIN_NE)
    # Zeff >= 1 whenever any ions exist; the floor also keeps beta_T's
    # 1/Zeff term finite in the fully-recombined limit.
    Zeff = np.maximum(A_zeff / dense, 1.0)
    Te = np.maximum(We / (1.5 * _EE * dense), _MIN_TE)   # [eV]
    Ti = np.maximum(Wi / (1.5 * _EE * densi), _MIN_TE)

    # --------------------------------------------------------------------------
    # ---- Electron-ion equilibrium rate [1/s], per cel
    # --------------------------------------------------------------------------
    neutral_term = sum(dens[sym][0, :] / _AMU[sym] for sym in layout.elements)
    rate_ei = 4.0e5 * neutral_term / densi

    # Coulomb contribution
    A_ei = 2.9e6 * log_lambda_ei(dense, Te, Zeff) / (densi * Te**1.5)
    B_ei = np.zeros(Nr)
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        Zs = np.arange(1, Zmax + 1)[:, None]
        B_ei += np.sum(dens[sym][1:, :] * Zs**2, axis=0) / _AMU[sym]

    rate_ei += A_ei * B_ei
    rate_ei *= 3.0 * dense / (1.0e12 * _MP_OVER_ME)  # Ave ion-e thermal eq. rate [1/s]

    # --------------------------------------------------------------------------
    # ---- Charge state evolution (local, vectorized over cells)
    # --------------------------------------------------------------------------
    dsolVdt = np.zeros(layout.size)
    dWe = np.zeros(Nr)             # accumulators [MJ/m^3/ms]
    prad_sum = np.zeros(Nr)        # sum_states rrad*n [eV/s] per cell
    dUdt = np.zeros(Nr)            # ionization-potential rate [eV cm^-3/ms]
    
    Tfac = np.sqrt(Te/Ti)           # scaling factor for optical depth (assuming Doppler broadening dominates)

    A_norm = 1e-3 * dense          # (Nr,) [cm^-3 * ms/s] : dn/dt = A_norm * (S n)

    for sym in layout.elements:
        Zmax = layout.Zmax[sym]

        # Call for AuroraRates
        #rion, rrec, rrad = rateStruct.all_rates(sym, dense, Te)  # (Zmax+1, Nr)

        # Call for CretinRates
        tau = Tfac * dens[sym] * Rmin * 100.0 # column density of this charge state [/cm^2] (Zmax+1, Nr)	
        rion, rrec, rrad = rateStruct.all_rates(sym, dense, Te, tau)  # (Zmax+1, Nr)

        rion = rion * dens[sym]
        rrec = rrec * dens[sym]
        rrad = rrad * dens[sym]

        prad_sum += np.sum(rrad, axis=0)

        # ---- Charge-state density derivatives (same stencil as 0-D)
        d = np.empty((Zmax + 1, Nr))
        d[0] = rrec[1] - rion[0]
        if Zmax >= 2:
            d[1:Zmax] = (
                rrec[2 : Zmax + 1] - rion[1:Zmax] - rrec[1:Zmax] + rion[0 : Zmax - 1]
            )
        d[Zmax] = -rrec[Zmax] + rion[Zmax - 1]

        dn = A_norm[None, :] * d                     # [cm^-3 / ms]
        layout.species(dsolVdt, sym)[:, :] = dn

        # Ionization-potential energy: sum_Z dn_Z/dt * E_cum(Z)  [eV cm^-3/ms]
        dUdt += np.einsum(
            "zr,z->r", dn[1 : Zmax + 1, :], rateStruct.Ei_cumulative[sym]
        )


    # --------------------------------------------------------------------------
    # ---- Injector deposition
    # --------------------------------------------------------------------------
    # 0-D-compatible state snapshot (volume averages) for Te-gated sources.
    state = {
        "Te": float(np.sum(Te * Vcell) / Vp),
        "Ti": float(np.sum(Ti * Vcell) / Vp),
        "ne": float(np.sum(dense * Vcell) / Vp),
        "ni": float(np.sum(densi * Vcell) / Vp),
        "time": t,
        # per-cell profiles, for future ablation models:
        "Te_prof": Te, "ne_prof": dense,
    }
    for inj in injectors:
        # Stateful injectors (Pellet): inventory lives in the aux block of
        # the state vector — deliver()-side counters would double-deplete
        # under BDF's rejected trial steps.
        if getattr(inj, "n_state", 0):
            y = layout.aux(solV)[inj.aux_slice]
            deposits, dydt = inj.ablate(t, state, y)
            layout.aux(dsolVdt)[inj.aux_slice] = dydt
        else:
            deposits = inj.deliver(t, state)

        for sym, Ndot in deposits.items():
            if sym not in layout or Ndot == 0.0:
                continue
            # Uniform volumetric deposition (step 3 replaces this with a
            # per-injector radial profile): Ndot [1e20 atoms/ms] spread over
            # Vp gives the same neutral-density rate in every cell.
            layout.species(dsolVdt, sym)[0, :] += Ndot * 1e14 / Vp  # [cm^-3/ms]

    # --------------------------------------------------------------------------
    # ---- Energy balance, per cell
    # --------------------------------------------------------------------------
    # Radiation cooling [MJ/m^3/ms]: e * ne * sum(rrad*n) * 1e-3
    arad = _EE * dense * np.maximum(prad_sum, 0.0) * 1e-3
    # Ionization-potential sink [MJ/m^3/ms]
    du = _EE * dUdt
    # e-i collisional transfer [MJ/m^3/ms]
    dWei = 1.5e-3 * dense * _EE * rate_ei * (Te - Ti)

    # ---- Resistivity per cell, lumped circuit ------------------------------
    eta = np.atleast_1d(eta_parallel(dense, Te, Zeff, Rmaj, Rmin))  # (Nr,) [Ohm*m]

    # Parallel conductances of nested current channels with cross-section
    # split in proportion to cell volume (exact in the 0-D limit; step 5
    # replaces the whole block with current diffusion).
    Ap_k = Ap * Vcell / Vp
    G_k = Ap_k / (2.0 * np.pi * eta * Rmaj)          # cell conductance [1/Ohm]
    G = G_k.sum()
    Rp = 1.0 / G                                      # plasma resistance [Ohm]
    # --- eta_eff is matplab equivilent * Vp / Vcell
    eta_eff = Rp * Ap / (2.0 * np.pi * Rmaj)          # conductance-averaged
    taup = 1e-4 * li * Ap / eta_eff                   # current decay time [ms]


    PJ = 1e3 * Rp * Ip**2                             # total Joule heating [MJ/ms]
    pj = PJ * (G_k / G) / Vcell                       # per-cell [MJ/m^3/ms]

    dWe[:] = pj - arad - du - dWei
    # 0-D freezes dWthe when Wthe <= 1e-6 MJ; the equivalent Nr-independent
    # criterion is on the energy density (1e-6 MJ spread over Vp).
    dWe[We <= _W_FLOOR_MJ / Vp] = 0.0
    layout.field(dsolVdt, "We")[:] = dWe
    layout.field(dsolVdt, "Wi")[:] = dWei

    # ---- Global circuit -----------------------------------------------------
    dsolVdt[layout.IP] = -Ip / taup + alphaL * Iw / tauw
    dsolVdt[layout.IW] = -dsolVdt[layout.IP] - Iw / tauw

    return dsolVdt
