"""
ODE right-hand side for the KPRAD thermal-quench / current-quench model.

READING THIS FILE
-----------------
The model is 0-D in physics but written over ``Nr`` radial cells so that
radially-resolved SPI ablation can be added later without restructuring.
For the literature benchmark cases ``Nr = 1``, and every array below has
length 1. Each block is annotated with its ``Nr = 1`` scalar reduction so
you can read the physics without unpacking the vectorization.

There is no radial transport anywhere in this file. Cells are coupled only
through the global circuit (Ip, Iw) and through the volume-averaged state
handed to Te-gated injectors. Every other term is strictly local.

STATE VECTOR (managed by ``layout.SolverLayout``)
-------------------------------------------------
    state = [Ip, Iw,
             We[0..Nr-1], Wi[0..Nr-1],
             n_H[Z=0..Zmax_H][0..Nr-1], n_C[...], ...,
             aux[0..n_aux-1]]

UNITS (absolute -- no dense0 normalization)
-------------------------------------------
    Ip, Iw    : MA
    We, Wi    : MJ / m^3      (We = 1.5 * e * ne[cm^-3] * Te[eV])
    densities : cm^-3
    time      : ms

``params`` keys
---------------
    rateStruct : AuroraRates or CretinRates
    tauw       : wall current decay time [ms]
    alphaL     : ratio of external to internal inductance
    Vp         : plasma volume [m^3] (informational; RHS uses grid Vcell)
    Rmaj       : major radius [m]
    Rmin       : minor radius [m]
    Ap         : plasma cross-section [m^2]
    li         : internal self-inductance parameter
    layout     : SolverLayout
    grid       : dict with ``Vcell`` (Nr,) cell volumes [m^3]
"""

import numpy as np

from kpradpy.util.constants import (
    _MP_OVER_ME,
    _MIN_NE,
    _MIN_TE,
    _AMU,
    _EE,
    _W_FLOOR_MJ,
)
from kpradpy.util.physics import log_lambda_ei, eta_parallel


def fkprad(
    t: float,
    state: np.ndarray,
    injectors: list,
    params: dict,
    return_diagnostics: bool = False,
):
    """Compute d(state)/dt for ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    t         : current time [ms]
    state     : state vector of length ``layout.size``
    injectors : list of Injector instances (MGI, SPI, WallSputter, ...)
    params    : see module docstring for required keys

    return_diagnostics
        When False (the default, and what ``solve_ivp`` always sees), return
        the derivative array alone. When True, return
        ``(dstate_dt, diagnostics)`` where ``diagnostics`` is a dict of
        volume-INTEGRATED power channels [GW = MJ/ms] evaluated from the very
        same expressions integrated below. ``postprocess.compute_energy_balance``
        uses this so its balance is consistent with the physics by
        construction; any residual then measures only reconstruction error on
        the output time grid, not a modelling inconsistency.
    """

    state = np.asarray(state)

    rateStruct = params["rateStruct"]
    tau_wall_ms = params["tauw"]
    inductance_ratio = params["alphaL"]
    Rmaj = params["Rmaj"]
    Rmin = params["Rmin"]
    Ap = params["Ap"]
    li = params["li"]
    layout = params["layout"]
    cell_volume = np.asarray(params["grid"]["Vcell"], dtype=float)  # (Nr,) [m^3]

    Nr = layout.Nr
    Vp = float(cell_volume.sum())  # [m^3]

    # =========================================================================
    # STEP 1 -- Unpack the state vector
    # =========================================================================
    # ``layout.species`` returns a zero-copy (Zmax+1, Nr) view into the flat
    # vector, so writing into it writes into the vector. No index arithmetic.
    Ip = state[layout.IP]  # [MA]
    Iw = state[layout.IW]  # [MA]
    We = layout.field(state, "We")  # (Nr,) [MJ/m^3]
    Wi = layout.field(state, "Wi")  # (Nr,) [MJ/m^3]
    n = {sym: layout.species(state, sym) for sym in layout.elements}
    # n[sym] : (Zmax+1, Nr) [cm^-3], row Z = charge state Z

    # =========================================================================
    # STEP 2 -- Derived plasma quantities: ne, ni, Zeff, Te, Ti
    # =========================================================================
    # Quasi-neutrality gives the free-electron density as the charge-weighted
    # sum over every ionized state of every element:
    #     ne   = sum_sym sum_{Z>=1} Z   * n_sym,Z
    #     ni   = sum_sym sum_{Z>=0}       n_sym,Z     (neutrals included)
    #     Zeff = (sum_sym sum_{Z>=1} Z^2 * n_sym,Z) / ne
    #
    # Nr = 1: each accumulator below is a single number.
    ne = np.zeros(Nr)  # free-electron density [cm^-3]
    ni = np.zeros(Nr)  # total ion + neutral density [cm^-3]
    sum_ni_Z2 = np.zeros(Nr)  # sum n_Z * Z^2, the Zeff numerator [cm^-3]

    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        charge = np.arange(1, Zmax + 1)[:, None]  # (Zmax, 1), Z = 1..Zmax
        ionized = n[sym][1:, :]  # (Zmax, Nr), drops the neutral row

        ne += np.sum(ionized * charge, axis=0)
        sum_ni_Z2 += np.sum(ionized * charge**2, axis=0)
        ni += np.sum(n[sym], axis=0)  # neutrals included

    # Floors protect the divisions below (Zeff, Te, Ti, equilibration rate)
    # against a deep-recombination transient driving ne or ni toward zero,
    # which would otherwise hand the implicit solver a NaN with no diagnostic.
    ne = np.maximum(ne, _MIN_NE)
    ni = np.maximum(ni, _MIN_NE)

    # Zeff >= 1 whenever ions exist; the floor also keeps the 1/Zeff term in
    # beta_T finite in the fully-recombined limit.
    Zeff = np.maximum(sum_ni_Z2 / ne, 1.0)

    # Invert We = 1.5 * e * ne * Te for the temperatures.
    Te = np.maximum(We / (1.5 * _EE * ne), _MIN_TE)  # [eV]
    Ti = np.maximum(Wi / (1.5 * _EE * ni), _MIN_TE)  # [eV]

    # =========================================================================
    # STEP 3 -- Electron-ion thermal equilibration rate [1/s]
    # =========================================================================
    # Two channels add together:
    #
    #   (a) Collisions with NEUTRALS, a fixed cross-section estimate with no
    #       Te dependence. Dominant once the plasma is cold and largely
    #       recombined, which is where the CQ spends most of its time.
    #
    #   (b) COULOMB collisions with ions, the classical Spitzer form
    #       ~ ln(Lambda) * Z^2 / (A * Te^1.5).
    #
    # Both are accumulated "per unit ni" and converted to a true rate by the
    # common factor on the last line.
    neutral_density_over_amu = sum(n[sym][0, :] / _AMU[sym] for sym in layout.elements)
    equilibration = 4.0e5 * neutral_density_over_amu / ni  # channel (a)

    coulomb_coeff = 2.9e6 * log_lambda_ei(ne, Te, Zeff) / (ni * Te**1.5)

    ion_Z2_over_amu = np.zeros(Nr)
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        charge = np.arange(1, Zmax + 1)[:, None]
        ion_Z2_over_amu += np.sum(n[sym][1:, :] * charge**2, axis=0) / _AMU[sym]

    equilibration += coulomb_coeff * ion_Z2_over_amu  # channel (b)

    # Common conversion to an average ion-electron equilibration rate [1/s].
    equilibration *= 3.0 * ne / (1.0e12 * _MP_OVER_ME)

    # =========================================================================
    # STEP 4 -- Charge-state evolution and radiated power
    # =========================================================================
    dstate_dt = np.zeros(layout.size)
    radiated_power_sum = np.zeros(Nr)  # sum_Z (rad coeff * n_Z) [eV/s]
    ionization_energy_rate = np.zeros(Nr)  # d/dt stored ionization potential
    #                                        [eV cm^-3 / ms]

    # Optical-depth scaling. Assumes Doppler broadening dominates the line
    # width, so the effective column density scales as sqrt(Te/Ti).
    doppler_factor = np.sqrt(Te / Ti)

    # Converts a rate-coefficient product into a per-millisecond density rate:
    #     dn/dt [cm^-3/ms] = (1e-3 * ne) * (S * n)
    # The 1e-3 is the s -> ms conversion; ne appears because every rate
    # coefficient S is per electron.
    ne_per_ms = 1e-3 * ne

    for sym in layout.elements:
        Zmax = layout.Zmax[sym]

        # --- Rate coefficients -----------------------------------------------
        # AuroraRates (ADAS) takes no opacity argument:
        #     S_ion, S_rec, S_rad = rateStruct.all_rates(sym, ne, Te)
        #
        # CretinRates is opacity-aware and needs the per-charge-state column
        # density. IMPORTANT: post-processing must use this same expression
        # when recomputing Prad, or the plotted radiation will not match the
        # energy the solver actually removed.
        column_density = doppler_factor * n[sym] * Rmin * 100.0  # [cm^-2]
        S_ion, S_rec, S_rad = rateStruct.all_rates(sym, ne, Te, column_density)
        # each (Zmax+1, Nr)

        # --- Weight coefficients by population to get fluxes -----------------
        ionization_flux = S_ion * n[sym]  # Z -> Z+1
        recombination_flux = S_rec * n[sym]  # Z -> Z-1
        radiated_power = S_rad * n[sym]  # [eV/s per electron]

        radiated_power_sum += np.sum(radiated_power, axis=0)

        # --- Coupled charge-state ladder -------------------------------------
        # Each state gains from the neighbours flowing in and loses to the
        # neighbours it flows out to:
        #
        #   Z = 0        : + rec[1]               - ion[0]
        #   0 < Z < Zmax : + rec[Z+1] + ion[Z-1]  - rec[Z] - ion[Z]
        #   Z = Zmax     : + ion[Zmax-1]          - rec[Zmax]
        #
        # rec[0] and ion[Zmax] are never referenced: a neutral cannot
        # recombine further and a bare nucleus cannot ionize further.
        net_flux = np.empty((Zmax + 1, Nr))
        net_flux[0] = recombination_flux[1] - ionization_flux[0]
        if Zmax >= 2:
            net_flux[1:Zmax] = (
                recombination_flux[2 : Zmax + 1]
                - ionization_flux[1:Zmax]
                - recombination_flux[1:Zmax]
                + ionization_flux[0 : Zmax - 1]
            )
        net_flux[Zmax] = ionization_flux[Zmax - 1] - recombination_flux[Zmax]

        dn_dt = ne_per_ms[None, :] * net_flux  # [cm^-3/ms]
        layout.species(dstate_dt, sym)[:, :] = dn_dt

        # --- Energy locked into ionization potential -------------------------
        # Ei_cumulative[sym][Z-1] is the total energy to strip Z electrons from
        # a neutral. Moving population up the ladder stores energy; moving it
        # back down releases it. Sums over Z >= 1 only.
        ionization_energy_rate += np.einsum(
            "zr,z->r", dn_dt[1 : Zmax + 1, :], rateStruct.Ei_cumulative[sym]
        )

    # =========================================================================
    # STEP 5 -- Injector particle deposition
    # =========================================================================
    # Te-gated and ablation-driven sources need a 0-D view of the plasma, so
    # build volume-averaged scalars. Per-cell profiles ride along for future
    # ablation models that resolve the deposition radially.
    plasma_state = {
        "Te": float(np.sum(Te * cell_volume) / Vp),
        "Ti": float(np.sum(Ti * cell_volume) / Vp),
        "ne": float(np.sum(ne * cell_volume) / Vp),
        "ni": float(np.sum(ni * cell_volume) / Vp),
        "time": t,
        "Te_prof": Te,
        "ne_prof": ne,
    }

    for injector in injectors:
        # Stateful injectors (pellets) keep their remaining inventory in the
        # aux block of the STATE VECTOR rather than on the object. An internal
        # counter would double-deplete, because BDF evaluates the RHS on trial
        # steps it later rejects.
        if getattr(injector, "n_state", 0):
            aux = layout.aux(state)[injector.aux_slice]
            deposits, daux_dt = injector.ablate(t, plasma_state, aux)
            layout.aux(dstate_dt)[injector.aux_slice] = daux_dt
        else:
            deposits = injector.deliver(t, plasma_state)

        for sym, Ndot in deposits.items():
            if sym not in layout or Ndot == 0.0:
                continue
            # Uniform volumetric deposition: Ndot [1e20 atoms/ms] spread over
            # Vp gives the same neutral-density rate in every cell. The 1e14
            # converts 1e20 atoms/m^3 to cm^-3. Material always arrives as
            # NEUTRALS (row 0) and ionizes through the Step 4 ladder.
            #
            # THIS IS THE HOOK FOR RADIALLY-RESOLVED SPI: replace the uniform
            # 1/Vp with a per-injector deposition profile over cell_volume.
            layout.species(dstate_dt, sym)[0, :] += Ndot * 1e14 / Vp  # [cm^-3/ms]

    # =========================================================================
    # STEP 6 -- Energy balance
    # =========================================================================
    # All terms in [MJ/m^3/ms]. The 1e-3 factors convert per-second rate
    # coefficients to per-millisecond.

    # Line + continuum radiation. Clipped at zero so a negative interpolated
    # coefficient cannot act as a heat source.
    radiation_sink = _EE * ne * np.maximum(radiated_power_sum, 0.0) * 1e-3

    # Energy spent ionizing (positive) or returned by recombination (negative).
    ionization_sink = _EE * ionization_energy_rate

    # Collisional transfer from electrons to ions. Positive when Te > Ti.
    ei_transfer = 1.5e-3 * ne * _EE * equilibration * (Te - Ti)

    # --- Resistivity and the lumped plasma resistance ------------------------
    eta_cell = np.atleast_1d(eta_parallel(ne, Te, Zeff, Rmaj, Rmin))  # (Nr,) [Ohm m]

    # Cells are treated as nested current channels in PARALLEL, each cell's
    # cross-section apportioned by its share of the volume. Exact in the 0-D
    # limit. A current-diffusion model would replace this whole block.
    #
    # Nr = 1: cell_area = Ap and R_plasma = 2*pi*Rmaj*eta/Ap -- the familiar
    # resistance of a torus of cross-section Ap and length 2*pi*Rmaj.
    cell_area = Ap * cell_volume / Vp  # [m^2]
    cell_conductance = cell_area / (2.0 * np.pi * eta_cell * Rmaj)  # [1/Ohm]
    total_conductance = cell_conductance.sum()
    R_plasma = 1.0 / total_conductance  # [Ohm]

    # Conductance-weighted effective resistivity. Equals the MATLAB KPRAD eta
    # scaled by Vp/Vcell.
    eta_effective = R_plasma * Ap / (2.0 * np.pi * Rmaj)  # [Ohm m]

    # L/R current decay time. The 1e-4 folds in the H -> ms conversion and the
    # mu0/(2*pi) geometry factor.
    tau_current_ms = 1e-4 * li * Ap / eta_effective  # [ms]

    # Ohmic heating. The 1e3 converts MA^2 * Ohm = 1e12 W into MJ/ms.
    ohmic_power_total = 1e3 * R_plasma * Ip**2  # [MJ/ms]
    # Distribute by conductance share, then convert to a density.
    ohmic_source = ohmic_power_total * (cell_conductance / total_conductance)
    ohmic_source = ohmic_source / cell_volume  # [MJ/m^3/ms]

    # The unconstrained electron balance, before the floor is applied.
    dWe_dt_raw = ohmic_source - radiation_sink - ionization_sink - ei_transfer

    # Freeze electron cooling once a cell is essentially empty of thermal
    # energy, so radiation cannot drive We negative. _W_FLOOR_MJ is a total
    # energy, so compare against it spread over the plasma volume.
    #
    # NOTE this is a real bookkeeping leak, not a no-op: the power that would
    # have been removed is simply discarded, while the ion side keeps
    # exchanging ei_transfer. It is reported as Pfrozen below rather than
    # hidden, so the energy balance stays honest.
    frozen = We <= _W_FLOOR_MJ / Vp
    dWe_dt = np.where(frozen, 0.0, dWe_dt_raw)

    layout.field(dstate_dt, "We")[:] = dWe_dt
    layout.field(dstate_dt, "Wi")[:] = ei_transfer  # ions only gain from e-i

    # =========================================================================
    # STEP 7 -- Global circuit
    # =========================================================================
    # Two inductively coupled loops: the plasma current and the image current
    # induced in the conducting wall.
    #
    #   dIp/dt = -Ip/tau_current + alphaL * Iw/tau_wall
    #   dIw/dt = -dIp/dt - Iw/tau_wall
    #
    # The second line enforces flux conservation: current lost by the plasma
    # appears in the wall, which then decays on its own L/R time. This is why
    # Ip and Iw are global scalars -- the circuit does not see the radial grid.
    dstate_dt[layout.IP] = -Ip / tau_current_ms + inductance_ratio * Iw / tau_wall_ms
    dstate_dt[layout.IW] = -dstate_dt[layout.IP] - Iw / tau_wall_ms

    if not return_diagnostics:
        return dstate_dt

    # =========================================================================
    # STEP 8 -- Optional diagnostics (not evaluated during integration)
    # =========================================================================
    # Volume-integrate each energy-density channel to a global power
    # [MJ/ms = GW]. By construction:
    #
    #     dWthe_dt = PJ - Prad - Pion - Pei - Pfrozen
    #     dWthi_dt = Pei
    #
    # Pfrozen is the electron power discarded by the We floor above. Ptransp
    # is identically zero in this model and exists so that a future transport
    # term has a slot -- and so the balance check stays valid when one is added.
    def integrate(power_density):
        return float(np.sum(power_density * cell_volume))

    diagnostics = {
        "PJ": float(ohmic_power_total),  # already a total
        "Prad": integrate(radiation_sink),
        "Pion": integrate(ionization_sink),
        "Pei": integrate(ei_transfer),
        "Pfrozen": integrate(np.where(frozen, dWe_dt_raw, 0.0)),
        "Ptransp": 0.0,
        "dWthe_dt": integrate(dWe_dt),
        "dWthi_dt": integrate(ei_transfer),
    }

    return dstate_dt, diagnostics
