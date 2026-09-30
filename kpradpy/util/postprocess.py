# postprocess.py
"""
Post-processing: physical observables from the raw ODE output.

READING THIS FILE
-----------------
``postprocess`` converts the flat state array ``sol.y`` into named physical
quantities. It returns volume-INTEGRATED energies, volume-AVERAGED densities
and temperatures, and per-species charge-state arrays.

The returned key names (``dense``, ``densi``, ``Wthe``, ...) are the PUBLIC
API of this module -- ``plotting.py``, ``main_script.py`` and the MATLAB
comparison script all index them by name, and the HDF5 files on disk use
them as dataset names. They are deliberately left unchanged even where the
names are poor, so old result files stay readable. Internal variables have
been renamed for clarity; the dict keys have not.

Radial information rides along under ``res["profiles"]`` so that 0-D
consumers scanning top-level keys never trip over it. At Nr = 1 the profile
arrays are just the 0-D traces with a leading axis of length 1.

A NOTE ON AVERAGING
-------------------
Two different averages appear here and they are not interchangeable:

  * Densities are VOLUME-AVERAGED:      <n> = sum_r n_r * Vcell_r / Vp
  * Temperatures are ENERGY-CONSISTENT: <Te> = Wthe_total / (1.5 * e * <ne> * Vp)

The second is not the volume average of Te. It is the temperature a uniform
plasma would need to hold the same total thermal energy at the same average
density. At Nr = 1 the two coincide exactly; for Nr > 1 they do not, and the
energy-consistent form is the one that makes the energy balance close.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from kpradpy.util.constants import _MIN_NE, _MIN_TE, _EE, _W_FLOOR_MJ
from kpradpy.util.layout import SolverLayout
from kpradpy.globals import OUTPUT_DIR


def postprocess(tV: np.ndarray, solY: np.ndarray, layout, Vcell: np.ndarray) -> dict:
    """Convert raw ODE state arrays into physical quantities.

    Parameters
    ----------
    tV     : (Nt,) time points [ms]
    solY   : (layout.size, Nt) -- ``sol.y`` straight from solve_ivp
    layout : SolverLayout
    Vcell  : (Nr,) cell volumes [m^3]

    Returns
    -------
    dict with the 0-D keys
        tV, Wthe, Wthi, Ip, Iw,
        dense, densi, Zeff, Zbar, Te, Ti, densetot,
        dens_<sym>   (Nt, Zmax+1) volume-averaged [cm^-3]
    plus ``profiles``: Vcell (Nr,), We/Wi/ne/Te (Nr, Nt),
    dens[sym] (Zmax+1, Nr, Nt).
    """
    Vcell = np.asarray(Vcell, dtype=float)
    Vp = float(Vcell.sum())  # [m^3]
    volume_weight = Vcell / Vp  # (Nr,) sums to 1
    Nr = layout.Nr
    Nt = len(tV)

    # ---- Global circuit traces (no radial structure) ------------------------
    Ip = solY[layout.IP]  # (Nt,) [MA]
    Iw = solY[layout.IW]  # (Nt,) [MA]

    # ---- Thermal energy densities -> volume-integrated energies -------------
    We = solY[layout.field_slice("We")]  # (Nr, Nt) [MJ/m^3]
    Wi = solY[layout.field_slice("Wi")]  # (Nr, Nt) [MJ/m^3]

    # Integrate over the grid: MJ/m^3 * m^3 -> MJ. Floored with the same
    # constant the RHS uses, so traces here match the integrated physics.
    Wthe = np.maximum(np.einsum("rt,r->t", We, Vcell), _W_FLOOR_MJ)  # (Nt,) [MJ]
    Wthi = np.maximum(np.einsum("rt,r->t", Wi, Vcell), _W_FLOOR_MJ)  # (Nt,) [MJ]

    # ---- Per-species charge-state populations -------------------------------
    # abs() guards against small negative excursions the stiff solver can
    # leave in nearly-empty charge states; they are numerical, not physical.
    density_profile = {}  # sym -> (Zmax+1, Nr, Nt) [cm^-3]
    density_avg = {}  # sym -> (Nt, Zmax+1) volume-averaged [cm^-3]
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        charge_states = np.abs(solY[layout.species_slice(sym)].reshape(Zmax + 1, Nr, Nt))
        density_profile[sym] = charge_states
        # Contract the radial axis against the volume weights, and transpose
        # to (time, charge) which is what every downstream plot expects.
        density_avg[sym] = np.einsum("znt,n->tz", charge_states, volume_weight)

    # ---- Electron / ion densities, Zeff, Zbar -------------------------------
    # Same definitions as the RHS (see solver.py Step 2), applied to the
    # volume-averaged populations:
    #     ne   = sum_sym sum_{Z>=1} Z   * n_Z
    #     ni   = sum_sym sum_{Z>=0}       n_Z
    #     Zeff = sum Z^2 n_Z / ne        Zbar = ne / ni
    ne_avg = np.zeros(Nt)  # [cm^-3]
    ni_avg = np.zeros(Nt)  # [cm^-3]
    sum_ni_Z2 = np.zeros(Nt)  # Zeff numerator [cm^-3]
    ne_profile = np.zeros((Nr, Nt))  # [cm^-3], radially resolved

    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        charge = np.arange(1, Zmax + 1)  # Z = 1..Zmax
        ionized = density_avg[sym][:, 1 : Zmax + 1]  # (Nt, Zmax)

        ne_avg += np.sum(ionized * charge, axis=1)
        sum_ni_Z2 += np.sum(ionized * charge**2, axis=1)
        ni_avg += np.sum(density_avg[sym], axis=1)  # neutrals included
        ne_profile += np.einsum(
            "znt,z->nt", density_profile[sym][1:, :, :], charge.astype(float)
        )

    # Floors matching solver.py. Without them a fully-recombined sample
    # divides by zero and puts inf/nan into the saved results.
    ne_avg = np.maximum(ne_avg, _MIN_NE)
    ni_avg = np.maximum(ni_avg, _MIN_NE)

    Zeff = sum_ni_Z2 / ne_avg  # effective charge, >= 1
    Zbar = ne_avg / ni_avg  # mean charge per nucleus

    # ---- Energy-consistent average temperatures -----------------------------
    # Invert Wth = 1.5 * e * Vp * n * T. See the module docstring on why this
    # is not the volume average of the temperature profile.
    Te = np.maximum(Wthe / (1.5 * Vp * _EE * ne_avg), _MIN_TE)  # (Nt,) [eV]
    Ti = np.maximum(Wthi / (1.5 * Vp * _EE * ni_avg), _MIN_TE)  # (Nt,) [eV]

    # ---- Total electron inventory for avalanche estimates -------------------
    # Runaway avalanche growth depends on ALL electrons, bound ones included,
    # because a runaway can knock an electron out of a partially-stripped ion.
    # A species with Zmax nuclear charge sitting in charge state iZ still
    # holds (Zmax - iZ) bound electrons. These are counted at HALF weight, the
    # usual approximation for their reduced availability at high energy.
    densetot = ne_avg.copy()
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        for iZ in range(Zmax):  # excludes the fully-stripped state
            bound_electrons = Zmax - iZ
            densetot += 0.5 * density_avg[sym][:, iZ] * bound_electrons

    # ---- Radial temperature profile -----------------------------------------
    # Cell-local inversion of We = 1.5 * e * ne * Te.
    Te_profile = np.maximum(We / (1.5 * _EE * np.maximum(ne_profile, 1.0)), _MIN_TE)

    # ---- Assemble. These key names are the public API; do not rename. -------
    result = dict(
        tV=tV,
        Wthe=Wthe,
        Wthi=Wthi,
        Ip=Ip,
        Iw=Iw,
        dense=ne_avg,
        densi=ni_avg,
        Zeff=Zeff,
        Zbar=Zbar,
        Te=Te,
        Ti=Ti,
        densetot=densetot,
    )
    for sym in layout.elements:
        result[f"dens_{sym}"] = density_avg[sym]

    # Radial extras live in their own sub-dict so 0-D consumers that scan
    # top-level keys (e.g. plotting's dens_* heuristic) never see them.
    result["profiles"] = dict(
        Vcell=Vcell,
        We=We,
        Wi=Wi,
        ne=ne_profile,
        Te=Te_profile,
        dens={sym: density_profile[sym] for sym in layout.elements},
    )

    # Auxiliary injector state, e.g. remaining pellet inventory vs time.
    if getattr(layout, "n_aux", 0):
        result["aux"] = solY[layout.aux_slice()]  # (n_aux, Nt)

    return result


def compute_quench_times(res: dict, Ip0: float, Te0_eV: float, tfirst: float) -> dict:
    """Locate the thermal-quench and current-quench boundaries.

    Both quenches are measured by the conventional 90 % -> 1 % crossings:
    the CQ on the plasma current, the TQ on the electron temperature.

    Because both signals fall monotonically once the quench starts, each
    boundary is found as the LAST index still above the threshold. That is
    more robust than the first crossing, which noise or a pre-quench wiggle
    can trigger early.

    Returns
    -------
    dict with
        i1, i2  : CQ start / end   (Ip at 90 % / 1 % of Ip0)
        i3      : mid-CQ           (Ip at ~50 % of Ip0)
        i4, i5  : TQ start / end   (Te at 90 % / 1 % of Te0)
        tTQ, tCQ: durations [ms], nan if the threshold is never crossed
    """
    tV, Ip, Te = res["tV"], res["Ip"], res["Te"]

    def last_above(trace, threshold):
        """Index of the last sample still above ``threshold``, or None."""
        above = np.where(trace > threshold)[0]
        return int(above[-1]) if above.size > 0 else None

    i1 = last_above(Ip, 0.90 * Ip0)  # CQ start
    i2 = last_above(Ip, 0.01 * Ip0)  # CQ end
    i4 = last_above(Te, 0.90 * Te0_eV)  # TQ start
    i5 = last_above(Te, 0.01 * Te0_eV)  # TQ end

    tCQ = (tV[i2] - tV[i1]) if (i1 is not None and i2 is not None) else np.nan
    tTQ = (tV[i5] - tV[i4]) if (i4 is not None and i5 is not None) else np.nan

    # Mid-CQ is bracketed by the 60 % and 40 % crossings and averaged, rather
    # than taken at a single 50 % crossing. On a coarse output grid the 50 %
    # sample can land well off the midpoint; bracketing halves that error.
    i_60 = last_above(Ip, 0.60 * Ip0)
    i_40 = last_above(Ip, 0.40 * Ip0)
    i3 = int(round((i_60 + i_40) / 2)) if (i_60 is not None and i_40 is not None) else None

    print(f"tTQ = {tTQ:.2f} ms   tCQ = {tCQ:.2f} ms")
    if i5 is not None:
        print(f"end of TQ at t = {tV[i5]:.2f} ms")
        if tfirst is not None:
            print(f"first light -> end of TQ delay = {tV[i5] - tfirst:.2f} ms")

    return dict(i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, tTQ=tTQ, tCQ=tCQ)


def compute_radiated_power(res: dict, params: dict) -> dict:
    """Per-species radiated power [GW] at each saved time.

    CONSISTENCY REQUIREMENT: the rate lookup must use the SAME per-charge-state
    column density as the ODE right-hand side,

        column_density_Z = sqrt(Te/Ti) * n_Z * Rmin * 100     [cm^-2]

    If a fixed column density is substituted here instead, the radiated power
    plotted will not be the energy the solver actually removed, and it cannot
    be checked against the drop in Wthe. Opacity-free backends (AuroraRates)
    ignore the argument and are unaffected either way.

    Unit conversion: rate coefficients come back in eV/(cm^3 s). Multiplying
    by e [J/eV] gives J/(cm^3 s); by Vp [m^3] and 1e-3 gives GW, since the
    cm^-3 density and m^3 volume combine with the J -> GJ factor.
    """
    tV = res["tV"]
    Te = res["Te"]
    Ti = res["Ti"]
    ne = res["dense"]
    Vp = params["Vp"]
    Rmin = params["Rmin"]
    rates = params["rateStruct"]
    layout = params["layout"]

    Nt = len(tV)
    Prad = {sym: np.zeros(Nt) for sym in layout.elements}

    for it in range(Nt):
        ne_t = ne[it]
        Te_t = Te[it]
        doppler_factor = np.sqrt(Te_t / Ti[it])  # as in solver.py Step 4

        for sym in layout.elements:
            n_Z = res[f"dens_{sym}"][it]  # (Zmax+1,) [cm^-3]
            column_density = doppler_factor * n_Z * Rmin * 100.0  # [cm^-2]

            # The backend wants shape (Zmax+1, Nr). ne and Te are scalars
            # here, so Nr = 1 and the column density needs a trailing axis.
            _, _, S_rad = rates.all_rates(sym, ne_t, Te_t, column_density[:, None])
            S_rad = np.asarray(S_rad)
            if S_rad.ndim == 2:
                S_rad = S_rad[:, 0]

            Prad[sym][it] = ne_t * np.sum(n_Z * S_rad)

    eV_cm3_s_to_GW = _EE * Vp * 1e-3
    for sym in Prad:
        Prad[sym] *= eV_cm3_s_to_GW

    return Prad


def compute_pellet_injection(res: dict, injectors: list) -> dict:
    """Delivery rate and cumulative delivery for STATEFUL sources (pellets).

    Reconstructed from the saved inventory trace ``res['aux']`` rather than
    recomputed: what left the pellet is what entered the plasma, so

        cumulative(t) = (N0 - N_remaining(t)) * atoms_per_molecule
        rate(t)       = d(cumulative)/dt

    This is exact and cannot drift from what the solver did, which a
    re-integration of the ablation model could.

    Complements ``compute_injection``, which covers state-INDEPENDENT sources.
    Returns {} when no stateful injectors are present.

    Returns
    -------
    dict ``{plasma_symbol: {'rate' [1e20 atoms/ms], 'cumulative' [1e20]}}``
    """
    aux = res.get("aux")
    if aux is None:
        return {}

    tV = res["tV"]
    out: dict = {}

    for injector in injectors:
        if getattr(injector, "n_state", 0) == 0:
            continue
        if getattr(injector, "aux_slice", None) is None:
            continue

        history = injector.injection_history(aux[injector.aux_slice], tV)
        # Several injectors may feed the same plasma species (e.g. two SPI
        # barrels of D2), so accumulate rather than overwrite.
        for sym, trace in history.items():
            if sym in out:
                out[sym]["rate"] = out[sym]["rate"] + trace["rate"]
                out[sym]["cumulative"] = out[sym]["cumulative"] + trace["cumulative"]
            else:
                out[sym] = trace

    return out


def compute_energy_balance(
    res: dict, solY: np.ndarray, injectors: list, params: dict, verbose: bool = True
) -> dict:
    """Global energy-balance diagnostic -- the main validation check.

    HOW IT WORKS
    ------------
    The RHS is re-invoked with ``return_diagnostics=True`` at every saved time
    point, so each power channel comes from exactly the expression the solver
    integrated: same rate tables, same opacity, same resistivity. The balance
    is therefore consistent with the physics BY CONSTRUCTION, and a non-zero
    residual can only mean one of two things:

        (a) trapezoidal reconstruction error on the output grid -- shrink it
            by saving more time points, not by changing the physics; or
        (b) a genuine bug in the RHS.

    That is what makes this worth running on a literature benchmark: it
    separates "the model disagrees with the paper" from "the code disagrees
    with itself".

    IDENTITIES CHECKED (energies in MJ, powers in GW = MJ/ms)
    ---------------------------------------------------------
        dWthe = E_J - E_rad - E_ion - E_ei - E_freeze     (electrons)
        dWthi = E_ei                                      (ions)
        dWth  = E_J - E_rad - E_ion - E_freeze            (total; E_ei cancels)

    ``E_freeze`` is electron-channel energy blocked by the We floor in the RHS
    -- a known bookkeeping leak, since the ion side keeps exchanging Pei while
    the electron side is frozen. It is reported, not hidden. A large E_freeze
    means the floor is doing real work and the result should be treated with
    suspicion.

    Injected neutrals carry no thermal energy in this model, so injection
    shows up as DILUTION (temperature falls at fixed W), not as a balance term.

    Parameters
    ----------
    res       : dict from ``postprocess`` (uses tV, Wthe, Wthi)
    solY      : (layout.size, Nt) raw state array from solve_ivp
    injectors : the injector list passed to the solver
    params    : the params dict passed to the solver
    verbose   : print a closing summary table

    Returns
    -------
    dict with tV, power (time traces), the cumulative integrals E_*,
    dWthe/dWthi, the residuals, Wth0, and ``resid_rel`` -- the single scalar
    figure of merit, max |residual| / initial thermal energy.
    """
    from scipy.integrate import cumulative_trapezoid
    from kpradpy.util.solver import fkprad

    tV = np.asarray(res["tV"], dtype=float)
    Nt = len(tV)

    channels = ("PJ", "Prad", "Pion", "Pei", "Pfrozen", "Ptransp", "dWthe_dt", "dWthi_dt")
    power = {name: np.zeros(Nt) for name in channels}

    for it in range(Nt):
        _, diagnostics = fkprad(
            float(tV[it]), solY[:, it], injectors, params, return_diagnostics=True
        )
        for name in channels:
            power[name][it] = diagnostics[name]

    def cumulative(name):
        """Time-integrate a power trace [GW] to a cumulative energy [MJ]."""
        return cumulative_trapezoid(power[name], tV, initial=0.0)

    E_J = cumulative("PJ")  # ohmic input
    E_rad = cumulative("Prad")  # radiated away
    E_ion = cumulative("Pion")  # stored as ionization potential
    E_ei = cumulative("Pei")  # electrons -> ions (internal transfer)
    E_freeze = cumulative("Pfrozen")  # discarded by the We floor

    # Stored-energy changes measured independently, from the state vector.
    dWthe = res["Wthe"] - res["Wthe"][0]
    dWthi = res["Wthi"] - res["Wthi"][0]
    Wth0 = float(res["Wthe"][0] + res["Wthi"][0])

    # Residuals: measured change minus the change the power channels imply.
    resid_e = dWthe - (E_J - E_rad - E_ion - E_ei - E_freeze)
    resid_i = dWthi - E_ei
    resid_tot = resid_e + resid_i
    resid_rel = float(np.max(np.abs(resid_tot)) / max(Wth0, 1e-30))

    out = dict(
        tV=tV,
        power=power,
        E_J=E_J,
        E_rad=E_rad,
        E_ion=E_ion,
        E_ei=E_ei,
        E_freeze=E_freeze,
        dWthe=dWthe,
        dWthi=dWthi,
        resid_e=resid_e,
        resid_i=resid_i,
        resid_tot=resid_tot,
        Wth0=Wth0,
        resid_rel=resid_rel,
    )

    if verbose:
        f = -1  # final index
        transport_check = float(np.max(np.abs(power["Ptransp"])))
        transport_line = (
            f"    transport net power (should be ~0): "
            f"max |Ptransp| = {transport_check:.2e} GW\n"
            if transport_check > 0.0
            else ""
        )
        print(
            "Energy balance [MJ]  (dWth = E_J - E_rad - E_ion - E_freeze):\n"
            f"    Wth0        = {Wth0:9.4f}\n"
            f"  + Joule       = {E_J[f]:+9.4f}\n"
            f"  - Radiated    = {-E_rad[f]:+9.4f}\n"
            f"  - Ionization  = {-E_ion[f]:+9.4f}\n"
            f"  - Frozen-We   = {-E_freeze[f]:+9.4f}\n"
            f"  = Sum         = {E_J[f]-E_rad[f]-E_ion[f]-E_freeze[f]:+9.4f}"
            f"   vs  dWth = {dWthe[f]+dWthi[f]:+9.4f}\n"
            f"    e->i transfer E_ei = {E_ei[f]:+9.4f} (internal)\n"
            f"{transport_line}"
            f"    residual: {resid_tot[f]:+.3e} MJ final, "
            f"{np.max(np.abs(resid_tot)):.3e} MJ max "
            f"({100*resid_rel:.3f}% of Wth0)"
        )

    return out


def compute_injection(injectors: list, tV: np.ndarray) -> dict:
    """Injection rate and cumulative delivery for STATE-INDEPENDENT sources.

    These are sources whose delivery is a prescribed function of time alone
    (MGI valve profiles, SPI flight-time distributions), so the whole trace
    can be evaluated without ever touching the plasma solution.

    ``WallSputter`` and any other source gated on plasma conditions returns
    {} when called without a state, and so drops out of this calculation --
    use ``compute_pellet_injection`` for stateful sources.

    Returns
    -------
    dict ``{symbol: {'rate' [1e20 atoms/ms], 'cumulative' [1e20 atoms]}}``,
    both arrays the same length as ``tV``.
    """
    from scipy.integrate import cumulative_trapezoid

    rate_by_symbol = {}
    for it, t in enumerate(tV):
        for injector in injectors:
            for sym, Ndot in injector.deliver(float(t)).items():
                if sym not in rate_by_symbol:
                    rate_by_symbol[sym] = np.zeros(len(tV))
                rate_by_symbol[sym][it] += float(Ndot)

    if not rate_by_symbol:
        return {}

    return {
        sym: {
            "rate": rate,
            "cumulative": cumulative_trapezoid(rate, tV, initial=0.0),
        }
        for sym, rate in rate_by_symbol.items()
    }


# ---------------------------------------------------------------------------
# ---- HDF5 output
# ---------------------------------------------------------------------------
def _sanitize(obj):
    """Coerce a result value into something h5py can store, or return None
    to skip it. Handles scalars, arrays, None, bool, and str."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return np.int8(1 if obj else 0)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return obj
    arr = np.asarray(obj)
    # Only numeric arrays are datasets; object/ragged arrays are skipped.
    if arr.dtype == object or arr.ndim == 0 and arr.dtype.kind not in "fiub":
        return None
    if arr.dtype.kind in "fiub":
        return arr
    return None


def _write_group(grp, mapping):
    """Recursively write a (possibly nested) dict into an h5py group.

    Nested dicts become sub-groups; None values are recorded as an empty
    attribute so the key's absence is distinguishable from a genuine zero.
    Non-serializable values (e.g. an Equilibrium object under a private key)
    are silently skipped.
    """
    for key, val in mapping.items():
        name = str(key)
        if key.startswith("_"):  # private stashes (e.g. _equilibrium)
            continue
        if isinstance(val, dict):
            _write_group(grp.create_group(name), val)
            continue
        clean = _sanitize(val)
        if clean is None:
            grp.attrs[name] = "None"  # marker for a present-but-null field
            continue
        if isinstance(clean, str):
            grp.attrs[name] = clean
        elif np.ndim(clean) == 0:
            grp.attrs[name] = clean  # scalar -> attribute
        else:
            grp.create_dataset(name, data=clean, compression="gzip")


def save_results_h5(
    res: dict,
    path: str | Path | None = None,
    *,
    config: dict | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> str:
    """Save a post-processed ``res`` dict to an HDF5 file.

    Layout mirrors the dict: top-level arrays/scalars become datasets or
    attributes of the root group, and nested dicts (``profiles``,
    ``energy_balance``, ``Prad``, ``injection``, …) become sub-groups. The
    YAML ``config`` (if given) is stored as a string under the ``/config``
    attribute for provenance.

    Path resolution (first hit wins):
        1. explicit ``path``
        2. ``output_dir`` argument
        3. ``config['simulation']['output_dir']``
        4. ``constants.OUTPUT_DIR``  (``$KPRAD_OUTPUT_DIR`` or ~/kprad_output)
    with ``filename`` defaulting to ``kprad_YYYYmmdd_HHMMSS.h5``. The
    directory is created if it does not exist.

    Returns the absolute path written.
    """
    import h5py

    if path is not None:
        out = Path(path)
    else:
        base = (
            output_dir
            or (config or {}).get("simulation", {}).get("output_dir")
            or OUTPUT_DIR
        )
        if filename is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kprad_{stamp}.h5"
        out = Path(base) / filename

    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["format"] = "kprad-results-1"
        if config is not None:
            try:
                import yaml

                f.attrs["config"] = yaml.safe_dump(
                    {k: v for k, v in config.items() if not str(k).startswith("_")},
                    default_flow_style=False,
                    sort_keys=False,
                )
            except Exception:
                pass  # provenance is best-effort; never block the save
        _write_group(f, res)

    print(f"→ Saved results to {out}")
    return str(out.resolve())


def load_results_h5(path: str | Path) -> dict:
    """Load a file written by ``save_results_h5`` back into a nested dict.

    Inverse of the writer: sub-groups become nested dicts, datasets become
    arrays, attributes become scalars/strings, and the ``"None"`` marker is
    mapped back to Python ``None``.
    """
    import h5py

    def _read_group(grp):
        out = {}
        for k, v in grp.attrs.items():
            if k in ("created", "format", "config"):
                continue
            out[k] = None if isinstance(v, str) and v == "None" else v
        for k, item in grp.items():
            if isinstance(item, h5py.Group):
                out[k] = _read_group(item)
            else:
                out[k] = item[()]
        return out

    with h5py.File(Path(path), "r") as f:
        res = _read_group(f)
        for meta in ("created", "format", "config"):
            if meta in f.attrs:
                res.setdefault("_meta", {})[meta] = f.attrs[meta]
    return res
