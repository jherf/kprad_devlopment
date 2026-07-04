# postprocess.py
"""
Post-processing: physical observables from raw ODE output.

``postprocess1d`` produces a dict with *exactly the same keys and shapes*
as the 0-D ``postprocess`` (volume-integrated energies, volume-averaged
densities/temperatures, per-species (Nt, Zmax+1) charge-state arrays), so
every existing consumer — ``compute_quench_times``, ``compute_radiated_power``,
``compute_injection``, ``plot_main_results`` — works unchanged.

On top of that it carries the radial information as extra keys
(``res["profiles"]`` sub-dict: Vcell, we, wi, ne, Te, dens[sym]) for (t, rho)
diagnostics; at Nr=1 these are just the 0-D traces with a leading axis.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from kprad.util.constants import _MIN_TE, _EE, _W_FLOOR_MJ, OUTPUT_DIR
from kprad.util.layout import SolverLayout


def postprocess(tV: np.ndarray, solY: np.ndarray, layout, Vcell: np.ndarray) -> dict:
    """
    Convert raw 1-D ODE state arrays to physical quantities.

    Parameters
    ----------
    tV     : 1-D array of time points [ms]
    solY   : 2-D array (layout.size, Nt) — ``sol.y`` straight from solve_ivp
    layout : Layout1D
    Vcell  : (Nr,) cell volumes [m^3]

    Returns
    -------
    dict with the 0-D keys
        tV, Wthe, Wthi, Ip, Iw,
        dense, densi, Zeff, Zbar, Te, Ti, densetot,
        dens_<sym>            (Nt, Zmax+1) volume-averaged [cm^-3]
    plus ``profiles``: a sub-dict of radial arrays —
        Vcell (Nr,), we/wi/ne/Te (Nr, Nt), dens[sym] (Zmax+1, Nr, Nt).
    """
    Vcell = np.asarray(Vcell, dtype=float)
    Vp = float(Vcell.sum())
    wgt = Vcell / Vp  # volume weights
    Nr = layout.Nr
    Nt = len(tV)

    Ip = solY[layout.IP]
    Iw = solY[layout.IW]
    We = solY[layout.field_slice("We")]  # (Nr, Nt) [MJ/m^3]
    Wi = solY[layout.field_slice("Wi")]

    # Volume-integrated thermal energies [MJ], same floor as the ODE RHS
    Wthe = np.maximum(np.einsum("rt,r->t", We, Vcell), _W_FLOOR_MJ)
    Wthi = np.maximum(np.einsum("rt,r->t", Wi, Vcell), _W_FLOOR_MJ)

    # Per-species charge-state profiles and volume averages
    dens_prof = {}
    dens_avg = {}
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        block = np.abs(solY[layout.species_slice(sym)].reshape(Zmax + 1, Nr, Nt))
        dens_prof[sym] = block  # (Z+1, Nr, Nt)
        dens_avg[sym] = np.einsum("znt,n->tz", block, wgt)  # (Nt, Z+1)

    # Volume-averaged electron / ion densities, Zeff, Zbar
    dense = np.zeros(Nt)
    A_sum = np.zeros(Nt)
    densi = np.zeros(Nt)
    ne_prof = np.zeros((Nr, Nt))
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        Zs = np.arange(1, Zmax + 1)
        ions = dens_avg[sym][:, 1 : Zmax + 1]
        dense += np.sum(ions * Zs, axis=1)
        A_sum += np.sum(ions * Zs**2, axis=1)
        densi += np.sum(dens_avg[sym], axis=1)
        ne_prof += np.einsum("znt,z->nt", dens_prof[sym][1:, :, :], Zs.astype(float))

    Zeff = A_sum / dense
    Zbar = dense / densi

    # Energy-consistent average temperatures: total thermal energy over
    # total particle inventory. Identical to the 0-D formula at Nr=1
    # (Wthfac = Vp * e).
    Te = np.maximum(Wthe / (1.5 * Vp * _EE * dense), _MIN_TE)
    Ti = np.maximum(Wthi / (1.5 * Vp * _EE * densi), _MIN_TE)

    # Total electron density for avalanche estimates: bound electrons at
    # half weight (charge state iZ of species with Zmax has Zmax - iZ bound)
    densetot = dense.copy()
    for sym in layout.elements:
        Zmax = layout.Zmax[sym]
        for iZ in range(Zmax):
            densetot += 0.5 * dens_avg[sym][:, iZ] * (Zmax - iZ)

    Te_prof = np.maximum(We / (1.5 * _EE * np.maximum(ne_prof, 1.0)), _MIN_TE)

    result = dict(
        tV=tV,
        Wthe=Wthe,
        Wthi=Wthi,
        Ip=Ip,
        Iw=Iw,
        dense=dense,
        densi=densi,
        Zeff=Zeff,
        Zbar=Zbar,
        Te=Te,
        Ti=Ti,
        densetot=densetot,
    )
    for sym in layout.elements:
        result[f"dens_{sym}"] = dens_avg[sym]

    # Radial extras live in their own sub-dict so 0-D consumers that scan
    # top-level keys (e.g. plotting's dens_* heuristic) never see them.
    result["profiles"] = dict(
        Vcell=Vcell,
        We=We,
        Wi=Wi,
        ne=ne_prof,
        Te=Te_prof,
        dens={sym: dens_prof[sym] for sym in layout.elements},
    )

    # Auxiliary (injector) state trace, e.g. remaining pellet inventory
    if getattr(layout, "n_aux", 0):
        result["aux"] = solY[layout.aux_slice()]  # (n_aux, Nt)

    return result


def compute_quench_times(res: dict, Ip0: float, Te0_eV: float, tfirst: float) -> dict:
    """Compute TQ / CQ times and indices marking their boundaries

    Returns a dict with keys:
    i1, i2  CQ 90 % / 1 % Ip indices
    i3      mid-CQ index (50 % Ip)
    i4, i5  TQ 90 % / 1 % Te indices
    tTQ, tCQ  durations [ms]  (nan if undefined)
    """
    tV, Ip, Te = res["tV"], res["Ip"], res["Te"]

    def last_above(arr, thr):
        idx = np.where(arr > thr)[0]
        return int(idx[-1]) if idx.size > 0 else None

    i1 = last_above(Ip, 0.9 * Ip0)
    i2 = last_above(Ip, 0.01 * Ip0)
    i4 = last_above(Te, 0.9 * Te0_eV)
    i5 = last_above(Te, 0.01 * Te0_eV)

    tCQ = (tV[i2] - tV[i1]) if (i1 is not None and i2 is not None) else np.nan
    tTQ = (tV[i5] - tV[i4]) if (i4 is not None and i5 is not None) else np.nan

    i3a = last_above(Ip, 0.6 * Ip0)
    i3b = last_above(Ip, 0.4 * Ip0)
    i3 = int(round((i3a + i3b) / 2)) if (i3a is not None and i3b is not None) else None

    print(f"tTQ = {tTQ:.2f} ms   tCQ = {tCQ:.2f} ms")
    if i5 is not None:
        print(f"end of TQ at t = {tV[i5]:.2f} ms")
        if tfirst is not None:
            print(f"first light -> end of TQ delay = {tV[i5] - tfirst:.2f} ms")

    return dict(i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, tTQ=tTQ, tCQ=tCQ)


def compute_radiated_power(res: dict, params: dict) -> dict:
    """
    Per-species radiated power [GW] at each saved time.

    Uses ``rateStruct.all_rates`` (one batched call per species per
    timestep) and the unit conversion ``ee * Vp * 1e-3`` to take
    eV/(cm^3 s) -> GW for plasma volume Vp.

    The rate lookup uses the *same* per-charge-state column density as the
    ODE RHS (``tau_Z = sqrt(Te/Ti) * n_Z * Rmin * 100``), so the integrated
    radiated energy is consistent with the energy the solver actually
    removed. Backends that ignore opacity (AuroraRates) are unaffected.
    """
    tV = res["tV"]
    Te = res["Te"]
    Ti = res["Ti"]
    dense = res["dense"]
    Vp = params["Vp"]
    Rmin = params["Rmin"]
    rates = params["rateStruct"]
    layout = params["layout"]

    Nt = len(tV)
    Prad = {sym: np.zeros(Nt) for sym in layout.elements}
    for it in range(Nt):
        ne_t = dense[it]
        Te_t = Te[it]
        Tfac = np.sqrt(Te_t / Ti[it])  # Doppler-broadening scaling, as in fkprad
        for sym in layout.elements:
            n_Z = res[f"dens_{sym}"][it]  # (Zmax+1,) [cm^-3]
            tau = Tfac * n_Z * Rmin * 100.0  # column density [1/cm^2]
            # Per-charge-state Ta requires shape (Zmax+1, Nr); ne/Te are
            # scalars here so Nr = 1.
            _, _, rrad = rates.all_rates(sym, ne_t, Te_t, tau[:, None])
            rrad = np.asarray(rrad)
            if rrad.ndim == 2:
                rrad = rrad[:, 0]
            Prad[sym][it] = ne_t * np.sum(n_Z * rrad)

    factor = _EE * Vp * 1e-3
    for sym in Prad:
        Prad[sym] *= factor
    return Prad


def compute_pellet_injection(res: dict, injectors: list) -> dict:
    """
    Delivery rate and cumulative delivery for stateful sources (pellets),
    reconstructed exactly from the saved inventory trace ``res['aux']``:
    what left the pellet is what entered the plasma, so

        cumulative(t) = (N0 - N_remaining(t)) * atoms_per_molecule
        rate(t)       = d(cumulative)/dt

    Complements ``compute_injection`` (which covers state-independent
    sources only). Returns {} if no stateful injectors are present.

    Returns
    -------
    dict ``{plasma_symbol: {'rate' [1e20 atoms/ms], 'cumulative' [1e20]}}``
    """
    aux = res.get("aux")
    if aux is None:
        return {}
    tV = res["tV"]

    out: dict = {}
    for inj in injectors:
        if getattr(inj, "n_state", 0) == 0:
            continue
        if getattr(inj, "aux_slice", None) is None:
            continue
        hist = inj.injection_history(aux[inj.aux_slice], tV)
        for sym, d in hist.items():
            if sym in out:
                out[sym]["rate"] = out[sym]["rate"] + d["rate"]
                out[sym]["cumulative"] = out[sym]["cumulative"] + d["cumulative"]
            else:
                out[sym] = d
    return out


def compute_energy_balance(
    res: dict, solY: np.ndarray, injectors: list, params: dict, verbose: bool = True
) -> dict:
    """
    Global energy-balance diagnostic.

    Re-invokes the ODE RHS with ``return_diagnostics=True`` at every saved
    time point, so every power channel is evaluated from *exactly* the
    expressions the solver integrated (same rates, same opacity, same
    resistivity) — the balance is consistent with the physics by
    construction, and any residual measures only (a) trapezoidal
    reconstruction error on the output grid and (b) genuine RHS bugs.

    Identities checked (all energies in MJ, powers in GW = MJ/ms):

        dWthe(t) = E_J - E_rad - E_ion - E_ei - E_freeze     (electrons)
        dWthi(t) = E_ei                                       (ions)
        dWth(t)  = E_J - E_rad - E_ion - E_freeze             (total)

    ``E_freeze`` is the electron-channel energy blocked by the We floor —
    the RHS's known bookkeeping leak (ions keep exchanging Pei while the
    electron side is frozen). It is reported, not hidden. Injected
    neutrals carry no thermal energy in this model, so injection appears
    as dilution (temperature drop at fixed W), not as a balance term.

    Parameters
    ----------
    res       : dict from ``postprocess`` (uses tV, Wthe, Wthi)
    solY      : (layout.size, Nt) raw state array, ``sol.y`` from solve_ivp
    injectors : the injector list passed to the solver
    params    : the params dict passed to the solver
    verbose   : print a closing summary table. Default True.

    Returns
    -------
    dict with
        tV                : (Nt,) [ms]
        power             : {PJ, Prad, Pion, Pei, Pfrozen, dWthe_dt,
                             dWthi_dt} time traces [GW]
        E_J, E_rad, E_ion, E_ei, E_freeze : cumulative integrals [MJ]
        dWthe, dWthi      : (Nt,) stored-energy changes from res [MJ]
        resid_e, resid_i, resid_tot : (Nt,) balance residuals [MJ]
        Wth0              : initial total thermal energy [MJ]
        resid_rel         : max |resid_tot| / Wth0  (scalar figure of merit)
    """
    from scipy.integrate import cumulative_trapezoid
    from kprad.util.solver import fkprad

    tV = np.asarray(res["tV"], dtype=float)
    Nt = len(tV)
    keys = ("PJ", "Prad", "Pion", "Pei", "Pfrozen", "Ptransp", "dWthe_dt", "dWthi_dt")
    power = {k: np.zeros(Nt) for k in keys}
    for it in range(Nt):
        _, diag = fkprad(float(tV[it]), solY[:, it], injectors, params)
        for k in keys:
            power[k][it] = diag[k]

    def _cum(k):
        return cumulative_trapezoid(power[k], tV, initial=0.0)

    E_J = _cum("PJ")
    E_rad = _cum("Prad")
    E_ion = _cum("Pion")
    E_ei = _cum("Pei")
    E_freeze = _cum("Pfrozen")

    dWthe = res["Wthe"] - res["Wthe"][0]
    dWthi = res["Wthi"] - res["Wthi"][0]
    Wth0 = float(res["Wthe"][0] + res["Wthi"][0])

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
        tr_check = float(np.max(np.abs(power["Ptransp"])))
        tr_line = (
            f"    transport net power (should be ~0): "
            f"max |Ptransp| = {tr_check:.2e} GW\n"
            if tr_check > 0.0
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
            f"{tr_line}"
            f"    residual: {resid_tot[f]:+.3e} MJ final, "
            f"{np.max(np.abs(resid_tot)):.3e} MJ max "
            f"({100*resid_rel:.3f}% of Wth0)"
        )

    return out


def compute_injection(injectors: list, tV: np.ndarray) -> dict:
    """
    Per-species injection rate [1e20 atoms/ms] and cumulative delivery
    [1e20 atoms] vs time, for state-independent sources (SPI pellets).

    ``WallSputter`` and any source whose ``deliver`` requires a plasma
    ``state`` will simply return ``{}`` when called without one — they are
    excluded from this calculation, which is purely pre-simulation geometry.

    Parameters
    ----------
    injectors : list of Injector instances
    tV        : 1-D array of time values [ms] from the ODE solver output

    Returns
    -------
    dict ``{symbol: {'rate': ndarray, 'cumulative': ndarray}}``
    where both arrays have the same length as ``tV``.
    """
    from scipy.integrate import cumulative_trapezoid

    rates = {}
    for it, t in enumerate(tV):
        for inj in injectors:
            for sym, ndot in inj.deliver(float(t)).items():
                if sym not in rates:
                    rates[sym] = np.zeros(len(tV))
                rates[sym][it] += float(ndot)

    if not rates:
        return {}

    return {
        sym: {
            "rate": rate,
            "cumulative": cumulative_trapezoid(rate, tV, initial=0.0),
        }
        for sym, rate in rates.items()
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
