# profile.py
"""
1-D electron density and temperature profiles → 0-D initial conditions.

Reads tabulated ne(ρ), Te(ρ) from an HDF5 file and derives the volume-
averaged quantities the simulator needs.

Volume averaging
----------------
Uses the circular-torus approximation  dV ∝ 2ρ dρ  (i.e. ρ = r/a for a
circular cross-section).  Under the mapping  ρ_pol = √ψ_n  this is exactly
equivalent to assuming a uniform ψ_n ↔ volume mapping, so the gfile pressure
integral and the profile integrals use the same volume element and the Ti
split is self-consistent.

For a more accurate result on highly non-circular equilibria, supply an
Equilibrium object (the full flux-surface volume profile is not yet
implemented but the API is reserved).

Supported ne_units
------------------
  '1e13_cm3'   1*10¹³ cm⁻³   (simulator native; no conversion)
  '1e19_m3'    1*10¹⁹ m⁻³   (= 1*10¹³ cm⁻³, no conversion)
  '1e20_m3'    1*10²⁰ m⁻³   (*10 to reach 1*10¹³)
  'm3'         m⁻³ absolute  (÷ 1*10¹⁹)
  'cm3'        cm⁻³ absolute (÷ 1*10¹³)

Supported Te_units
------------------
  'eV'   electron-volts
  'keV'  kilo-electron-volts (*1000)
  'J'    joules (÷ 1.6*10⁻¹⁹)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid as _trapz

_EV = 1.6e-19  # J per eV


# ── Data container ────────────────────────────────────────────────────────────


@dataclass
class PlasmaProfiles:
    """
    Tabulated 1-D electron profiles on a ρ = [0, 1] grid.

    Attributes
    ----------
    rho  : 1-D array sorted in ascending order and clipped to [0, 1].
    ne   : electron density [1*10¹³ cm⁻³]
    Te   : electron temperature [eV]
    Zeff : optional effective charge
    source : file path or other provenance string
    """

    rho: np.ndarray
    ne: np.ndarray  # [1e13 cm^-3]
    Te: np.ndarray  # [eV]
    Zeff: Optional[np.ndarray] = None
    source: Optional[str] = None

    def __post_init__(self):
        # Sort by rho and clip domain to [0, 1]
        idx = np.argsort(self.rho)
        self.rho = np.clip(self.rho[idx], 0.0, 1.0)
        self.ne = np.maximum(self.ne[idx], 0.0)
        self.Te = np.maximum(self.Te[idx], 0.0)
        if self.Zeff is not None:
            self.Zeff = self.Zeff[idx]


# ── HDF5 reader ───────────────────────────────────────────────────────────────


def read_profiles_h5(
    path: str | Path,
    *,
    rho_key: str = "rho",
    ne_key: str = "ne",
    Te_key: str = "Te",
    Zeff_key: Optional[str] = None,
    ne_units: str = "1e19_m3",
    Te_units: str = "eV",
    time_idx: int = -1,
) -> PlasmaProfiles:
    """
    Read 1-D profiles from an HDF5 file.

    Parameters
    ----------
    path      : path to the HDF5 file.
    rho_key   : dataset name for the ρ array.  Default ``'rho'``.
    ne_key    : dataset name for ne.  Default ``'ne'``.
    Te_key    : dataset name for Te.  Default ``'Te'``.
    Zeff_key  : dataset name for Zeff (optional).
    ne_units  : unit string for ne  (see module docstring).
    Te_units  : unit string for Te.
    time_idx  : if a dataset is 2-D (time, rho), select this row.
                Default ``-1`` = last time point.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        rho = np.asarray(f[rho_key]).ravel()
        ne_raw = np.asarray(f[ne_key])
        Te_raw = np.asarray(f[Te_key])
        Zeff_raw = np.asarray(f[Zeff_key]) if Zeff_key and Zeff_key in f else None

    # Select time slice if 2-D
    def _pick(arr):
        return arr[time_idx, :] if arr.ndim == 2 else arr.ravel()

    ne_raw, Te_raw = _pick(ne_raw), _pick(Te_raw)
    if Zeff_raw is not None:
        Zeff_raw = _pick(Zeff_raw)

    return PlasmaProfiles(
        rho=rho,
        ne=_ne_to_1e13(ne_raw, ne_units),
        Te=_Te_to_eV(Te_raw, Te_units),
        Zeff=Zeff_raw,
        source=str(path),
    )


# ── Unit converters ───────────────────────────────────────────────────────────


def _ne_to_1e13(values, units: str) -> np.ndarray:
    """Convert any ne unit string to 1*10¹³ cm⁻³."""
    u = units.strip().lower().replace("^", "").replace("-", "").replace(" ", "")
    if u in ("1e13cm3", "1e13_cm3", "native"):
        return np.asarray(values, float)
    if u in ("1e19m3", "1e19_m3"):
        return np.asarray(values, float)  # 1e19 m^-3 ≡ 1e13 cm^-3
    if u in ("1e20m3", "1e20_m3"):
        return np.asarray(values, float) * 10.0
    if u in ("m3", "si", "/m3"):
        return np.asarray(values, float) * 1e-19  # 1 m^-3 → 1e-19 * 1e13 cm^-3
    if u in ("cm3", "/cm3"):
        return np.asarray(values, float) * 1e-13
    raise ValueError(
        f"Unknown ne_units {units!r}.  " "Accepted: 1e13_cm3, 1e19_m3, 1e20_m3, m3, cm3"
    )


def _Te_to_eV(values, units: str) -> np.ndarray:
    """Convert any Te unit string to eV."""
    u = units.strip().lower()
    if u in ("ev", "electron_volt", "electronvolt"):
        return np.asarray(values, float)
    if u in ("kev",):
        return np.asarray(values, float) * 1e3
    if u in ("j", "joule"):
        return np.asarray(values, float) / _EV
    raise ValueError(f"Unknown Te_units {units!r}.  Accepted: eV, keV, J")


# ── Derived initial conditions ────────────────────────────────────────────────


def derive_initial_conditions(
    profiles: PlasmaProfiles,
    *,
    eq=None,  # equilibrium.Equilibrium object (optional)
    fC: float = 0.02,  # carbon fraction: n_C / n_e  [dimensionless]
    Vp: Optional[float] = None,  # fallback volume [m^3] if no eq
) -> dict:
    """
    Derive 0-D initial conditions from 1-D profiles.

    Volume integrals use the circular-torus weight  w(ρ) = 2ρ,
    equivalent to uniform ψ_n ↔ volume spacing under ρ = √ψ_n.

    Parameters
    ----------
    profiles : PlasmaProfiles (ne in 1e13 cm⁻³, Te in eV)
    eq       : Equilibrium object from equilibrium.load_gfile  (optional)
               If present, derives Ti from gfile total pressure.
               If absent, falls back to Ti = Te (equal temperatures).
    fC       : carbon impurity fraction = n_C / n_e.  Default 0.02.
    Vp       : plasma volume [m³] used only if eq is None.

    Returns
    -------
    dict
        Te_keV      : density-weighted electron temperature [keV]
        Ti_keV      : volume-averaged ion temperature [keV]
        nH          : H ion density [1e13 cm⁻³]
        nC          : C ion density [1e13 cm⁻³]
        ne_avg      : volume-averaged electron density [1e13 cm⁻³]
        ne_central  : central (ρ=0) electron density [1e13 cm⁻³]
        Wthe_MJ     : electron thermal energy [MJ]
        Wth_MJ      : total thermal energy [MJ]  (nan if no eq)
    """
    rho = profiles.rho
    ne = profiles.ne  # [1e13 cm^-3]
    Te = profiles.Te  # [eV]

    # Volume-weighted integrals: ∫ f dV / V ≈ 2 ∫₀¹ f(ρ) ρ dρ
    w = 2.0 * rho
    norm = _trapz(w, rho)  # → 1.0 for rho=[0,1]
    ne_avg = _trapz(ne * w, rho) / norm  # [1e13 cm^-3]
    neTe_avg = _trapz(ne * Te * w, rho) / norm  # [1e13 cm^-3 eV]
    Te_eff = neTe_avg / max(ne_avg, 1e-10)  # density-weighted [eV]

    ne_central = float(np.interp(0.0, rho, ne))  # [1e13 cm^-3]

    # Species densities (fully ionised charge-balance: ne = nH + 6 nC)
    Z_C = 6.0
    nC = max(ne_central * fC, 0.0)
    nH = max(ne_central * (1.0 - Z_C * fC), 0.0)

    # Electron thermal energy
    # neTe_avg [1e13 cm^-3 eV] * 1e19 → [m^-3 eV]  (since 1e13 cm^-3 = 1e19 m^-3)
    # NaN-safe volume resolution: `or`-chaining is wrong here because NaN is
    # truthy, so a gfile with an undefined volume would poison Wthe.
    Vp_eff = getattr(eq, "V", None) if eq is not None else None
    if Vp_eff is None or not np.isfinite(Vp_eff) or Vp_eff <= 0.0:
        Vp_eff = Vp if (Vp is not None and np.isfinite(Vp) and Vp > 0.0) else 20.0
    Wthe_MJ = 1.5 * _EV * Vp_eff * (neTe_avg * 1e19) * 1e-6  # [MJ]

    # Ion temperature from gfile total pressure (if available)
    if eq is not None:
        Ti_eV, Wth_MJ = _Ti_from_gfile_pressure(eq, neTe_avg, ne_avg)
    else:
        Ti_eV = Te_eff  # fallback: Ti = Te
        Wth_MJ = float("nan")

    return dict(
        Te_keV=Te_eff * 1e-3,
        Ti_keV=Ti_eV * 1e-3,
        nH=nH,
        nC=nC,
        ne_avg=ne_avg,
        ne_central=ne_central,
        Wthe_MJ=Wthe_MJ,
        Wth_MJ=Wth_MJ,
    )


def _Ti_from_gfile_pressure(eq, neTe_avg: float, ne_avg: float) -> tuple[float, float]:
    """
    Ti from gfile total pressure minus electron pressure.

    Wth_total = 1.5 · V · ∫₀¹ p(ψ_n) dψ_n       [from gfile, in J]
    Wthe      = 1.5 · eV · V · ⟨ne Te⟩_SI        [from profiles, in J]
    Wthi      = Wth_total - Wthe
    Ti        = Wthi / (1.5 · eV · V · ni_avg)

    Both integrals use the uniform ψ_n ↔ volume mapping, which is self-
    consistent when ρ = ρ_pol = √ψ_n (the circular-torus approximation).

    Returns
    -------
    (Ti_eV, Wth_MJ)
    """
    Vp = eq.V  # [m^3]

    # Total stored energy from MHD equilibrium [J]
    Wth_J = 1.5 * Vp * _trapz(eq.pressure, eq.psi_n)

    # Electron stored energy from profiles [J]
    # neTe_avg in [1e13 cm^-3 eV] = [1e19 m^-3 eV]
    Wthe_J = 1.5 * _EV * Vp * (neTe_avg * 1e19)

    Wthi_J = Wth_J - Wthe_J

    if Wthi_J <= 0.0:
        # Thomson data implies more energy than MHD total: fall back to Ti = Te
        return float(neTe_avg / max(ne_avg, 1e-10)), Wth_J * 1e-6

    # ni ≈ ne for a fully-ionised deuterium-dominated plasma
    ni_SI = ne_avg * 1e19  # [m^-3]
    Ti_eV = float(Wthi_J / (1.5 * _EV * Vp * ni_SI))
    Ti_eV = max(Ti_eV, 0.3)  # floor

    return Ti_eV, Wth_J * 1e-6  # (Ti [eV], Wth [MJ])


def uniform_grid(Vp: float, Nr: int) -> dict:
    """Minimal radial grid: Nr equal-volume cells filling plasma volume Vp.

    Returns a dict with
        rho_c : cell-centre coordinates in [0, 1]
        Vcell : (Nr,) cell volumes [m^3], sum = Vp

    Placeholder for step 1 (real V'(rho) from the equilibrium); everything
    downstream only consumes ``Vcell`` and ``rho_c`` so the upgrade is a
    drop-in.
    """
    Nr = int(Nr)
    rho_edges = np.linspace(0.0, 1.0, Nr + 1)
    rho_c = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    Vcell = np.full(Nr, Vp / Nr)
    return {"rho_c": rho_c, "Vcell": Vcell}
