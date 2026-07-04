# config.py
"""
Simulation setup: configuration loading, initial-state construction,
and injector factory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import yaml
import numpy as np


from main.util.injectors import MGI, Pellet, WallSputter
from main.util.layout import SolverLayout
from main.util.constants import _EE


# ---- Config loader
def config_loader(path: Union[str, Path], verbose: bool = False) -> dict | None:
    """
    Load and pre-process a kprad YAML configuration file.

    Adds a convenience ``config['elements']`` list built by collecting every
    element symbol that appears in ``initial.species``, ``sources``, and
    ``simulation.sputter_species`` (in that order, no duplicates).

    Returns ``None`` and prints a diagnostic if the file cannot be read.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        if verbose:
            print(f"Loaded config: {path}")

        # ---- Build unified element list (preserves insertion order) -------------

        # Add elements from the initial plasma species
        seen = set()
        elements: list[str] = []
        for sym in config["initial"]["species"]:
            if sym not in seen:
                seen.add(sym)
                elements.append(sym)

        # Add the elements from the particle sources
        for src in config.get("sources", []):
            syms = src.get("species", [])
            # Making sure Ar doesn't convert to 'A' and 'r'
            if isinstance(syms, str):
                syms = [syms]
            for sym in syms:
                if sym not in seen:
                    seen.add(sym)
                    elements.append(sym)

        config["elements"] = elements

        # ---- Optional: derive geometry from gfile -------------------------------
        eq = None
        eq_cfg = config.get("equilibrium") or {}
        gfile_path = eq_cfg.get("gfile")
        # PyYAML parses a bare ``None`` as the *string* 'None' (only ``null``,
        # ``~`` or an empty value map to Python None); normalise those here so
        # a placeholder never triggers a file open on a file named "None".
        if isinstance(gfile_path, str) and gfile_path.strip().lower() in (
            "",
            "none",
            "null",
        ):
            gfile_path = None
        if gfile_path:
            eq = _load_equilibrium(
                gfile_path, cocos=eq_cfg.get("cocos", 1), verbose=verbose
            )
            if eq is not None:
                _apply_equilibrium(config, eq)

        # ---- Optional: derive initial conditions from profiles ------------------
        prof_cfg = config.get("profiles", {})
        h5_path = prof_cfg.get("h5_file")
        if h5_path:
            profiles = _load_profiles(h5_path, prof_cfg)
            if profiles is not None:
                _apply_profiles(config, profiles, eq=eq)

        return config

    except Exception as e:
        print(f"Could not load config {path}: {e}")
        return None


# ---- Equilibrium loaders
def _load_equilibrium(gfile_path: str, cocos: int = 1, verbose: bool = False):
    """Load and summarise a G-EQDSK file.  Returns None on failure."""
    try:
        from main.util.equilibrium import load_gfile, summary as eq_summary

        eq = load_gfile(gfile_path, cocos=cocos)
        if verbose:
            print(eq_summary(eq))
        return eq
    except FileNotFoundError:
        print(f"  [!] gfile not found: {gfile_path}")
    except Exception as exc:
        print(f"  [!] gfile load failed ({gfile_path}): {exc}")
    return None


def _apply_equilibrium(config: dict, eq) -> None:
    """
    Inject gfile-derived quantities into config, overriding manually typed values.

    Sets
    ----
    tokamak.Rmaj, initial.Ip, initial.li, initial.a, initial.Ap, initial.Vp

    Does NOT touch wall parameters (Rw, dw, Cw, etaw) — those are machine
    geometry, not equilibrium geometry.
    """
    config.setdefault("tokamak", {})
    config.setdefault("initial", {})

    config["tokamak"]["Rmaj"] = eq.R0
    config["initial"]["Ip"] = abs(float(eq.Ip)) / 1e6  # [MA]
    config["initial"]["li"] = float(eq.li3)
    config["initial"]["a"] = float(eq.a)
    config["initial"]["Ap"] = float(eq.A_cs)
    config["initial"]["Vp"] = float(eq.V)

    print(
        f"→ Loaded gfile: Rmaj={eq.R0:.3f} m  Ip={abs(eq.Ip)/1e6:.3f} MA  "
        f"a={eq.a:.3f} m  Vp={eq.V:.2f} m³  li={eq.li3:.3f}"
    )


# ---- Profile loaders
def _load_profiles(h5_path: str, prof_cfg: dict):
    """Load profiles from HDF5.  Returns None on failure."""
    try:
        from main.util.profile import read_profiles_h5

        return read_profiles_h5(
            h5_path,
            rho_key=prof_cfg.get("rho_key", "rho"),
            ne_key=prof_cfg.get("ne_key", "ne"),
            Te_key=prof_cfg.get("Te_key", "Te"),
            Zeff_key=prof_cfg.get("Zeff_key"),
            ne_units=prof_cfg.get("ne_units", "1e19_m3"),
            Te_units=prof_cfg.get("Te_units", "eV"),
            time_idx=prof_cfg.get("time_idx", -1),
        )
    except FileNotFoundError:
        print(f"  [!] profiles h5 not found: {h5_path}")
    except Exception as exc:
        print(f"  [!] profiles load failed ({h5_path}): {exc}")
    return None


def _apply_profiles(config: dict, profiles, eq=None) -> None:
    """
    Inject profile-derived initial conditions into config.

    Sets
    ----
    initial.Te, initial.Ti, initial.species.H, initial.species.C

    Does NOT set initial.species.Ne — pre-existing Ne is a simulation
    choice (typically 0.0 for a pure-krypton or pure-neon SPI shot).
    """
    from main.util.profile import derive_initial_conditions

    fC = config.get("profiles", {}).get("fC", 0.02)
    Vp = config.get("initial", {}).get("Vp", 20.0)
    ic = derive_initial_conditions(profiles, eq=eq, fC=fC, Vp=Vp)

    config.setdefault("initial", {})
    config["initial"]["Te"] = ic["Te_keV"]
    config["initial"]["Ti"] = ic["Ti_keV"]
    config["initial"].setdefault("species", {})
    config["initial"]["species"]["H"] = ic["nH"]
    config["initial"]["species"]["C"] = ic["nC"]
    config["initial"]["species"].setdefault("Ne", 0.0)

    print(
        f"  profiles → Te={ic['Te_keV']:.2f} keV  Ti={ic['Ti_keV']:.2f} keV  "
        f"ne_avg={ic['ne_avg']:.3f}×10¹³ cm⁻³  "
        f"nH={ic['nH']:.3f}  nC={ic['nC']:.4f}  [10¹³ cm⁻³]"
    )
    if not np.isnan(ic.get("Wth_MJ", float("nan"))):
        print(
            f"            Wthe={ic['Wthe_MJ']:.3f} MJ  "
            f"Wth_total={ic['Wth_MJ']:.3f} MJ  "
            f"(Ti from gfile pressure split)"
        )


# ---- Initial solver state
def build_initial_state(
    layout: SolverLayout,
    species_densities: dict,
    Te0_eV,
    Ti0_eV,
    Ip0_MA: float,
    Iw0_MA: float,
    injectors=(),
):
    """
    Construct the 1-D ODE initial state vector.

    Parameters
    ----------
    layout            : Layout1D
    species_densities : {sym: n0}, fully-ionized densities in [1e13 cm^-3]
                        (same convention as the 0-D config). Scalars give
                        flat profiles; (Nr,) arrays give shaped profiles.
    Te0_eV, Ti0_eV    : initial temperatures [eV], scalar or (Nr,).
    Ip0_MA, Iw0_MA    : initial currents [MA].
    injectors         : injector list; stateful injectors (n_state > 0,
                        e.g. Pellet) must already have their ``aux_slice``
                        assigned — their ``state0()`` fills the aux block.

    Returns
    -------
    solV0  : ndarray of length layout.size
    dense0 : (Nr,) initial electron density [cm^-3]
    densi0 : (Nr,) initial ion density [cm^-3]
    """

    Nr = layout.Nr
    dense0 = np.zeros(Nr)
    densi0 = np.zeros(Nr)
    dens = {}

    # Put the densities in the fully-stripped spots
    for sym, n in species_densities.items():
        # Initial density from config file, convert to cm^-3
        n = np.broadcast_to(np.asarray(n, dtype=float) * 1.0e13, (Nr,))
        block = np.zeros((layout.Zmax[sym] + 1, Nr))
        block[-1, :] = n  # fully stripped

        dens[sym] = block
        dense0 += n * layout.Zmax[sym]
        densi0 += n

    # --- Constant Te, Ti profiles (UPDATE IN THE FUTURE!)
    Te0 = np.broadcast_to(np.asarray(Te0_eV, dtype=float), (Nr,))
    Ti0 = np.broadcast_to(np.asarray(Ti0_eV, dtype=float), (Nr,))
    We0 = 1.5 * _EE * dense0 * Te0  # [MJ/m^3]
    Wi0 = 1.5 * _EE * densi0 * Ti0

    # --- Injector ODE state (e.g. remaining pellet inventory)
    aux0 = None
    if getattr(layout, "n_aux", 0):
        aux0 = np.zeros(layout.n_aux)
        for inj in injectors:
            if getattr(inj, "n_state", 0):
                if inj.aux_slice is None:
                    raise ValueError(f"{inj!r} has ODE state but no aux_slice assigned")
                aux0[inj.aux_slice] = inj.state0()

    solV0 = layout.pack(Ip0_MA, Iw0_MA, We0, Wi0, dens, aux=aux0)

    return solV0, dense0, densi0


# ---- Combing injectors to a list
def build_injectors(config: dict, Te0_eV: float) -> list:
    """
    Construct the list of Injector instances from the YAML config.

    Supported ``sources[*].type`` values
    -------------------------------------
    ``mgi``
        Keys: species, V_torrL, profile (optional, default 'exponential'),
              profile_kwargs, T_K (optional), torrL_to_1e20 (optional,
              e.g. 0.322 to reproduce the MATLAB conversion factor).

    ``pellet``
        Single spherical D2/Ne (or mixed) pellet, Parks NGS ablation.
        Keys: species (subset of ['D2','Ne']), and exactly one of
        V_torrL (gas-equivalent [Torr-L]) or N_1e20 ([1e20 molecules]);
        t_start (optional, [ms]), T_K / torrL_to_1e20 (optional, as MGI).

    ``wall_sputter``
        Keys: species, Ndot_TQ, Ndot_CQ, Te0_eV (optional; defaults to the
        initial plasma Te, which sets the upper TQ-window edge as
        0.9 * Te0_eV). The source is only appended when ``Ndot_TQ > 0``
        or ``Ndot_CQ > 0``.
    """
    injectors = []

    for src in config.get("sources", []):
        t = src["type"]
        if t.upper() == "MGI":
            injectors.append(
                MGI(
                    species=src["species"],
                    V_TorrL=src["V_torrL"],
                    profile=src.get("profile", "exponential"),
                    T_K=float(src.get("T_K", 293.15)),
                    torrL_to_1e20=src.get("torrL_to_1e20"),
                    **src.get("profile_kwargs", {}),
                )
            )
        elif t.lower() == "pellet":
            injectors.append(
                Pellet(
                    species=src["species"],
                    V_TorrL=src.get("V_torrL"),
                    t_start=float(src.get("t_start", 0.0)),
                    T_K=float(src.get("T_K", 293.15)),
                    torrL_to_1e20=src.get("torrL_to_1e20"),
                )
            )
        elif t == "wall_sputter":
            ndot_tq = src.get("Ndot_TQ", 0.0)
            ndot_cq = src.get("Ndot_CQ", 0.0)
            if ndot_tq > 0 or ndot_cq > 0:
                injectors.append(
                    WallSputter(
                        species=src["species"],
                        Ndot_TQ=ndot_tq,
                        Ndot_CQ=ndot_cq,
                        # Config may override the TQ-window reference Te;
                        # default is the initial plasma Te.
                        Te0_eV=float(src.get("Te0_eV", Te0_eV)),
                    )
                )
        else:
            raise ValueError(f"Unknown source type: {t!r}")

    return injectors
