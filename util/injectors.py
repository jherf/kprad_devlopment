# injectors.py
"""
Particle-source models for the kprad simulation.

A source is anything that adds neutral atoms to the plasma at each timestep.
Sources share a single interface:

    source.deliver(t, state=None) -> dict {symbol: ndot}

where ndot is in 1e20 atoms / ms and ``symbol`` is always a *plasma* element
symbol that exists in the solver layout ('H', 'Ne', 'C', ...). Molecular
input species ('D2', 'H2', ...) are converted to atoms at delivery time.

The ``state`` argument carries the instantaneous plasma state
(keys 'Te', 'Ti', 'ne', 'ni', 'time') so that plasma-dependent sources
(ablation models, Te-gated sputtering) can use it. State-independent
sources simply ignore it.

Sources
-------
MGI          Multi-species massive gas injection with a configurable
             delivery profile (gaussian / square / exponential).

Pellet       Single spherical D2/Ne (or mixed) pellet ablated by the Parks
             NGS model; remaining inventory lives in the ODE aux block.

WallSputter  Te-gated wall-material source active during TQ and CQ.
"""

import numpy as np
from .util_injectors import torr_l_to_particles, make_profile, DeliveryProfile
from .constants import _RHO_SOLID

_MOLECULAR = {
    "H2": ("H", 2.0),
    "D2": ("H", 2.0),
    "T2": ("H", 2.0),
    "DT": ("H", 2.0),
}


def _to_plasma_species(sym: str) -> tuple[str, float]:
    """Returns 2 for D2-like, 1 otherwise"""
    return _MOLECULAR.get(str(sym).upper(), (sym, 1.0))


# ---------------------------------------------------------------------------
# ---- Base class
# ---------------------------------------------------------------------------
class Injector:
    """Base class for injector sources."""

    species = None

    def deliver(self, t: float, state: dict | None = None) -> dict:
        """Return ``{symbol: ndot}`` of neutral deposition rates at time ``t``
        [ms], with ndot in [1e20 atoms / ms].

        ``state`` (optional) carries the current plasma state for
        state-dependent sources: keys ``'Te'``, ``'Ti'``, ``'ne'``, ``'ni'``,
        ``'time'``.
        """
        return {}

    def first_light_time(self, threshold_frac: float = 0.01) -> float | None:
        """Time of delivery onset [ms], or None if not time-gated."""
        return None

    def total_atoms_1e20(self) -> dict:
        """Total particles delivered over all time, per species [x 1e20]."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(species={self.species!r})"


# ---------------------------------------------------------------------------
# ---- Massive gas injection
# ---------------------------------------------------------------------------
class MGI(Injector):
    """
    Multi-species massive gas injector with a configurable delivery profile.

    The total particle inventory is fixed by the Torr-L quantities; the time
    history is set by a normalized ``DeliveryProfile`` (its ``shape(t)``
    integrates to 1 over all time), so

        ndot_sym(t) = N_tot_sym [1e20] * profile.shape(t) [1/ms].

    Molecular species ('D2', 'H2', ...) are converted at construction time:
    1 Torr-L of D2 delivers 2x the molecule count as plasma species 'H'
    (this reproduces the MATLAB ``B * 2 * fracD2`` term, and guarantees the
    delivered symbol exists in the solver layout).

    ``deliver`` accepts the plasma ``state`` so a future ablation /
    assimilation model can scale delivery on Te, ne, etc. without changing
    the interface; the base MGI ignores it.

    Parameters
    ----------
    species         : str or list of input gas symbols, e.g. 'Ne' or ['Ne', 'D2'].
    V_TorrL         : float or list, gas quantity per species [Torr-L],
                      same length as ``species``.
    profile         : a DeliveryProfile instance, or a profile name passed to
                      ``make_profile``: 'gaussian', 'square', 'exponential'.
                      Default 'exponential' (valve opens, flow decays).
    T_K             : gas temperature for the Torr-L conversion [K].
                      Default 293.15 K gives 0.3294e20 molecules/Torr-L.
    torrL_to_1e20   : optional explicit conversion factor
                      [1e20 molecules / Torr-L], overriding the ideal-gas
                      conversion at T_K. Use 0.322 to reproduce the MATLAB
                      KPRAD value (equivalent to T ~ 300 K). Molecular
                      species are still doubled to atoms either way.
    **profile_kwargs: forwarded to ``make_profile`` when ``profile`` is a
                      name, e.g.
                        gaussian:    t_peak=5.0, dt_pulse=1.8
                        square:      t_start=4.0, t_end=7.0
                        exponential: t_start=4.0, tau=1.0

    Examples
    --------
    >>> mgi = MGI(['Ar'], [300.0], profile='exponential', t_start=2.0, tau=1.5)
    >>> mgi.deliver(3.0)
    >>> spi = MGI(['Ne', 'D2'], [65.0, 50.0], profile='gaussian',
    ...           t_peak=5.0, dt_pulse=1.8)
    >>> spi.deliver(5.0)            # {'Ne': ..., 'H': ...}  (D2 -> 2 H atoms)
    >>> spi.total_atoms_1e20()
    """

    def __init__(
        self,
        species,
        V_TorrL,
        profile="exponential",
        T_K: float = 293.15,
        torrL_to_1e20: float | None = None,
        **profile_kwargs,
    ) -> None:
        super().__init__()

        # --- Make sure species and V_torrL are in a list form
        if isinstance(species, str):
            species = [species]
        if np.isscalar(V_TorrL):
            V_TorrL = [V_TorrL]

        if len(species) != len(V_TorrL):
            raise ValueError(
                f"species ({len(species)}) and V_TorrL ({len(V_TorrL)}) "
                "must have the same length"
            )

        self.species = list(species)
        self.V_TorrL = [float(v) for v in V_TorrL]  # type: ignore

        if isinstance(profile, DeliveryProfile):
            if profile_kwargs:
                raise ValueError(
                    "profile_kwargs are only valid when `profile` is a name"
                )
            self.profile = profile
        else:
            self.profile = make_profile(profile, **profile_kwargs)

        # ---- Total delivered atoms per *plasma* species [1e20 atoms]
        self.N_tot_1e20: dict[str, float] = {}
        for sym, V in zip(self.species, self.V_TorrL):
            psym, atoms_per_molecule = _to_plasma_species(sym)
            if torrL_to_1e20 is not None:
                # Explicit molecules-per-Torr-L factor (e.g. 0.322 for
                # MATLAB-KPRAD comparison runs).
                N = V * float(torrL_to_1e20) * atoms_per_molecule
            else:
                N = torr_l_to_particles(V, T_K) * atoms_per_molecule / 1e20

            # Add value to the dict
            self.N_tot_1e20[psym] = self.N_tot_1e20.get(psym, 0.0) + N

    # ------------------------------------------------------------------
    def deliver(self, t: float, state: dict | None = None) -> dict:
        s = float(self.profile.shape(float(t)))
        if s <= 0.0:
            return {}
        return {sym: N * s for sym, N in self.N_tot_1e20.items()}

    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return self.profile.first_light_time(threshold_frac)

    def total_atoms_1e20(self) -> dict:
        return dict(self.N_tot_1e20)

    def __repr__(self) -> str:
        qty = ", ".join(f"{s}={v:g} TL" for s, v in zip(self.species, self.V_TorrL))
        return f"{type(self).__name__}({qty}, profile={self.profile!r})"


# ---------------------------------------------------------------------------
# ---- Pellet
# ---------------------------------------------------------------------------
# Parks 2017 composite (Ne/D2) neutral-gas-shielding constants
# [P.B. Parks, TSDW 2017; as implemented in DREAM, INDEX, JOREK-SPI]
_PARKS_LAMBDA_A = 27.0837  # lambda(X) = A + tan(B*X)  [g/s]
_PARKS_LAMBDA_B = 1.48709  # X = N_D2 / (N_D2 + N_Ne), molecular fraction
_W_MOL = {"D2": 4.0282, "Ne": 20.183}  # molecular weight [g/mol] (Parks)
_N_AVOGADRO = 6.02214076e23

# Rate below which the pellet is considered fully ablated
# [1e20 molecules] (= 1e11 molecules — utterly negligible)
_PELLET_N_FLOOR = 1e-9


class Pellet(Injector):
    """
    Single spherical cryogenic pellet of solid D2, Ne, or a uniform D2/Ne
    mixture, ablated by the neutral-gas-shielding (NGS) model.

    Ablation model (Parks 2017 composite scaling)
    ---------------------------------------------
        G [g/s] = lambda(X) * (Te[eV]/2000)^(5/3)
                            * (r_p[cm]/0.2)^(4/3)
                            * (ne[1e14 cm^-3])^(1/3)
        lambda(X) = 27.0837 + tan(1.48709 * X),   X = N_D2/(N_D2 + N_Ne)

    lambda(1) = 39.0 g/s and lambda(0) = 27.08 g/s reproduce Parks' pure-D2
    and pure-Ne rates. Te and ne are the instantaneous (volume-averaged)
    plasma electron temperature and density from the solver ``state``.
    The molar ablation rate is split congruently (uniform mixture):

        dN_i/dt = frac_i * G / mubar,    mubar = sum_i frac_i * W_i

    ODE state
    ---------
    The remaining inventory (one slot per pellet species, [1e20 molecules])
    lives in the solver state vector's aux block — deliver()-side counters
    would double-deplete under BDF's rejected trial steps. The driver
    assigns ``aux_slice`` and calls ``ablate(t, state, y)``, which returns
    both the neutral deposition rates and d(inventory)/dt. Depletion is
    self-limiting: r_p ~ N^(1/3) so G -> 0 smoothly as the pellet vanishes.

    The pellet radius is derived from the remaining inventory and the solid
    densities (constants._RHO_SOLID), assuming ideal volume mixing.

    Parameters
    ----------
    species       : str or list from {'D2', 'Ne'} — the Parks composite law
                    is specific to this pair.
    V_TorrL       : float or list, gas-equivalent inventory per species
                    [Torr-L] (same convention as MGI). Give this OR N_1e20.
    N_1e20        : float or list, inventory per species [1e20 molecules].
    t_start       : time the pellet enters the plasma [ms]. Default 0.
    T_K           : gas temperature for the Torr-L conversion [K].
    torrL_to_1e20 : optional explicit conversion factor
                    [1e20 molecules / Torr-L] (e.g. 0.322 for MATLAB parity).

    Examples
    --------
    >>> p = Pellet(['D2', 'Ne'], V_TorrL=[200.0, 65.0], t_start=2.0)
    >>> p.rp0_cm            # initial radius from inventory + solid densities
    >>> p = Pellet('Ne', N_1e20=[21.4], t_start=1.0)
    """

    n_state = 0  # set per-instance in __init__

    def __init__(
        self,
        species,
        V_TorrL=None,
        t_start: float = 0.0,
        T_K: float = 293.15,
        torrL_to_1e20: float | None = None,
    ) -> None:
        super().__init__()

        if isinstance(species, str):
            species = [species]
        self.pellet_species = [str(s) for s in species]

        unknown = [s for s in self.pellet_species if s not in _W_MOL]
        if unknown:
            raise ValueError(
                f"Pellet species {unknown} not supported: the Parks composite "
                f"ablation law covers {sorted(_W_MOL)} only."
            )
        if len(set(self.pellet_species)) != len(self.pellet_species):
            raise ValueError("Duplicate pellet species")
        for s in self.pellet_species:
            if s not in _RHO_SOLID:
                raise ValueError(f"No solid density for {s} in constants._RHO_SOLID")

        # ---- Convert V_torrL to # of particles / 1e20 -------------------
        if V_TorrL is not None:
            if np.isscalar(V_TorrL):
                V_TorrL = [V_TorrL]
            if len(V_TorrL) != len(self.pellet_species):
                raise ValueError("species and V_TorrL must have the same length")
            if torrL_to_1e20 is not None:
                N0 = [float(v) * float(torrL_to_1e20) for v in V_TorrL]  # type: ignore
            else:
                N0 = [torr_l_to_particles(float(v), T_K) / 1e20 for v in V_TorrL]  # type: ignore
        else:
            raise RuntimeError("V_TorrL must not be None!")

        # NOTE: inventory is in MOLECULES (D2 molecules / Ne atoms), [1e20]
        self.N0_1e20 = np.asarray(N0, dtype=float)
        self.n_state = len(self.pellet_species)
        self.t_start = float(t_start)
        self.aux_slice: slice | None = None  # assigned by the driver

        # Plasma symbols this pellet feeds (for repr / bookkeeping)
        self.species = sorted({_to_plasma_species(s)[0] for s in self.pellet_species})

        self.rp0_cm = self.radius_cm(self.N0_1e20)

    # ------------------------------------------------------------------
    def state0(self) -> list:
        """Initial aux-block values: remaining molecules [1e20] per species."""
        return list(self.N0_1e20)

    def radius_cm(self, y) -> float:
        """Pellet radius [cm] from remaining inventory (ideal volume mixing)."""
        y = np.maximum(np.asarray(y, dtype=float), 0.0)
        vol = 0.0  # [cm^3]
        for yi, s in zip(y, self.pellet_species):
            vol += (yi * 1e20 / _N_AVOGADRO) * _W_MOL[s] / _RHO_SOLID[s]
        return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    def ablate(self, t: float, state: dict | None, y) -> tuple[dict, np.ndarray]:
        """Ablation at time ``t`` given plasma ``state`` and inventory ``y``.

        Parameters
        ----------
        t     : time [ms]
        state : plasma state dict with 'Te' [eV] and 'ne' [cm^-3]
        y     : (n_state,) remaining molecules [1e20], from the aux block

        Returns
        -------
        deposits : {plasma_symbol: Ndot [1e20 atoms/ms]}  (D2 -> 2 H atoms)
        dydt     : (n_state,) d(inventory)/dt [1e20 molecules/ms]
        """
        zero = ({}, np.zeros(self.n_state))
        if state is None or t < self.t_start:
            return zero

        y = np.maximum(np.asarray(y, dtype=float), 0.0)
        Ntot = float(y.sum())
        if Ntot <= _PELLET_N_FLOOR:
            return zero

        Te = float(state.get("Te", 0.0))  # [eV]
        ne = float(state.get("ne", 0.0))  # [cm^-3]
        if Te <= 0.0 or ne <= 0.0:
            return zero

        frac = y / Ntot
        # D2 molecular fraction X (0 if the pellet has no D2)
        X = 0.0
        for fi, s in zip(frac, self.pellet_species):
            if s == "D2":
                X = float(fi)

        rp = self.radius_cm(y)  # [cm]
        lam = _PARKS_LAMBDA_A + np.tan(_PARKS_LAMBDA_B * X)  # [g/s]
        G = (
            lam
            * (Te / 2000.0) ** (5.0 / 3.0)
            * (rp / 0.2) ** (4.0 / 3.0)
            * (ne / 1.0e14) ** (1.0 / 3.0)
        )  # mass ablation rate [g/s]

        mubar = float(sum(fi * _W_MOL[s] for fi, s in zip(frac, self.pellet_species)))
        # total molecular ablation rate [1e20 molecules/ms]:
        #   (G/mubar) [mol/s] * N_A [1/mol] * 1e-3 [s/ms] / 1e20
        Rtot = (G / mubar) * _N_AVOGADRO * 1.0e-3 / 1.0e20

        dydt = -frac * Rtot
        deposits: dict[str, float] = {}
        for fi, s in zip(frac, self.pellet_species):
            psym, apm = _to_plasma_species(s)
            deposits[psym] = deposits.get(psym, 0.0) + fi * Rtot * apm
        return deposits, dydt

    # ------------------------------------------------------------------
    def injection_history(self, y_traj: np.ndarray, tV: np.ndarray) -> dict:
        """Reconstruct delivery vs. time from the saved aux trajectory.

        Parameters
        ----------
        y_traj : (n_state, Nt) inventory trace [1e20 molecules]
        tV     : (Nt,) time points [ms]

        Returns
        -------
        {plasma_symbol: {'rate' [1e20 atoms/ms], 'cumulative' [1e20 atoms]}}
        """
        y = np.maximum(np.atleast_2d(np.asarray(y_traj, dtype=float)), 0.0)
        tV = np.asarray(tV, dtype=float)
        out: dict = {}
        for i, s in enumerate(self.pellet_species):
            psym, apm = _to_plasma_species(s)
            cum = np.maximum(self.N0_1e20[i] - y[i], 0.0) * apm
            cum = np.maximum.accumulate(cum)  # guard tiny integrator wiggles
            rate = (
                np.gradient(cum, tV, edge_order=1)
                if len(tV) > 1
                else np.zeros_like(cum)
            )
            if psym in out:
                out[psym]["rate"] = out[psym]["rate"] + rate
                out[psym]["cumulative"] = out[psym]["cumulative"] + cum
            else:
                out[psym] = {"rate": rate, "cumulative": cum}
        return out

    # ------------------------------------------------------------------
    def deliver(self, t: float, state: dict | None = None) -> dict:
        """Stateful source: delivery goes through ``ablate`` (needs the aux
        inventory), so the state-independent interface returns nothing.
        This also keeps ``compute_injection`` (pre-simulation geometry)
        from double-counting the pellet."""
        return {}

    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return self.t_start

    def total_atoms_1e20(self) -> dict:
        out: dict[str, float] = {}
        for n0, s in zip(self.N0_1e20, self.pellet_species):
            psym, apm = _to_plasma_species(s)
            out[psym] = out.get(psym, 0.0) + n0 * apm
        return out

    def __repr__(self) -> str:
        qty = ", ".join(
            f"{s}={n:g}e20" for s, n in zip(self.pellet_species, self.N0_1e20)
        )
        return (
            f"{type(self).__name__}({qty}, rp0={self.rp0_cm*10:.2f} mm, "
            f"t_start={self.t_start} ms)"
        )


# ---------------------------------------------------------------------------
# ---- Wall sputtering
# ---------------------------------------------------------------------------
class WallSputter(Injector):
    """
    Te-gated wall-sputtering source. Produces ``Ndot_TQ`` while the plasma is
    in the thermal-quench window (``Te_CQ < Te < Te_TQ_frac * Te0_eV``) and
    ``Ndot_CQ`` once Te has fallen into the current-quench window
    (``Te <= Te_CQ``).

    Parameters
    ----------
    species         : symbol of the sputtered material, e.g. 'C'.
    Ndot_TQ, Ndot_CQ: source rates during TQ / CQ phases [1e20 atoms/ms].
    Te0_eV          : initial electron temperature [eV]; sets the upper
                      TQ-window edge as Te_TQ_frac * Te0_eV.
    Te_CQ_threshold : TQ/CQ boundary [eV]. Default 10.
    Te_TQ_frac      : upper TQ edge as fraction of Te0_eV. Default 0.9.
    """

    def __init__(
        self,
        species: str,
        Ndot_TQ: float,
        Ndot_CQ: float,
        Te0_eV: float,
        Te_CQ_threshold: float = 10.0,
        Te_TQ_frac: float = 0.9,
    ) -> None:
        super().__init__()
        self.species = species
        self.Ndot_TQ = float(Ndot_TQ)
        self.Ndot_CQ = float(Ndot_CQ)
        self.Te0_eV = float(Te0_eV)
        self.Te_CQ = float(Te_CQ_threshold)
        self.Te_TQ_max = float(Te_TQ_frac) * self.Te0_eV

    def deliver(self, t: float, state: dict | None = None) -> dict:
        if state is None or "Te" not in state:
            return {}
        Te = state["Te"]
        if self.Te_CQ < Te < self.Te_TQ_max:
            return {self.species: self.Ndot_TQ}
        if Te <= self.Te_CQ:
            return {self.species: self.Ndot_CQ}
        return {}
