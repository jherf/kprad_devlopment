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

CSP          Cryogenic shell pellet: Ne core inside a D2 outer shell,
             ablated sequentially (shell first, then the exposed core).

SPI          Shattered pellet injection: a train of fragments with Parks /
             Mott-Linfoot sizes and a velocity spread, so arrival at the
             plasma is staggered in time.

GradedCSP    Shell pellet with a CONTINUOUS radial Ne/D2 composition
             profile, so the plume composition evolves as the surface
             recedes. One aux slot (remaining volume).

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
                raise ValueError("profile_kwargs are only valid when `profile` is a name")
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
        N_1e20=None,
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

        # ---- Initial inventory, in MOLECULES [1e20] ----------------------
        # Two equivalent ways to specify it. V_TorrL is the experimental
        # convention (what a gas-handling system reads out); N_1e20 is the
        # direct particle count, useful when reproducing a published case
        # that quotes atoms rather than Torr-L.
        if (V_TorrL is None) == (N_1e20 is None):
            raise ValueError(
                "Give exactly one of V_TorrL or N_1e20 (got "
                f"V_TorrL={V_TorrL!r}, N_1e20={N_1e20!r})"
            )

        if V_TorrL is not None:
            if np.isscalar(V_TorrL):
                V_TorrL = [V_TorrL]
            if len(V_TorrL) != len(self.pellet_species):
                raise ValueError("species and V_TorrL must have the same length")
            if torrL_to_1e20 is not None:
                # Explicit molecules-per-Torr-L factor, e.g. 0.322 for
                # MATLAB-KPRAD parity. Same convention as MGI.
                N0 = [float(v) * float(torrL_to_1e20) for v in V_TorrL]  # type: ignore
            else:
                N0 = [torr_l_to_particles(float(v), T_K) / 1e20 for v in V_TorrL]  # type: ignore
        else:
            if np.isscalar(N_1e20):
                N_1e20 = [N_1e20]
            if len(N_1e20) != len(self.pellet_species):
                raise ValueError("species and N_1e20 must have the same length")
            N0 = [float(v) for v in N_1e20]  # type: ignore

        if any(v < 0 for v in N0):
            raise ValueError("Pellet inventories must be non-negative")

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

        inventory = np.maximum(np.asarray(y, dtype=float), 0.0)
        N_remaining = float(inventory.sum())
        if N_remaining <= _PELLET_N_FLOOR:
            return zero

        Te = float(state.get("Te", 0.0))  # [eV]
        ne = float(state.get("ne", 0.0))  # [cm^-3]
        if Te <= 0.0 or ne <= 0.0:
            return zero

        # ---- Composition -----------------------------------------------
        # Congruent ablation: the pellet ablates in its own proportions, so
        # the mixture stays uniform and these fractions are constant in
        # practice. They are recomputed each call rather than cached because
        # CSP and SPI subclass this and do change composition.
        molecular_fraction = inventory / N_remaining

        # Parks' composite law is parameterised by the D2 MOLECULAR fraction
        # X = N_D2 / (N_D2 + N_Ne), not the atom or mass fraction. Pure Ne
        # is X = 0, pure D2 is X = 1.
        D2_fraction = 0.0
        for frac_i, sym in zip(molecular_fraction, self.pellet_species):
            if sym == "D2":
                D2_fraction = float(frac_i)

        # ---- Parks NGS mass ablation rate [g/s] -------------------------
        #   G = lambda(X) * (Te/2000)^(5/3) * (r_p/0.2)^(4/3) * (ne/1e14)^(1/3)
        #
        # B*X never exceeds 1.48709 < pi/2, so the tan() cannot blow up.
        # lambda(0) = 27.08 g/s (pure Ne), lambda(1) = 39.0 g/s (pure D2).
        pellet_radius_cm = self.radius_cm(inventory)
        lambda_g_per_s = _PARKS_LAMBDA_A + np.tan(_PARKS_LAMBDA_B * D2_fraction)
        mass_ablation_rate = (
            lambda_g_per_s
            * (Te / 2000.0) ** (5.0 / 3.0)
            * (pellet_radius_cm / 0.2) ** (4.0 / 3.0)
            * (ne / 1.0e14) ** (1.0 / 3.0)
        )  # [g/s]

        # ---- Mass rate -> molecular rate --------------------------------
        # Divide by the mixture's mean molecular weight to get mol/s, then
        #   [mol/s] * N_A [1/mol] * 1e-3 [s/ms] / 1e20  ->  [1e20 molecules/ms]
        mean_molecular_weight = float(
            sum(
                frac_i * _W_MOL[sym]
                for frac_i, sym in zip(molecular_fraction, self.pellet_species)
            )
        )  # [g/mol]
        molecule_rate = (
            (mass_ablation_rate / mean_molecular_weight) * _N_AVOGADRO * 1.0e-3 / 1.0e20
        )

        # Depletion is self-limiting: r_p ~ N^(1/3), so G -> 0 smoothly as
        # the pellet vanishes and the inventory never crosses zero abruptly.
        dydt = -molecular_fraction * molecule_rate

        deposits: dict[str, float] = {}
        for frac_i, sym in zip(molecular_fraction, self.pellet_species):
            plasma_sym, atoms_per_molecule = _to_plasma_species(sym)
            deposits[plasma_sym] = (
                deposits.get(plasma_sym, 0.0) + frac_i * molecule_rate * atoms_per_molecule
            )
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
            rate = np.gradient(cum, tV, edge_order=1) if len(tV) > 1 else np.zeros_like(cum)
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
        qty = ", ".join(f"{s}={n:g}e20" for s, n in zip(self.pellet_species, self.N0_1e20))
        return (
            f"{type(self).__name__}({qty}, rp0={self.rp0_cm*10:.2f} mm, "
            f"t_start={self.t_start} ms)"
        )


# ---------------------------------------------------------------------------
# ---- Cryogenic shell pellet (CSP)
# ---------------------------------------------------------------------------
class CSP(Pellet):
    """
    Cryogenic shell pellet: a solid Ne core wrapped in a D2 outer shell.

    Ablation proceeds sequentially, using the same Parks NGS scaling as
    ``Pellet`` but with the material-appropriate pure-species coefficient
    and the radius of the surface actually exposed to the plasma:

      1. Shell phase — only D2 ablates: lambda(X=1) = 39.0 g/s, evaluated at
         the *outer* radius r_out (core + shell). The Ne core is shielded
         and its inventory stays frozen.
      2. Core phase  — once the shell is gone, only Ne ablates:
         lambda(X=0) = 27.0837 g/s, evaluated at the core radius r_core.

    Breach smoothing (numerics, not physics)
    ----------------------------------------
    The shell reaches zero at *finite* ablation rate (r_out stays large
    because of the core underneath), so a hard D2 -> Ne switch would be a
    discontinuous RHS and make BDF chatter at the transition. When the
    shell is thinner than ``breach_um`` (default 10 um), the core is
    treated as fractionally exposed: the D2 rate is scaled by
    f = (r_out - r_core)/breach and the Ne rate by (1 - f). This confines
    the hand-off to a sliver of the ablation history (um-scale on a
    mm-scale pellet), makes the RHS continuous, and drives the shell
    inventory to zero exponentially instead of overshooting negative.

    ODE state (aux block, assigned by the driver)
    ---------------------------------------------
        y[0] : remaining D2 shell  [1e20 molecules]
        y[1] : remaining Ne core   [1e20 atoms]

    Parameters
    ----------
    shell_TorrL, core_TorrL : gas-equivalent inventory of the D2 shell and
                    Ne core [Torr-L]. Give this pair OR the 1e20 pair.
    shell_1e20, core_1e20   : inventories in [1e20 molecules].
    t_start       : time the pellet enters the plasma [ms]. Default 0.
    T_K           : gas temperature for the Torr-L conversion [K].
    torrL_to_1e20 : optional explicit conversion factor (as MGI/Pellet).
    breach_um     : shell thickness [um] over which the shell->core
                    transition is smoothed. Default 10.

    Examples
    --------
    >>> p = CSP(shell_TorrL=200.0, core_TorrL=65.0, t_start=2.0)
    >>> p.rp0_cm, p.rcore0_cm     # outer and core radii from inventories
    """

    _I_SHELL, _I_CORE = 0, 1  # aux ordering: [D2 shell, Ne core]

    def __init__(
        self,
        shell_TorrL: float | None = None,
        core_TorrL: float | None = None,
        shell_1e20: float | None = None,
        core_1e20: float | None = None,
        t_start: float = 0.0,
        T_K: float = 293.15,
        torrL_to_1e20: float | None = None,
        breach_um: float = 10.0,
    ) -> None:
        torr_pair = (shell_TorrL is not None, core_TorrL is not None)
        n20_pair = (shell_1e20 is not None, core_1e20 is not None)
        if any(torr_pair) and any(n20_pair):
            raise ValueError("Give either the *_TorrL pair or the *_1e20 pair, not both")
        if all(torr_pair):
            super().__init__(
                ["D2", "Ne"],
                V_TorrL=[shell_TorrL, core_TorrL],
                t_start=t_start,
                T_K=T_K,
                torrL_to_1e20=torrL_to_1e20,
            )
        elif all(n20_pair):
            # NOTE: this branch previously dropped the inventories entirely
            # and fell through to Pellet's "V_TorrL must not be None" error,
            # making the whole documented *_1e20 path unusable.
            super().__init__(
                ["D2", "Ne"],
                N_1e20=[shell_1e20, core_1e20],
                t_start=t_start,
            )
        else:
            raise ValueError(
                "CSP needs both shell and core inventories "
                "(shell_TorrL + core_TorrL, or shell_1e20 + core_1e20)"
            )

        if breach_um <= 0:
            raise ValueError("breach_um must be positive")
        self.breach_cm = float(breach_um) * 1e-4
        self.rcore0_cm = self.radius_cm([0.0, self.N0_1e20[self._I_CORE]])

    # ------------------------------------------------------------------
    def ablate(self, t: float, state: dict | None, y) -> tuple[dict, np.ndarray]:
        """Sequential shell-then-core ablation. Same interface as Pellet."""
        zero = ({}, np.zeros(self.n_state))
        if state is None or t < self.t_start:
            return zero

        inventory = np.maximum(np.asarray(y, dtype=float), 0.0)
        N_shell = float(inventory[self._I_SHELL])  # D2 [1e20 molecules]
        N_core = float(inventory[self._I_CORE])  # Ne [1e20 atoms]
        if N_shell + N_core <= _PELLET_N_FLOOR:
            return zero

        Te = float(state.get("Te", 0.0))  # [eV]
        ne = float(state.get("ne", 0.0))  # [cm^-3]
        if Te <= 0.0 or ne <= 0.0:
            return zero

        # The Te/ne part of the Parks law is common to both phases; only
        # lambda and the exposed radius differ, so factor it out.
        plasma_factor = (Te / 2000.0) ** (5.0 / 3.0) * (ne / 1.0e14) ** (1.0 / 3.0)

        r_outer_cm = self.radius_cm(inventory)  # core + shell
        r_core_cm = self.radius_cm([0.0, N_core])  # core alone
        shell_thickness_cm = r_outer_cm - r_core_cm

        # ---- Shell coverage: 1 = intact shell, 0 = core fully exposed ----
        # A hard D2 -> Ne switch would make the RHS discontinuous, because
        # the shell hits zero at FINITE ablation rate (r_outer stays large,
        # propped up by the core beneath it). Ramping over the last
        # breach_cm of thickness keeps the RHS continuous and drives the
        # shell inventory to zero exponentially instead of overshooting
        # negative. This is numerics, not physics: the ramp spans microns
        # on a millimetre-scale pellet.
        shell_coverage = 0.0
        if N_shell > _PELLET_N_FLOOR:
            shell_coverage = min(shell_thickness_cm / self.breach_cm, 1.0)

        deposits: dict[str, float] = {}
        dydt = np.zeros(self.n_state)

        if shell_coverage > 0.0:
            # ---- Phase 1: D2 shell ablating at the OUTER surface (X = 1) --
            lambda_D2 = _PARKS_LAMBDA_A + np.tan(_PARKS_LAMBDA_B)  # 39.0 g/s
            mass_rate = (
                shell_coverage
                * lambda_D2
                * plasma_factor
                * (r_outer_cm / 0.2) ** (4.0 / 3.0)
            )  # [g/s]
            molecule_rate = (
                (mass_rate / _W_MOL["D2"]) * _N_AVOGADRO * 1.0e-3 / 1.0e20
            )  # [1e20 molecules/ms]
            dydt[self._I_SHELL] = -molecule_rate
            deposits["H"] = 2.0 * molecule_rate  # one D2 -> two D atoms

        if shell_coverage < 1.0 and N_core > _PELLET_N_FLOOR:
            # ---- Phase 2: exposed Ne core at the CORE surface (X = 0) -----
            # Note the radius: the core ablates at r_core, not r_outer. Using
            # the outer radius here would overestimate the Ne rate by
            # (r_outer/r_core)^(4/3) for as long as any shell remains.
            mass_rate = (
                (1.0 - shell_coverage)
                * _PARKS_LAMBDA_A
                * plasma_factor
                * (r_core_cm / 0.2) ** (4.0 / 3.0)
            )  # [g/s]
            atom_rate = (
                (mass_rate / _W_MOL["Ne"]) * _N_AVOGADRO * 1.0e-3 / 1.0e20
            )  # [1e20 atoms/ms]
            dydt[self._I_CORE] = -atom_rate
            deposits["Ne"] = atom_rate

        return deposits, dydt

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(D2 shell={self.N0_1e20[self._I_SHELL]:g}e20, "
            f"Ne core={self.N0_1e20[self._I_CORE]:g}e20, "
            f"rp0={self.rp0_cm*10:.2f} mm, rcore0={self.rcore0_cm*10:.2f} mm, "
            f"t_start={self.t_start} ms)"
        )


# ---------------------------------------------------------------------------
# ---- Shattered pellet injection (SPI)
# ---------------------------------------------------------------------------
class SPI(Pellet):
    """
    Shattered pellet injection: a train of fragments from a single shattered
    D2/Ne (or mixed) pellet, arriving at the plasma staggered in time by
    their velocity spread, each ablating independently by the Parks NGS law.

    Fragment sizes (Parks 2016 / Mott & Linfoot 1943)
    -------------------------------------------------
    Fragment diameters follow the statistical-fragmentation distribution

        f(d) = beta^2 * d * K0(beta * d)

    (K0 = modified Bessel function of the second kind), the same form used
    by DREAM/INDEX/JOREK SPI modelling. Published implementations set beta
    from shatter-impact material correlations (Gebhart 2020); here beta is
    instead calibrated to a target fragment count via the analytic mean
    fragment volume <V> = 3*pi^2/(4*beta^3), i.e.
    beta = (3*pi^2*N_frag / (4*V_pellet))^(1/3). Fragments are sampled
    sequentially (inverse-CDF, F(d) = 1 - beta*d*K1(beta*d)) until the
    pellet volume is exhausted and the last fragment is shrunk to conserve
    volume exactly — so the realised count varies around ``N_frag`` with
    the random seed, as in the reference codes. ``size_dist='equal'``
    instead gives exactly N_frag identical fragments.

    Velocities and arrival times
    ----------------------------
    Fragment speeds are Normal(v_mean, dv_frac*v_mean), uncorrelated with
    size (standard assumption; spread ~20% per AUG measurements), clipped
    at 0.1*v_mean. Fragment k arrives at the plasma at

        t_k = t_shatter + 1e3 * L_flight / v_k   [ms]

    so faster fragments arrive first and the plume spans a finite time.
    Each fragment's ablation switches on over a short linear ramp
    ``dt_ramp`` (numerical smoothing of the arrival discontinuity).

    ODE state (aux block): y[k] = remaining molecules of fragment k
    [1e20], one slot per fragment; composition is uniform across fragments
    and constant under congruent ablation, so per-species inventories per
    fragment would be redundant state.

    All randomness is drawn from a seeded generator (default seed=0), so a
    given config is exactly reproducible; vary ``seed`` for
    sensitivity/ensemble studies.

    Limitations: 0-D — fragments see the volume-averaged plasma from their
    arrival time onward and ablate until consumed (no trajectory, no
    flight-through-and-exit, no plasmoid drift).

    Parameters
    ----------
    species       : str or list from {'D2', 'Ne'} (as Pellet).
    V_TorrL / N_1e20 : total pellet inventory per species (as Pellet).
    N_frag        : target number of fragments. Default 30.
    v_mean        : mean fragment speed [m/s]. Default 200.
    dv_frac       : fractional velocity spread (sigma/v_mean). Default 0.2.
    L_flight      : shatter-point-to-plasma distance [m]. Default 1.0.
    t_shatter     : shatter time [ms]. Default 0.
    size_dist     : 'parks' (K0 distribution) or 'equal'. Default 'parks'.
    seed          : RNG seed for sizes and velocities. Default 0.
    dt_ramp       : per-fragment turn-on ramp [ms]. Default 0.02.
    T_K, torrL_to_1e20 : as Pellet/MGI.

    Examples
    --------
    >>> spi = SPI(['Ne'], V_TorrL=[65.0], N_frag=30, v_mean=200.0,
    ...           L_flight=1.0, t_shatter=2.0)
    >>> spi.n_state                 # realised fragment count
    >>> spi.t_arrive_ms             # staggered arrival times [ms]
    """

    def __init__(
        self,
        species,
        V_TorrL=None,
        N_1e20=None,
        N_frag: int = 30,
        v_mean: float = 200.0,
        dv_frac: float = 0.2,
        L_flight: float = 1.0,
        t_shatter: float = 0.0,
        size_dist: str = "parks",
        seed: int = 0,
        dt_ramp: float = 0.02,
        T_K: float = 293.15,
        torrL_to_1e20: float | None = None,
    ) -> None:
        # Parent parses/validates species + total inventory, sets
        # N0_1e20 (per species), pellet_species, rp0_cm (intact pellet).
        super().__init__(
            species,
            V_TorrL=V_TorrL,
            N_1e20=N_1e20,
            t_start=t_shatter,
            T_K=T_K,
            torrL_to_1e20=torrL_to_1e20,
        )
        if N_frag < 1:
            raise ValueError("N_frag must be >= 1")
        if not (0.0 <= dv_frac < 1.0):
            raise ValueError("dv_frac must be in [0, 1)")
        if v_mean <= 0 or L_flight < 0 or dt_ramp <= 0:
            raise ValueError("v_mean, dt_ramp must be > 0; L_flight >= 0")
        if size_dist not in ("parks", "equal"):
            raise ValueError("size_dist must be 'parks' or 'equal'")

        self.t_shatter = float(t_shatter)
        self.v_mean = float(v_mean)
        self.dv_frac = float(dv_frac)
        self.L_flight = float(L_flight)
        self.dt_ramp = float(dt_ramp)
        self.size_dist = size_dist
        self.seed = int(seed)

        # ---- Fixed mixture properties (congruent ablation) ---------------
        # Every fragment has the pellet's composition, and congruent ablation
        # preserves it, so these three quantities are constants of the run.
        # Computing them once here is why SPI.ablate only has to vary the
        # per-fragment radius.
        N_total_1e20 = float(self.N0_1e20.sum())  # molecules [1e20]
        self._molecular_fraction = self.N0_1e20 / N_total_1e20  # per species

        # Parks' X: the D2 MOLECULAR fraction (0 for pure Ne, 1 for pure D2).
        self._D2_fraction = 0.0
        for frac_i, sym in zip(self._molecular_fraction, self.pellet_species):
            if sym == "D2":
                self._D2_fraction = float(frac_i)

        # Mean molecular weight [g/mol], for mass rate -> molecular rate.
        self._mean_molecular_weight = float(
            sum(
                frac_i * _W_MOL[sym]
                for frac_i, sym in zip(self._molecular_fraction, self.pellet_species)
            )
        )

        # Solid volume occupied by 1e20 molecules of the mixture [cm^3],
        # assuming ideal volume mixing. Converts inventory <-> radius.
        self._volume_per_1e20_cm3 = float(
            sum(
                frac_i * 1e20 / _N_AVOGADRO * _W_MOL[sym] / _RHO_SOLID[sym]
                for frac_i, sym in zip(self._molecular_fraction, self.pellet_species)
            )
        )

        # One generator seeds both sizes and velocities, so a given seed
        # reproduces the whole fragment plume exactly.
        rng = np.random.default_rng(self.seed)

        # ---- Fragment sizes [1e20 molecules per fragment] -----------------
        # 'equal' gives exactly N_frag identical fragments, which is the
        # right choice when benchmarking against a paper that assumed a
        # monodisperse plume. 'parks' samples the real size distribution and
        # so returns a count NEAR but not equal to N_frag.
        if size_dist == "equal":
            fragment_inventory = np.full(int(N_frag), N_total_1e20 / int(N_frag))
        else:
            fragment_inventory = self._sample_parks_fragments(
                N_total_1e20, int(N_frag), rng
            )
        self._fragment_N0_1e20 = fragment_inventory
        self.n_state = len(fragment_inventory)

        # ---- Velocities and arrival times ---------------------------------
        # Speeds are Normal and uncorrelated with fragment size (the standard
        # assumption; AUG measurements give ~20% spread). The floor at
        # 0.1*v_mean keeps the Normal tail from producing a zero or negative
        # speed, which would put an infinite arrival time into t_arrive_ms.
        speed = rng.normal(self.v_mean, self.dv_frac * self.v_mean, self.n_state)
        speed = np.maximum(speed, 0.1 * self.v_mean)

        # Flight time converted m/(m/s) = s -> ms. Faster fragments arrive
        # first, so the plume is staggered over a finite window.
        arrival_time = self.t_shatter + 1.0e3 * self.L_flight / speed  # [ms]

        # Sort everything by arrival so ablate() can early-return on
        # t < t_arrive_ms[0] and first_light_time() is just element 0.
        order = np.argsort(arrival_time)
        self.v_frag = speed[order]
        self.t_arrive_ms = arrival_time[order]
        self._fragment_N0_1e20 = self._fragment_N0_1e20[order]

    # ------------------------------------------------------------------
    def _sample_parks_fragments(self, N_tot: float, N_frag_target: int, rng) -> np.ndarray:
        """Sample fragment inventories [1e20 molecules] from the Parks /
        Mott-Linfoot size distribution.

            f(d) = beta^2 * d * K0(beta * d)        (K0 = modified Bessel, 2nd kind)
            F(d) = 1 - (beta * d) * K1(beta * d)    (its CDF)

        ``beta`` is the single shape parameter and sets the scale. Published
        SPI codes fix it from shatter-impact correlations (Gebhart 2020);
        here it is calibrated instead so the MEAN fragment volume yields the
        requested count, using the analytic result

            <V_frag> = (pi/6) <d^3> = 3*pi^2 / (4*beta^3)

        Fragments are drawn one at a time until the pellet volume runs out,
        so the REALISED count varies around N_frag_target with the seed --
        the same behaviour as the reference codes. Use size_dist='equal' if
        you need an exact count.
        """
        from scipy.special import k1
        from scipy.optimize import brentq

        V_total_cm3 = N_tot * self._volume_per_1e20_cm3
        beta = (3.0 * np.pi**2 * N_frag_target / (4.0 * V_total_cm3)) ** (1.0 / 3.0)

        def sample_diameter():
            """Inverse-CDF draw. Solves F(d) = u for d, in the scaled
            variable x = beta*d so the bracket is beta-independent."""
            u = rng.uniform()
            residual = lambda x: x * k1(x) - (1.0 - u)
            return brentq(residual, 1e-12, 60.0) / beta

        fragment_volumes = []
        volume_used = 0.0
        while volume_used < V_total_cm3:
            diameter = sample_diameter()
            volume = np.pi / 6.0 * diameter**3
            if volume_used + volume > V_total_cm3:
                # Shrink the last fragment so total volume is conserved
                # exactly -- mass conservation beats distribution fidelity
                # for a single fragment.
                volume = V_total_cm3 - volume_used
            fragment_volumes.append(volume)
            volume_used += volume

        return np.asarray(fragment_volumes) / self._volume_per_1e20_cm3

    # ------------------------------------------------------------------
    def state0(self) -> list:
        """Initial aux values: remaining molecules per fragment [1e20]."""
        return list(self._fragment_N0_1e20)

    def _frag_radius_cm(self, y: np.ndarray) -> np.ndarray:
        """Per-fragment radii [cm] from remaining inventories (vectorized)."""
        vol = np.maximum(y, 0.0) * self._volume_per_1e20_cm3
        return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    def ablate(self, t: float, state: dict | None, y) -> tuple[dict, np.ndarray]:
        """Summed Parks ablation of all fragments that have arrived."""
        zero = ({}, np.zeros(self.n_state))
        if state is None or t < self.t_arrive_ms[0]:
            return zero

        inventory = np.maximum(np.asarray(y, dtype=float), 0.0)
        if inventory.sum() <= _PELLET_N_FLOOR:
            return zero

        Te = float(state.get("Te", 0.0))  # [eV]
        ne = float(state.get("ne", 0.0))  # [cm^-3]
        if Te <= 0.0 or ne <= 0.0:
            return zero

        # ---- Arrival gate, one value per fragment ------------------------
        # Fragment k does nothing until t reaches its arrival time, then
        # ramps to full ablation over dt_ramp. The ramp is purely numerical:
        # a step function at each of ~30 arrival times would make the RHS
        # discontinuous 30 times and force BDF to restart at each one.
        arrival_ramp = np.clip(
            (t - self.t_arrive_ms) / self.dt_ramp, 0.0, 1.0
        )  # (n_state,)

        # ---- Parks NGS, applied per fragment -----------------------------
        # Composition is uniform across fragments and constant under
        # congruent ablation, so lambda and the mean molecular weight are
        # computed once in __init__ and only the radius varies per fragment.
        plasma_factor = (Te / 2000.0) ** (5.0 / 3.0) * (ne / 1.0e14) ** (1.0 / 3.0)
        lambda_g_per_s = _PARKS_LAMBDA_A + np.tan(_PARKS_LAMBDA_B * self._D2_fraction)
        fragment_radius_cm = self._frag_radius_cm(inventory)  # (n_state,)

        mass_ablation_rate = (
            arrival_ramp
            * lambda_g_per_s
            * plasma_factor
            * (fragment_radius_cm / 0.2) ** (4.0 / 3.0)
        )  # [g/s] per fragment
        molecule_rate = (
            (mass_ablation_rate / self._mean_molecular_weight)
            * _N_AVOGADRO
            * 1.0e-3
            / 1.0e20
        )  # [1e20 molecules/ms] per fragment

        dydt = -molecule_rate
        total_molecule_rate = float(molecule_rate.sum())

        # Split the plume's total by the (fixed) mixture composition.
        deposits: dict[str, float] = {}
        for frac_i, sym in zip(self._molecular_fraction, self.pellet_species):
            plasma_sym, atoms_per_molecule = _to_plasma_species(sym)
            deposits[plasma_sym] = (
                deposits.get(plasma_sym, 0.0)
                + frac_i * total_molecule_rate * atoms_per_molecule
            )
        return deposits, dydt

    # ------------------------------------------------------------------
    def injection_history(self, y_traj: np.ndarray, tV: np.ndarray) -> dict:
        """Delivery vs. time from the fragment-inventory trace: the plume's
        total ablated molecules split by the (constant) composition."""
        y = np.maximum(np.atleast_2d(np.asarray(y_traj, dtype=float)), 0.0)
        tV = np.asarray(tV, dtype=float)
        N_tot = float(self._fragment_N0_1e20.sum())
        ablated = np.maximum(N_tot - y.sum(axis=0), 0.0)  # [1e20 molecules]
        ablated = np.maximum.accumulate(ablated)
        out: dict = {}
        for fi, sp in zip(self._molecular_fraction, self.pellet_species):
            psym, apm = _to_plasma_species(sp)
            cum = fi * ablated * apm
            rate = np.gradient(cum, tV, edge_order=1) if len(tV) > 1 else np.zeros_like(cum)
            if psym in out:
                out[psym]["rate"] = out[psym]["rate"] + rate
                out[psym]["cumulative"] = out[psym]["cumulative"] + cum
            else:
                out[psym] = {"rate": rate, "cumulative": cum}
        return out

    # ------------------------------------------------------------------
    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return float(self.t_arrive_ms[0])

    def __repr__(self) -> str:
        qty = "+".join(f"{s}={n:g}e20" for s, n in zip(self.pellet_species, self.N0_1e20))
        return (
            f"{type(self).__name__}({qty}, Nfrag={self.n_state} ({self.size_dist}), "
            f"v={self.v_mean:g}±{self.dv_frac*self.v_mean:g} m/s, "
            f"arrivals {self.t_arrive_ms[0]:.2f}–{self.t_arrive_ms[-1]:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# ---- Graded cryogenic shell pellet (continuous radial composition)
# ---------------------------------------------------------------------------
# Built-in profile shapes. Each takes the normalized radius s = r/R0 in [0, 1]
# and an interface position s_i, and returns the local Ne MOLECULAR fraction.
# All are written Ne-rich at the centre (s -> 0) and D2-rich at the surface
# (s -> 1), matching the physical shell-pellet arrangement.
def _profile_tanh(s, s_i, width):
    """Smooth graded interface. width -> 0 recovers a sharp shell/core."""
    return 0.5 * (1.0 - np.tanh((s - s_i) / width))


def _profile_linear(s, s_i, width):
    """Linear ramp of total extent ``width``, centred on s_i."""
    return np.clip((s_i + 0.5 * width - s) / width, 0.0, 1.0)


def _profile_sharp(s, s_i, width):
    """Step: pure Ne inside s_i, pure D2 outside. The GradedCSP limit that
    reproduces the two-layer CSP. Numerically stiff -- prefer 'tanh' with a
    small width unless you specifically want the discontinuity."""
    return np.where(np.asarray(s) < s_i, 1.0, 0.0)


_GRADED_PROFILES = {
    "tanh": _profile_tanh,
    "linear": _profile_linear,
    "sharp": _profile_sharp,
}


class GradedCSP(Pellet):
    """
    Cryogenic shell pellet with a CONTINUOUS radial composition profile.

    Where ``CSP`` has a pure D2 shell around a pure Ne core and hands off
    between them, this class carries a Ne molecular fraction x_Ne(r) that
    varies smoothly with radius. The surface composition therefore evolves
    as the pellet ablates: an initially D2-dominated plume becomes
    progressively Ne-rich as the surface recedes toward the core.

    WHY THE STATE IS ONE SLOT
    -------------------------
    Composition is a prescribed function of radius, so the remaining
    inventory of each species is fully determined by how far the surface has
    receded. There is nothing to track per species. The single aux slot is
    the pellet's remaining VOLUME [cm^3], not its radius, because

        G ~ r^(4/3)   =>   dV/dt ~ -V^(4/9)      (smooth, -> 0 as V -> 0)
                           dr/dt ~ -r^(-2/3)     (diverges as r -> 0)

    so volume-as-state lets the pellet vanish smoothly while radius-as-state
    hands the integrator a singularity in the final moments.

    ABLATION
    --------
    At each step the surface radius is r = (3V/4pi)^(1/3), the normalized
    radius is s = r/R0, and the LOCAL composition x_Ne(s) sets:

      * the Parks coefficient, evaluated at the ablating surface,
            lambda = 27.0837 + tan(1.48709 * X),   X = 1 - x_Ne(s)
      * the local mean molecular weight, mass rate -> molar rate
      * the local solid density (ideal volume mixing), molar rate -> dV/dt
      * the split of the delivered atoms between H and Ne

    The Parks mass rate itself is unchanged:
        G [g/s] = lambda * (Te/2000)^(5/3) * (r/0.2)^(4/3) * (ne/1e14)^(1/3)

    HOW THE PROFILE IS NORMALIZED
    -----------------------------
    The shape is specified in normalized radius, so it fixes the Ne:D2 molar
    RATIO but not the absolute size. Construction therefore does two things:

      1. Solve the interface position ``s_i`` so the profile's implied
         Ne:D2 ratio matches the requested inventories exactly.
      2. Scale R0 so the absolute inventories match.

    Both are exact for the built-in shapes. A user-supplied callable has no
    free parameter to solve, so its implied ratio is used as-is and the
    requested inventories are rescaled to match it; the realised values are
    always available in ``N0_1e20``.

    Parameters
    ----------
    D2_TorrL, Ne_TorrL : gas-equivalent inventories [Torr-L]. Give this pair
                    OR the 1e20 pair.
    D2_1e20, Ne_1e20   : inventories [1e20 molecules / atoms].
    profile       : 'tanh' (default), 'linear', 'sharp', or a callable
                    f(s) -> Ne molecular fraction for s in [0, 1].
    grade_width   : width of the composition transition in normalized
                    radius. Default 0.15. Small values approach ``CSP``.
    t_start       : time the pellet enters the plasma [ms]. Default 0.
    T_K, torrL_to_1e20 : as Pellet/MGI.
    n_quad        : radial quadrature points for the profile integrals.
                    Default 2001.

    Attributes
    ----------
    R0_cm         : initial outer radius [cm]
    s_interface   : solved interface position (normalized radius)
    N0_1e20       : realised [D2, Ne] inventories [1e20]

    Examples
    --------
    >>> p = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0, grade_width=0.2)
    >>> p.surface_composition(p.state0()[0])   # Ne fraction at the surface
    """

    _I_D2, _I_NE = 0, 1  # ordering of N0_1e20 (NOT the aux block)

    def __init__(
        self,
        D2_TorrL: float | None = None,
        Ne_TorrL: float | None = None,
        D2_1e20: float | None = None,
        Ne_1e20: float | None = None,
        profile="tanh",
        grade_width: float = 0.15,
        t_start: float = 0.0,
        T_K: float = 293.15,
        torrL_to_1e20: float | None = None,
        n_quad: int = 2001,
    ) -> None:
        torr_pair = (D2_TorrL is not None, Ne_TorrL is not None)
        n20_pair = (D2_1e20 is not None, Ne_1e20 is not None)
        if any(torr_pair) and any(n20_pair):
            raise ValueError("Give either the *_TorrL pair or the *_1e20 pair, not both")

        if all(torr_pair):
            super().__init__(
                ["D2", "Ne"],
                V_TorrL=[D2_TorrL, Ne_TorrL],
                t_start=t_start,
                T_K=T_K,
                torrL_to_1e20=torrL_to_1e20,
            )
        elif all(n20_pair):
            super().__init__(["D2", "Ne"], N_1e20=[D2_1e20, Ne_1e20], t_start=t_start)
        else:
            raise ValueError(
                "GradedCSP needs both inventories (D2_TorrL + Ne_TorrL, "
                "or D2_1e20 + Ne_1e20)"
            )

        if not (0.0 < grade_width <= 1.0):
            raise ValueError("grade_width must be in (0, 1]")
        self.grade_width = float(grade_width)
        self.n_quad = int(n_quad)

        # ---- Resolve the profile shape -----------------------------------
        if callable(profile):
            self.profile_name = getattr(profile, "__name__", "custom")
            self._shape = lambda s, s_i: np.clip(
                np.asarray(profile(s), dtype=float), 0.0, 1.0
            )
            self._solvable = False
        else:
            key = str(profile).lower()
            if key not in _GRADED_PROFILES:
                raise ValueError(
                    f"Unknown profile {profile!r}. "
                    f"Options: {sorted(_GRADED_PROFILES)} or a callable."
                )
            self.profile_name = key
            shape_fn = _GRADED_PROFILES[key]
            self._shape = lambda s, s_i: np.clip(
                shape_fn(s, s_i, self.grade_width), 0.0, 1.0
            )
            self._solvable = True

        # ---- Radial quadrature grid --------------------------------------
        self._s_grid = np.linspace(0.0, 1.0, self.n_quad)

        # ---- Solve the interface position for the requested ratio ---------
        # Per unit R0^3, the molar content of each species is
        #     I_k = 4*pi * integral_0^1 s^2 * x_k(s) * n_mol(s) ds
        # so I_Ne/I_D2 depends only on the SHAPE. Solve s_i to match.
        N_D2_req = float(self.N0_1e20[self._I_D2])
        N_Ne_req = float(self.N0_1e20[self._I_NE])
        target_ratio = N_Ne_req / max(N_D2_req, 1e-300)

        if self._solvable:
            self.s_interface = self._solve_interface(target_ratio)
        else:
            self.s_interface = 0.5  # unused by a custom callable

        # ---- Scale R0 so the absolute inventories match -------------------
        I_D2, I_Ne = self._shape_integrals(self.s_interface)
        # Work in 1e20 particles to match the inventory convention.
        I_D2_1e20 = I_D2 * _N_AVOGADRO / 1e20
        I_Ne_1e20 = I_Ne * _N_AVOGADRO / 1e20

        if self._solvable:
            # Ratio already matches, so either species fixes R0.
            R0_cubed = N_D2_req / I_D2_1e20 if I_D2_1e20 > 0 else N_Ne_req / I_Ne_1e20
        else:
            # A custom shape has no free parameter: preserve the TOTAL
            # particle count and accept the shape's own split.
            R0_cubed = (N_D2_req + N_Ne_req) / (I_D2_1e20 + I_Ne_1e20)

        self.R0_cm = float(R0_cubed ** (1.0 / 3.0))
        self.V0_cm3 = 4.0 / 3.0 * np.pi * self.R0_cm**3

        # Realised inventories (exact for built-ins; the shape's split for
        # a custom callable).
        self.N0_1e20 = np.array([I_D2_1e20 * R0_cubed, I_Ne_1e20 * R0_cubed], dtype=float)
        self.rp0_cm = self.R0_cm

        # ---- Cumulative inventory vs radius, for injection_history --------
        # C_k(s) = particles of species k inside normalized radius s [1e20].
        x_Ne = self._shape(self._s_grid, self.s_interface)
        integrand_D2 = (
            4.0 * np.pi * self._s_grid**2 * (1.0 - x_Ne) * self._molar_density(x_Ne)
        )
        integrand_Ne = 4.0 * np.pi * self._s_grid**2 * x_Ne * self._molar_density(x_Ne)
        scale = R0_cubed * _N_AVOGADRO / 1e20
        self._cum_D2 = _cumtrapz0(integrand_D2, self._s_grid) * scale
        self._cum_Ne = _cumtrapz0(integrand_Ne, self._s_grid) * scale

        # One aux slot: remaining volume [cm^3].
        self.n_state = 1
        self.species = ["H", "Ne"]

    # ------------------------------------------------------------------
    # ---- Profile helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _molar_density(x_Ne):
        """Molar density [mol/cm^3] of the local mixture, ideal volume mixing.

        Molar volume is the composition-weighted sum of each component's own
        molar volume W/rho, so the density is its reciprocal.
        """
        x_Ne = np.asarray(x_Ne, dtype=float)
        x_D2 = 1.0 - x_Ne
        molar_volume = (
            x_D2 * _W_MOL["D2"] / _RHO_SOLID["D2"] + x_Ne * _W_MOL["Ne"] / _RHO_SOLID["Ne"]
        )  # [cm^3/mol]
        return 1.0 / molar_volume

    def _shape_integrals(self, s_i: float) -> tuple[float, float]:
        """Molar content per unit R0^3 [mol/cm^3 * cm^3], per species."""
        s = self._s_grid
        x_Ne = self._shape(s, s_i)
        n_mol = self._molar_density(x_Ne)
        common = 4.0 * np.pi * s**2 * n_mol
        I_D2 = float(np.trapezoid(common * (1.0 - x_Ne), s))
        I_Ne = float(np.trapezoid(common * x_Ne, s))
        return I_D2, I_Ne

    def _solve_interface(self, target_ratio: float) -> float:
        """Find s_i such that the profile's Ne:D2 molar ratio hits the target.

        The ratio increases monotonically with s_i (a bigger Ne core), so a
        bracketed root find is safe.
        """
        from scipy.optimize import brentq

        def residual(s_i):
            I_D2, I_Ne = self._shape_integrals(s_i)
            return I_Ne / max(I_D2, 1e-300) - target_ratio

        lo, hi = 1e-6, 1.0 - 1e-6
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo > 0.0:
            return lo  # requested Ne fraction below what the shape can reach
        if f_hi < 0.0:
            return hi  # requested Ne fraction above what the shape can reach
        return float(brentq(residual, lo, hi, xtol=1e-10))

    def surface_composition(self, volume_cm3: float) -> float:
        """Ne molecular fraction at the surface, given remaining volume."""
        r = (3.0 * max(float(volume_cm3), 0.0) / (4.0 * np.pi)) ** (1.0 / 3.0)
        s = min(r / self.R0_cm, 1.0)
        return float(self._shape(np.array([s]), self.s_interface)[0])

    # ------------------------------------------------------------------
    def state0(self) -> list:
        """Initial aux value: remaining pellet volume [cm^3]."""
        return [self.V0_cm3]

    def radius_cm(self, y) -> float:
        """Outer radius [cm]. Accepts the aux volume, or (during Pellet's
        __init__, before the profile exists) a per-species inventory."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if getattr(self, "n_state", 0) == 1 and y.size == 1:
            return float((3.0 * max(y[0], 0.0) / (4.0 * np.pi)) ** (1.0 / 3.0))
        return super().radius_cm(y)

    # ------------------------------------------------------------------
    def ablate(self, t: float, state: dict | None, y) -> tuple[dict, np.ndarray]:
        """Parks ablation with the composition taken at the ablating surface."""
        zero = ({}, np.zeros(self.n_state))
        if state is None or t < self.t_start:
            return zero

        volume_cm3 = float(np.maximum(np.asarray(y, dtype=float).ravel()[0], 0.0))
        if volume_cm3 <= 0.0:
            return zero

        radius_cm = (3.0 * volume_cm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        if radius_cm <= 0.0:
            return zero

        Te = float(state.get("Te", 0.0))  # [eV]
        ne = float(state.get("ne", 0.0))  # [cm^-3]
        if Te <= 0.0 or ne <= 0.0:
            return zero

        # ---- Local composition at the ablating surface -------------------
        s_surface = min(radius_cm / self.R0_cm, 1.0)
        x_Ne = float(self._shape(np.array([s_surface]), self.s_interface)[0])
        x_D2 = 1.0 - x_Ne

        # Parks X is the D2 molecular fraction, evaluated HERE rather than
        # globally -- this is what makes the plume composition evolve.
        lambda_g_per_s = _PARKS_LAMBDA_A + np.tan(_PARKS_LAMBDA_B * x_D2)

        mass_ablation_rate = (
            lambda_g_per_s
            * (Te / 2000.0) ** (5.0 / 3.0)
            * (radius_cm / 0.2) ** (4.0 / 3.0)
            * (ne / 1.0e14) ** (1.0 / 3.0)
        )  # [g/s]

        # ---- Local material properties -----------------------------------
        mean_molecular_weight = x_D2 * _W_MOL["D2"] + x_Ne * _W_MOL["Ne"]  # [g/mol]
        molar_density = float(self._molar_density(x_Ne))  # [mol/cm^3]
        local_solid_density = mean_molecular_weight * molar_density  # [g/cm^3]

        # ---- Rates --------------------------------------------------------
        # Volume recession: mass rate / local density, s -> ms.
        dV_dt = -(mass_ablation_rate / local_solid_density) * 1.0e-3  # [cm^3/ms]

        # Molar rate -> particles: [mol/s] * N_A * 1e-3 [s/ms] / 1e20
        molecule_rate = (
            (mass_ablation_rate / mean_molecular_weight) * _N_AVOGADRO * 1.0e-3 / 1.0e20
        )  # [1e20 molecules/ms], mixture total

        # Split by the local composition. One D2 molecule -> two D atoms.
        deposits = {
            "H": 2.0 * x_D2 * molecule_rate,
            "Ne": x_Ne * molecule_rate,
        }
        return deposits, np.array([dV_dt])

    # ------------------------------------------------------------------
    def injection_history(self, y_traj: np.ndarray, tV: np.ndarray) -> dict:
        """Delivery vs time, reconstructed from the volume trace.

        Everything outside the current surface radius has been ablated, so
        the cumulative delivery is the profile integral from the current
        radius out to R0 -- read off the precomputed tables.
        """
        volume = np.maximum(np.asarray(y_traj, dtype=float).ravel(), 0.0)
        tV = np.asarray(tV, dtype=float)

        radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        s = np.clip(radius / self.R0_cm, 0.0, 1.0)

        remaining_D2 = np.interp(s, self._s_grid, self._cum_D2)
        remaining_Ne = np.interp(s, self._s_grid, self._cum_Ne)

        cum_D2 = np.maximum.accumulate(np.maximum(self._cum_D2[-1] - remaining_D2, 0.0))
        cum_Ne = np.maximum.accumulate(np.maximum(self._cum_Ne[-1] - remaining_Ne, 0.0))

        cum_H = 2.0 * cum_D2  # D2 -> 2 atoms
        out = {}
        for sym, cum in (("H", cum_H), ("Ne", cum_Ne)):
            rate = np.gradient(cum, tV, edge_order=1) if len(tV) > 1 else np.zeros_like(cum)
            out[sym] = {"rate": rate, "cumulative": cum}
        return out

    # ------------------------------------------------------------------
    def total_atoms_1e20(self) -> dict:
        return {
            "H": 2.0 * float(self.N0_1e20[self._I_D2]),
            "Ne": float(self.N0_1e20[self._I_NE]),
        }

    def __repr__(self) -> str:
        return (
            f"GradedCSP(D2={self.N0_1e20[self._I_D2]:g}e20, "
            f"Ne={self.N0_1e20[self._I_NE]:g}e20, "
            f"profile={self.profile_name}(w={self.grade_width:g}, "
            f"s_i={self.s_interface:.3f}), "
            f"R0={self.R0_cm*10:.2f} mm, t_start={self.t_start} ms)"
        )


def _cumtrapz0(values, x):
    """Cumulative trapezoid starting at zero, same length as the input."""
    out = np.zeros_like(np.asarray(values, dtype=float))
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    return out


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
