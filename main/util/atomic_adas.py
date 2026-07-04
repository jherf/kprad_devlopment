# atomic.py
"""
Aurora / Open-ADAS backed atomic rate class

Charge-state convention
-----------------------
Single-rate methods (Sion, Srec, Prad) take Z as a 1-based index:
    Z = 1           neutral
    Z = Zmax + 1    fully stripped
Vectorized lookup (all_rates) returns 0-based arrays of length Zmax + 1
where index i corresponds to charge state i (0 = neutral)

ADAS block convention (important):
    scd[k], plt[k]  belong to ionizing stage k        (0-based charge k)
    acd[k], prb[k]  belong to the recombining stage   (0-based charge k+1)
so Sion/line-radiation index with Z-1 while Srec/continuum index with Z-2.

The following parameters are loaded from ADAS:
https://open.adas.ac.uk/terminology
scd     : Effective ionization coefficients
acd     : Effection recombination coefficients
plt     : Effective low-level line power
prb     : Effective recombination + Bremsstrahlung power
- Below are not implimented -
ccd     : Effective charge exchange recombination
prc     : Effective charge exchange power


Units
-----
Sion, Srec      : cm^3 / s
Prad            : eV cm^3 / s
"""

import numpy as np
import aurora
from scipy.interpolate import RegularGridInterpolator
from main.util.constants import _EE, _MIN_NE, _MIN_TE

# ---- Atomic tables --------------------------------------------------
_RATE_KEYS = ["scd", "acd", "plt", "prb"]
_PERIODIC = {"H": 1, "He": 2, "C": 6, "Ne": 10, "Ar": 18, "W": 74}
_IONIZATION_ENERGIES = {
    "H": [13.6],
    "C": [11.26, 24.384, 47.888, 64.5, 392.1, 490.0],
    "Ne": [21.56, 40.96, 63.42, 97.2, 126.2, 157.9, 207.3, 239.1, 1196.0, 1362.0],
    "Ar": [
        15.760,
        27.630,
        40.74,
        59.81,
        75.02,
        91.01,
        124.32,
        143.46,
        422.45,
        478.69,
        538.96,
        618.26,
        686.10,
        755.74,
        854.77,
        918.03,
        4120.87,
        4426.23,
    ],
}
_ADAS_ALIAS = {"D": "H", "D2": "H", "T": "H", "H2": "H", "T2": "H", "DT": "H"}


def _adas_element(sym: str) -> str:
    """Returns the canonical ADAS element symbol for ``sym``."""
    return _ADAS_ALIAS.get(sym, sym)


# ---- Main class --------------------------------------------------
class AuroraRates:
    """
    ADAS atomic-rate lookups for a configuration set of elements

    Rates are pre-interpolated onto a (log10 ne, log10 Te) grid at
    construction; lookups are then cheap RegularGridInterpolator calls.

    Parameters
    ----------
    elements     : sequence of element symbols, e.g. ['H', 'Ne', 'C'].
    ne_range     : (float, float), optional. ne bounds [cm^-3] of the
                   pre-computed grid (interpolation is done in log10 space).
    n_ne         : int, optional. Number of points on the log10(ne) grid.
    Te_range     : (float, float), optional. Te bounds [eV].
    n_Te         : int, optional. Number of points on the log10(Te) grid.

    Attributes
    ----------
    elements       : list of canonical symbols
    Zmax           : dict[str, int] — atomic number per loaded element
    Ei_cumulative  : dict[str, ndarray] — cumulative ionization energy [eV],
                     length Zmax (index k = energy to reach charge k+1)

    Examples
    --------
    >>> rates = AuroraRates(['H', 'C', 'Ne'])
    >>> rates.Sion('Ne', Z=5, ne=1e14, Te=100.0)        # scalar [cm^3/s]
    >>> rates.all_rates('Ne', 1e14, 100.0)              # three (Zmax+1,) arrays
    >>> rates.all_rates('Ne', ne_prof, Te_prof)         # three (Zmax+1, Nr) arrays
    """

    def __init__(
        self,
        elements: list,
        ne_range: tuple = (1e9, 3e15),
        n_ne: int = 30,
        Te_range: tuple = (0.3, 1e5),
        n_Te: int = 120,
    ):

        # ---------------------------------------------------------------------
        # ---- Validate elements
        # ---------------------------------------------------------------------
        if isinstance(elements, str):
            raise TypeError(
                f" `elements` must be a list of symbols, not a single string. "
                f"Use ['{elements}'] if you want just one element"
            )

        if not elements:
            raise ValueError("AuroraRates requires at least one element")

        # ---------------------------------------------------------------------
        # ---- Convert D2, T, etc. to H
        # ---------------------------------------------------------------------
        seen, canon = set(), []
        for s in elements:
            e = _adas_element(s)
            if e not in seen:
                seen.add(e)
                canon.append(e)
        self.elements = canon

        unknown = [s for s in self.elements if s not in _PERIODIC]
        if unknown:
            raise ValueError(
                f"Unknown element(s) {unknown}. Known: {sorted(_PERIODIC)}"
            )
        missing_ei = [s for s in self.elements if s not in _IONIZATION_ENERGIES]
        if missing_ei:
            raise ValueError(
                f"No ionization-energy table for {missing_ei}; add entries to "
                f"_IONIZATION_ENERGIES in atomic.py before using them."
            )

        self._log_ne = np.linspace(np.log10(ne_range[0]), np.log10(ne_range[1]), n_ne)
        self._log_te = np.linspace(np.log10(Te_range[0]), np.log10(Te_range[1]), n_Te)

        # ---- Maximum charge state per elemente (= atomic number)
        self.Zmax = {sym: _PERIODIC[sym] for sym in self.elements}

        # ---- Pre-build the (log_ne, log_Te) rate interpolator for every element
        self._interp = {sym: self._build_interpolators(sym) for sym in self.elements}

        # ---- Cumulative ionization energies, length Zmax
        self.Ei_cumulative = {
            s: np.cumsum(np.asarray(_IONIZATION_ENERGIES[s], dtype=float))
            for s in self.elements
        }

    # ---------------------------------------------------------------------------
    # ---- Internal lookups
    # ---------------------------------------------------------------------------
    def _check_element(self, symbol: str) -> str:
        sym = _adas_element(symbol)
        if sym not in self._interp:
            raise KeyError(
                f"Element {symbol} not loaded. Loaded elements {self.elements}"
            )
        return sym

    @staticmethod
    def _log_points(ne, Te) -> np.ndarray:
        """Floored, stacked log10 (ne, Te) points of shape (N, 2)."""
        ne = np.atleast_1d(np.asarray(ne, dtype=float))
        Te = np.atleast_1d(np.asarray(Te, dtype=float))
        ne, Te = np.broadcast_arrays(ne, Te)
        log_ne = np.log10(np.maximum(ne, _MIN_NE))
        log_Te = np.log10(np.maximum(Te, _MIN_TE))
        return np.column_stack([log_ne.ravel(), log_Te.ravel()])

    def _evaluate_many(
        self, symbol: str, key: str, indx: int, pts: np.ndarray
    ) -> np.ndarray:
        """Evaluate one charge-state interpolator at ``pts`` (N, 2).

        Returns zeros for out-of-range charge-state indices, mirroring the
        'no such transition' convention.
        """
        symbol = _adas_element(symbol)
        table = self._interp[symbol][key]
        if indx < 0 or indx >= len(table):
            return np.zeros(len(pts))
        return 10.0 ** table[indx](pts)

    def _evaluate(self, symbol: str, key: str, indx: int, ne, Te) -> float:
        """Scalar convenience wrapper around _evaluate_many."""
        symbol = _adas_element(symbol)
        return float(
            self._evaluate_many(symbol, key, indx, self._log_points(ne, Te))[0]
        )

    # --------------------------------------------------------------------------
    # ---- Building functions
    # --------------------------------------------------------------------------

    def _build_interpolators(self, symbol: str) -> dict:
        """
        Loads ADAS data for each `symbol` (Ne, H, etc.), returns a dict mapping
        each _RATE_KEY to a list of scipy interpolators (one per charge state),
        each taking (log_ne, log_Te) as an input and returning the rate
        """

        symbol = _adas_element(symbol)

        interpolators = {}
        LTe, Lne = np.meshgrid(self._log_te, self._log_ne, indexing="ij")

        # Get the atomic data
        atom_table = aurora.get_atom_data(symbol, _RATE_KEYS)

        # Interpolate the atomic data along the grid
        for key in _RATE_KEYS:
            r = aurora.interp_atom_prof(
                atom_table[key], Lne, LTe, log_val=False, x_multiply=False
            )  # Format n_Te, Z (atomic number), n_ne

            _, n_charge, _ = r.shape

            # Interpolate in log space
            log_r = np.log10(np.maximum(r, 1e-99))

            interpolators[key] = [
                RegularGridInterpolator(
                    (self._log_ne, self._log_te),
                    log_r[:, iZ, :].T,
                    method="linear",
                    bounds_error=False,
                    fill_value=-99.0,
                )
                for iZ in range(n_charge)
            ]

        return interpolators

    # --------------------------------------------------------------------------
    # ---- Ionization rate, power, recomb. rate, etc.
    # --------------------------------------------------------------------------
    def Sion(self, symbol: str, Z: int, ne: float, Te: float) -> float:
        """Ionization rate coefficient [cm^3/s] for charge state Z (Z=1 neutral)."""
        sym = self._check_element(symbol)
        if Z > self.Zmax[sym]:
            return 0.0
        return self._evaluate(sym, "scd", Z - 1, ne, Te)

    def Srec(self, symbol: str, Z: int, ne: float, Te: float) -> float:
        """Recombination rate coefficient [cm^3/s] for charge state Z."""
        sym = self._check_element(symbol)
        if Z <= 1:
            return 0.0
        return self._evaluate(sym, "acd", Z - 2, ne, Te)

    def Prad(self, symbol: str, Z: int, ne: float, Te: float) -> float:
        """Radiative cooling rate coefficient [eV cm^3 / s] for charge state Z
        (line + recombination + Bremsstrahlung).

        prb follows the acd (recombining-stage) block convention, hence
        index Z-2: the continuum power of stage Z is radiated as it
        recombines toward Z-1. Z=1 (neutral) therefore has line power only;
        Z = Zmax+1 (fully stripped) has continuum power only.
        """
        sym = self._check_element(symbol)
        plt_v = self._evaluate(sym, "plt", Z - 1, ne, Te)
        prb_v = self._evaluate(sym, "prb", Z - 2, ne, Te)
        return float(plt_v + prb_v) / float(_EE)  # W*cm^3 -> eV*cm^3/s

    # --------------------------------------------------------------------------
    # ---- Vectorized lookup ------------------------------------------------------
    # --------------------------------------------------------------------------
    def all_rates(
        self, symbol: str, ne, Te, Ta
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rates for every charge state, vectorized over plasma points.

        Parameters
        ----------
        ne, Te : float or 1-D arrays of shape (Nr,) [cm^-3, eV].
                 Scalars and arrays may be mixed (NumPy broadcasting).

        Ta     : not used, in for compatability with CRETIN all_rates function

        Returns
        -------
        rion, rrec, rrad : ndarrays, 0-based charge-state index along axis 0.
            Shape (Zmax+1,) for scalar input, (Zmax+1, Nr) for array input —
            i.e. directly broadcastable against a species density block of
            shape (Zmax+1, Nr).

            rion[i] : ionization rate coefficient i -> i+1  [cm^3/s],
                      rion[Zmax] = 0 (fully stripped cannot ionize)
            rrec[i] : recombination rate coefficient i -> i-1 [cm^3/s],
                      rrec[0] = 0 (neutral cannot recombine)
            rrad[i] : radiative cooling coefficient of charge state i
                      [eV cm^3 / s]
        """
        sym = self._check_element(symbol)
        Zmax = self.Zmax[sym]
        n = Zmax + 1

        scalar_input = np.isscalar(ne) or np.asarray(ne).ndim == 0
        scalar_input &= np.isscalar(Te) or np.asarray(Te).ndim == 0

        pts = self._log_points(ne, Te)
        Nr = len(pts)

        rion = np.zeros((n, Nr))
        rrec = np.zeros((n, Nr))
        rrad = np.zeros((n, Nr))

        for iZ in range(n):
            if iZ < Zmax:  # fully stripped cannot ionize
                rion[iZ] = self._evaluate_many(sym, "scd", iZ, pts)
            if iZ >= 1:  # neutral cannot recombine
                rrec[iZ] = self._evaluate_many(sym, "acd", iZ - 1, pts)
            # line power (ionizing stage) + continuum power (recombining stage)
            rrad[iZ] = (
                self._evaluate_many(sym, "plt", iZ, pts)
                + self._evaluate_many(sym, "prb", iZ - 1, pts)
            ) / _EE

        if scalar_input:
            return rion[:, 0], rrec[:, 0], rrad[:, 0]
        return rion, rrec, rrad
