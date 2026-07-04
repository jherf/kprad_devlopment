# rate_make_nLTE.py
"""
Loads CRETIN non-LTE rates files via read_rate_nLTE then builds an
interpolator so the rate coefficients can be reconstructed for arbitrary
ne, Te, Ta.

Two back-ends are available (selected with `method=` at construction):

    method="rgi"  (default) : a scipy RegularGridInterpolator per charge state,
                              built on the (log ne, log Te, log Ta) grid in
                              log(rate). Robust, never oscillates, and matches
                              the approach used by AuroraRates in atomic.py.

    method="poly"           : the original nested polynomial fit
                              (degree NTepp-1 in log Te, cubic in log ne and
                              log Ta). Kept for comparison; the high Te degree
                              can ring (Runge) and is ill-conditioned.

Three quantities that are stored:
    ASrad   radiation cooling rate coefficient  Srad/na [eV cm^3 /s]
    ASion   ionization rate coefficient         Sion/ne [cm^3 / s]
    ASrec   recombination rate coefficient      Srec/ne [cm^3 / s]

Charge state indexing is 0-based here (z=0 neutral, ..., z = Z-1 fully stripped)
"""

import numpy as np
import warnings
from scipy.interpolate import RegularGridInterpolator
from main.util.read_rate_nLTE import read_rate_nLTE
from main.util.atomic_adas import _PERIODIC, _IONIZATION_ENERGIES, _ADAS_ALIAS
from main.util.constants import _MIN_NE, _MIN_TE, _MIN_TA
from main.globals import CRETIN_PATH

# Default grid (matches all three MATLAB scripts)
_DEFAULT_neV = [1e10, 1e12, 1e14, 1e16]  # electron densities      [1/cm^3]
_DEFAULT_TaV = [1e10, 1e12, 1e14, 1e16]  # element column densities[1/cm^2]


# Floor used before taking log of a rate coefficient inside the interpolators.
_LOG_FLOOR = 1e-300


def _adas_element(sym: str) -> str:
    """Returns the canonical ADAS element symbol for ``sym``."""
    return _ADAS_ALIAS.get(sym, sym)


class CretinRates:
    """CRETIN atomic-rate lookups, pre-compiled into an interpolator over the
    (log ne, log Te, log Ta) grid.

    Parameters
    ----------
    elements   : sequence of element symbols, e.g. ['H', 'Ne', 'C'].
    method     : {"rgi", "poly"}, optional. Back-end used to reconstruct the
                 rates between grid points. "rgi" (default) builds a scipy
                 RegularGridInterpolator per charge state; "poly" uses the
                 original nested polynomial fit.
    rgi_method : str, optional. Interpolation method passed to
                 RegularGridInterpolator (e.g. "linear", "cubic", "pchip").
                 Applies to all three axes. Default "linear" (robust; ne and
                 Ta only have 4 grid points, so higher orders there can
                 overshoot). Only used when method="rgi".
    NTepp      : int, optional. Number of Te polynomial coefficients
                 (degree NTepp-1). Only used when method="poly". Default 15.
    data_path  : str, optional. Template path to the CRETIN .dat files with
                 {el}, {ne}, {ta} placeholders. Resolution order:
                 this argument > $KPRAD_CRETIN_PATH > built-in default.

    Attributes
    ----------
    elements       : list of canonical symbols
    Zmax           : dict[str, int] -- atomic number per loaded element
    Ei_cumulative  : dict[str, ndarray] -- cumulative ionization energy [eV],
                     length Zmax (index k = energy to reach charge k+1)

    Examples
    --------
    >>> rates = CretinRates(['D2', 'C', 'Ne'])                 # RGI back-end
    >>> rates = CretinRates(['Ne'], method="poly")             # old poly fit
    >>> rates.Sion('Ne', Z=5, ne=1e14, Te=100.0, Ta=1e12)      # scalar [cm^3/s]
    >>> rates.all_rates('Ne', ne=1e14, Te=100.0, Ta=1e12)      # three (Zmax+1,) arrays
    """

    def __init__(
        self,
        elements: list,
        method: str = "rgi",
        rgi_method: str = "linear",
        NTepp: int = 15,
        data_path: str | None = None,
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
            raise ValueError("CretinRates requires at least one element")

        if method not in ("rgi", "poly"):
            raise ValueError(f"method must be 'rgi' or 'poly', got {method!r}")
        self._method = method
        self._rgi_method = rgi_method
        self._NTepp = NTepp
        self._path_template = data_path or CRETIN_PATH

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

        # ---- Maximum charge state per element (= atomic number)
        self.Zmax = {sym: _PERIODIC[sym] for sym in self.elements}

        # ---- Pre-build the rate interpolator for every element
        self._interp = {sym: self._build_fit(sym) for sym in self.elements}

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

    def _exp_tag(self, value):
        """Two-character exponent tag, as the MATLAB sprintf('%0.0e',v)(4:5) produced
        (e.g. 1e10 -> '10', 1e16 -> '16')."""
        return ("%0.0e" % value)[3:5]

    def _polyfit(self, x, y, deg):
        """Preforms a polyfit on x, y with degree = deg"""
        return np.polyfit(x, y, deg)

    def _build_fit(self, symbol):
        """Loads tables, builds the selected interpolator back-end."""
        symbol = _adas_element(symbol)
        ASrad, ASion, ASrec, Te, neV, TaV = self.load_rate_tables(symbol)
        if self._method == "rgi":
            return self.build_rgi_tables(
                ASrad, ASion, ASrec, Te, neV, TaV, rgi_method=self._rgi_method
            )
        return self.fit_rate_tables(
            ASrad, ASion, ASrec, Te, neV, TaV, NTepp=self._NTepp
        )

    @staticmethod
    def _log_points(ne, Te, Ta) -> np.ndarray:
        """Floored, stacked natural-log (ne, Te, Ta) points of shape (N, 3)."""
        ne = np.atleast_1d(np.asarray(ne, dtype=float))
        Te = np.atleast_1d(np.asarray(Te, dtype=float))
        Ta = np.atleast_1d(np.asarray(Ta, dtype=float))
        ne, Te, Ta = np.broadcast_arrays(ne, Te, Ta)
        log_ne = np.log(np.maximum(ne, _MIN_NE))
        log_Te = np.log(np.maximum(Te, _MIN_TE))
        log_Ta = np.log(np.maximum(Ta, _MIN_TA))
        return np.column_stack([log_ne.ravel(), log_Te.ravel(), log_Ta.ravel()])

    # ---------------------------------------------------------------------------
    # ---- Data loaders / fits
    # ---------------------------------------------------------------------------
    def load_rate_tables(self, element):
        """Reads the CRETIN rate tables for the selected element

        Returns
        -------
        ASrad, ASion, ASrec : ndarray, shape (NTa, Nne, NZ, NTe)
            Rate coefficients, floored at `Smin`.
        Te   : ndarray (NTe,)   electron temperatures [eV]
        neV  : ndarray (Nne,)   electron densities     [1/cm^3]
        TaV  : ndarray (NTa,)   column densities       [1/cm^2]
        """

        # Allocated later
        ASrad = ASion = ASrec = None

        neV = np.asarray(_DEFAULT_neV)
        TaV = np.asarray(_DEFAULT_TaV)
        Nne, NTa = len(neV), len(TaV)
        Smin = np.asarray(1e-20)  # Minimum number for taking log of rate coefficients

        Te_ref = None
        for iTa in range(NTa):
            for iNe in range(Nne):
                fname = self._path_template.format(
                    el=element, ne=self._exp_tag(neV[iNe]), ta=self._exp_tag(TaV[iTa])
                )
                ne, na, Lo, Te, Sion, Srec, Srad = read_rate_nLTE(
                    filename=fname, element=element
                )

                # --- All files must share one Te grid: the arrays are stacked
                # on a single axis, so a mismatch silently corrupts the fit.
                if Te_ref is None:
                    Te_ref = np.asarray(Te, float)
                elif len(Te) != len(Te_ref) or not np.allclose(Te, Te_ref):
                    raise ValueError(
                        f"Te grid in {fname} differs from the first file's "
                        f"grid; all CRETIN tables for {element} must share "
                        f"one Te grid."
                    )

                # --- Consistency checks (mirror the MATLAB *_ratemake_nLTE scripts)
                if not np.isclose(ne, neV[iNe]):
                    warnings.warn(f"density from {fname} ({ne}) != expected {neV[iNe]}")
                if not np.isclose(na * Lo, TaV[iTa]):
                    warnings.warn(
                        f"column density from {fname} ({na*Lo}) != expected {TaV[iTa]}"
                    )

                if ASrad is None or ASion is None or ASrec is None:
                    NZ, NTe = Srad.shape
                    ASrad = np.zeros((NTa, Nne, NZ, NTe))
                    ASion = np.zeros((NTa, Nne, NZ, NTe))
                    ASrec = np.zeros((NTa, Nne, NZ, NTe))

                ASrad[iTa, iNe, :, :] = Srad / na  # radiation cooling rate [eV*cm^3/s]
                ASion[iTa, iNe, :, :] = Sion / ne  # ionization rate [cm^3/s]
                ASrec[iTa, iNe, :, :] = Srec / ne  # recombination rate [cm^3/s]

        ASrad = np.maximum(ASrad, Smin)  # type: ignore
        ASion = np.maximum(ASion, Smin)  # type: ignore
        ASrec = np.maximum(ASrec, Smin)  # type: ignore
        return ASrad, ASion, ASrec, np.asarray(Te, float), neV, TaV  # type: ignore

    def build_rgi_tables(
        self, ASrad, ASion, ASrec, Te, neV, TaV, rgi_method="linear"
    ) -> dict:
        """Build a RegularGridInterpolator per (rate type, charge state).

        Each interpolator maps (log ne, log Te, log Ta) -> log(rate); the caller
        clamps the query point to the grid limits and exponentiates the result.

        The grid axis order is (log ne, log Te, log Ta), so the stored values
        have shape (Nne, NTe, NTa) for every charge state.
        """
        ASrad = np.asarray(ASrad, float)
        ASion = np.asarray(ASion, float)
        ASrec = np.asarray(ASrec, float)

        NTa, Nne, NZ, NTe = ASrad.shape

        logTe = np.log(np.asarray(Te, float))
        logne = np.log(np.asarray(neV, float))
        logTa = np.log(np.asarray(TaV, float))

        # RegularGridInterpolator requires strictly increasing coordinates, sort Te to be safe
        te_order = np.argsort(logTe)
        logTe = logTe[te_order]
        points = (logne, logTe, logTa)  # axis order: (ne, Te, Ta)

        vals = {"ASrad": ASrad, "ASion": ASion, "ASrec": ASrec}
        rgi = {k: [] for k in vals}
        for k, As_ in vals.items():
            for iZ in range(NZ):
                # As_[:, :, iZ, :] has shape (NTa, Nne, NTe);
                # move to (Nne, NTe, NTa) and reorder Te.
                cube = np.transpose(As_[:, :, iZ, :], (1, 2, 0))  # (Nne, NTe, NTa)
                cube = cube[:, te_order, :]
                log_cube = np.log(np.maximum(cube, _LOG_FLOOR))
                rgi[k].append(
                    RegularGridInterpolator(
                        points,
                        log_cube,
                        method=rgi_method,
                        bounds_error=False,
                        fill_value=float(np.log(1e-20)),  # For out-of-bounds queries
                    )
                )

        return {
            "method": "rgi",
            "NZ": NZ,
            "NTe": NTe,
            "NTa": NTa,
            "Nne": Nne,
            "Telim": (float(np.exp(logTe.min())), float(np.exp(logTe.max()))),
            "nelim": (float(neV.min()), float(neV.max())),
            "Talim": (float(TaV.min()), float(TaV.max())),
            "rgi": rgi,
        }

    def fit_rate_tables(self, ASrad, ASion, ASrec, Te, neV, TaV, NTepp=15) -> dict:
        """
        Fit the three raw coefficient grids and return the coefficient struct (dict).

        `NTepp` is the number of Te polynomial coefficients (degree NTepp-1); it must
        not exceed the number of Te data points.
        """
        ASrad = np.asarray(ASrad, float)
        ASion = np.asarray(ASion, float)
        ASrec = np.asarray(ASrec, float)
        Te = np.asarray(Te, float)
        neV = np.asarray(neV, float)
        TaV = np.asarray(TaV, float)

        NTa, Nne, NZ, NTe = ASrad.shape
        if NTepp > NTe:
            raise ValueError(f"NTepp ({NTepp}) must be <= number of Te points ({NTe}).")

        logTe, logne, logTa = np.log(Te), np.log(neV), np.log(TaV)
        logTeS = logTe - logTe.mean()
        logneS = logne - logne.mean()
        logTaS = logTa - logTa.mean()

        vals = {"ASrad": ASrad, "ASion": ASion, "ASrec": ASrec}

        # --- stage 1: fit log(coeff) vs logTe  -> (NTa, Nne, NZ, NTepp) ----------
        s1 = {k: np.zeros((NTa, Nne, NZ, NTepp)) for k in vals}
        for k, As_ in vals.items():
            for iTa in range(NTa):  # Step through the column density
                for ine in range(Nne):  # Step through the electron density
                    for iZ in range(NZ):  # Step through each charge state
                        s1[k][iTa, ine, iZ] = self._polyfit(
                            logTeS, np.log(As_[iTa, ine, iZ]), NTepp - 1
                        )

        # --- stage 2: fit each Te-coefficient vs logne -> (NTa, Nne, NZ, NTepp) --
        s2 = {k: np.zeros((NTa, Nne, NZ, NTepp)) for k in vals}
        for k in vals:
            for iTa in range(NTa):
                for iTe in range(NTepp):
                    for iZ in range(NZ):
                        s2[k][iTa, :, iZ, iTe] = self._polyfit(
                            logneS, s1[k][iTa, :, iZ, iTe], Nne - 1
                        )

        # --- stage 3: fit each ne-coefficient vs logTa -> (NTa, Nne, NZ, NTepp) --
        s3 = {k: np.zeros((NTa, Nne, NZ, NTepp)) for k in vals}
        for k in vals:
            for ine in range(Nne):
                for iTe in range(NTepp):
                    for iZ in range(NZ):
                        s3[k][:, ine, iZ, iTe] = self._polyfit(
                            logTaS, s2[k][:, ine, iZ, iTe], NTa - 1
                        )

        return {
            "method": "poly",
            "NZ": NZ,
            "NTe": NTepp,
            "NTa": NTa,
            "Nne": Nne,
            "Telim": (float(Te.min()), float(Te.max())),
            "nelim": (float(neV.min()), float(neV.max())),
            "Talim": (float(TaV.min()), float(TaV.max())),
            "logTemean": float(logTe.mean()),
            "lognemean": float(logne.mean()),
            "logTamean": float(logTa.mean()),
            "ASrad": s3["ASrad"],
            "ASion": s3["ASion"],
            "ASrec": s3["ASrec"],
        }

    # ---------------------------------------------------------------------------
    # ---- Evaluation
    # ---------------------------------------------------------------------------
    def _evaluate(self, element, which_, iZ, Ta, ne, Te) -> float:
        """Scalar wrapper around :meth:`_evaluate_many`.

        iZ = 0 = neutral, iZ = ZMax = fully stripped. Ta, ne, Te are scalars.
        """
        return float(self._evaluate_many(element, which_, iZ, Ta, ne, Te)[0])

    def _evaluate_many(self, element, which_, iZ, Ta, ne, Te) -> np.ndarray:
        """Vectorized evaluation: returns a 1-D array of rate coefficients.

        Ta, ne, Te may be scalars or 1-D arrays; they are broadcast against one
        another to a common shape (Nr,) and a (Nr,) array is returned.
        """
        element = _adas_element(element)
        if which_ not in ["ASrad", "ASion", "ASrec"]:
            raise ValueError(
                f"Subtype not found, avaible subtypes: {['ASrad', 'ASion', 'ASrec']}"
            )

        Zmax = _PERIODIC[element]
        fit_ = self._interp[element]
        if iZ > Zmax:
            raise ValueError(
                f"Input iZ value cannot be larger than {Zmax} for {element}"
            )

        # Clamp the (0-based) charge-state index to a valid array index [0, NZ-1]
        iZuse = int(np.clip(iZ, 0, fit_["NZ"] - 1))

        ne, Te, Ta = np.broadcast_arrays(
            np.asarray(ne, float), np.asarray(Te, float), np.asarray(Ta, float)
        )
        ne, Te, Ta = np.atleast_1d(ne), np.atleast_1d(Te), np.atleast_1d(Ta)

        if fit_.get("method") == "rgi":
            return self._evaluate_rgi_many(which_, iZuse, Ta, ne, Te, fit_)
        return self._evaluate_poly_many(which_, iZuse, Ta, ne, Te, fit_)

    def _evaluate_rgi_many(self, which_, iZuse, Ta, ne, Te, fit_) -> np.ndarray:
        """RegularGridInterpolator evaluation (inputs clamped to the grid)."""
        log_ne = np.log(np.clip(ne, fit_["nelim"][0], fit_["nelim"][1]))
        log_Te = np.log(np.clip(Te, fit_["Telim"][0], fit_["Telim"][1]))
        log_Ta = np.log(np.clip(Ta, fit_["Talim"][0], fit_["Talim"][1]))

        # axis order matches build_rgi_tables: (log ne, log Te, log Ta)
        pts = np.column_stack([log_ne.ravel(), log_Te.ravel(), log_Ta.ravel()])
        interp = fit_["rgi"][which_][iZuse]
        return np.exp(interp(pts))

    def _evaluate_poly_many(self, which_, iZuse, Ta, ne, Te, fit_) -> np.ndarray:
        """Nested-polynomial evaluation (original back-end), vectorized."""
        log_ne = (
            np.log(np.clip(ne, fit_["nelim"][0], fit_["nelim"][1])) - fit_["lognemean"]
        )
        log_Te = (
            np.log(np.clip(Te, fit_["Telim"][0], fit_["Telim"][1])) - fit_["logTemean"]
        )
        log_Ta = (
            np.log(np.clip(Ta, fit_["Talim"][0], fit_["Talim"][1])) - fit_["logTamean"]
        )
        log_Ta, log_ne, log_Te = log_Ta.ravel(), log_ne.ravel(), log_Te.ravel()

        NTa, Nne, NTe = fit_["NTa"], fit_["Nne"], fit_["NTe"]
        # Vandermonde matrices, one row per query point
        log_TaV = log_Ta[:, None] ** np.arange(NTa - 1, -1, -1)[None, :]  # (Nr, NTa)
        log_neV = log_ne[:, None] ** np.arange(Nne - 1, -1, -1)[None, :]  # (Nr, Nne)
        log_TeV = log_Te[:, None] ** np.arange(NTe - 1, -1, -1)[None, :]  # (Nr, NTe)

        ASval = fit_[which_]
        dumV = np.empty((log_Ta.shape[0], NTa))
        for iiTa in range(NTa):
            AM = ASval[iiTa, :, iZuse, :]  # (Nne, NTe)
            # MATLAB per-point: dumV(iiTa) = logneV*(AM*logTeV')
            dumV[:, iiTa] = np.einsum("ri,ij,rj->r", log_neV, AM, log_TeV)

        logSval = np.einsum("rt,rt->r", dumV, log_TaV)  # sum over Ta
        return np.exp(logSval)

    # --------------------------------------------------------------------------
    # ---- Ionization rate, power, recomb. rate, etc.
    # --------------------------------------------------------------------------
    def Sion(self, symbol: str, Z: int, ne: float, Te: float, Ta: float) -> float:
        """Ionization rate coefficient [cm^3/s] for charge state Z (Z=0 neutral)."""
        sym = self._check_element(symbol)
        if Z > self.Zmax[sym]:
            return 0.0
        return self._evaluate(sym, "ASion", Z, Ta, ne, Te)

    def Srec(self, symbol: str, Z: int, ne: float, Te: float, Ta: float) -> float:
        """Recombination rate coefficient [cm^3/s] for charge state Z (Z->Z-1).

        The neutral (Z=0) has no lower state to recombine into, so it returns 0.
        """
        sym = self._check_element(symbol)
        if Z < 1:
            return 0.0

        # Special case: He II recombination (NZ==3, Z=1 in 0-based indexing).
        # CRETIN rates are too high here, so use the 3-body + radiative formula
        # (mirrors Srec_nLTE.m).
        if self.Zmax[sym] == 2 and Z == 1:
            return 8.75e-27 * ne / Te**4.5 + 2.7e-13 * np.sqrt(Te)

        return self._evaluate(sym, "ASrec", Z, Ta, ne, Te)

    def Prad(self, symbol: str, Z: int, ne: float, Te: float, Ta: float) -> float:
        """Radiative cooling rate coefficient [eV cm^3 / s] for charge state Z."""
        sym = self._check_element(symbol)
        return self._evaluate(sym, "ASrad", Z, Ta, ne, Te)

    def all_rates(self, symbol: str, ne, Te, Ta):
        """Evaluate all three rate coefficients for every charge state.

        Parameters
        ----------
        symbol : element symbol, e.g. 'Ne'.
        ne     : electron density       [1/cm^3]
        Te     : electron temperature   [eV]
        Ta     : element column density [1/cm^2]
            ne and Te may be floats or 1-D arrays of shape (Nr,);
        Returns
        -------
        Sion, Srec, Prad : ndarray, 0-based charge-state index along axis 0.
        """
        sym = self._check_element(symbol)
        Zmax = self.Zmax[sym]
        n = Zmax + 1

        Ta = np.asarray(Ta, float)
        per_charge_Ta = Ta.ndim == 2  # (n, Nr): one Ta per charge state
        if per_charge_Ta:
            ne_b, Te_b = np.broadcast_arrays(
                np.asarray(ne, float), np.asarray(Te, float)
            )
            ne_f = np.atleast_1d(ne_b).ravel()
            Te_f = np.atleast_1d(Te_b).ravel()
            Nr = ne_f.size
            if Ta.shape != (n, Nr):
                raise ValueError(
                    f"per-charge-state Ta must have shape {(n, Nr)}, got {Ta.shape}"
                )
            scalar_input = False
        else:  # scalar or (Nr,) -> shared Ta
            scalar_input = np.ndim(ne) == 0 and np.ndim(Te) == 0 and Ta.ndim == 0
            ne_b, Te_b, Ta_b = np.broadcast_arrays(
                np.asarray(ne, float), np.asarray(Te, float), Ta
            )
            ne_f = np.atleast_1d(ne_b).ravel()
            Te_f = np.atleast_1d(Te_b).ravel()
            Ta_f = np.atleast_1d(Ta_b).ravel()
            Nr = ne_f.size

        Sion = np.zeros((n, Nr))
        Srec = np.zeros((n, Nr))
        Prad = np.zeros((n, Nr))

        is_he = Zmax == 2
        for Z in range(n):
            Ta_use = Ta[Z] if per_charge_Ta else Ta_f  # type: ignore
            Prad[Z] = self._evaluate_many(sym, "ASrad", Z, Ta_use, ne_f, Te_f)
            Sion[Z] = self._evaluate_many(sym, "ASion", Z, Ta_use, ne_f, Te_f)
            if Z >= 1:  # neutral cannot recombine -> stays 0
                if is_he and Z == 1:  # He II special case (mirrors Srec)
                    Srec[Z] = 8.75e-27 * ne_f / Te_f**4.5 + 2.7e-13 * np.sqrt(Te_f)
                else:
                    Srec[Z] = self._evaluate_many(sym, "ASrec", Z, Ta_use, ne_f, Te_f)

        if scalar_input:
            return Sion[:, 0], Srec[:, 0], Prad[:, 0]
        return Sion, Srec, Prad
