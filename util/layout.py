# layout.py
"""
1-D radial state-vector layout

The kprad state vector is organized as:

    [
    Ip, Iw,                                    <- global circuit scalars
    We[0..Nr-1],                               <- electron thermal energy density
    Wi[0..Nr-1],                               <- ion thermal energy density
    n_H[Z=0][0..Nr-1], n_H[Z=1][0..Nr-1],      <- per charge state, per cell
    ... next element ...
    aux[0..n_aux-1],                           <- injector ODE state (optional)
    ]

    Units (absolute — the 0-D dense0 normalization is gone):
    Ip, Iw : MA
    We, Wi : MJ / m^3        (We = 1.5 * e * ne[cm^-3] * Te[eV], see note)
    n      : cm^-3

Note on the energy-density unit: with ne in cm^-3 and Te in eV,
    We/Wi [MJ/m^3] = 1.5 * e * ne * Te
because 1 eV*cm^-3 = e * 1e6 J/m^3 = e MJ/m^3 with e = 1.602e-19. This is
the per-volume version of the 0-D convention Wthe[MJ] = 1.5 * (Vp*e) * Te
* ne, so at Nr=1: We = Wthe / Vp exactly.

Because every field occupies a contiguous slice, ``species(solV, sym)``
returns a zero-copy (Zmax+1, Nr) reshaped *view*: writing into it writes
into the flat vector. The RHS therefore never does index arithmetic.
"""

import numpy as np

_HEADER_SIZE = 2 # Ip, Iw


class SolverLayout:
    """Maps element symbols to slot ranges in the 1-D ODE state vector.

    The first ``_HEADER_SIZE`` slots are reserved for the bulk plasma
    variables (Ip, Iw). Each element then occupies a contigous
    block of ``Zmax + 1`` slots starting at ``offset(sym)``, with the neutral
    at the lowest and fully-stripped at the highes

    Parameters
    ----------
    rates : object with ``.elements`` (list[str]) and ``.Zmax`` (dict),
            e.g. an AuroraRates instance.
    Nr    : number of radial cells. Nr=1 reproduces the 0-D model.
    n_aux : number of auxiliary scalar slots appended after the species
            blocks, used for injector ODE state (e.g. remaining pellet
            inventory). Default 0. The aux block lives at the *end* of the
            vector so header/field/species offsets are independent of it.

    Attributes
    ----------
    IP, IW   : int — absolute indices of the circuit scalars.
    Nr       : int
    elements : list[str]
    Zmax     : dict[str, int]
    n_aux    : int
    size     : int — total length of the state vector.
    """

    HEADER_SIZE = _HEADER_SIZE
    IP, IW = 0, 1

    def __init__(self, rates, Nr: int, n_aux: int = 0):
        if Nr < 1:
            raise ValueError("Nr must be >= 1")
        if n_aux < 0:
            raise ValueError("n_aux must be >= 0")

        self.Nr = int(Nr)
        self.elements = list(rates.elements)
        self.Zmax = dict(rates.Zmax)

        # Main counter
        offset = _HEADER_SIZE

        # ---- Add We and Wi regions for the number of Nr values
        # These are called fields
        self._fields: dict[str, slice] = {}
        for name in ("We", "Wi"):
            self._fields[name] = slice(offset, offset + self.Nr)
            offset += self.Nr

        # --- Add space for each element, these are called _species
        self._species: dict[str, slice] = {}
        for sym in self.elements:
            nslots = (self.Zmax[sym] + 1) * self.Nr
            self._species[sym] = slice(offset, offset + nslots)
            offset += nslots

        # --- Auxiliary slots (injector ODE state, e.g. pellet inventory)
        self.n_aux = int(n_aux)
        self._aux = slice(offset, offset + self.n_aux)
        offset += self.n_aux

        self.size = offset


    # ----------------------------------------------------------------
    # ---- Index lookups
    # ----------------------------------------------------------------
    def field_slice(self, name: str) -> slice:
        """Flat-index slice of 'We' or 'Wi'"""
        return self._fields[name]

    def species_slice(self, sym: str) -> slice:
        """Flat-index slice of all charge states in ``sym``"""
        return self._species[sym]

    def aux_slice(self) -> slice:
        """Flat-index slice of the auxiliary (injector-state) block."""
        return self._aux

    def aux(self, solV: np.ndarray) -> np.ndarray:
        """(n_aux,) view of the auxiliary block. Writing into it writes
        into the flat vector (or, for a 2-D solY, returns the aux rows)."""
        return solV[self._aux]

    def field(self, solV: np.ndarray, name: str) -> np.ndarray:
        """Returns `We` or `Wi` values within the solV array"""
        return solV[self._fields[name]]

    def species(self, solV: np.ndarray, sym: str) -> np.ndarray:
        """(Zmax+1, Nr) view of all charge states of ``sym``.

        Row i = charge state i (0 = neutral), column k = radial cell k.
        Length Nr
        """
        block = solV[self._species[sym]]
        return block.reshape(self.Zmax[sym] + 1, self.Nr)

    def view(self, solV: np.ndarray) -> dict:
        """Dict of views into ``solV``: Ip/Iw (0-d), We/Wi (Nr,), n[sym], aux."""
        return {
            "Ip": solV[self.IP],
            "Iw": solV[self.IW],
            "We": self.field(solV, "We"),
            "Wi": self.field(solV, "Wi"),
            "n": {sym: self.species(solV, sym) for sym in self.elements},
            "aux": self.aux(solV),
        }

    # ------------------------------------------------------------------
    # ---- Create initial state
    # ------------------------------------------------------------------
    def pack(self, Ip: float, Iw: float, We, Wi, dens: dict, aux=None) -> np.ndarray:
        """Build a flat state vector from physical fields.

        We, Wi      : scalar or (Nr,) [MJ/m^3]
        dens[sym]   : (Zmax+1,) or (Zmax+1, Nr) charge-state densities [cm^-3]
        aux         : optional (n_aux,) auxiliary initial values
        """
        solV = np.zeros(self.size)
        solV[self.IP] = Ip
        solV[self.IW] = Iw
        self.field(solV, "We")[:] = We
        self.field(solV, "Wi")[:] = Wi
        for sym, arr in dens.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]  # broadcast flat profile across cells
            self.species(solV, sym)[:, :] = arr
        if aux is not None:
            if self.n_aux == 0:
                raise ValueError("aux values given but layout has n_aux=0")
            self.aux(solV)[:] = np.asarray(aux, dtype=float)
        return solV


    def __contains__(self, sym):
        return sym in self._species

    def __repr__(self):
        return (
            f"SolverLayout(elements={self.elements}, Nr={self.Nr}, "
            f"n_aux={self.n_aux}, size={self.size})"
        )