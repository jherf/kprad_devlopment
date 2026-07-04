"""
rate_read_nLTE.py
=================

Universal reader for CRETIN non-LTE rate files of the form

    <El>_rate_nLTE_*.dat

(e.g. ``H_rate_nLTE_*.dat``, ``C_rate_nLTE_*.dat``, ``Ne_rate_nLTE_*.dat``),
produced by the corresponding CRETIN generator files ``<El>_rate_nLTE.gen``.

This single function replaces the three element-specific MATLAB readers
``H_rate_read_nLTE.m``, ``C_rate_read_nLTE.m`` and ``Ne_rate_read_nLTE.m``.

Returned quantities (matching the MATLAB output order)
------------------------------------------------------
    ne    electron density                       [1/cm^3]   (scalar)
    na    atom density of the element            [1/cm^3]   (scalar)
    Lo    scale length / optical depth           [cm]       (scalar)
    Te    electron temperatures used in the run  [eV]       (n_Te,)
    Sion  ionization rate vs charge state, Te    [1/s]      (NZ, n_Te)
    Srec  recombination rate vs charge state, Te [1/s]      (NZ, n_Te)
    Srad  electron radiative cooling rate        [eV/s]     (NZ, n_Te)

Charge-state indexing
---------------------
``NZ = Z + 1`` rows are returned.  Row 0 is the neutral atom and row
``NZ-1`` is the fully stripped ion (this is the MATLAB index 1..NZ shifted
to 0-based Python).  By convention:

    * the fully stripped ion has no ionization        -> Sion[NZ-1, :] = 0
    * the neutral atom has no recombination           -> Srec[0,    :] = 0

File format notes
-----------------
CRETIN prints the ionization / recombination rate matrices with at most
``max_cols`` (= 9) data columns per "page"; when there are more transitions
than that (e.g. neon, with 10 ionizing stages) the table is continued on a
second page.  This reader handles an arbitrary number of pages, so it works
for any element, not just the three the original scripts were written for.
"""

from __future__ import annotations

import math
import os
import re
from collections import namedtuple

import numpy as np

# ---------------------------------------------------------------------------
# Result container (supports both tuple unpacking and attribute access)
# ---------------------------------------------------------------------------
RateData = namedtuple("RateData", ["ne", "na", "Lo", "Te", "Sion", "Srec", "Srad"])

# ---------------------------------------------------------------------------
# Per-element parameters.
#   Z        : atomic number  (number of charge states NZ = Z + 1)
#   t0_skip  : number of "t = 0" data lines to skip (not used by the analysis).
#              This count is specific to the standard 40-Te-step CRETIN files;
#              pass `t0_skip=` explicitly to override it for a non-standard file.
# ---------------------------------------------------------------------------
ELEMENT_CONFIG = {
    "H": {"Z": 1},
    "C": {"Z": 6},
    "Ne": {"Z": 10},
}

# Fixed column geometry of the CRETIN text format (1-indexed, like MATLAB).
_MAX_COLS = 9  # maximum rate columns printed per page
_COL_WIDTH = 11  # character stride between successive columns
_COL_START = 20  # 1-indexed start column of the first rate field


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------
def _to_float(text: str) -> float:
    """Robust float parse (handles blanks and Fortran 'D'/space exponents)."""
    s = text.strip()
    if not s:
        return np.nan
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        # Fortran sometimes drops the 'E', e.g. '1.234-105' -> 1.234e-105
        m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", s)
        if m:
            return float(m.group(1) + "e" + m.group(2))
        raise


def fastforward(lines, str, start_pos=0):
    ii = 0
    for ii in range(start_pos, len(lines)):
        if str in lines[ii]:
            break
        if ii == len(lines) - 1:
            # print(f"ERROR: {str} not found in file!")
            ii = -1
    return ii


def fastforward_equal(lines, str, start_pos=0):
    ii = 0
    for ii in range(start_pos, len(lines)):
        if str == lines[ii]:
            break
        if ii == len(lines) - 1:
            print(f"ERROR: {str} not found in file!")
            ii = -1
    return ii


def _field(line: str, start1: int, stop1: int) -> str:
    """MATLAB-style inclusive 1-indexed slice  line(start1:stop1)."""
    return line[start1 - 1 : stop1]


def _col_value(line: str, col: int) -> float:
    """Value of the `col`-th (1-indexed) rate field on a data line."""
    ilo = _COL_START + _COL_WIDTH * (col - 1)
    ihi = ilo + _COL_WIDTH
    return _to_float(_field(line, ilo, ihi))


def _str2num(s):
    """Mimic MATLAB str2num for a single fixed-width numeric field.

    The original .dat columns hold one clean number per field, so a stripped
    float() reproduces str2num here. (If a file ever uses Fortran 'D' exponents,
    str2num would have failed on it too.)
    """
    return float(s.strip())


def _read_rate_pages(lines, p, NZ, n_Te, n_trans, max_cols, S, mode):
    """
    Read a (possibly multi-page) ionization or recombination block and fill S.

    Each page is preceded by 7 separator lines and contains NZ * n_Te data
    lines (all charge states are printed even though only the relevant
    transition is extracted for each one).

    mode == 'ion':  charge state z (1..NZ-1) ionizes via transition  t = z
    mode == 'rec':  charge state z (2..NZ)   recombines via transition t = z-1
    Transition t is printed on page  (t-1)//max_cols  in column  t - page*max_cols.
    """
    n_pages = math.ceil(n_trans / max_cols) if n_trans > 0 else 0
    for page in range(n_pages):
        p += 7  # skip separator lines that precede every page
        for z in range(1, NZ + 1):
            for it in range(1, n_Te + 1):
                line = lines[p]
                p += 1
                if mode == "ion":
                    trans, valid = z, (z <= n_trans)
                else:  # 'rec'
                    trans, valid = z - 1, (z >= 2)
                if valid and (trans - 1) // max_cols == page:
                    col = trans - page * max_cols
                    S[z - 1, it - 1] = _col_value(line, col)
    return p


def _extract_cycle_number(line):
    """Extracts the number after cycle #"""

    match = re.search(r"cycle\s*#\s*(\d+)\s*:", line)
    if match:
        cycle = int(match.group(1))
        return cycle
    else:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def read_rate_nLTE(filename, element=None, max_cols=_MAX_COLS):
    """
    Read a CRETIN ``*_rate_nLTE_*.dat`` file for any element.

    Parameters
    ----------
    filename : str
        Path to the .dat file.
    element : str, optional
        Element symbol ('H', 'C', 'Ne', ...).  If omitted, it is inferred
        from the leading token of the file name (e.g. 'Ne_rate_nLTE_...').
    Z : int, optional
        Atomic number.  Overrides / supplies the value if `element` is unknown.
    t0_skip : int, optional
        Number of t=0 data lines to skip.  Overrides / supplies the value if
        `element` is unknown (required for elements not in ELEMENT_CONFIG).
    n_Te : int, optional
        Number of electron-temperature steps.  Defaults to the value read
        from the file header (the original MATLAB assumed 40).
    max_cols : int, optional
        Maximum rate columns per printed page (CRETIN default 9).

    Returns
    -------
    RateData namedtuple : (ne, na, Lo, Te, Sion, Srec, Srad)
        Unpack like the MATLAB call:
            ne, na, Lo, Te, Sion, Srec, Srad = read_rate_nLTE(fname)
    """
    # --- resolve element parameters ----------------------------------------
    if element is None:
        m = re.match(r"([A-Za-z]+)_rate", os.path.basename(filename))
        if m:
            element = m.group(1)

    if element not in ELEMENT_CONFIG:
        raise ValueError(
            f"{element} not found, known elements: {ELEMENT_CONFIG.keys()}"
        )

    cfg = ELEMENT_CONFIG.get(element, {}) if element else {}

    Z = cfg.get("Z")
    NZ = 0
    if Z is not None:
        NZ = Z + 1  # number of charge states (neutral .. fully stripped)

    with open(filename, "r") as f:
        lines = f.readlines()

    # --- header ------------------------------------------------------------
    # Cycle forward until 'optical depth' is in the data
    pos = fastforward(lines, "optical depth", start_pos=10)
    # Line with optical depth
    Lo = _str2num(lines[pos].split(" ")[3])

    # Line with ion density
    pos = fastforward(lines, "ion density", start_pos=pos)
    na = _str2num(lines[pos].split(" ")[3])

    # Line with electron density
    pos = fastforward(lines, "electron density", start_pos=pos)
    ne = _str2num(lines[pos].split(" ")[3])

    # Fast forward until 'temperature steps' is reached
    pos = fastforward(lines, "temperature steps", start_pos=pos)
    n_Te = int(_str2num(lines[pos].split(" ")[3]))
    NSTEPS = n_Te * NZ

    # --- allocate ----------------------------------------------------------
    Te = np.zeros(n_Te)
    Srad = np.zeros((NZ, n_Te))
    Sion = np.zeros((NZ, n_Te))
    Srec = np.zeros((NZ, n_Te))

    # --- Fast-forward through the t = 0 data (not used) --------------------
    cycles = {}
    pos2 = 0
    count = 0
    while True:
        pos2 = fastforward(lines, "cycle", start_pos=pos2 + 1)
        val = _extract_cycle_number(lines[pos2])

        # Break when reached the end of file
        if pos2 == -1:
            break

        if val not in cycles:
            cycles[val] = pos2

        if pos2 < cycles[val]:
            cycles[val] = pos2

        # Shouldn't trigger, but added backup break statement
        count += 1
        if count >= len(lines):
            break

    max_ = max(list(cycles.keys()))
    pos = cycles[max_]

    # ---- radiation cooling rate (and the Te grid) ------------------------
    pos = fastforward(lines, "electron cooling rate", start_pos=pos + 1)
    pos += 3
    for z in range(1, NZ + 1):
        for it in range(1, n_Te + 1):
            myString = lines[pos]
            vals = myString.split(" ")
            Te[it - 1] = _str2num(vals[-2])  # MATLAB myString(9:19)
            Srad[z - 1, it - 1] = -_str2num(vals[-1])  # MATLAB myString(20:31)
            pos += 1

    n_trans = NZ - 1  # number of ionizing / recombining transitions

    # --- ionization then recombination -------------------------------------
    pos = _read_rate_pages(lines, pos, NZ, n_Te, n_trans, max_cols, Sion, "ion")
    pos = _read_rate_pages(lines, pos, NZ, n_Te, n_trans, max_cols, Srec, "rec")

    Sion[NZ - 1, :] = 0.0  # no ionization for fully stripped
    Srec[0, :] = 0.0  # no recombination for neutral

    return RateData(ne=ne, na=na, Lo=Lo, Te=Te, Sion=Sion, Srec=Srec, Srad=Srad)
