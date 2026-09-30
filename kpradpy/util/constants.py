# constants.py
"""
File containing program-wide constants
"""

import scipy.constants

# ---- Physical constants ------------------------------------------------
_EE = scipy.constants.elementary_charge
_MU0 = scipy.constants.mu_0
_MP_OVER_ME = scipy.constants.proton_mass / scipy.constants.electron_mass


# ---- Model parameters --------------------------------------------------
# Hard floors applied to ODE to avoid singularities
_MIN_NE = 0.3  # electron density [cm^-3]
_MIN_TE = 0.3  # electron temperature [eV]
_MIN_TA = 1.0  # [1/cm^2]
# Energy floor below which electron heating/cooling is frozen, expressed as
# total energy in a cell [MJ]. Shared by the ODE RHS (solver.py) and the
# post-processing floors so displayed traces match the integrated physics.
_W_FLOOR_MJ = 1e-6
# Please note that this also applies to Spitzer's resitivity (see below)
# Resitivity saturation temperature [eV]. Below this temperature the Spitzer
# eta ~ Te^-1.5 scaling is replaced by this value. Without this
# cap, Te^-1.5 goes out of control for Te < 1, which increases the resistance
# and explodes the ohmic heating term.


# ---- Per-species material properties ------------------------------------
# Atomic mass [amu]
_AMU = {"H": 1.008, "D2": 2.0141, "C": 12.0107, "Ne": 20.2, "Ar": 39.948, "W": 183.84}
# Solid density [g / cm3]
_RHO_SOLID = {"H": 0.086, "D2": 0.201, "Ne": 1.44, "Ar": 1.62}
