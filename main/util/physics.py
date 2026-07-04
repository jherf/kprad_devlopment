# physics.py
"""
Plasma physics utility functions

All functions accept scalars or NumPy arrays (per-cell evaluation in 1-D).
"""

import numpy as np
from main.util.constants import _MIN_TE


# ---- Coulomb Logarithm
def log_lambda_ei(ne, Te, Zeff):
    """
    Coulomb Logarithm for electron-ion collisions. Inputs can either be
    np.ndarrays or a float.

    Forumlas from NRL Plasma Formulary, definition automatically
    chooses which one is valid for the input parameters.

    Parameters
    ----------
    ne      : Electron density [cm^-3]
    Te      : Electron temperature [eV]
    Zeff    : Effection ion charge

    Returns
    -------
    log(Lambda)
    """
    ne, Te, Zeff = np.broadcast_arrays(
        np.asarray(ne, dtype=float),
        np.asarray(Te, dtype=float),
        np.asarray(Zeff, dtype=float),
    )

    # --- Make sure no Te values are below the floor
    Te = np.maximum(Te, _MIN_TE)

    logLambda = np.zeros(ne.shape)

    # --- Case 1, Te < 10 * Zeff^2:  lnL = 23 - ln( ne^1/2 * Z * Te^-3/2 )
    loc = Te < 10 * Zeff**2
    if loc.any():
        val = np.clip(np.sqrt(ne[loc]) * Zeff[loc] * Te[loc] ** (-1.5), 1e-9, None)
        logLambda[loc] = 23.0 - np.log(val)

    # --- Case 2, 10 * Zeff^2 < Te:  lnL = 24 - ln( ne^1/2 / Te )
    if (~loc).any():
        logLambda[~loc] = 24.0 - np.log(np.sqrt(ne[~loc]) / Te[~loc])

    return logLambda if logLambda.ndim else float(logLambda)


# ---- Spitzer-Härm resistivity correction
def beta_e(Zeff):
    """
    Spitzer-Härm parallel resistivity correction factor.

    Used as:
        eta_Spitzer = 5.24e-5 * lnLambda * Zeff * beta_e(Zeff) / Te^1.5  [Ohm*m]

    The prefactor (5.24e-5 * Z) corresponds to the Hydrogen normalized limit,
    so beta_e(1) ~ 1 and beta_e(inf) -> 0.58.

    Simple Pade-like fit reproducing the Spitzer & Härm (1953) tabulated values.
    """
    if False:
        # ---- Script to reproduce values used in the MATLAB Kprad's code
        from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt

        def beta_e_model(Zeff, a, b, c):
            """Function to reproduce KPRAD's MatLab's implementation of beta_e"""
            return a + b / (1 + c * Zeff)

        ZEFF = np.array([1, 2, 4, 16, 1e6])
        GAMMA_E = np.array([0.5816, 0.6833, 0.7849, 0.9255, 1.0])
        BETA_E = GAMMA_E[0] / GAMMA_E

        (a, b, c), _ = curve_fit(beta_e_model, ZEFF, BETA_E, p0=[0.582, 0.94, 1.25])
        # Results: a = 0.582, b = 0.921, c = 1.207
        f = plt.figure()
        ax = f.add_subplot()
        ax.scatter(ZEFF, BETA_E, marker="s", color="black")
        z_new = np.linspace(1, 16, 100)
        beta_e_new = beta_e_model(z_new, a, b, c)
        ax.plot(z_new, beta_e_new, color="tab:red", label="New Fit")
        ax.plot(
            z_new,
            beta_e_model(z_new, 0.582, 0.94, 1.25),
            color="tab:blue",
            label="Old Fit",
        )
        ax.set_ylabel("Zeff")
        ax.set_xlabel("beta_e")
        ax.legend()
        ax.set_xlim(1, 16)
        plt.tight_layout()
        plt.show()

    # return 0.582 + 0.94/(1 + 1.25*Zeff)
    # ---- New fit
    return 0.5823 + 0.9213 / (1 + 1.2073 * Zeff)


# ---- Neoclassical trapped-electron correction
def beta_T(ne, Te, Zeff, Rmaj, Rmin, q=2.0):
    """Updated beta_T based on MATLAB KPRAD's code"""

    LL = log_lambda_ei(
        ne,
        Te,
        Zeff,
    )
    tau_e = 3.45e5 * Te**1.5 / (ne * LL)  #  % electron-electron collision time [s]
    eps = Rmin / Rmaj  # inverse aspect ratio
    fT = 1 - (1 - eps) ** 2 / (
        np.sqrt(1 - eps**2) * (1 + 1.46 * np.sqrt(eps))
    )  # trapping fraction
    vbar_e = 4.19e7 * np.sqrt(Te)  # electron thermal velocity [cm/s]
    nustar_e = (
        100 * Rmaj * q / (vbar_e * tau_e * eps**1.5)
    )  # electron collision rate normalized to orbit frequency
    C = 0.56 * (3 - Zeff) / (Zeff * (3 + Zeff))
    phi = fT / (1 + (0.58 + 0.2 * Zeff) * nustar_e)
    B = 1 / ((1 - phi) * (1 - C * phi))
    return B


def eta_parallel(ne, Te, Zeff, Rmaj, Rmin, Te_floor=_MIN_TE):
    """Parallel (neoclassical) plasma resistivity [Ohm*m] with a low-Te cap.

        eta = beta_T * eta_Spitzer
        eta_Spitzer = 5.24e-5 * log_lambda * Zeff * beta_e / Te^1.5

    Accepts scalars or arrays (per-cell evaluation); broadcasting follows
    NumPy rules. Returns a float for scalar input, an ndarray otherwise.
    """
    Te_eff = np.maximum(np.asarray(Te, dtype=float), float(Te_floor))

    etas = (
        5.24e-5
        * log_lambda_ei(ne, Te_eff, Zeff)
        * np.asarray(Zeff, dtype=float)
        * beta_e(Zeff)
        / Te_eff**1.5
    )
    out = beta_T(ne, Te_eff, Zeff, Rmaj, Rmin, q=2.0) * etas
    out = np.asarray(out)
    return out.item() if out.ndim == 0 else out
