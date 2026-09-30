"""Tests for the ODE right-hand side and the energy-balance diagnostic.

The central guarantee protected here: the radial (Nr-cell) machinery must
reduce EXACTLY to the 0-D model when Nr = 1. Literature benchmarks are 0-D,
so any drift in this reduction silently invalidates every one of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kpradpy.util.solver import fkprad
from kpradpy.util.postprocess import postprocess, compute_energy_balance
from kpradpy.util.constants import _EE, _MIN_NE, _MIN_TE


def _scalar_reference_rhs(state, params):
    """A deliberately naive, fully scalar 0-D re-implementation of the RHS.

    Written independently of solver.py, with plain floats and loops, so it
    can catch a vectorization bug that the vectorized code cannot see in
    itself. Only the energy and circuit terms are checked here -- the
    charge-state ladder is exercised separately.
    """
    from kpradpy.util.physics import log_lambda_ei, eta_parallel

    layout = params["layout"]
    rates = params["rateStruct"]
    Vp = float(params["grid"]["Vcell"][0])

    Ip = state[layout.IP]
    Iw = state[layout.IW]
    We = float(layout.field(state, "We")[0])
    Wi = float(layout.field(state, "Wi")[0])

    ne = ni = sum_Z2 = 0.0
    for sym in layout.elements:
        n = layout.species(state, sym)[:, 0]
        for Z in range(1, layout.Zmax[sym] + 1):
            ne += Z * n[Z]
            sum_Z2 += Z * Z * n[Z]
        ni += n.sum()
    ne = max(ne, _MIN_NE)
    ni = max(ni, _MIN_NE)
    Zeff = max(sum_Z2 / ne, 1.0)
    Te = max(We / (1.5 * _EE * ne), _MIN_TE)

    eta = float(np.atleast_1d(eta_parallel(ne, Te, Zeff, params["Rmaj"], params["Rmin"]))[0])
    Rp = 2.0 * np.pi * params["Rmaj"] * eta / params["Ap"]
    taup = 1e-4 * params["li"] * params["Ap"] / eta
    PJ = 1e3 * Rp * Ip**2

    dIp = -Ip / taup + params["alphaL"] * Iw / params["tauw"]
    dIw = -dIp - Iw / params["tauw"]
    return dict(PJ=PJ, dIp=dIp, dIw=dIw, Rp=Rp)


def test_nr1_matches_scalar_reference(problem_nr1):
    layout, state, params = problem_nr1
    _, diag = fkprad(1.0, state, [], params, return_diagnostics=True)
    ref = _scalar_reference_rhs(state, params)
    d = fkprad(1.0, state, [], params)
    assert np.isclose(diag["PJ"], ref["PJ"], rtol=1e-12)
    assert np.isclose(d[layout.IP], ref["dIp"], rtol=1e-12)
    assert np.isclose(d[layout.IW], ref["dIw"], rtol=1e-12)


def test_diagnostics_flag_does_not_change_derivative(problem_nr1, problem_nr3):
    for layout, state, params in (problem_nr1, problem_nr3):
        plain = fkprad(2.0, state.copy(), [], params)
        with_diag, _ = fkprad(2.0, state.copy(), [], params, return_diagnostics=True)
        assert np.array_equal(plain, with_diag)


def test_instantaneous_energy_identity(problem_nr1, problem_nr3):
    """dWthe_dt == PJ - Prad - Pion - Pei - Pfrozen must hold pointwise."""
    for layout, state, params in (problem_nr1, problem_nr3):
        _, d = fkprad(1.0, state, [], params, return_diagnostics=True)
        lhs = d["dWthe_dt"]
        rhs = d["PJ"] - d["Prad"] - d["Pion"] - d["Pei"] - d["Pfrozen"]
        assert abs(lhs - rhs) < 1e-12 * max(1.0, abs(lhs))
        assert d["dWthi_dt"] == pytest.approx(d["Pei"])
        assert d["Ptransp"] == 0.0


def test_circuit_flux_conservation(problem_nr1):
    """dIp/dt + dIw/dt = -Iw/tauw: current lost by the plasma appears in the wall."""
    layout, state, params = problem_nr1
    d = fkprad(0.5, state, [], params)
    assert np.isclose(d[layout.IP] + d[layout.IW], -state[layout.IW] / params["tauw"])


def test_particle_conservation_in_charge_ladder(problem_nr1):
    """Ionization/recombination move population between Z but never create it."""
    layout, state, params = problem_nr1
    d = fkprad(1.0, state, [], params)
    for sym in layout.elements:
        total_rate = layout.species(d, sym).sum()
        assert abs(total_rate) < 1e-6 * layout.species(state, sym).sum()


@pytest.mark.slow
def test_energy_balance_converges_with_output_grid(problem_nr1):
    """The residual must be reconstruction error only, so it shrinks with nt.

    A residual that plateaus above the integrator's tolerance floor is the
    signature of a genuine RHS bug.

    Two regimes are checked, because a real run contains both:

    * TRANSIENT -- the random initial state is far from ionization
      equilibrium, so the first microseconds are a violent recombination
      burst, much like a thermal quench. Trapezoidal reconstruction only
      becomes asymptotic once the grid resolves it, so the grids here start
      fine. Second-order convergence (~16x per 4x refinement) is expected;
      4x is demanded.
    * SMOOTH -- after burning through the transient, the residual should be
      tiny on any reasonable grid, and bounded by the integrator tolerance
      rather than by the output grid.
    """
    layout, state, params = problem_nr1
    Vcell = params["grid"]["Vcell"]

    def resid_rel(y0, Nt, t_end=5.0):
        tV = np.linspace(0.0, t_end, Nt)
        sol = solve_ivp(
            fkprad, [0.0, t_end], y0, t_eval=tV, args=([], params),
            method="BDF", rtol=1e-11, atol=1e-11,
        )
        assert sol.success
        res = postprocess(tV, sol.y, layout, Vcell)
        return compute_energy_balance(res, sol.y, [], params, verbose=False)["resid_rel"]

    # --- transient regime: genuine convergence ---
    r = [resid_rel(state, Nt) for Nt in (1601, 6401, 25601)]
    assert r[1] < r[0] / 4.0, r
    assert r[2] < r[1] / 4.0, r
    assert r[2] < 1e-3, r

    # --- smooth regime: already negligible ---
    burn = solve_ivp(fkprad, [0.0, 0.1], state, args=([], params),
                     method="BDF", rtol=1e-11, atol=1e-11)
    assert burn.success
    assert resid_rel(burn.y[:, -1], 401) < 1e-6
