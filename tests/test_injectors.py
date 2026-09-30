"""Tests for the particle-source classes in kpradpy.util.injectors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kpradpy.util.injectors import (
    MGI, Pellet, CSP, GradedCSP, SPI, WallSputter, torr_l_to_particles,
)


# ---------------------------------------------------------------------------
# Construction paths
# ---------------------------------------------------------------------------
def test_pellet_accepts_both_inventory_conventions():
    a = Pellet(["D2", "Ne"], V_TorrL=[200.0, 65.0])
    b = Pellet(["D2", "Ne"], N_1e20=list(a.N0_1e20))
    assert np.allclose(a.N0_1e20, b.N0_1e20)
    assert a.rp0_cm == pytest.approx(b.rp0_cm)


def test_pellet_rejects_ambiguous_inventory():
    with pytest.raises(ValueError):
        Pellet("Ne", V_TorrL=[65.0], N_1e20=[21.4])
    with pytest.raises(ValueError):
        Pellet("Ne")


def test_csp_1e20_path_works():
    """Regression: this branch used to fall through to a RuntimeError."""
    ref = CSP(shell_TorrL=200.0, core_TorrL=65.0)
    c = CSP(shell_1e20=float(ref.N0_1e20[0]), core_1e20=float(ref.N0_1e20[1]))
    assert c.rp0_cm == pytest.approx(ref.rp0_cm)
    assert c.rcore0_cm == pytest.approx(ref.rcore0_cm)


def test_spi_accepts_N_1e20():
    s = SPI(["Ne"], N_1e20=[21.4], N_frag=20, t_shatter=2.0, seed=1)
    assert s.N0_1e20[0] == pytest.approx(21.4)
    assert s.n_state > 0


def test_torrL_factor_matches_matlab_convention():
    """torrL_to_1e20=0.322 reproduces the MATLAB KPRAD conversion exactly."""
    m = MGI(["Ne"], [100.0], profile="gaussian", t_peak=5.0, dt_pulse=1.8,
            torrL_to_1e20=0.322)
    assert m.total_atoms_1e20()["Ne"] == pytest.approx(32.2)
    # and the physical value differs from it by ~2%, as documented
    phys = torr_l_to_particles(100.0, 293.15) / 1e20
    assert 0.02 < abs(phys - 32.2) / 32.2 < 0.03


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------
def _burn(injector, plasma_state, t_end=200.0):
    def rhs(t, y):
        _, dy = injector.ablate(t, plasma_state, y)
        return dy
    sol = solve_ivp(rhs, [0.0, t_end], injector.state0(), method="BDF",
                    rtol=1e-10, atol=1e-14, t_eval=np.linspace(0, t_end, 2001))
    assert sol.success
    return sol


def test_pellet_full_burn_conserves_atoms(plasma_state):
    p = Pellet(["D2", "Ne"], V_TorrL=[50.0, 20.0])
    sol = _burn(p, plasma_state)
    hist = p.injection_history(sol.y, sol.t)
    for sym, want in p.total_atoms_1e20().items():
        assert hist[sym]["cumulative"][-1] == pytest.approx(want, rel=1e-8)


def test_graded_csp_full_burn_conserves_atoms(plasma_state):
    g = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0, grade_width=0.15)
    sol = _burn(g, plasma_state)
    assert sol.y[0, -1] / g.V0_cm3 < 1e-8
    hist = g.injection_history(sol.y, sol.t)
    for sym, want in g.total_atoms_1e20().items():
        assert hist[sym]["cumulative"][-1] == pytest.approx(want, rel=1e-10)


def test_spi_fragment_sampler_conserves_mass():
    for seed in range(4):
        s = SPI(["Ne", "D2"], V_TorrL=[65.0, 200.0], N_frag=40, seed=seed)
        assert s._fragment_N0_1e20.sum() == pytest.approx(s.N0_1e20.sum(), rel=1e-12)
        assert np.all(s._fragment_N0_1e20 > 0)


# ---------------------------------------------------------------------------
# GradedCSP physics
# ---------------------------------------------------------------------------
def test_graded_csp_normalization_is_exact():
    ref = CSP(shell_TorrL=200.0, core_TorrL=65.0)
    for w in (0.02, 0.15, 0.40):
        g = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0, grade_width=w)
        assert np.allclose(g.N0_1e20, ref.N0_1e20, rtol=1e-12)
        assert 0.0 < g.s_interface < 1.0


def test_graded_csp_sharp_limit_reproduces_csp():
    ref = CSP(shell_TorrL=200.0, core_TorrL=65.0)
    g = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0, profile="sharp")
    assert g.R0_cm == pytest.approx(ref.rp0_cm, rel=2e-4)
    assert g.s_interface * g.R0_cm == pytest.approx(ref.rcore0_cm, rel=2e-4)


def test_graded_csp_surface_composition_evolves(plasma_state):
    g = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0, grade_width=0.15)
    fractions = [1.0, 0.6, 0.25, 0.05]
    x_Ne = [g.surface_composition(g.V0_cm3 * f) for f in fractions]
    # Ne fraction at the surface must rise monotonically as the pellet shrinks
    assert all(b > a for a, b in zip(x_Ne, x_Ne[1:]))
    assert x_Ne[0] < 0.05 and x_Ne[-1] > 0.9
    # and the delivered plume must reflect it
    d_early, _ = g.ablate(1.0, plasma_state, [g.V0_cm3])
    d_late, _ = g.ablate(1.0, plasma_state, [g.V0_cm3 * 0.05])
    ne_frac = lambda d: d["Ne"] / (d["H"] / 2 + d["Ne"])
    assert ne_frac(d_late) > 10 * ne_frac(d_early)


def test_graded_csp_recession_is_smooth_at_end(plasma_state):
    """dV/dt -> 0 as V -> 0; volume-as-state must not diverge."""
    g = GradedCSP(D2_TorrL=200.0, Ne_TorrL=65.0)
    rates = []
    for f in (1e-2, 1e-4, 1e-6):
        _, dV = g.ablate(1.0, plasma_state, [g.V0_cm3 * f])
        rates.append(abs(dV[0]))
    assert rates[0] > rates[1] > rates[2]


def test_graded_csp_from_config():
    from kpradpy.util.config import build_injectors
    cfg = {"sources": [{"type": "graded_csp", "D2_torrL": 200.0, "Ne_torrL": 65.0,
                        "profile": "tanh", "grade_width": 0.2, "t_start": 2.0}]}
    (inj,) = build_injectors(cfg, Te0_eV=4300.0)
    assert isinstance(inj, GradedCSP)
    assert inj.n_state == 1


# ---------------------------------------------------------------------------
# Stateless sources
# ---------------------------------------------------------------------------
def test_mgi_gaussian_integrates_to_total():
    m = MGI(["Ne"], [65.0], profile="gaussian", t_peak=5.0, dt_pulse=1.8)
    t = np.linspace(0.0, 30.0, 30001)
    rate = np.array([m.deliver(float(ti)).get("Ne", 0.0) for ti in t])
    assert np.trapezoid(rate, t) == pytest.approx(m.total_atoms_1e20()["Ne"], rel=1e-3)


def test_wall_sputter_te_gating():
    w = WallSputter("C", Ndot_TQ=0.5, Ndot_CQ=0.1, Te0_eV=4300.0)
    assert w.deliver(1.0, {"Te": 4000.0}) == {}          # above TQ threshold
    assert w.deliver(1.0, {"Te": 1000.0}).get("C") == 0.5  # in TQ
    assert w.deliver(1.0, {"Te": 5.0}).get("C") == 0.1     # in CQ
    assert w.deliver(1.0) == {}                            # no state -> silent
