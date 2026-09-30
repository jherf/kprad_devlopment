"""Shared fixtures for the kpradpy test suite.

The physics tests here do NOT need the CRETIN or ADAS tables. They use a
deterministic mock rate backend so the suite runs on a bare clone. Tests
that require real atomic data are marked ``@pytest.mark.needs_data`` and
skip automatically when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from kpradpy.util.layout import SolverLayout


class MockRates:
    """Deterministic, smooth, Te-dependent pseudo rate coefficients.

    Shapes and calling convention match ``CretinRates.all_rates``. The
    numbers are not physical; they exist so the ODE machinery can be tested
    end-to-end without external data.
    """

    elements = ["H", "Ne"]
    Zmax = {"H": 1, "Ne": 10}
    Ei_cumulative = {
        "H": np.array([13.6]),
        "Ne": np.cumsum(
            [21.6, 41.0, 63.5, 97.1, 126.2, 157.9, 207.3, 239.1, 1195.8, 1362.2]
        ),
    }

    def all_rates(self, sym, ne, Te, tau=None):
        Z = self.Zmax[sym] + 1
        Te = np.atleast_1d(np.asarray(Te, dtype=float))
        shape = (Z, Te.size)
        k = np.arange(Z)[:, None]
        Te_safe = np.maximum(Te, 1.0)
        S_ion = 1e-8 * np.exp(-k * 20.0 / Te_safe) * np.ones(shape)
        S_rec = 1e-12 * np.ones(shape) / Te_safe
        S_rad = 1e-9 * k * np.ones(shape) / np.sqrt(Te_safe)
        return S_ion, S_rec, S_rad


@pytest.fixture(scope="session")
def rates():
    return MockRates()


def _build_problem(rates, Nr, seed=0):
    layout = SolverLayout(rates, Nr=Nr, n_aux=0)
    rng = np.random.default_rng(seed)
    state = np.abs(rng.random(layout.size)) * 1e13
    state[layout.IP] = 1.6
    state[layout.IW] = 0.05
    layout.field(state, "We")[:] = 0.12
    layout.field(state, "Wi")[:] = 0.08
    if Nr == 1:
        Vcell = np.array([20.0])
    else:
        Vcell = np.linspace(4.0, 8.0, Nr)
        Vcell *= 20.0 / Vcell.sum()
    params = dict(
        rateStruct=rates,
        tauw=50.0,
        alphaL=5.0,
        Vp=20.0,
        Rmaj=1.7,
        Rmin=0.8,
        Ap=2.0,
        li=0.8,
        layout=layout,
        grid={"Vcell": Vcell},
    )
    return layout, state, params


@pytest.fixture
def problem_nr1(rates):
    """(layout, state, params) for a one-cell 0-D problem."""
    return _build_problem(rates, Nr=1)


@pytest.fixture
def problem_nr3(rates):
    """(layout, state, params) for a three-cell problem."""
    return _build_problem(rates, Nr=3)


@pytest.fixture
def plasma_state():
    """A representative plasma_state dict for injector tests."""
    return {"Te": 1500.0, "Ti": 1000.0, "ne": 8e13, "ni": 8e13, "time": 3.0}
