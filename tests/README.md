# tests/

Run with `pytest` from the repository root (after `pip install -e ".[test]"`).

| File | What it protects |
|---|---|
| `test_imports.py` | every module imports on a bare clone; no stale `main.`/`kprad.` imports; no machine-specific default paths |
| `test_solver.py` | Nr=1 reduces exactly to a scalar 0-D reference; diagnostics flag leaves the derivative untouched; the instantaneous energy identity; circuit flux conservation; particle conservation in the charge ladder; energy-balance residual converges with output grid |
| `test_injectors.py` | all inventory construction paths; MATLAB Torr-L factor; full-burn atom conservation for `Pellet` and `GradedCSP`; SPI fragment mass conservation; GradedCSP normalization, sharp limit, and evolving composition; MGI profile normalization; WallSputter gating |
| `test_pellet_legacy.py` | the original ad-hoc pellet script, kept for reference |
| `compare_rates.py` | CRETIN vs ADAS rate comparison (needs data; run manually) |

None of the physics tests need the CRETIN or ADAS tables — they use a
deterministic mock backend from `conftest.py`. Mark anything that needs real
data with `@pytest.mark.needs_data`.

`pytest -m "not slow"` skips the grid-convergence study (~10 s).
