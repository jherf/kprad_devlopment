# Changelog

## 0.2.0 — 2026-09-15

### Package and layout
- Renamed package `main/` → `kpradpy/`; fixed the stale `kprad.` imports
  left by the previous rename (`config.py`, MATLAB comparison script).
- Added `pyproject.toml`: `pip install -e .`, console script `kprad`,
  declared dependencies (`freeqdsk` was undeclared).
- `aurora` and `freeqdsk` are now lazy imports; the CRETIN backend and the
  whole package import without them.
- All machine-specific paths removed. `globals.py` resolves everything from
  `KPRAD_DATA`, `KPRAD_OUTPUT`, `KPRAD_CRETIN`, `KPRAD_CONFIG` with
  repo-relative defaults. Config paths resolve via `resolve_data_path`.
- `tests/` (pytest, 39 tests, no external data) and `benchmarks/` (three
  literature cases with READMEs and starting configs).
- Added `README.md`, `tests/README.md`, `benchmarks/README.md`.

### solver.py
- Rewritten with descriptive names and step-by-step annotation, including
  the `Nr = 1` scalar form of every vectorised expression. Verified bitwise
  identical to the previous RHS at `Nr = 1` and `Nr = 3`.
- New `return_diagnostics` flag returns volume-integrated power channels
  (`PJ, Prad, Pion, Pei, Pfrozen, Ptransp, dWthe_dt, dWthi_dt`).
- The `We` floor now reports the power it discards (`Pfrozen`) instead of
  losing it silently.

### postprocess.py
- `compute_energy_balance` was non-functional (expected a diagnostics tuple
  the solver never produced; its call in `main_script` was commented out).
  It now works and is the primary validation check: its residual is
  reconstruction error only and must converge with `nt`.
- Analysis functions annotated and internally renamed; public dict keys are
  unchanged (they are the HDF5 dataset names).
- `Zeff`/`Zbar` now use the same density floors as the solver instead of
  writing inf/nan on full recombination.

### injectors.py
- **Bug:** `Pellet` documented an `N_1e20` parameter that did not exist.
  Added, and threaded through `SPI`.
- **Bug:** `CSP(shell_1e20=, core_1e20=)` always raised; the branch dropped
  the inventories. Fixed.
- New `GradedCSP`: continuous radial Ne/D2 composition profile
  (`tanh`/`linear`/`sharp`/callable). Parks λ(X) evaluated at the ablating
  surface; single aux state (remaining volume); interface position solved
  so requested inventories are matched exactly; sharp limit reproduces
  `CSP`; mass conserved to machine precision through full burn-through.
- Ablation methods annotated and renamed. Verified identical against a
  pre-change baseline for every class.

### config.py / main_script.py
- `graded_csp` source type.
- `simulation.cretin_method` (`rgi` | `poly`) exposed for MATLAB parity.
- `kprad` CLI with argparse.
- Configs: removed the phantom `transport:` block (never implemented);
  added a `graded_csp` example; gfile paths made `KPRAD_DATA`-relative.
