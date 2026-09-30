# kpradpy

Python port of **KPRAD**, the 0-D tokamak disruption / radiative-shutdown
model (Whyte, Hollmann *et al.*). It evolves the plasma current, wall image
current, electron and ion thermal energy, and every charge state of every
species through a thermal quench and current quench driven by massive gas
injection (MGI), pellets, cryogenic shell pellets (CSP), or shattered
pellet injection (SPI).

The physics is 0-D. The code is written over `Nr` radial cells so that a
radially-resolved SPI deposition model can be added later without
restructuring; for all literature benchmarks `Nr = 1`, and the test suite
guarantees that limit reduces exactly to the scalar model.

---

## Setup

```bash
git clone <this repo>
cd kprad_devlopment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"            # add ,adas for the AuroraRates backend
pytest                              # 39 tests, no external data needed
```

External data lives outside the repo, found through one environment variable:

```bash
export KPRAD_DATA=~/research/kprad_data
#   $KPRAD_DATA/cretin/<El>_rates_CRETIN/<El>_rate_ne<i>_Ta<j>.dat
#   $KPRAD_DATA/gfiles/g<shot>.<time>
#   $KPRAD_DATA/matlab/Jeffs_kprad.mat
export KPRAD_OUTPUT=~/research/kprad_runs   # optional; default ./output
```

Config files refer to data relatively (`gfile: gfiles/g206990.01480`) and
`kpradpy.globals.resolve_data_path` resolves them against `KPRAD_DATA`.
See `kpradpy/globals.py` for every variable.

## Run

```bash
kprad configs/180016_SPI.yaml            # console entry point
python -m kpradpy.main_script configs/180016_SPI.yaml
```

Results go to `$KPRAD_OUTPUT` as HDF5 plus figures.

---

## Layout

```
kpradpy/                    the package
├── globals.py              paths (env-driven, no machine-specific defaults)
├── main_script.py          entry point: config -> rates -> solve_ivp -> postprocess
└── util/
    ├── solver.py           fkprad: the ODE right-hand side, annotated step by step
    ├── layout.py           SolverLayout: named views into the flat state vector
    ├── injectors.py        MGI, Pellet, CSP, GradedCSP, SPI, WallSputter
    ├── util_injectors.py   MGI delivery-profile shapes
    ├── postprocess.py      observables, quench times, Prad, energy balance, HDF5
    ├── physics.py          ln Lambda, parallel resistivity
    ├── constants.py        floors, unit factors, solid densities
    ├── config.py           YAML loader, build_initial_state, build_injectors
    ├── atomic_cretin.py    CretinRates (opacity-aware tables; 'rgi' or 'poly')
    ├── atomic_adas.py      AuroraRates (ADAS via aurora-fusion, optional)
    ├── equilibrium.py      G-EQDSK loading (freeqdsk)
    ├── profile.py          1-D profile helpers and radial grids
    └── plotting.py
configs/                    working YAML inputs (DIII-D 180016, 206990)
benchmarks/                 literature cases to reproduce — see benchmarks/README.md
tests/                      pytest suite — see tests/README.md
data/, output/              gitignored; default locations if env vars are unset
```

### Reading `solver.py`

`fkprad` is organised in eight labelled steps: unpack the state → derived
quantities (ne, ni, Zeff, Te, Ti) → e–i equilibration rate → charge-state
ladder and radiation → injector deposition → energy balance → global circuit
→ optional diagnostics. Each vectorised expression carries a comment giving
its `Nr = 1` scalar form. The one line to replace for radially-resolved SPI
is marked in Step 5.

### Energy balance

`postprocess.compute_energy_balance` re-invokes `fkprad(..., return_diagnostics=True)`
on the saved time grid, so each power channel comes from the same expression
the solver integrated. Its residual therefore measures only trapezoidal
reconstruction error and must fall as `simulation.nt` is increased — a
residual that plateaus is a bug. Use this on every benchmark.

---

## Workflow toward literature validation

0. **MATLAB parity** — `benchmarks/matlab_parity/`. Same model, same inputs.
1. **Hollmann 2008 Ar MGI** — canonical 0-D case; check the opacity,
   wall-current, and wall-impurity sensitivities individually.
2. **Shiraki 2026 SPI** — 0-D KPRAD with a resolved fragment plume; the
   closest published analogue of this code.
3. Freeze each reproduced case as a regression snapshot.
4. Only then: radial deposition profiles and inter-cell transport.

## Conventions worth knowing

* Densities are `cm^-3`; energies `MJ/m^3` (per cell) or `MJ` (integrated);
  time `ms`; currents `MA`. There is no `dense0` normalisation.
* `torrL_to_1e20: 0.322` reproduces MATLAB's Torr-L conversion; the
  physical value at 293.15 K is 0.3294. Set it explicitly for parity runs.
* `cretin_method: poly` matches MATLAB's polynomial fit; `rgi` (default)
  interpolates the table and is preferred for new work.
* Public keys returned by `postprocess` (`dense`, `densi`, `Wthe`, …) are the
  on-disk HDF5 dataset names and are deliberately not renamed.
* The `206990_*` configs predate the pellet classes and model the shell
  pellet as two Gaussian MGI pulses; migrate them to `csp` / `graded_csp`
  before using them as benchmarks.
