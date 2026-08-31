# SHburning

Spectral hole burning in Pr:YSO, and atomic frequency comb (AFC) design.

The simulation is a Python port of the MATLAB "Bloch-Simulator"
(<https://zenodo.org/records/16962167>). It was originally written across the
`SHsimulation_v1*.ipynb` notebooks; the reusable code now lives in a single
module, `shb.py`, with a PyQt6 GUI (`shb_gui.py`) on top.

## Install

```bash
pip install -r requirements.txt
```

## Run

Desktop GUI:

```bash
python shb_gui.py
```

Module smoke test (burns one default comb, prints the OD summary):

```bash
python shb.py
```

Use it as a library:

```python
import shb

params = shb.DEFAULT_PARAMS
params["output_params"]["norm"] = shb.normalisation_factor(params)

out = shb.run_burn(shb.single_hole(0.0), params, pump_passes=10)
print(out["spec"].min(), out["spec"].max())
```

## What it does

`params` is a nested dict:

| section | keys |
| --- | --- |
| `crystal_params` | `g`, `e` level energies (MHz); `dm` `n×n` branching / relative-dipole matrix |
| `linewidth_params` | `h_width` (homogeneous FWHM), `in_width` (inhomogeneous FWHM) |
| `numerical_params` | `fmin`, `fmax` spectrum window; `df` mesh step |
| `burn_params` | `pump_passes` |
| `output_params` | `alpha` (absorption scale), `norm` (unburned peak, computed) |

### Crystal presets

`shb.PRESETS` (GUI: the crystal dropdown) holds ready-made configurations. The
default `Pr:YSO (site 1, tuned)` uses the Nilsson/Sharma branching matrix; the
others are transcribed from the original MATLAB source
(`zenodo.org/records/16962167`):

| preset | levels | `dm` | h_width / in_width | window |
| --- | --- | --- | --- | --- |
| `Pr:YSO (site 1, tuned)` | 3 g / 3 e | measured 3×3 | 0.001 / 10000 | ±50 |
| `Pr:YSO (MATLAB source)` | 3 g / 3 e | uniform | 0.02 / 2000 | ±200 |
| `151Eu:YSO (site 1)` | 3 g / 3 e | uniform | 0.1 / 2000 | ±1000 |
| `171Yb:YSO (site 2)` | 4 g / 4 e | 4×4 (D2 axis) | 0.5 / 1000 | −2000 … 10000 |
| `Zeeman doublet` | 2 g / 2 e | uniform | 0.1 / 2000 | ±1000 |

Each preset also carries a suggested `fmin/fmax/df`; the GUI writes those into the
fields when you pick it. The core handles any square `n×n` level structure.

Three ways to use it, mirrored by the GUI's **mode** selector:

1. **Single hole** - one pump frequency, burned `pump_passes` times.
2. **One comb** - a comb of teeth (`tooth_spacing`, `burn_width` in MHz), snapped
   to the mesh by `comb_at`.
3. **Scan** - `run_scan(storage_min, storage_max, n_points, finesse_min,
   finesse_max, n_finesse, target_window, top_n, pump_passes, ...)` sweeps AFC
   **storage time** x comb **finesse** (`F = tooth_spacing / tooth_width`) and
   scores each spectrum by its **AFC contrast**
   `(A_max - A_min) / (2 * A_unburned)` over the central `target_window` - the
   Michelson visibility of the comb with the background pinned to the unburned
   absorption level (computed once for the sweep), so it is a strict rescaling
   of the raw delta absorption and its maximum sits on the same `(storage,
   finesse)` cell - keeping the top *N*. Both grids are **linear**: `n_points`
   storage times (µs) evenly across `[storage_min, storage_max]` and `n_finesse`
   finesse values evenly across their range, so the heatmap axes are uniformly
   spaced. Each cell keeps its requested `(storage_us, finesse)`; the burn behind
   it uses the tooth spacing `1 / t` MHz rounded to the `df` mesh (so any
   `n_points` works - nothing has to divide `df` evenly). `results` are dicts
   `{spacing_MHz, width_MHz, finesse, storage_us, contrast}` where `spacing_MHz`
   is the mesh-snapped value actually burned. The tooth spacing `1/t` is
   quantised to the mesh, so two close storage times can round to the same comb.
   `run_scan` handles this itself: if the requested `n_points` cannot be resolved
   at the config `df`, it **refines `df`** by a whole factor (so it still divides
   the crystal levels and window edges) until every requested storage time lands
   on a distinct comb, bounded below by what is physically meaningful (~
   `h_width/5`) and by a spectrum-array-size cap. The refined value is returned
   as `df_used` and both the refinement and any remaining unresolved points are
   logged. A finer mesh means slower burns - the up-front time estimate reflects
   it.
   Runs on a `multiprocessing.Pool` **by default** (`parallel=True`,
   `n_workers=os.cpu_count()`; GUI: the *multiprocessing* checkbox, now ticked by
   default); pass `parallel=False` for a single-process sweep.

### The `df` rule

`df` must divide every level energy **and** both window edges exactly, or the
transition table and spectrum axis fall between bins and the result is silently
wrong. `validate_params` and `to_steps` raise `ValueError` when it does not. For
the default crystal (levels given to 0.01 MHz) that means `df` of 0.01, 0.005 or
0.001. Finer `df` costs more, but far less than it used to (see below).

### Speed

`spectrum()` is the per-scan-point hot spot. It now evaluates the homogeneous
Lorentzian sum as one FFT convolution per ground state (`scipy.fft`) instead of a
Python loop over every detuning bin, and `burn_parallel()` places every pump
frequency with a single `np.bincount` instead of a loop. `run_scan` builds the
transition table, axes and FFT kernel once and reuses them for the whole sweep
(and hands them to the worker processes once, not per task), and only computes the
unique `(spacing, width)` burns even when several finesse points map to the same
one. The result matches the original triple-loop (`shb._spectrum_loop`, the
MATLAB reference) to ~1e-13. A high-resolution scan that used to take minutes now
takes seconds.

## GUI layout

The left column has two tabs.

**Simulation** tab:
- *Import config (.json)…* button (see below).
- **Crystal parameters** - the crystal-preset dropdown (see *Crystal presets*
  above; picking one loads its levels, linewidths and a suggested window/mesh),
  a *Relative dipole strength* toggle (off = every transition equally strong),
  and an *Edit levels…* dialog for `g` / `e` / `dm`.
- **Simulation Parameters** - `fmin`/`fmax`/`df`, `h_width`/`in_width`,
  `pump_passes`, burn mode, `alpha`, the mode selector and its fields (Scan:
  storage-time min/max (µs) + **storage-time points**, finesse min/max +
  **finesse points**, target window, keep top N, multiprocessing on by default),
  **Run** (becomes **Cancel** while busy) and a progress bar. The bar shows an
  elapsed timer for a single burn, and for a scan an up-front time estimate
  followed by a live `done/total · ~mm:ss left` readout.

**Export** tab - bundle the current run to a folder. Tick any of *Run log*,
*Simulation config (JSON)*, *Figures*, *Data*, set a filename prefix (auto-filled
from the run), then *Export…* picks the folder. Written files, for prefix `P`:

| item | burn | scan |
| --- | --- | --- |
| run log | `P_runlog.txt` | `P_runlog.txt` |
| config | `P_config.json` (`{params, gui}`) | same |
| figures | `P_plot.png` | `P_plot.png` + `P_heatmap.png`, `P_bestcontrast.png`, `P_topspectra.png` |
| data | `P_spectrum.npy` (`vstack[f,spec]`), `P_spectrum.csv` | `P_results.csv` (`spacing_MHz,width_MHz,finesse,storage_us,contrast`), `P_top_spectra.npy` (row 0 = `f`, then each kept spectrum), `P_scan.pkl` |

Right column: one figure with the **pump comb** on top and the **spectrum**
below, sharing the Frequency axis - **left-drag across it to zoom the x-axis**
(as in the notebook's plotly plots), **double-click or the Home button to reset**,
or use the toolbar. A scan replaces both with a single full-height view: the
**AFC-contrast heatmap** over storage time (`1/tooth_spacing`, µs) x finesse
(`tooth_spacing/tooth_width`), *best contrast vs storage time*, or *top spectra* -
chosen by the *Scan view* dropdown. Below is the **Simulation run log**.

**Import config** reads a `P_config.json` back and restores the whole session -
crystal preset, levels, linewidths, numerical fields, mode and its fields, the
dipole toggle. A bare `params` JSON (no `gui` key, e.g. from
`shb.params_to_json`) also loads.

**File** menu: *Import config*, *Export*, *Quit*.

## Figure style

The matplotlib figures use `signature.mplstyle` (Times New Roman, grid, inferno
heatmaps) - the same look as the notebook write-up figures. `shb.apply_style()`
applies it process-wide; the GUI calls it at startup. If the file is missing the
module falls back to a small inline rcParams set.

## Files

| file | purpose |
| --- | --- |
| `shb.py` | the simulator: config, core physics, combs, scan, plot helpers, `apply_style()` |
| `shb_gui.py` | the PyQt6 GUI |
| `signature.mplstyle` | shared matplotlib style, ported from `SHsimulation_v1_7.ipynb` |
| `worker.py` | thin `from shb import *` shim, kept so `SHsimulation_v1_3test2multi.ipynb` still runs |
| `SHsimulation_v1*.ipynb` | the original working notebooks, left as they are |

All computation in the GUI runs on a background thread, so the window stays
responsive and a scan can be cancelled mid-run.
