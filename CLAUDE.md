# CLAUDE.md

Guidance for Claude Code (and anyone else) working in this repo. See
[README.md](README.md) for the physics and the user-facing feature tour -
this file is about how the code is put together and what not to break.

## Layout

Three Python files. Everything else (notebooks, `*.pkl`/`*.npy`/`*.html`
output) is not source.

- **`shb.py`** - the whole simulator, in sections (marked with `# ====`
  banners): Config, Core, Combs, Scan, Checkpointing, Plotting, CLI. Keep new
  code in its section rather than appending to the end of the file.
- **`shb_gui.py`** - the PyQt6 GUI. `SimWorker` (a `QObject` moved to a
  `QThread`) runs norm/burn/scan jobs off the UI thread and talks back to
  `MainWindow` over signals; all computation itself lives in `shb.py`, not
  here.
- **`worker.py`** - a backwards-compatible shim (`from shb import *`) so
  `SHsimulation_v1_3test2multi.ipynb`'s `from worker import *` still works.
  Not a place to add code.

## Running it

```bash
pip install -r requirements.txt
python shb_gui.py                     # the GUI
python shb.py                         # smoke test (no test suite exists)
python shb.py scan --sim-dir runs/afc1 --storage-min 2 --storage-max 10 \
    --n-points 40 --finesse-min 2 --finesse-max 20 --n-finesse 20
python shb.py scan --sim-dir runs/afc1 --resume     # after a crash (default)
python shb.py status  --sim-dir runs/afc1
python shb.py collect --sim-dir runs/afc1
```

There is no test suite. Before and after touching `shb.py`'s scan path,
sanity-check with the smoke test and a small before/after `run_scan` diff -
see *Verifying a change* below.

## The one invariant that silently corrupts results

`df` must divide every ground and excited level energy, and both spectrum
window edges, to within `1e-6` of a whole step. `validate_params` and
`to_steps` enforce this and raise `ValueError` otherwise - never bypass them.
The core works in integer *steps*, not raw MHz; a frequency that hasn't been
through `to_steps` should not be turned into a bin index by hand.

## Scan architecture

`run_scan` sweeps storage time x finesse. Each `(t, F)` cell maps to a
`(spacing_steps, width_steps)` pair on the `df` mesh; cells that land on the
same pair dedup to one burn (`uniq`). `df` can auto-refine (by a whole
integer factor, bounded by mesh size, time, and memory - see
`_resolve_scan_df`) when requested storage times are too close together to
resolve on the configured mesh.

The sweep itself is **score-only**: each worker returns
`(quality, n, spacing_MHz, width_MHz)`, not the spectrum - a spectrum and its
frequency axis are ~1.6 MB at `df=0.001`, so returning them per point would
pickle gigabytes back to the parent to keep a few dozen. Once the sweep
picks its top-N winners (by quality), a short **collect pass** re-burns just
those and shares one frequency axis across every kept record. Preserve this
shape when touching `run_scan` - don't have workers return full spectra again
without re-reading the memory analysis in git history (commit introducing
`_init_worker`/`_run_point`/`_collect`).

Each worker process builds its own `_scan_context` once via a `Pool`
initializer (`_init_worker`), not per task - a `functools.partial` handed to
`Pool.imap` looks like it pickles the context once, but CPython pickles
`(func, chunk)` per *chunk*, which is what silently multiplied the context
across the task queue before.

## Memory rules for the scan path

If you're touching `burn_parallel`, `compute_one`, or `run_scan`, keep these
invariants:

- **No full spectrum crosses the `Pool` pipe per point.** Only scores do; see
  *Scan architecture* above.
- **`burn_parallel`'s buffers (`hits`, `landing`, etc.) are allocated once
  outside the `pump_passes` loop and refilled, not rebuilt.** `landing` is
  `n_g x n_g x n_det` - tens to hundreds of MB at a fine mesh - and one lives
  per worker process. The in-place `divide`/`multiply` sequence there is
  provably bit-identical to the old three-array version; see the comment
  above it before changing it.
- **Every kept spectrum shares one frequency axis** (`ctx["f"]`), not a copy
  per record - this is also why pickling a `scan` dict (`.pkl` export) stays
  small (pickle memoises by object identity).
- **`run_scan` runs a pre-flight memory budget** (`_estimate_worker_bytes` /
  `_available_bytes`) before starting the `Pool`, and clamps `n_workers` /
  `top_n` - or refuses with an actionable message - rather than letting a
  sweep swap or get OOM-killed. If you add a new per-worker allocation,
  update `_estimate_worker_bytes`'s breakdown to account for it.

## `QUALITY_VERSION`

`_score_quality` is still being tuned (see git log). **Bump
`QUALITY_VERSION`** (near `DEFAULT_PARAMS`) whenever its formula changes -
it is part of every simulation folder's resume contract (`burn_spec`), and a
missed bump means a resumed scan silently mixes qualities computed two
different ways into one results table.

## Simulation folders

`run_scan(..., sim_dir=...)` checkpoints into a folder:

- `manifest.json` - written once; `burn_spec` (everything that changes what
  a checkpointed quality *means* - `df_used`, the crystal, the window, pump
  passes, burn mode, target window, `QUALITY_VERSION`) is the resume
  contract, checked field-by-field on resume. `grid_spec` (which cells are
  requested) is deliberately *not* part of that check - widening `n_points`
  or extending the storage range still resumes, reusing whatever `(s, w)`
  keys land on the same mesh points.
- `progress.csv` - **append-only**, one row per completed burn. This is the
  actual checkpoint; it is what makes a crash lose at most a few seconds of
  work. Never rewrite it wholesale.
- `status.json` - atomically replaced (`_atomic_write_json`, tmp + rename),
  advisory only (`_read_progress` never consults it - a resume works even
  with no `status.json` at all). It doubles as the run's lock: a folder is
  "live" only when its recorded `state` is `"running"` **and** the pid/host
  it names is still alive with a fresh heartbeat - check `state` first, or a
  long-lived process (a test driving several scans, say) will see its own
  pid as "live" forever.

`shb.load_run` / `shb.inspect_run` / `shb.collect_top` / `shb.write_run_outputs`
read a folder back in any state (complete, cancelled, crashed). The GUI's
cancel/crash path and its "Open run folder" both go through `load_run`, on
purpose - there is one reconstruction path, not two that could drift apart.

## Style

- Numpydoc-ish prose docstrings that explain the physics/reasoning, not just
  the signature; British spellings ("finesse", "colour" in comments,
  "optimise" etc.).
- Matplotlib styling via `signature.mplstyle` / `apply_style()`; plotting
  helpers draw onto a supplied `Axes` rather than creating their own figure.
- No new hard dependencies without a good reason (`psutil` is optional -
  `_available_bytes` degrades gracefully without it).

## Verifying a change to the scan path

There's no test suite, so lean on these:

1. `python shb.py` - the smoke test.
2. A small-grid regression: run `run_scan` on the same tiny grid before and
   after your change, serial and parallel, and diff `results` and `best`'s
   qualities/shapes. They should be bit-identical unless your change is
   deliberately numerical.
3. If touching checkpointing: start a scan with `sim_dir` set, `kill -9` it
   mid-sweep, resume it, and confirm the result equals an uninterrupted run
   of the same grid.

## Don't commit generated artefacts

The repo root already carries some (an 80 MB `.pkl`, a 26 MB `.html`, PNGs) -
`.gitignore` now excludes `*.pkl`, `*.npy`, `*.png`, `*.html`, `sim_*/`, and
`runs/`. Don't add more; regenerate them instead.
