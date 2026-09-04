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
see *Verifying a change* below. The smoke test also burns a GEM preparation
sequence, so it covers the staged path too.

## The one invariant that silently corrupts results

`df` must divide every ground and excited level energy, and both spectrum
window edges, to within `1e-6` of a whole step. `validate_params` and
`to_steps` enforce this and raise `ValueError` otherwise - never bypass them.
The core works in integer *steps*, not raw MHz; a frequency that hasn't been
through `to_steps` should not be turned into a bin index by hand.

## Preparation sequences

`run_scan` and `run_burn` both burn one flat set of pump frequencies.
`run_sequence` burns a *sequence*: `burn_sequence` calls the chosen burn
function once per stage, each on the populations the last stage left. Ordering
is physics - pit, then burn back, then clean - so never pool stages into one
frequency array "to save a call".

The row -> stage rule lives in `sequence_stages`: **consecutive rows sharing a
label are one stage**, burned together. That is what keeps comb teeth
simultaneous while a pit and its burn-back stay ordered; don't change it to
"one row, one stage" without re-reading why.

`run_sequence` returns `run_burn`'s dict plus `stages`, which is what lets the
GUI's plot and export paths stay unchanged - keep that shape.

Both `burn_sequence` and `run_sequence` take an optional `should_cancel`
callback, checked between stages (never mid-stage - there is no cheaper
checkpoint inside one `burn` call) and raising `shb.Cancelled` when it
returns true. This is what makes the GUI's Cancel button effective on a
sequence burn; `SimWorker.run_sequence_job` (`shb_gui.py`) wires it to
`self._cancel` and catches `shb.Cancelled`. `run_burn`/`run_burn_job` (a flat,
one-stage burn) has no equivalent hook - there's no stage boundary to check
at.

Two helpers hold the physics, and both are derived from the crystal rather than
written down, so they follow whichever level table is loaded:

- `anti_hole_pumps` - which pump frequencies refill the ground state that
  absorbs at a given frequency (and why candidates inside the pit are useless).
- `peak_shadow_bands` - the other frequencies at which a peak's own ions
  absorb. `prep_gem` cuts these out of its cleaning sweeps; without that the
  cleaning stage erases the anti-hole (measured: centre absorption 1.31 -> 0.22).
  If you touch the cleaning rows, re-check the centre absorption stays above
  `alpha`.

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

`run_scan` calls its optional `on_done(state, out, error)` at every terminal
exit - the pre-flight memory refusal, the `except` clause (only when
`_state == "failed"`, never `"cancelled"`), and right before the final
`return out` on success. This is the one hook a scan's outcome is reported
through (`scan_teams_notifier` builds Teams alerts from it) - if you touch
`run_scan`'s try/except/finally structure, keep calling `on_done` at all
three points, or a notification silently stops firing on some outcome.

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
4. If touching the sequence path: a one-stage sequence must give a spectrum
   bit-identical to `run_burn` on the same frequencies, and reordering the GEM
   stages must change the answer (burn back before the pit kills the peak).

## Don't commit generated artefacts

The repo root already carries some (an 80 MB `.pkl`, a 26 MB `.html`, PNGs) -
`.gitignore` now excludes `*.pkl`, `*.npy`, `*.png`, `*.html`, `sim_*/`, and
`runs/`. Don't add more; regenerate them instead.

`.gitignore` also excludes two local, personal files that must never be
committed: `.teams_webhook` (the Teams alert webhook URL - a bearer secret,
see `resolve_teams_webhook`) and `prep_presets.json` (saved
preparation-sequence presets, see `load_user_presets`). Neither should ever
show up in `git status` as anything but ignored.

## Not pip-installable, but still consumed elsewhere

`shb.py` is deliberately **not** a pip package - it's a single file other
projects vendor a copy of rather than install. [Flyswatter](https://github.com/Tymin0912/flyswatter)'s
"Spectral Feature (SHB)" tab is the one doing this today, keeping its own copy
under `third_party/shb/` (see that repo's `third_party/shb/NOTICE.md` for the
update procedure). Two things in this file exist *because* of that, and stay
true regardless of how the copy gets there:

- `cycler` (a matplotlib transitive dependency) is imported lazily, inside
  `apply_style()`'s fallback branch, not at module import time - `_STYLE_FALLBACK`
  stores its colours as a plain list (`_prop_cycle_colors`) and only becomes a
  `cycler.Cycler` when a `Figure` is actually about to be styled. `import shb`
  costs numpy + scipy + stdlib only; keep it that way - Flyswatter's frozen
  macOS app excludes matplotlib from the bundle entirely, so a vendored copy
  that imported `cycler` eagerly would crash on `import shb`. A venv with
  numpy and scipy but no matplotlib/cycler installed is the regression test
  (`python -c "import shb; shb.validate_params(shb.DEFAULT_PARAMS)"`).
- `USER_PRESETS_PATH` is resolved by `_user_presets_path()`, not hardcoded to
  `Path(__file__).with_name(...)`, because that would write into whatever
  directory a vendored copy sits in - read-only inside a frozen `.app` bundle.
  Resolution order: `$SHB_PREP_PRESETS` -> next to `shb.py` if that directory
  is writable and not inside a `site-packages`/`dist-packages` tree
  (preserves today's checkout behaviour) -> the platform user-config dir
  (`~/.config/shb/`, `%APPDATA%\shb\`, `~/Library/Application Support/shb/`).
  `load_user_presets`/`save_user_preset`/`delete_user_preset` all call
  `_user_presets_path()` fresh rather than closing over the module attribute,
  so changing `$SHB_PREP_PRESETS` mid-process (tests) takes effect
  immediately.
