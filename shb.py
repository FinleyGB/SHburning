"""
Spectral hole burning in Pr:YSO, and atomic frequency comb (AFC) design.

A single-file port of the MATLAB "Bloch-Simulator" from
https://zenodo.org/records/16962167, lifted out of the project notebooks
(``SHsimulation_v1*.ipynb``) so it can be imported, scripted and driven from a
GUI.

Layout
------
    Config      DEFAULT_PARAMS, settings, flatten/nest, JSON I/O, validate_params
    Core        to_steps, build_transitions, make_axes, initial_populations,
                burn_parallel, burn_series, spectrum, run_burn,
                normalisation_factor
    Combs       comb, comb_for, comb_at, single_hole
    Scan        compute_one, run_scan  (serial or multiprocessing)
    Plotting    matplotlib helpers that draw onto a supplied Axes, plus a few
                plotly builders kept for the notebooks' standalone HTML exports

The physics is a straight copy of the notebook code - only the packaging,
docstrings and naming have been tidied. See the NOTE in ``burn_parallel``.

Frequencies are in MHz throughout. ``df`` must divide every level energy and
both spectrum-window edges exactly; ``validate_params`` and ``to_steps`` enforce
this.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import pickle
import socket
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# Single-thread BLAS and FFT. These variables bind when numpy is imported, so
# they have to be set before it - which is also why worker.py carries the same
# block, for the notebooks that import it ahead of this module. A scan gets its
# parallelism from multiprocessing, so a threaded BLAS inside each of a dozen
# worker processes only fights itself for cores and adds a thread stack per core
# per worker.
_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")
for _var in _BLAS_THREAD_VARS:
    os.environ.setdefault(_var, "1")
del _var

import numpy as np

__all__ = [
    # config
    "DEFAULT_PARAMS", "PRESETS", "settings", "flatten", "nest",
    "params_to_json", "params_from_json", "validate_params",
    "uniform_dipole_params",
    # style
    "STYLE_PATH", "apply_style",
    # core
    "to_steps", "build_transitions", "make_axes", "initial_populations",
    "burn_parallel", "burn_series", "spectrum", "run_burn",
    "burn_sequence", "run_sequence", "normalisation_factor", "Cancelled",
    # combs / preparation sequences
    "comb", "comb_steps", "comb_for", "comb_at", "comb_in", "single_hole",
    "pulse_group", "sequence_stages", "anti_hole_pumps",
    "peak_shadow_bands", "default_pit_width",
    "prep_gem", "prep_narrowband_afc", "prep_broadband_afc", "PREP_PRESETS",
    "USER_PRESETS_PATH", "load_user_presets", "save_user_preset",
    "delete_user_preset",
    # scan
    "compute_one", "run_scan", "QUALITY_VERSION",
    # notifications
    "resolve_teams_webhook", "notify_teams", "scan_teams_notifier",
    # checkpointing / simulation folders
    "load_run", "inspect_run", "collect_top", "write_run_outputs",
    "RESULTS_COLUMNS",
    # plotting (matplotlib, draw onto an Axes)
    "plot_spectrum", "plot_comb", "plot_sequence", "plot_populations",
    "plot_level_diagram", "plot_scan_heatmap", "plot_scan_measured_finesse",
    "plot_finesse_written_vs_measured", "plot_scan_max", "plot_top_spectra",
    "plot_top_spectrum", "save_top_spectrum_figures", "SCAN_FIGURES",
    # plotting (plotly, return a Figure)
    "plot_spectrum_plotly", "plot_scan_heatmap_plotly",
    "plot_scan_max_plotly", "plot_top_spectra_plotly",
]


# ===========================================================================
# Plot style
# ===========================================================================

STYLE_PATH = Path(__file__).with_name("signature.mplstyle")

# Used only if signature.mplstyle is missing, so the module still runs
# standalone. The full set lives in the style sheet.
_STYLE_FALLBACK = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Plain list here, not a cycler.Cycler: cycler is a matplotlib transitive
    # dependency, and this dict is built at *module import time*. Turning it
    # into an ``axes.prop_cycle`` happens in apply_style()'s fallback branch
    # below, where matplotlib (and hence cycler) is known to be importable -
    # so `import shb` alone costs numpy + stdlib only, which matters for the
    # frozen app that excludes matplotlib. See CLAUDE.md.
    "_prop_cycle_colors": ["#1f77b4", "#e05c2a", "#2ca02c", "#9467bd",
                           "#8c564b", "#d62728", "#17becf", "#bcbd22"],
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "image.cmap": "inferno",
    "image.origin": "lower",
    "image.aspect": "auto",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style():
    """
    Apply the shared figure style (``signature.mplstyle``) process-wide.

    Call this once, before any Figure is created, so both the GUI canvases and
    any standalone plots match the notebook write-up figures. Falls back to a
    small inline rcParams update if the style sheet is not next to this module.
    """
    import matplotlib.pyplot as plt
    try:
        plt.style.use(str(STYLE_PATH))
    except (OSError, ValueError):
        import matplotlib as mpl
        from cycler import cycler
        rc = dict(_STYLE_FALLBACK)
        rc["axes.prop_cycle"] = cycler("color", rc.pop("_prop_cycle_colors"))
        mpl.rcParams.update(rc)


# ===========================================================================
# Config
# ===========================================================================

def _default_params():
    """A fresh copy of the Pr:YSO configuration (from SHsimulation_v1_7 cell 2)."""
    return dict(

        crystal_params={

            #        +-5/2(g)          +-3/2(g)  +-1/2(g)
            "g": [-(10.19 + 17.30),    -10.19,      0.00],   # MHz

            #        +-1/2(e)  +-3/2(e)     +-5/2(e)
            "e": [      0.00,      4.58,   4.58 + 4.84],      # MHz

            # Relative transition strengths, Nilsson et al. 2004 Table 1
            # and Sharma et al. 2023.
            #                      +-1/2(e)  +-3/2(e)  +-5/2(e)
            "dm": np.array([
                [                    0.052,    0.022,    0.926],   # +-5/2(g)
                [                    0.396,    0.596,    0.008],   # +-3/2(g)
                [                    0.552,    0.382,    0.066],   # +-1/2(g)
            ]),
        },

        linewidth_params={
            "h_width":  0.001,       # homogeneous FWHM (MHz), Lorentzian
            "in_width": 10000.0,     # inhomogeneous FWHM (MHz), Gaussian
        },

        numerical_params={
            "fmin": -50.0,           # spectrum window, low edge (MHz)
            "fmax":  50.0,           # spectrum window, high edge (MHz)
            "df":     0.001,         # mesh step (MHz); fine steps are slow.
                                     #
                                     # df MUST DIVIDE THE LEVEL ENERGIES ABOVE
                                     # AND THE WINDOW EDGES EXACTLY. The levels
                                     # are given to 0.01 MHz, so 0.1, 0.05,
                                     # 0.02, 0.01, 0.005 and 0.001 all work;
                                     # 0.03, 0.04 and 0.07 do not.
        },

        burn_params={
            "pump_passes": 1,        # how many times to repeat the pump list
        },

        output_params={
            "alpha": 1.0,            # absorption coefficient; 1.0 => normalised
            "norm":  None,           # peak of the unburned spectrum; computed
                                     # by normalisation_factor()
        },
    )


DEFAULT_PARAMS = _default_params()

# Bump whenever _score_quality's formula changes (see the reminder on that
# function). It goes into every simulation folder's manifest, so a resume
# against an old checkpoint is refused rather than silently mixing two
# versions of the metric into one results table.
QUALITY_VERSION = 3


def _preset(g, e, dm, h_width, in_width, fmin, fmax, df):
    """Build a full params dict for a crystal preset."""
    return dict(
        crystal_params={"g": list(g), "e": list(e),
                        "dm": np.asarray(dm, dtype=float)},
        linewidth_params={"h_width": float(h_width),
                          "in_width": float(in_width)},
        numerical_params={"fmin": float(fmin), "fmax": float(fmax),
                          "df": float(df)},
        burn_params={"pump_passes": 1},
        output_params={"alpha": 1.0, "norm": None},
    )


# Crystal presets. "Pr:YSO (site 1, tuned)" is the notebook's refined Pr:YSO
# (Nilsson et al. 2004 / Sharma et al. 2023 branching matrix); the rest are
# transcribed verbatim from the original MATLAB source,
# zenodo.org/records/16962167, file
# test_SHB_simulator_single_hole_examples.m. Each preset also carries a
# suggested fmin/fmax/df (df = h_width, the MATLAB default).
PRESETS = {
    "Pr:YSO (site 1, tuned)": _default_params(),

    "Pr:YSO (MATLAB source)": _preset(
        g=[0.0, -10.2, -27.5], e=[0.0, 4.6, 9.4],
        dm=np.full((3, 3), 1 / 3), h_width=0.02, in_width=2000.0,
        fmin=-200.0, fmax=200.0, df=0.02),

    # 1.4% Pr:La2(WO4)3 - not from the MATLAB source: hyperfine levels and
    # linewidths (25.3 kHz homogeneous, 18.8 GHz inhomogeneous) supplied
    # directly; dm is a uniform placeholder until a measured branching matrix
    # is available. df = 0.01 is the coarsest mesh dividing the levels.
    "1.4% Pr:LaWO": _preset(
        g=[0.0, -14.87, -39.31], e=[0.0, 4.94, 12.17],
        dm=np.full((3, 3), 1 / 3), h_width=0.0253, in_width=18800.0,
        fmin=-100.0, fmax=100.0, df=0.01),

    # 500 ppm Pr:CaWO4 - Yang et al., Phys. Rev. A 107, 022802 (2023). The
    # splittings are measured (ground 50.3 / 99.3 MHz, excited 4.2 / 6.4 MHz,
    # Fig. 4) but their ordering is only inferred, from dual-frequency hole
    # burning (Fig. 6): (a)/(b) are the two candidate ground orderings, (d)/(e)
    # the two excited ones. All four combinations are offered; the paper
    # concludes (b) + (e). h_width = 1/(pi T2) with T2 = 74.2 us at 1.7 K;
    # in_width is the 85.46 GHz optical FWHM (fitted as a Lorentzian in the
    # paper, Gaussian here - immaterial at this width). dm is a uniform
    # placeholder: no branching ratios were measured. The levels are given to
    # 0.1 MHz; df = 0.01 keeps the mesh at ~2 bins per homogeneous linewidth.
    "Pr:CaWO4 (Fig. 6a+d)": _preset(
        g=[0.0, -99.3, -149.6], e=[0.0, 4.2, 10.6],
        dm=np.full((3, 3), 1 / 3), h_width=0.00428, in_width=85460.0,
        fmin=-200.0, fmax=200.0, df=0.01),

    "Pr:CaWO4 (Fig. 6a+e)": _preset(
        g=[0.0, -99.3, -149.6], e=[0.0, 6.4, 10.6],
        dm=np.full((3, 3), 1 / 3), h_width=0.00428, in_width=85460.0,
        fmin=-200.0, fmax=200.0, df=0.01),

    "Pr:CaWO4 (Fig. 6b+d)": _preset(
        g=[0.0, -50.3, -149.6], e=[0.0, 4.2, 10.6],
        dm=np.full((3, 3), 1 / 3), h_width=0.00428, in_width=85460.0,
        fmin=-200.0, fmax=200.0, df=0.01),

    "Pr:CaWO4 (Fig. 6b+e, paper)": _preset(
        g=[0.0, -50.3, -149.6], e=[0.0, 6.4, 10.6],
        dm=np.full((3, 3), 1 / 3), h_width=0.00428, in_width=85460.0,
        fmin=-200.0, fmax=200.0, df=0.01),

    "151Eu:YSO (site 1)": _preset(
        g=[0.0, -34.5, -80.7], e=[0.0, 75.0, 176.6],
        dm=np.full((3, 3), 1 / 3), h_width=0.1, in_width=2000.0,
        fmin=-1000.0, fmax=1000.0, df=0.1),

    "171Yb:YSO (site 2)": _preset(
        g=[0.0, -655.0, -2497.0, -3025.0],
        e=[0.0, 2623.0, 4417.0, 4705.0],
        # branching ratio table, laser polarised along the crystal D2 axis
        dm=[[0.15, 0.06, 0.08, 0.71],
            [0.06, 0.19, 0.71, 0.04],
            [0.07, 0.71, 0.16, 0.06],
            [0.72, 0.04, 0.05, 0.19]],
        h_width=0.5, in_width=1000.0, fmin=-2000.0, fmax=10000.0, df=0.5),

    "Zeeman doublet": _preset(
        g=[0.0, 100.0], e=[0.0, 25.0],
        dm=np.full((2, 2), 0.5), h_width=0.1, in_width=2000.0,
        fmin=-1000.0, fmax=1000.0, df=0.1),
}


def settings(params):
    """Flatten the nested ``params`` into the one flat dict the core reads."""
    flat = {}
    for section in params.values():
        flat.update(section)
    return flat


# ``flatten`` is a public alias; ``settings`` is kept because the notebook code
# and worker.py both call it by that name.
flatten = settings


def nest(flat, template=None):
    """
    Fold a flat dict back into the nested ``params`` shape.

    Keys not present in ``template`` (default: DEFAULT_PARAMS) are dropped.
    """
    template = DEFAULT_PARAMS if template is None else template
    out = {}
    for section, contents in template.items():
        out[section] = {k: flat[k] for k in contents if k in flat}
    return out


def uniform_dipole_params(params):
    """
    A copy of ``params`` with the ``dm`` branching matrix replaced by a uniform
    one (every excited state reached with equal probability, rows summing to 1).

    Used by the GUI's "Relative dipole strength" toggle to show what the
    branching ratios are doing.
    """
    out = copy.deepcopy(params)
    dm = np.asarray(out["crystal_params"]["dm"], dtype=float)
    out["crystal_params"]["dm"] = np.full_like(dm, 1.0 / dm.shape[1])
    return out


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def params_to_json(params, indent=2):
    """Serialise ``params`` to a JSON string (numpy arrays become nested lists)."""
    return json.dumps(params, default=_json_default, indent=indent)


def params_from_json(text):
    """
    Parse a JSON string written by :func:`params_to_json` back into ``params``,
    restoring ``dm`` as a numpy array and merging onto DEFAULT_PARAMS so missing
    keys keep their defaults.
    """
    loaded = json.loads(text)
    out = _default_params()
    for section, contents in loaded.items():
        if section not in out:
            continue
        out[section].update(contents)
    dm = out["crystal_params"].get("dm")
    if dm is not None:
        out["crystal_params"]["dm"] = np.asarray(dm, dtype=float)
    return out


def validate_params(params):
    """
    Check the mesh is self-consistent, raising ``ValueError`` if not.

    ``df`` must divide every ground and excited level energy, and both window
    edges, to within 1e-6 of a whole step - otherwise the transition table and
    the spectrum axis fall between bins and the results are silently wrong.
    """
    p = settings(params)
    df = float(p["df"])
    if df <= 0:
        raise ValueError(f"df must be positive, got {df!r}")
    if p["fmax"] <= p["fmin"]:
        raise ValueError(f"fmax ({p['fmax']}) must be greater than fmin "
                         f"({p['fmin']})")

    checks = {
        "ground levels g": np.asarray(p["g"], dtype=float),
        "excited levels e": np.asarray(p["e"], dtype=float),
        "window edges": np.array([p["fmin"], p["fmax"]], dtype=float),
    }
    offenders = []
    for name, values in checks.items():
        steps = values / df
        if np.any(np.abs(steps - np.round(steps)) > 1e-6):
            offenders.append(name)
    if offenders:
        raise ValueError(
            f"df = {df:g} MHz does not divide {', '.join(offenders)} exactly. "
            f"Pick a df that does: for the default levels (given to 0.01 MHz) "
            f"that means 0.01, 0.005 or 0.001."
        )

    dm = np.asarray(p["dm"], dtype=float)
    if dm.shape != (len(p["g"]), len(p["e"])):
        raise ValueError(f"dm shape {dm.shape} does not match "
                         f"{len(p['g'])} ground x {len(p['e'])} excited levels")
    return True


# ===========================================================================
# Core  (verbatim from SHsimulation_v1_7.ipynb cell 6 / worker.py)
# ===========================================================================

def to_steps(values, df):
    """
    MHz to whole df steps. df must divide every frequency in the simulation
    exactly.
    """
    steps = np.asarray(values, dtype=float) / df
    if np.any(np.abs(steps - np.round(steps)) > 1e-6):
        raise ValueError(f"not a whole number of df = {df:g} steps - see "
                         f"the note in the config")
    return np.round(steps).astype(np.int64)


def build_transitions(params):
    """
    The table of transition frequencies, carried in whole df steps.

    Returns
        g, e         sorted energies (MHz)
        trans        [Ng, Ne], trans[k, m] = e[m] - g[k]   (MHz)
        trans_steps  the same table in whole df steps      (int)
    """
    p = settings(params)
    df = p["df"]
    g = np.asarray(p["g"], dtype=float)
    e = np.asarray(p["e"], dtype=float)

    g_steps = to_steps(g, df)
    e_steps = to_steps(e, df)

    trans_steps = e_steps[None, :] - g_steps[:, None]
    return g, e, trans_steps * df, trans_steps


def make_axes(params, trans_steps, fmin=None, fmax=None):
    """
    The spectrum axis f and the atomic detuning axis det.

    Returns f, det, and det0 - the bin index of det[0], which is what turns a
    frequency in steps into a column of rho.
    """
    p = settings(params)
    df = p["df"]
    fmin = p["fmin"] if fmin is None else fmin
    fmax = p["fmax"] if fmax is None else fmax

    f0 = int(to_steps(fmin, df))
    f1 = int(to_steps(fmax, df))
    pad = int(np.abs(trans_steps).max())

    f = df * np.arange(f0, f1 + 1)
    det = df * np.arange(f0 - pad, f1 + pad + 1)
    return f, det, f0 - pad


def initial_populations(g, det):
    """Equal population in every ground state, for every ion class."""
    return np.ones((len(g), len(det))) / len(g)


def burn_parallel(rho, pump_freqs, trans_steps, det, det0, params,
                  pump_passes=None):
    """
    PARALLEL burn - every pump frequency acts at once.

    All the frequencies are present together, so they all act on the SAME
    populations: the ions go up together and decay back down together. The
    order of pump_freqs therefore makes no difference to the answer.

    rho goes in, a new rho comes out; the input is not modified.
    """
    p = settings(params)
    pump_passes = p["pump_passes"] if pump_passes is None else pump_passes
    df, dm = p["df"], p["dm"]
    n_g, n_e = trans_steps.shape
    rho = rho.copy()

    # The pump frequencies as whole bins. This is where an off-mesh comb is
    # caught, before any of it is burned.
    pump_steps = to_steps(pump_freqs, df)

    n_det = len(det)

    # Working buffers, allocated once and refilled each pass rather than built
    # anew. ``landing`` is the big one - n_g x n_g x n_det, so tens of MB at a
    # fine mesh and hundreds at a refined one - and a scan holds one of these
    # per worker process, so rebuilding it every pass is what pushed a long
    # sweep into swap.
    hits = np.zeros_like(rho)
    landing = np.zeros((n_g, n_g, n_det))
    outer_buf = np.empty((n_g, n_det))
    mask = np.empty(rho.shape, dtype=bool)
    excited = np.empty_like(rho)
    acc = np.empty_like(rho)

    for _ in range(pump_passes):

        # hits[k, d]        how many addressed transitions take ions out of
        #                   ground state k in ion class d
        # landing[j, k, d]  the branchings of those same transitions, summed -
        #                   so landing/hits is where the ions actually go
        # NOTE: carried over from the notebook - the author flagged this block
        # for a final correctness check. Behaviour is left exactly as it was;
        # only the per-pump-frequency Python loop is replaced by a bincount, so
        # a fine comb (tens of thousands of pump frequencies) is no longer slow.
        hits.fill(0.0)
        landing.fill(0.0)

        for k in range(n_g):
            for m in range(n_e):
                # Resonant detuning bin for every pump frequency at once.
                # Integer subtraction - the bin is computed, not rounded to.
                d = pump_steps - int(trans_steps[k, m]) - det0
                d = d[(d >= 0) & (d < n_det)]
                if d.size == 0:
                    continue
                counts = np.bincount(d, minlength=n_det)
                hits[k] += counts
                np.multiply(dm[:, m][:, None], counts, out=outer_buf)
                landing[:, k, :] += outer_buf

        # The ions leave. A class addressed three times over still only has
        # one population to give, so this is a mask, not a count. The mask is
        # kept because the divide below needs the same test.
        np.greater(hits, 0, out=mask)
        np.multiply(rho, mask, out=excited)

        # And come back down, sharing equally between the branchings of
        # whichever transitions took them.
        #
        # The share is written back over ``landing`` instead of into a second
        # array that size. That is safe rather than merely thrifty: hits[k, d]
        # and landing[:, k, d] are accumulated from the same ``counts``, and a
        # bincount is never negative, so hits[k, d] == 0 implies
        # landing[:, k, d] == 0 exactly - the entries ``where`` skips already
        # hold the 0.0 that the old ``out=np.zeros_like(landing)`` supplied.
        np.divide(landing, hits, out=landing, where=mask)
        landing *= excited
        np.sum(landing, axis=1, out=acc)
        rho -= excited
        rho += acc

    return rho


def burn_series(rho, pump_freqs, trans_steps, det, det0, params,
                pump_passes=None):
    """
    SERIES burn - one pump frequency after another.

    This is the original MATLAB ``SHB_simulator.m`` pumping model. It is
    genuinely sequential (each pump frequency sees the populations the previous
    one left), so it cannot be vectorised the way ``burn_parallel`` is and stays
    slower for a fine comb. It is not the scan default.
    """
    p = settings(params)
    pump_passes = p["pump_passes"] if pump_passes is None else pump_passes
    df, dm = p["df"], p["dm"]
    n_g, n_e = trans_steps.shape
    n_det = len(det)
    rho = rho.copy()

    pump_steps = to_steps(pump_freqs, df)
    # d_table[p, k, m] = resonant detuning bin of pump p on transition (k, m)
    d_table = pump_steps[:, None, None] - trans_steps[None, :, :] - det0

    for _ in range(pump_passes):
        for pd in d_table:
            for k in range(n_g):
                for m in range(n_e):
                    d = int(pd[k, m])
                    if not (0 <= d < n_det):
                        continue
                    removed = rho[k, d]
                    rho[k, d] = 0.0
                    rho[:, d] += removed * dm[:, m]

    return rho


def _spectrum_loop(rho, trans, det, f, params, norm=None, alpha=None):
    """
    Reference absorption spectrum - a direct transcription of the triple loop in
    the MATLAB ``SHB_simulator.m``. Kept as the oracle for the equivalence test
    and as readable documentation of the model; ``spectrum`` computes the same
    sum far faster. O(n_det * n_g * n_e * n_f).
    """
    p = settings(params)
    norm = (p["norm"] if p["norm"] is not None else 1.0) if norm is None else norm
    alpha = p["alpha"] if alpha is None else alpha
    h_width, in_width, dm = p["h_width"], p["in_width"], p["dm"]

    spec = np.zeros(len(f))
    half = h_width / 2.0
    n_g, n_e = trans.shape

    for d in range(len(det)):
        gauss = np.exp(-4 * np.log(2) * det[d]**2 / in_width**2)
        for k in range(n_g):
            pop = rho[k, d]
            if pop == 0.0:
                continue
            for m in range(n_e):
                spec += (dm[k, m] * pop * half**2
                         / ((trans[k, m] + det[d] - f)**2 + half**2)
                         * gauss)

    return (spec / norm) * alpha


class _SpectrumCtx:
    """
    Everything ``spectrum`` needs that does not depend on ``rho`` - so it can be
    built once and reused for every point of a scan.

    gauss     inhomogeneous Gaussian envelope, sampled on ``det``
    Lf_hat    rfft of the (full, untruncated) Lorentzian kernel
    nfft      transform length
    conv_len  useful length of the linear convolution, 3*n_det - 2
    n_f       len(f)
    bases     bases[k, m] - where transition (k, m) reads out of conv_k
    n_g, n_e  transition-table shape
    """

    def __init__(self, params, det, f):
        p = settings(params)
        df = p["df"]
        half = p["h_width"] / 2.0
        n_det = len(det)
        self.n_f = len(f)

        self.gauss = np.exp(-4 * np.log(2) * det**2 / p["in_width"]**2)

        # Lorentzian over every integer lag the convolution can need,
        # lag in [-(n_det-1) .. n_det-1]; even, untruncated (1/x**2 tails kept).
        lags = np.arange(-(n_det - 1), n_det) * df
        Lf = half**2 / (lags**2 + half**2)

        self.conv_len = 3 * n_det - 2
        try:
            from scipy.fft import next_fast_len
            self._use_scipy = True
            self.nfft = int(next_fast_len(self.conv_len))
        except ImportError:                              # numpy fallback
            self._use_scipy = False
            self.nfft = 1 << int(np.ceil(np.log2(self.conv_len)))
        self.Lf_hat = self._rfft(Lf)

        g_steps = to_steps(p["g"], df)
        e_steps = to_steps(p["e"], df)
        trans_steps = e_steps[None, :] - g_steps[:, None]
        self.n_g, self.n_e = trans_steps.shape
        pad = (n_det - self.n_f) // 2
        self.bases = pad + (n_det - 1) - trans_steps      # (n_g, n_e), >= 0

    # Function handles are resolved per call rather than stored, so the context
    # pickles cleanly when it is handed to a multiprocessing worker.
    def _rfft(self, x):
        if self._use_scipy:
            from scipy.fft import rfft
            return rfft(x, self.nfft, workers=1)
        return np.fft.rfft(x, self.nfft)

    def _irfft(self, x):
        if self._use_scipy:
            from scipy.fft import irfft
            return irfft(x, self.nfft, workers=1)
        return np.fft.irfft(x, self.nfft)


def spectrum(rho, trans, det, f, params, norm=None, alpha=None, ctx=None):
    """
    Absorption spectrum from the populations.

    Same result as :func:`_spectrum_loop` (the MATLAB triple loop) to floating
    point, but computed as one FFT convolution per ground state instead of a
    loop over every detuning bin. Pass a prebuilt ``ctx`` (see ``_SpectrumCtx``)
    to skip the per-call setup - :func:`run_scan` does this once for the whole
    sweep.

    norm and alpha default to config; the spectrum is divided by norm and
    multiplied by alpha, which puts the unburned crystal at alpha.
    """
    p = settings(params)
    norm = (p["norm"] if p["norm"] is not None else 1.0) if norm is None else norm
    alpha = p["alpha"] if alpha is None else alpha
    dm = np.asarray(p["dm"], dtype=float)

    if ctx is None:
        ctx = _SpectrumCtx(params, det, f)

    w = rho * ctx.gauss                                   # (n_g, n_det)
    spec = np.zeros(ctx.n_f)
    for k in range(ctx.n_g):
        conv_k = ctx._irfft(ctx._rfft(w[k]) * ctx.Lf_hat)[:ctx.conv_len]
        for m in range(ctx.n_e):
            b = int(ctx.bases[k, m])
            spec += dm[k, m] * conv_k[b:b + ctx.n_f]

    return (spec / norm) * alpha


def run_burn(pump_freqs, params, burn=burn_parallel,
             pump_passes=None, fmin=None, fmax=None, ctx=None):
    """
    The whole thing: set up, burn, take the spectrum.

    Pass ``burn=burn_series`` for a series burn instead of a parallel one.
    Pass a prebuilt ``ctx`` (a ``_SpectrumCtx``) to reuse the spectrum setup
    across many calls - :func:`run_scan` does this.

    Returns a dictionary with
        f, spec       the spectrum and its frequency axis
        rho, rho0     populations after and before the burn
        det           the atomic detuning axis (the x axis for rho)
        trans         the transition table (MHz)
        pump_freqs    what was burned
    """
    g, e, trans, trans_steps = build_transitions(params)
    f, det, det0 = make_axes(params, trans_steps, fmin, fmax)
    rho0 = initial_populations(g, det)
    rho = burn(rho0, pump_freqs, trans_steps, det, det0, params, pump_passes)
    spec = spectrum(rho, trans, det, f, params, ctx=ctx)

    return {"f": f, "spec": spec, "rho": rho, "rho0": rho0,
            "det": det, "trans": trans, "trans_steps": trans_steps,
            "pump_freqs": np.asarray(pump_freqs)}


class Cancelled(Exception):
    """Raised by :func:`burn_sequence` / :func:`run_sequence` when their
    ``should_cancel`` callback returns true. Caught by the caller (typically a
    GUI worker) to stop cleanly between stages rather than mid-stage."""


def burn_sequence(rho, stages, trans_steps, det, det0, params,
                  burn=burn_parallel, log=None, should_cancel=None):
    """
    Burn a preparation sequence: one stage after another, each acting on the
    populations the previous stage left.

    A preparation sequence is ordered by physics, not by bookkeeping. Burning
    a pit and its burn-back pulses at the same time would empty the anti-hole
    as fast as it was filled, and cleaning pulses only mean anything once
    there is a peak to clean, so the stages cannot be pooled into one flat set
    of pump frequencies the way a comb can.

    Within a stage the pump frequencies are still handed to ``burn`` in one
    call, so ``burn_parallel``'s "all present together" model (and
    ``burn_series``'s one-after-another model) applies inside the stage exactly
    as it does for a comb.

    ``stages`` are as :func:`sequence_stages` returns them: dicts with
    ``freqs``, an optional ``passes`` (``None`` uses the configured
    ``pump_passes``) and a ``name`` for the log.

    ``should_cancel``, if given, is called between stages (a stage itself
    still runs to completion once started - there is no cheaper checkpoint
    inside a single ``burn`` call); when it returns true this raises
    :class:`Cancelled` rather than continuing to the next stage.
    """
    for st in stages:
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        freqs = np.asarray(st["freqs"], dtype=float)
        if freqs.size == 0:
            continue
        rho = burn(rho, freqs, trans_steps, det, det0, params,
                   st.get("passes"))
        if log is not None:
            log(f"  {st.get('name', 'stage')}: {freqs.size} pump frequencies"
                f" x {st.get('passes') or settings(params)['pump_passes']}"
                f" passes")
    return rho


def run_sequence(stages, params, burn=burn_parallel, fmin=None, fmax=None,
                 ctx=None, log=None, should_cancel=None):
    """
    :func:`run_burn` for a preparation sequence: set up, burn the stages in
    order, take the spectrum.

    Returns the same dictionary as :func:`run_burn` - so every consumer of a
    burn result (plots, exports) keeps working unchanged - plus

        stages        the stages that were burned, in order

    with ``pump_freqs`` holding every frequency the sequence used.

    ``should_cancel``, if given, is forwarded to :func:`burn_sequence` and
    checked between stages; see there for what "between stages" means.
    """
    g, e, trans, trans_steps = build_transitions(params)
    f, det, det0 = make_axes(params, trans_steps, fmin, fmax)
    rho0 = initial_populations(g, det)
    rho = burn_sequence(rho0, stages, trans_steps, det, det0, params,
                        burn=burn, log=log, should_cancel=should_cancel)
    spec = spectrum(rho, trans, det, f, params, ctx=ctx)

    parts = [np.asarray(st["freqs"], dtype=float) for st in stages
             if len(st["freqs"])]
    pump_freqs = (np.unique(np.concatenate(parts)) if parts
                  else np.array([], dtype=float))

    return {"f": f, "spec": spec, "rho": rho, "rho0": rho0,
            "det": det, "trans": trans, "trans_steps": trans_steps,
            "pump_freqs": pump_freqs, "stages": stages}

def normalisation_factor(params):
    """
    Peak absorption of the unburned crystal.

    Dividing a spectrum by this and multiplying by alpha puts the unburned
    crystal at alpha, with holes below it and antiholes above. It has to be
    computed rather than written down.

    Measured over a window that always spans the transition manifold
    (+-max(50, max|trans|) MHz): +-50 MHz for Pr:YSO, but wide enough that
    Eu:YSO and Yb:YSO, whose transitions sit hundreds to thousands of MHz out,
    still see their line rather than an empty window.
    """
    g, e, trans, trans_steps = build_transitions(params)
    lim = max(50.0, float(np.abs(trans).max()))
    f, det, det0 = make_axes(params, trans_steps, fmin=-lim, fmax=lim)
    rho = initial_populations(g, det)
    spec = spectrum(rho, trans, det, f, params, norm=1.0, alpha=1.0)
    return float(spec.max())


# ===========================================================================
# Combs
# ===========================================================================

def single_hole(centre=0.0):
    """One pump frequency, at ``centre`` MHz."""
    return np.array([centre])


def comb(spacing_steps, width_steps, params):
    """
    Teeth every ``spacing_steps`` bins across the spectrum window, each tooth
    ``width_steps`` bins wide. ``width_steps`` is a whole number of df steps.

    ``spacing_steps`` may be fractional (see :func:`period_steps`): tooth ``i``
    then starts at ``round(i * spacing_steps)``, so every tooth sits on the
    mesh, its position off by at most half a step, while the *mean* period -
    which is what sets the echo time - is exactly ``spacing_steps``. A whole
    ``spacing_steps`` gives the plain, perfectly regular comb, unchanged.
    """
    p = settings(params)
    df = p["df"]
    f0 = int(to_steps(p["fmin"], df))
    f1 = int(to_steps(p["fmax"], df))

    n_teeth = int((f1 - f0) // spacing_steps) + 1
    starts = f0 + np.rint(spacing_steps * np.arange(n_teeth)).astype(int)
    return df * (starts[:, None] + np.arange(width_steps)[None, :]).ravel()


def period_steps(storage_time, df):
    """
    The AFC tooth period, in df steps, for a storage time of ``storage_time``
    us: ``1 / (storage_time * df)``.

    Not rounded to a whole step - the mesh only offers storage times
    ``1 / (k * df)``, which bunch up at short storage times and spread out at
    long ones (the gap grows as ``t**2 * df``), so rounding the period would
    sample a linear grid of storage times unevenly. :func:`comb` burns a
    fractional period exactly on average instead. A period within ``1e-6`` of
    a whole step snaps to that ``int``, so an on-mesh storage time burns the
    same perfectly regular comb (and checkpoints under the same key) as it
    always has. Never below one step.
    """
    s = 1.0 / (float(storage_time) * float(df))
    k = round(s)
    if abs(s - k) < 1e-6:
        return int(max(1, k))
    return max(1.0, s)


def comb_steps(storage_time, finesse, df):
    """
    The ``(spacing_steps, width_steps)`` of the comb for a storage time of
    ``storage_time`` us and a finesse of ``finesse``.

    The spacing is :func:`period_steps` (possibly fractional); the tooth width
    is ``spacing / finesse`` rounded to a whole step, never below one step nor
    wider than the period. This is the one rule :func:`run_scan` and
    :func:`comb_for` share, so a scan cell's ``(t, F)`` re-burned as a single
    comb is the same burn.
    """
    s = period_steps(storage_time, df)
    w = int(max(1, min(int(s), round(s / finesse))))
    return s, w


def comb_for(storage_time, finesse, params):
    """
    A comb across the whole config window for a storage time of
    ``storage_time`` us and a finesse of ``finesse`` - the experimentally
    meaningful pair, rather than the tooth spacing and width themselves.

    Mapped to the mesh by :func:`comb_steps` and burned by :func:`comb`, so
    neither value needs to land on the ``df`` mesh: the storage time is
    honoured exactly on average (a fractional period), the finesse to the
    nearest whole step of tooth width.

    Returns
        freqs           the pump frequencies (MHz), exactly on the mesh
        actual_storage  the storage time the mean period gives (us)
        actual_finesse  the spacing over the mesh-snapped tooth width
        n_teeth         how many teeth fitted in the window
    """
    df = settings(params)["df"]
    s, w = comb_steps(storage_time, finesse, df)
    freqs = comb(s, w, params)
    return freqs, 1.0 / (s * df), s / w, freqs.size // w


def comb_at(tooth_spacing, burn_width, params):
    """
    A comb across the whole config window: teeth every ``tooth_spacing`` MHz,
    each tooth ``burn_width`` MHz wide.

    Returns
        freqs           the pump frequencies (MHz), exactly on the mesh
        actual_spacing  the spacing, snapped to the mesh (MHz)
        actual_width    the width, snapped to the mesh (MHz)
        n_teeth         how many teeth fitted in the window
    """
    p = settings(params)
    df = p["df"]

    # to_steps rather than round: an off-mesh spacing or width stops here.
    spacing_steps = int(to_steps(tooth_spacing, df))
    width_steps = int(to_steps(burn_width, df))

    f0 = int(to_steps(p["fmin"], df))
    f1 = int(to_steps(p["fmax"], df))

    n_teeth = (f1 - f0) // spacing_steps + 1
    starts = f0 + spacing_steps * np.arange(n_teeth)
    freqs = df * (starts[:, None] + np.arange(width_steps)[None, :]).ravel()

    return freqs, spacing_steps * df, width_steps * df, n_teeth


def comb_in(tooth_spacing, burn_width, lo, hi, params):
    """
    A comb confined to ``[lo, hi]`` MHz rather than the whole config window.

    Same geometry as :func:`comb_at` - teeth every ``tooth_spacing`` MHz, each
    ``burn_width`` MHz wide, the first starting at ``lo`` - but bounded by a
    band the caller chooses. A narrowband AFC burns its comb over a few MHz
    inside a prepared pit, not across the whole inhomogeneous line, which is
    what this is for.

    Returns
        freqs           the pump frequencies (MHz), exactly on the mesh
        actual_spacing  the spacing, snapped to the mesh (MHz)
        actual_width    the width, snapped to the mesh (MHz)
        n_teeth         how many teeth fitted in the band
    """
    p = settings(params)
    df = p["df"]

    # to_steps rather than round: an off-mesh spacing, width or edge stops here.
    spacing_steps = int(to_steps(tooth_spacing, df))
    width_steps = int(to_steps(burn_width, df))
    f0 = int(to_steps(lo, df))
    f1 = int(to_steps(hi, df))

    if spacing_steps < 1 or width_steps < 1:
        raise ValueError("tooth spacing and burn width must be at least one "
                         f"df step ({df} MHz)")
    if f1 < f0:
        raise ValueError(f"empty comb band: hi ({hi}) is below lo ({lo})")

    n_teeth = (f1 - f0) // spacing_steps + 1
    starts = f0 + spacing_steps * np.arange(n_teeth)
    freqs = df * (starts[:, None] + np.arange(width_steps)[None, :]).ravel()

    return freqs, spacing_steps * df, width_steps * df, n_teeth


def pulse_group(centre, sweep, params):
    """
    Every pump frequency across ``centre +- sweep/2`` MHz, one per df step.

    This is the model's picture of a swept (chirped) burning pulse. The burn
    engine sees a *set of frequencies* - it carries no intensity and no pulse
    duration - so a sweep is represented by the frequencies it visits, always
    at the crystal's own configured resolution: ``sweep=0`` is a single pulse
    at ``centre``; otherwise every mesh bin across the span is hit. There is
    no coarser "stepped chirp" option - a gap in the sweep is never what a
    pit, burn-back, or cleaning stage wants.

    The arithmetic is done in whole df steps, so every frequency returned is on
    the mesh by construction; ``to_steps`` gates ``centre`` and ``sweep``, so an
    off-mesh value stops here rather than being quietly rounded.
    """
    p = settings(params)
    df = p["df"]

    c = int(to_steps(centre, df))
    s = abs(int(to_steps(sweep, df)))
    if s == 0:
        return np.array([c * df])

    offsets = np.arange(s + 1) - s // 2
    return df * (c + offsets)


def sequence_stages(rows, params):
    """
    Turn preparation-sequence rows into the stages :func:`burn_sequence` burns.

    A row is ``{"label", "centre", "sweep", "passes", "on"}`` - one swept (or
    single) burning pulse, as entered in the GUI's sequence editor.
    **Consecutive rows sharing a label become one stage**, burned together;
    a change of label starts a new stage, burned afterwards on whatever
    populations the previous stage left. That one rule covers both cases that
    matter: the teeth of a comb (many rows, one label) go in at once, the way a
    comb burn actually works, while a pit and its burn-back (different labels)
    stay strictly ordered - burn them together and the anti-hole is simply
    erased again.

    A stage takes its pump-pass count from the first row of the group; ``None``
    means "use the configured ``pump_passes``". Rows with ``on=False`` are
    skipped.

    Returns a list of ``{"name", "freqs", "passes"}``.
    """
    grouped = []
    for row in rows:
        if not row.get("on", True):
            continue
        label = str(row.get("label", "")).strip() or "stage"
        freqs = pulse_group(row.get("centre", 0.0), row.get("sweep", 0.0),
                            params)
        if grouped and grouped[-1]["name"] == label:
            grouped[-1]["parts"].append(freqs)
            continue
        passes = row.get("passes")
        passes = None if passes in (None, "") else int(passes)
        grouped.append({"name": label, "parts": [freqs], "passes": passes})

    return [{"name": st["name"],
             "freqs": np.unique(np.concatenate(st["parts"])),
             "passes": st["passes"]}
            for st in grouped]


def anti_hole_pumps(centre, params, *, avoid=None, limit=None):
    """
    The "burn back" pump frequencies that put absorption back at ``centre`` MHz.

    Worked from the crystal's own level structure rather than written down, so
    it follows whichever preset (or hand-edited level table) is loaded.

    The reasoning: absorption at ``centre`` comes from the ion class at
    detuning ``d0 = centre - trans[k0, m0]``, through whatever population it
    holds in ground state ``k0``. A pit burned at ``centre`` has emptied that
    ``k0`` - that is what makes the pit. Pumping any *other* transition
    ``(k, m)``, ``k != k0``, of the same class - which sits at
    ``trans[k, m] + d0`` - lifts those ions out of ``k`` and drops a share
    ``dm[k0, m]`` of them back into ``k0``, so the class absorbs at ``centre``
    again: an anti-hole inside the pit.

    ``avoid=(lo, hi)`` excludes the pit itself. A pump inside the pit would
    address ions the pit has already emptied (nothing left to move) and would
    re-burn the pit at the same time, so those candidates are dropped.

    Candidates are ranked by ``dm[k0, m0] * dm[k0, m]`` - how strongly the
    route fills ``k0``, times how strongly ``k0`` then absorbs - and deduped by
    frequency, since several routes can land on the same pump frequency.

    Returns a list of ``{"freq", "target": (k0, m0), "via": (k, m), "weight"}``,
    strongest first, truncated to ``limit`` if given.
    """
    p = settings(params)
    df, dm = p["df"], p["dm"]
    g, e, trans, trans_steps = build_transitions(params)
    n_g, n_e = trans_steps.shape

    c_steps = int(to_steps(centre, df))
    lo = hi = None
    if avoid is not None:
        lo, hi = float(min(avoid)), float(max(avoid))

    found = []
    for k0 in range(n_g):
        for m0 in range(n_e):
            for k in range(n_g):
                if k == k0:
                    continue            # pumping k0 empties it again
                for m in range(n_e):
                    weight = float(dm[k0, m0] * dm[k0, m])
                    if weight <= 0.0:
                        continue
                    # Integer steps throughout: the pump frequency is computed
                    # on the mesh, not rounded onto it.
                    nu_steps = (c_steps - int(trans_steps[k0, m0])
                                + int(trans_steps[k, m]))
                    nu = nu_steps * df
                    if not (p["fmin"] <= nu <= p["fmax"]):
                        continue
                    if lo is not None and lo <= nu <= hi:
                        continue
                    found.append({"freq": float(nu), "target": (k0, m0),
                                  "via": (k, m), "weight": weight,
                                  "steps": nu_steps})

    found.sort(key=lambda r: (-r["weight"], abs(r["steps"] - c_steps)))

    seen, uniq = set(), []
    for r in found:
        if r["steps"] in seen:
            continue
        seen.add(r["steps"])
        uniq.append({k: v for k, v in r.items() if k != "steps"})

    return uniq if limit is None else uniq[:int(limit)]


# ---------------------------------------------------------------------------
# Preparation sequences
#
# Each builder returns editor *rows* - not frequencies - so a preset is a
# starting point the sequence editor then lets you rearrange, retune or delete.
# The defaults are derived from the loaded crystal wherever there is a sensible
# way to derive them; they are a reasonable place to start, not a claim about
# the one right recipe for your experiment.
# ---------------------------------------------------------------------------

def _span_row(label, lo_steps, hi_steps, df, passes):
    """
    A row filling every mesh bin from ``lo_steps`` to ``hi_steps`` inclusive.

    The span is nudged up by one bin when it is odd, so the row's centre lands
    on the mesh rather than half a step off it.
    """
    if hi_steps < lo_steps:
        return None
    span = hi_steps - lo_steps
    if span % 2:
        span += 1
    return {"label": label, "centre": (lo_steps + span // 2) * df,
            "sweep": span * df, "passes": int(passes), "on": True}


def peak_shadow_bands(centre, params, targets, peak_width):
    """
    The frequency bands a cleaning sweep must leave alone.

    The ions that make an anti-hole at ``centre`` - class
    ``d0 = centre - trans[k0, m0]``, holding population in ground state ``k0``
    - do not only absorb at ``centre``. The same population absorbs on every
    other transition out of ``k0``, at ``centre + trans[k0, m] - trans[k0, m0]``,
    and several of those land inside the pit. A cleaning sweep that ignores
    them pumps the peak's own ions straight back out and erases the feature it
    was meant to sharpen - which is exactly what happens if you sweep the whole
    pit bar the peak.

    ``targets`` are the ``(k0, m0)`` pairs of the burn-back pulses being used
    (the ``"target"`` field of :func:`anti_hole_pumps`). Returns merged,
    ascending ``(lo, hi)`` MHz bands, each ``peak_width`` wide, including the
    peak itself.
    """
    p = settings(params)
    df = p["df"]
    g, e, trans, trans_steps = build_transitions(params)
    n_e = trans_steps.shape[1]

    c = int(to_steps(centre, df))
    half = max(0, int(round(float(peak_width) / df)) // 2)

    raw = []
    for k0, m0 in targets:
        for m in range(n_e):
            mid = c + int(trans_steps[k0, m]) - int(trans_steps[k0, m0])
            raw.append((mid - half, mid + half))

    merged = []
    for lo, hi in sorted(raw):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo * df, hi * df) for lo, hi in merged]


def _gap_rows(label, lo_steps, hi_steps, blocked_steps, df, passes):
    """
    Rows filling ``[lo_steps, hi_steps]`` except the ``blocked_steps`` bands -
    a cleaning sweep with holes cut in it where the peak's ions absorb.
    """
    rows, cursor = [], lo_steps
    for b_lo, b_hi in sorted(blocked_steps):
        if b_hi < cursor or b_lo > hi_steps:
            continue
        rows.append(_span_row(label, cursor, min(b_lo - 1, hi_steps), df,
                              passes))
        cursor = max(cursor, b_hi + 1)
    rows.append(_span_row(label, cursor, hi_steps, df, passes))
    return [r for r in rows if r is not None]

def default_pit_width(params):
    """
    A starting pit width: the full span of the ground-state manifold.

    A pit has to be at least as wide as the ground splittings to empty ions
    rather than just shuffle them between ground states, so the manifold span
    (27.49 MHz for the tuned Pr:YSO preset) is the natural first guess.
    """
    p = settings(params)
    gg = np.asarray(p["g"], dtype=float)
    return float(gg.max() - gg.min())


def _shuffle_row(centre, sweep, passes, params):
    """
    A ``"shuffle"`` sequence-editor row: one powerful sweep, meant to erase
    whatever feature sits at ``centre`` (a peak, a comb, an anti-hole...)
    rather than shape one. ``sweep`` needs to clear more than the hyperfine
    separation (36.9 MHz for the tuned Pr:YSO preset) or it leaves
    hyperfine-shifted remnants of the old feature behind instead of erasing
    it outright - the field is passed through as given, though, since a
    narrower deliberate sweep is a caller's call to make, not this
    function's to forbid.
    """
    p = settings(params)
    df = p["df"]
    c_steps = int(to_steps(centre, df))
    sweep_steps = abs(int(to_steps(sweep, df)))
    return {"label": "shuffle", "centre": c_steps * df,
            "sweep": sweep_steps * df, "passes": int(passes), "on": True}


def prep_gem(params, *, centre=0.0, pit_width=12.0, peak_width=1.0,
             n_burn_back=1, burn_back_freq=36.9, pit_passes=100,
             burn_back_passes=50, clean=False, clean_passes=20,
             shuffle=False, shuffle_centre=0.0, shuffle_sweep=40.0,
             shuffle_passes=100):
    """
    A gradient-echo-memory (GEM) preparation, as sequence-editor rows.

    Up to four stages, burned in this order:

    1. **pit** - a wide sweep centred on ``centre`` that empties every ion
       class in the band, leaving a transparent window.
    2. **burn back** - single pulses outside the pit, from
       :func:`anti_hole_pumps`, that drop population back into the ground state
       absorbing at ``centre``: a narrow anti-hole in the middle of the pit.
    3. **clean**, only when ``clean=True`` - sweeps over the rest of the pit,
       either side of a ``peak_width`` MHz keep-out around ``centre``,
       removing the broad skirts the burn-back pulses leave and narrowing the
       peak.
    4. **shuffle**, only when ``shuffle=True`` - one powerful sweep (see
       :func:`_shuffle_row`) that erases whatever this sequence just burned,
       last, once it has done its job.

    ``pit_width`` defaults to a fixed 12 MHz; pass ``pit_width=None`` to fall
    back to :func:`default_pit_width` (the crystal's ground-manifold span)
    instead.

    The burn-back stays crystal-derived - it is still picked from
    :func:`anti_hole_pumps`'s real candidates, not a hardcoded pump - but by
    default picks the ``n_burn_back`` candidates nearest ``burn_back_freq``
    (an MHz hint) rather than the strongest by weight. Pass
    ``burn_back_freq=None`` to go back to picking the top-weighted
    candidates.
    """
    p = settings(params)
    df = p["df"]

    c = int(to_steps(centre, df))
    if pit_width is None:
        pit_width = default_pit_width(params)
    half = max(1, int(round(float(pit_width) / df)) // 2)
    half_peak = max(0, int(round(float(peak_width) / df)) // 2)

    pit_lo, pit_hi = (c - half) * df, (c + half) * df

    rows = [_span_row("pit", c - half, c + half, df, pit_passes)]

    candidates = anti_hole_pumps(centre, params, avoid=(pit_lo, pit_hi),
                                 limit=None)
    if burn_back_freq is not None:
        candidates = sorted(candidates,
                            key=lambda cnd: abs(cnd["freq"] - burn_back_freq))
    candidates = candidates[:int(n_burn_back)]

    targets = []
    for cand in candidates:
        rows.append({"label": "burn back", "centre": cand["freq"],
                     "sweep": 0.0, "passes": int(burn_back_passes),
                     "on": True})
        targets.append(cand["target"])

    if clean:
        # The cleaning sweep fills the rest of the pit, but skips the peak
        # *and* the other frequencies at which the peak's own ions absorb -
        # see peak_shadow_bands. Sweeping straight over those erases the peak.
        blocked = [(int(round(lo / df)), int(round(hi / df)))
                   for lo, hi in peak_shadow_bands(centre, params, targets,
                                                   2 * half_peak * df)]
        if not blocked:
            blocked = [(c - half_peak, c + half_peak)]
        rows += _gap_rows("clean", c - half, c + half, blocked, df,
                          clean_passes)

    if shuffle:
        rows.append(_shuffle_row(shuffle_centre, shuffle_sweep,
                                 shuffle_passes, params))

    return [r for r in rows if r is not None]


def _comb_teeth_rows(centre, span, tooth_spacing, finesse, passes, params):
    """
    ``"afc comb"`` sequence-editor rows: as many whole teeth, each
    ``tooth_spacing`` MHz apart and ``tooth_spacing / finesse`` MHz wide, as
    fit within ``span`` MHz - laid down as one block centred on ``centre``,
    not started from the block's low edge the way :func:`comb_at`/
    :func:`comb_in` do, which would skew the comb toward one side of the
    requested span instead of straddling ``centre``. All the teeth share the
    "afc comb" label, so they burn together the way a comb burn does.
    """
    p = settings(params)
    df = p["df"]

    c = int(to_steps(centre, df))
    band = max(2, int(round(float(span) / df)))
    half = band // 2

    spacing_steps = int(to_steps(tooth_spacing, df))
    tooth_width = max(df, float(tooth_spacing) / max(1e-9, float(finesse)))
    width_steps = max(1, int(round(tooth_width / df)))
    span_steps = 2 * half
    n_teeth = max(1, (span_steps - width_steps) // spacing_steps + 1)
    footprint = (n_teeth - 1) * spacing_steps + width_steps
    start0 = c - footprint // 2

    rows = []
    for i in range(n_teeth):
        s0 = start0 + i * spacing_steps
        rows.append(_span_row("afc comb", s0, s0 + width_steps - 1, df, passes))
    return [r for r in rows if r is not None]


def prep_narrowband_afc(params, *, centre=0.0, bandwidth=4.0,
                        tooth_spacing=0.5, finesse=2.0, guard=1.0,
                        n_burn_back=2, passes=20, shuffle=False,
                        shuffle_centre=0.0, shuffle_sweep=40.0,
                        shuffle_passes=100):
    """
    A narrowband AFC preparation, as sequence-editor rows.

    The broadband AFC of :func:`comb_at` burns its comb straight into the
    inhomogeneous line. A narrowband AFC instead prepares a clean band first
    and then burns a comb over only a few MHz of it:

    1. **pit** - a sweep over ``bandwidth + 2*guard`` MHz, emptying the band.
    2. **burn back** - the :func:`anti_hole_pumps` frequencies, each *swept
       across the bandwidth* rather than fired at a single frequency, so
       absorption is restored across the whole AFC band instead of at one spot.
    3. **afc comb** - teeth every ``tooth_spacing`` MHz, each
       ``tooth_spacing / finesse`` MHz wide, placed as one block centred on
       ``centre`` and sized to fit within ``bandwidth`` MHz. All the teeth
       share one label, so they burn together the way a comb burn does.
    4. **shuffle**, only when ``shuffle=True`` - one powerful sweep (see
       :func:`_shuffle_row`) that erases whatever this sequence just burned,
       last, once it has done its job.

    The storage time the comb gives is ``1 / tooth_spacing`` (0.5 MHz -> 2 us).
    """
    p = settings(params)
    df = p["df"]

    c = int(to_steps(centre, df))
    band = max(2, int(round(float(bandwidth) / df)))
    guard_steps = max(0, int(round(float(guard) / df)))
    half_band = band // 2

    pit_lo_steps = c - half_band - guard_steps
    pit_hi_steps = c + half_band + guard_steps
    rows = [_span_row("pit", pit_lo_steps, pit_hi_steps, df, passes)]

    for cand in anti_hole_pumps(centre, params,
                                avoid=(pit_lo_steps * df, pit_hi_steps * df),
                                limit=int(n_burn_back)):
        rows.append({"label": "burn back", "centre": cand["freq"],
                     "sweep": (2 * half_band) * df,
                     "passes": int(passes), "on": True})

    rows += _comb_teeth_rows(centre, bandwidth, tooth_spacing, finesse,
                             passes, params)

    if shuffle:
        rows.append(_shuffle_row(shuffle_centre, shuffle_sweep,
                                 shuffle_passes, params))

    return [r for r in rows if r is not None]


def prep_broadband_afc(params, *, width=100.0, storage_time=2.0, finesse=2.0,
                       centre=0.0, passes=20, shuffle=False,
                       shuffle_centre=0.0, shuffle_sweep=40.0,
                       shuffle_passes=100):
    """
    A broadband AFC preparation, as sequence-editor rows.

    Unlike :func:`prep_narrowband_afc`, the comb is burned straight into the
    inhomogeneous line rather than into a prepared pit: no pit, burn-back or
    cleaning stage, just the comb itself (see :func:`_comb_teeth_rows`), and
    an optional shuffle pulse after it.

    Up to two stages, burned in this order:

    1. **afc comb** - teeth ``1 / storage_time`` MHz apart (the storage time
       an AFC of tooth spacing ``s`` gives is ``1 / s``, so this is that
       relation used in reverse - the field that matters experimentally,
       rather than the spacing itself), each tooth ``storage_time / finesse``
       wide, filling ``width`` MHz centred on ``centre``.
    2. **shuffle**, only when ``shuffle=True`` - one powerful sweep (see
       :func:`_shuffle_row`) that erases the comb, last, once it has done
       its job.

    ``storage_time`` is free-form - unlike a pump frequency, it names no
    position on the mesh itself, only its reciprocal (the tooth spacing)
    does, and the reciprocal of an on-mesh spacing is essentially never
    exactly representable in a finite decimal. So rather than gate it through
    :func:`to_steps` (which would reject almost every value typed), the
    implied tooth spacing is rounded to the nearest ``df`` step - the comb
    this actually burns has ``1 / (spacing rounded to the mesh)`` for its
    true storage time, which is what the sequence-editor row you get back
    reflects.
    """
    p = settings(params)
    df = p["df"]
    spacing_steps = max(1, int(round(1.0 / float(storage_time) / df)))
    tooth_spacing = spacing_steps * df
    rows = _comb_teeth_rows(centre, width, tooth_spacing, finesse, passes,
                            params)

    if shuffle:
        rows.append(_shuffle_row(shuffle_centre, shuffle_sweep,
                                 shuffle_passes, params))

    return rows


#: The preparation presets offered by the GUI's sequence editor. Each entry
#: carries its builder plus the fields the editor exposes for it, as
#: ``(kwarg, label, default, decimals)`` - or ``(..., depends_on)`` with a 5th
#: element naming another (boolean) field this one only makes sense alongside,
#: e.g. a shuffle pulse's width/passes next to its on/off checkbox; the GUI
#: hides such a field until its named field is checked. A callable default is
#: resolved against the current params, ``decimals=0`` means an integer
#: field, and ``decimals="bool"`` means a checkbox. For a float field
#: ``decimals`` only sizes the editor's spinner step - how many places can be
#: *typed* is the editor's own ``SPIN_DECIMALS``, not this.
#:
#: Every builder shares the same shuffle-pulse kwargs (``shuffle``,
#: ``shuffle_centre``, ``shuffle_sweep``, ``shuffle_passes`` - see
#: _shuffle_row); ``shuffle_centre`` is deliberately not exposed here, since
#: the useful default (the sequence's own ``centre``, 0.0 for all three
#: presets below) is what every preset already targets.
_SHUFFLE_FIELDS = [
    ("shuffle", "shuffle pulse after", False, "bool"),
    ("shuffle_sweep", "shuffle width (MHz)", 40.0, 3, "shuffle"),
    ("shuffle_passes", "shuffle passes", 100, 0, "shuffle"),
]

PREP_PRESETS = {
    "Gradient echo memory (GEM)": {
        "builder": prep_gem,
        "fields": [
            ("centre", "centre (MHz)", 0.0, 3),
            ("pit_width", "pit width (MHz)", 12.0, 3),
            ("peak_width", "peak width (MHz)", 1.0, 3),
            ("n_burn_back", "burn-back pulses", 1, 0),
            ("burn_back_freq", "burn-back frequency hint (MHz)", 36.9, 3),
            ("pit_passes", "pit passes", 100, 0),
            ("burn_back_passes", "burn-back passes", 50, 0),
            *_SHUFFLE_FIELDS,
        ],
    },
    "Narrowband AFC": {
        "builder": prep_narrowband_afc,
        "fields": [
            ("centre", "centre (MHz)", 0.0, 3),
            ("bandwidth", "AFC bandwidth (MHz)", 4.0, 3),
            ("tooth_spacing", "tooth spacing (MHz)", 0.5, 3),
            ("finesse", "finesse", 2.0, 2),
            ("guard", "pit guard band (MHz)", 1.0, 3),
            ("n_burn_back", "burn-back pulses", 2, 0),
            ("passes", "pump passes per stage", 20, 0),
            *_SHUFFLE_FIELDS,
        ],
    },
    "Broadband AFC": {
        "builder": prep_broadband_afc,
        "fields": [
            ("width", "AFC width (MHz)", 100.0, 3),
            ("storage_time", "storage time (us)", 2.0, 3),
            ("finesse", "finesse", 2.0, 2),
            *_SHUFFLE_FIELDS,
        ],
    },
}


def _user_presets_path():
    """
    Where saved sequence-editor presets live.

    ``Path(__file__).with_name(...)`` (the old, unconditional choice) writes
    next to this module - fine for a checkout, but that becomes
    ``site-packages`` once ``shb`` is pip-installed (no business writing
    there) and a read-only bundle when frozen (can't write there at all). So:

    1. ``$SHB_PREP_PRESETS``, if set - an explicit override wins outright.
    2. Next to this module, if that directory is writable and is not inside a
       ``site-packages``/``dist-packages`` tree - preserves today's checkout
       behaviour exactly (this is also why the module attribute below is
       still called ``USER_PRESETS_PATH`` and still resolves to
       ``./prep_presets.json`` for anyone running from a clone).
    3. Otherwise the platform's per-user config directory
       (``~/.config/shb/prep_presets.json`` on Linux,
       ``~/Library/Application Support/shb/prep_presets.json`` on macOS,
       ``%APPDATA%\\shb\\prep_presets.json`` on Windows), created if needed.
    """
    env = os.environ.get("SHB_PREP_PRESETS")
    if env:
        return Path(env)

    here = Path(__file__).resolve().parent
    parts = {p.lower() for p in here.parts}
    in_site_packages = "site-packages" in parts or "dist-packages" in parts
    if not in_site_packages and os.access(here, os.W_OK):
        return here / "prep_presets.json"

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "shb" / "prep_presets.json"


#: Resolved once at import time - see :func:`_user_presets_path`. Kept as a
#: module attribute (rather than only a function) so existing code/tests that
#: read ``shb.USER_PRESETS_PATH`` keep working unchanged.
USER_PRESETS_PATH = _user_presets_path()


def load_user_presets():
    """
    Sequence-editor presets you have saved yourself, as ``{name: rows}``
    (see ``PulseSequenceDialog``'s "Save as preset..." button, shb_gui.py).

    Unlike :data:`PREP_PRESETS` (a formula plus adjustable fields), a saved
    preset is a literal snapshot of rows - nothing to tune, just what was in
    the table when it was saved. Local to this checkout (``prep_presets.json``,
    gitignored) rather than committed, so it never enters a code review.

    Returns ``{}`` if the file does not exist or cannot be parsed - a missing
    or corrupt preset file should never stop the sequence editor from opening.
    """
    path = _user_presets_path()
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_preset(name, rows):
    """Add or overwrite one saved preset - ``name`` -> a list of sequence rows."""
    name = str(name).strip()
    if not name:
        raise ValueError("a preset needs a name")
    presets = load_user_presets()
    presets[name] = rows
    path = _user_presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, presets)


def delete_user_preset(name):
    """Remove one saved preset, if it exists."""
    presets = load_user_presets()
    if name in presets:
        del presets[name]
        path = _user_presets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, presets)


# ===========================================================================
# Scan
# ===========================================================================

def _fmt_dur(seconds):
    """A compact ``1h04m`` / ``3m20s`` / ``12s`` duration for the run log."""
    s = int(max(0.0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# Tuning knob for the squareness term in _score_quality: a peak or dip inside
# one tooth period must exceed this fraction of that period's own
# peak-to-floor range to count as a genuine defect (ringing / sub-structure)
# rather than numerical noise.
_SQUARENESS_PROMINENCE_FRAC = 0.05


def _local_maxima_1d(x):
    """Indices of strict local maxima of ``x`` (higher than both neighbours);
    endpoints never count. A tiny hand-rolled stand-in for
    ``scipy.signal.find_peaks`` - see :func:`_peak_prominences_1d` for why."""
    if len(x) < 3:
        return np.empty(0, dtype=int)
    mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    return np.where(mask)[0] + 1


def _peak_prominences_1d(x, peak_idx):
    """
    Topographic prominence of each index in ``peak_idx`` (as returned by
    :func:`_local_maxima_1d`): from each peak, walk outward in both
    directions until either the array boundary or a point higher than the
    peak itself, tracking the lowest point crossed; the prominence is the
    peak's height above the higher of its two lowest crossings. This is the
    same definition ``scipy.signal.find_peaks(..., prominence=...)`` uses.

    Reimplemented here in plain numpy/Python rather than imported from
    ``scipy.signal`` deliberately: unlike the ``scipy.fft`` calls used
    elsewhere in this module (pure C, no Fortran/LAPACK), importing
    ``scipy.signal`` pulls in ``scipy.linalg``, which on a typical Windows
    Anaconda build is linked against Intel MKL's Fortran runtime. That
    runtime installs its own Windows console Ctrl+C handler, which can
    broadcast a spurious ``CTRL_C_EVENT`` to the whole console process group
    when a linked subprocess exits - hitting every worker *and* the GUI's
    main thread as a bare ``KeyboardInterrupt``, with no keypress involved.
    ``_score_quality`` runs inside every scan worker, so it must not import
    anything that can trigger this.
    """
    n = len(x)
    proms = np.empty(len(peak_idx), dtype=float)
    for k, i in enumerate(peak_idx):
        peak = x[i]
        left_min = peak
        j = i
        while j > 0:
            j -= 1
            if x[j] > peak:
                break
            if x[j] < left_min:
                left_min = x[j]
        right_min = peak
        j = i
        while j < n - 1:
            j += 1
            if x[j] > peak:
                break
            if x[j] < right_min:
                right_min = x[j]
        proms[k] = peak - max(left_min, right_min)
    return proms


def _measured_finesse(template):
    """
    The finesse a comb *actually* has, read off one averaged tooth period.

    ``template`` is one period of the spectrum (periods averaged together, as
    :func:`_score_quality` builds it). The tooth's FWHM is taken as the number
    of samples above half height - halfway between the template's floor and
    its peak - so the measured finesse is ``F_a = len(template) / n_above``.
    Counting samples rather than locating edges makes this circular: it does
    not matter where in the period the tooth sits, or whether it wraps round
    the period boundary. Returns ``nan`` for a flat template (no tooth).

    This is the finesse the comb was burned *into*, not the one it was written
    with: power broadening, the hole's own homogeneous width and side-hole
    refilling all spread the teeth, so a comb written at ``F = 10`` routinely
    comes out near ``F = 2``.
    """
    template = np.asarray(template, dtype=float)
    lo, hi = float(template.min()), float(template.max())
    if hi <= lo:
        return float("nan")
    n_above = int(np.count_nonzero(template > 0.5 * (lo + hi)))
    return template.size / max(n_above, 1)


def _score_quality(f, spec, target_window, spacing_mhz, width_mhz=None):
    """
    AFC comb-quality score over ``target_window`` MHz at the centre of ``f``.
    The score alone - see :func:`_score_quality_and_finesse` for the formula,
    and for the comb's measured finesse alongside it.
    """
    return _score_quality_and_finesse(f, spec, target_window, spacing_mhz,
                                      width_mhz)[0]


# NOTE: this function is still being tuned (see the git history) - bump
# QUALITY_VERSION above whenever its formula changes, or a resumed scan will
# silently mix qualities computed two different ways into one results table.
# _measured_finesse feeds finesse_match, so changing it is a formula change
# too (and it also changes the measured_finesse column a scan records).
def _score_quality_and_finesse(f, spec, target_window, spacing_mhz,
                               width_mhz=None):
    """
    AFC comb-quality score over ``target_window`` MHz at the centre of ``f``,
    and the finesse the comb actually came out with.

    Returns ``(quality, F_a)``. ``F_a`` is :func:`_measured_finesse` of the
    averaged one-period template the score builds anyway, so a scan gets it
    for free rather than re-cropping the spectrum; it is ``nan`` wherever there
    is no template to read it from (fewer than two whole periods in the window,
    no spacing given, or a flat segment). Measured whether or not
    ``width_mhz`` is given - only the ``finesse_match`` factor needs that.

    A clean broadband AFC has regular, equal-height teeth with deep flat dips
    and a single flat-topped bump per period; a bad one has irregular teeth,
    ringing / sub-structure inside a tooth, or aperiodic drift between
    periods. A plain Michelson visibility ``(A_max - A_min) / (A_max + A_min)``
    cannot tell them apart - both hit the same extreme samples. This score
    multiplies six factors, each in ``[0, 1]``, so a comb has to be deep,
    regular, periodic, square, empty between its teeth *and* as fine as it was
    written to score well::

        Q = mean_j(v_j) * uniformity * periodicity * squareness
            * background * finesse_match

    where the window is cropped to a whole number of tooth periods
    (``spacing_mhz`` is the known tooth spacing, so ``per = spacing_mhz / df`` is
    an exact integer and there is no spectral leakage):

    * ``v_j = (p95_j - p05_j) / (p95_j + p05_j)`` is the robust per-period depth
      (5th/95th percentiles, so ringing spikes do not dominate);
    * ``uniformity = 1 - std(v_j) / mean(v_j)`` penalises tooth-to-tooth scatter
      in that same per-period depth - both irregular peak heights and irregular
      dip floors count against it;
    * ``periodicity`` is the fraction of the windowed AC power sitting at
      harmonics of the fundamental comb frequency (every multiple of
      ``1 / spacing_mhz``, via a real FFT of the truncated segment) - genuine
      period-to-period drift or aperiodicity leaks power off those bins and
      drops it. An earlier version used only the fundamental bin, which is
      duty-cycle biased: a mathematically perfect, ringing-free narrow-duty
      rectangular pulse train legitimately puts most of its power in
      harmonics rather than the fundamental, so that version penalised high
      finesse on its own, independent of any real raggedness. Summing every
      harmonic fixes that - it stays ~1 for a perfect comb at any finesse;
    * ``squareness`` penalises extra bumps/dips *inside* one tooth period
      (ringing / sub-structure) - the one defect nothing above catches, since
      a period can be deep, regular and duty-cycle-fair yet still ragged. The
      per-period template (periods averaged together) is searched for
      significant peaks and dips (topographic prominence over
      ``_SQUARENESS_PROMINENCE_FRAC`` of the template's own range - see
      :func:`_peak_prominences_1d`); a clean tooth has exactly one dominant
      feature, so the largest is exempted and every other peak/dip's
      prominence is charged against the score;
    * ``background = exp(-d0)`` charges the absorption left *between* the
      teeth. Every term above is a ratio (``v_j`` sees only ``d0 / d``), so a
      comb whose antihole teeth grew tall enough could hide half-refilled gaps.
      But gap absorption is absolute loss - it takes up input light that is
      never rephased - and AFC theory carries it as a separate
      ``eta ~ exp(-d0)`` factor (Afzelius et al., PRA 79, 052329 (2009)).
      ``d0`` is the mean of every sample below half height in each period
      (then averaged over periods), not the per-period floor, so a sloped or
      partly refilled gap and the teeth's Lorentzian tails all count. The
      spectrum is in ``alpha`` units (the unburned crystal sits at ``alpha``),
      so this is the physical ``exp(-d0)`` only when ``alpha`` is set to the
      crystal's real optical depth; at the default ``alpha = 1`` it is still
      monotonic in how far the gaps have refilled;
    * ``finesse_match = min(F_a, F_w) / max(F_a, F_w)`` compares the finesse
      the comb was written with, ``F_w = spacing_mhz / width_mhz``, to the one
      it actually came out with, ``F_a`` (:func:`_measured_finesse`, from the
      half-height duty cycle of the averaged period). Broadening routinely
      spreads a comb written at ``F = 10`` to ``F ~ 2``; the scan's finesse
      axis is ``F_w``, so without this factor a cell would be credited with a
      finesse its comb does not have. Skipped (``1``) when ``width_mhz`` is
      not given.

    With fewer than two whole periods in the window there is no comb to speak of
    and the score falls back to the robust whole-window depth times
    ``exp(-p05)`` (the window floor standing in for ``d0``); there is no whole
    period to measure a finesse from.
    """
    nan = float("nan")
    f = np.asarray(f)
    spec = np.asarray(spec, dtype=float)
    centre = f[len(f) // 2]
    lo = int(np.searchsorted(f, centre - target_window / 2))
    hi = int(np.searchsorted(f, centre + target_window / 2))
    seg = spec[lo:hi]
    fseg = f[lo:hi]
    if seg.size == 0:
        return 0.0, nan

    p05, p95 = np.percentile(seg, [5.0, 95.0])
    if p95 + p05 <= 0.0:
        return 0.0, nan
    depth = float(np.clip((p95 - p05) / (p95 + p05), 0.0, 1.0))
    depth *= float(np.exp(-max(float(p05), 0.0)))      # background, see docstring

    df_seg = float(fseg[1] - fseg[0]) if fseg.size >= 2 else 0.0
    if not (spacing_mhz and spacing_mhz > 0.0 and df_seg > 0.0):
        return depth, nan
    per_f = spacing_mhz / df_seg
    per = int(round(per_f))                           # == spacing_steps
    if per >= 2 and abs(per_f - per) > 1e-6:
        # A fractional period (see period_steps): resample the window onto a
        # grid of exactly ``per`` samples per period, so the reshape below
        # still lines periods up instead of drifting by up to half a step a
        # period. A whole period skips this, so its score is unchanged.
        step = spacing_mhz / per
        x = fseg[0] + step * np.arange(int((fseg[-1] - fseg[0]) / step) + 1)
        seg = np.interp(x, fseg, seg)
    n_per = seg.size // per if per > 0 else 0
    if per < 2 or n_per < 2:                          # < 2 teeth in the window
        return depth, nan

    keep = n_per * per
    s = seg[:keep]

    # per-period robust depth + tooth-to-tooth uniformity
    chunks = s.reshape(n_per, per)
    lo_j, hi_j = np.percentile(chunks, [5.0, 95.0], axis=1)
    denom = hi_j + lo_j
    v_j = np.where(denom > 0.0, (hi_j - lo_j) / denom, 0.0)
    mean_v = float(v_j.mean())
    uniformity = (1.0 - float(v_j.std()) / mean_v) if mean_v > 0.0 else 0.0
    uniformity = float(np.clip(uniformity, 0.0, 1.0))

    # background: absolute absorption left in the gaps (mean below half height)
    gap = chunks < 0.5 * (lo_j + hi_j)[:, None]
    n_gap = gap.sum(axis=1)
    d0_j = np.where(n_gap > 0,
                    np.where(gap, chunks, 0.0).sum(axis=1) / np.maximum(n_gap, 1),
                    lo_j)
    background = float(np.exp(-max(float(d0_j.mean()), 0.0)))

    # periodicity: AC power fraction sitting at harmonics of the comb's
    # fundamental frequency (not just the fundamental itself - see docstring)
    g = s - s.mean()
    power = np.abs(np.fft.rfft(g)) ** 2
    total_power = float(power[1:].sum())               # drop DC (index 0)
    if total_power <= 0.0:
        return 0.0, nan                                # flat: no tooth to measure
    harmonics = np.arange(n_per, power.size, n_per)
    periodicity = float(np.clip(power[harmonics].sum() / total_power, 0.0, 1.0))

    # squareness: penalise extra peaks/dips inside one tooth period
    template = chunks.mean(axis=0)
    template_range = float(template.max() - template.min())
    if template_range <= 0.0:
        squareness = 1.0
    else:
        thresh = _SQUARENESS_PROMINENCE_FRAC * template_range
        peak_idx = _local_maxima_1d(template)
        dip_idx = _local_maxima_1d(-template)
        peak_proms = _peak_prominences_1d(template, peak_idx)
        dip_proms = _peak_prominences_1d(-template, dip_idx)
        proms = np.concatenate([peak_proms, dip_proms])
        proms = proms[proms >= thresh]
        excess = float(proms.sum() - proms.max()) if proms.size else 0.0
        squareness = float(np.clip(1.0 - excess / template_range, 0.0, 1.0))

    # finesse_match: written finesse vs the finesse the comb actually has
    f_a = _measured_finesse(template)
    finesse_match = 1.0
    if width_mhz and width_mhz > 0.0:
        f_w = spacing_mhz / width_mhz
        finesse_match = (min(f_a, f_w) / max(f_a, f_w)
                         if np.isfinite(f_a) else 0.0)

    return (float(np.clip(mean_v * uniformity * periodicity * squareness
                          * background * finesse_match, 0.0, 1.0)),
            float(f_a))


def _estimate_worker_bytes(params, min_finesse=1.0):
    """
    Rough peak-RSS estimate for one scan worker, in bytes.

    Cheap and allocation-free (``build_transitions`` only touches arrays sized
    by the crystal, n_g x n_e): the context (axes, initial populations, the FFT
    kernel), one burn pass's buffers (:func:`burn_parallel`, since the scan
    default), the spectrum's FFT transients, and the worst-case pump-frequency
    arrays at ``min_finesse`` (a narrow tooth walks the whole comb). The flat
    120 MB is the interpreter/numpy/scipy baseline, measured rather than
    derived from the code.

    Returns ``(total_bytes, breakdown)`` where ``breakdown`` is a dict of the
    same pieces, for an honest log line.
    """
    p = settings(params)
    df = float(p["df"])
    fmin, fmax = float(p["fmin"]), float(p["fmax"])
    n_f = int(round((fmax - fmin) / df)) + 1

    _, _, _, trans_steps = build_transitions(params)
    n_g, n_e = trans_steps.shape
    pad = int(np.abs(trans_steps).max())
    n_det = n_f + 2 * pad

    conv_len = 3 * n_det - 2
    try:
        from scipy.fft import next_fast_len
        nfft = int(next_fast_len(conv_len))
    except ImportError:
        nfft = 1 << int(np.ceil(np.log2(max(conv_len, 1))))

    context = 8 * (n_f + 2 * n_det + n_g * n_det) + 8 * nfft
    burn = 8 * n_det * (n_g * n_g + 4 * n_g + 1) + n_g * n_det
    spectrum = 24 * nfft + 8 * n_g * n_det
    n_pump = (fmax - fmin) / max(min_finesse * df, df)
    pump = 6 * 8 * n_pump
    baseline = 120_000_000

    breakdown = {"context": context, "burn": burn, "spectrum": spectrum,
                "pump": pump, "baseline": baseline}
    return sum(breakdown.values()), breakdown


def _available_bytes():
    """
    Best-effort available RAM, in bytes, and where the number came from.

    ``psutil`` (optional - see requirements.txt) gives "available", which
    accounts for a reclaimable page cache and is what actually matters; without
    it, ``/proc/meminfo``'s ``MemAvailable`` on Linux, or half of total physical
    memory elsewhere (macOS has no ``SC_AVPHYS_PAGES``; the total is all
    ``os.sysconf`` can give, so the half is a deliberately coarse margin - said
    so in the log line, not left implicit).
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available), "psutil"
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, "/proc/meminfo"
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(0.5 * pages * page_size), "half of total (coarse)"
    except (ValueError, OSError, AttributeError):
        return 4_000_000_000, "unknown - guessed 4 GB"


def _fmt_bytes(n):
    """A byte count as a short human string: '280 MB', '5.6 GB'."""
    n = float(n)
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("bytes", "kB") else f"{n:.1f} {unit}"
        n /= 1024


def _resolve_scan_df(t_targets, params, finesse_min):
    """
    The df a scan actually burns at, after auto-refinement.

    Tooth spacing is 1/t, burned as a fractional period on the df mesh
    (:func:`period_steps`), so every storage time gets its own comb, but each
    tooth sits up to half a step off its ideal position. This refines df by a
    whole factor (so it still divides the crystal levels and the window edges)
    until consecutive spacings are at least one step apart, which keeps that
    jitter below the difference between neighbouring combs - bounded below by what is
    physically meaningful (~h_width/5), by a cap on the spectrum array size,
    and by what a worker can hold in memory.

    Factored out of :func:`run_scan` so anything previewing a scan before
    running it - the GUI's resume/mismatch check, chiefly - computes the same
    df_used a checkpoint will actually be keyed on, rather than comparing
    against the unrefined df and reporting a spurious mismatch.

    Returns ``(df, capped_by_memory)``.
    """
    p = settings(params)
    df_user = p["df"]
    _avail_bytes, _ = _available_bytes()
    # Leave room for the parent (which builds one context itself, for the
    # timing probe) and the OS; a single refined worker still has to fit.
    _refine_ceiling = max(1, 0.5 * _avail_bytes)
    df = df_user
    spacings = 1.0 / t_targets
    min_gap = (float(np.min(np.abs(np.diff(spacings)))) if len(spacings) > 1
               else df_user)
    capped_by_memory = False
    if min_gap < df_user:
        floor = min(df_user, max(1e-4, 0.2 * float(p["h_width"])))
        floor = max(floor,
                    (float(p["fmax"]) - float(p["fmin"])) / 2_000_000)
        # Capped well below the old 500x: a refine that steep is a runaway in
        # burn time as much as memory (more, smaller-spaced burns, each one
        # over a finer, slower mesh), and it was reachable silently before.
        want = min(20, int(np.ceil(df_user / max(min_gap, floor))))
        edges = np.concatenate([np.asarray(p["g"], float),
                                np.asarray(p["e"], float),
                                [float(p["fmin"]), float(p["fmax"])]])
        for k in range(max(1, want), 0, -1):
            cand = df_user / k
            if cand < floor and k > 1:
                continue
            try:
                to_steps(edges, cand)
            except ValueError:
                continue
            cand_params = copy.deepcopy(params)
            cand_params["numerical_params"]["df"] = cand
            if _estimate_worker_bytes(cand_params,
                                      min_finesse=finesse_min)[0] > _refine_ceiling:
                capped_by_memory = True
                continue
            df = cand
            break
    return df, capped_by_memory


def _scan_context(params, burn, pump_passes, target_window):
    """
    The parts of a scan point that are the same for every point - built once by
    :func:`run_scan` and handed to every :func:`compute_one` call (pickled once
    per worker process, not once per task).
    """
    g, e, trans, trans_steps = build_transitions(params)
    f, det, det0 = make_axes(params, trans_steps)
    rho0 = initial_populations(g, det)
    sctx = _SpectrumCtx(params, det, f)

    return {
        "params": params, "burn": burn, "pump_passes": pump_passes,
        "target_window": target_window, "df": settings(params)["df"],
        "trans": trans, "trans_steps": trans_steps,
        "f": f, "det": det, "det0": det0,
        "rho0": rho0, "sctx": sctx,
    }


def compute_one(item, params=None, burn=None, pump_passes=None,
                df=None, target_window=None, ctx=None, want_spec=True):
    """
    One scan point, pulled out of the loop so it can run in a worker process.

    ``item`` is ``(n, (spacing_steps, width_steps))``. Returns
    ``(result_row, record)`` where
        result_row = (spacing_MHz, width_MHz, quality)
        record     = (quality, n, spec, spacing_MHz, width_MHz, f)
    ``quality`` is the AFC comb-quality score (:func:`_score_quality`).

    With ``want_spec=False`` the record is the short form
    ``(quality, n, spacing_MHz, width_MHz, measured_finesse)`` - no spectrum,
    no frequency axis. That is what a scan sweep uses: a spectrum and its axis
    are ~1.6 MB at df 0.001, so returning them for every point pickles
    gigabytes back into the parent process to keep a few dozen.
    :func:`run_scan` recomputes the winners' spectra afterwards instead.
    ``measured_finesse`` is the finesse the comb actually came out with
    (:func:`_measured_finesse`, ``nan`` where there is no whole period to read
    it from) - one float, read off the template the score builds anyway. The
    default is ``True`` so the notebooks still get the six-tuple.

    Pass ``ctx`` (from :func:`_scan_context`) for the fast path: the transition
    table, axes, initial populations and spectrum FFT setup are reused instead of
    rebuilt every call. Without ``ctx`` it falls back to a plain ``run_burn``.
    """
    n, (spacing_steps, width_steps) = item

    if ctx is not None:
        params, burn, df = ctx["params"], ctx["burn"], ctx["df"]
        pump_passes, target_window = ctx["pump_passes"], ctx["target_window"]
        freqs = comb(spacing_steps, width_steps, params)
        rho = burn(ctx["rho0"], freqs, ctx["trans_steps"], ctx["det"],
                   ctx["det0"], params, pump_passes)
        f = ctx["f"]
        spec = spectrum(rho, ctx["trans"], ctx["det"], f, params,
                        ctx=ctx["sctx"])
    else:
        out = run_burn(comb(spacing_steps, width_steps, params), params,
                       burn=burn, pump_passes=pump_passes)
        f, spec = out["f"], out["spec"]

    spacing_mhz = spacing_steps * df
    width_mhz = width_steps * df

    quality, f_meas = _score_quality_and_finesse(f, spec, target_window,
                                                 spacing_mhz, width_mhz)

    result_row = (spacing_mhz, width_mhz, quality)
    if want_spec:
        record = (quality, n, spec, spacing_mhz, width_mhz, f)
    else:
        record = (quality, n, spacing_mhz, width_mhz, f_meas)
    return result_row, record


# Set by :func:`_init_worker` in each pool process. The scan context is built
# once per worker and left here, rather than travelling with every task.
_WORKER_CTX = None


def _init_worker(params, burn, pump_passes, target_window):
    """
    Pool initialiser: build the scan context inside the worker process.

    Only small things cross the pipe - ``params`` (scalars plus the branching
    matrix) and a reference to the burn function. The axes, initial populations
    and FFT setup, 12 MB at df 0.001 and ten times that at a refined mesh, are
    built here.

    Handing ``partial(compute_one, ctx=ctx)`` to ``Pool.imap`` instead looks
    like it pickles the context once, but CPython pickles ``(func, chunk)``
    once per task *chunk*, so the whole context went down the task queue
    ``4 * n_workers`` times a sweep.
    """
    global _WORKER_CTX
    _WORKER_CTX = _scan_context(params, burn, pump_passes, target_window)


def _run_point(item):
    """One scan point in a worker process: the score record, no spectrum."""
    return compute_one(item, ctx=_WORKER_CTX, want_spec=False)[1]


def _spectrum_at(item, ctx):
    """
    The spectrum of one scan point, for :func:`run_scan`'s collect pass.

    ``item`` is ``(row, (spacing_steps, width_steps))``, where ``row`` is the
    slot to fill in the caller's array. Returns ``(row, spec)`` - the frequency
    axis is left behind, because every kept record shares the parent's.
    """
    row, (spacing_steps, width_steps) = item
    params = ctx["params"]
    freqs = comb(spacing_steps, width_steps, params)
    rho = ctx["burn"](ctx["rho0"], freqs, ctx["trans_steps"], ctx["det"],
                      ctx["det0"], params, ctx["pump_passes"])
    spec = spectrum(rho, ctx["trans"], ctx["det"], ctx["f"], params,
                    ctx=ctx["sctx"])
    return row, spec


def _spectrum_for(item):
    """Worker-process wrapper for :func:`_spectrum_at`."""
    return _spectrum_at(item, _WORKER_CTX)


def _target_match(t_target, f_target, spacing_mhz, width_mhz):
    """
    How closely a burnt comb's actual storage time/finesse match a scan
    cell's request, in ``[0, 1]``.

    Each requested ``(t, F)`` cell is snapped to the nearest df-mesh comb
    before burning (see :func:`run_scan`); at coarse ``df``, or at high
    finesse where the width rounds to very few distinct steps, many adjacent
    cells collide onto the same actual comb and would otherwise all be
    stamped with that one comb's shape quality regardless of how far their
    own request sat from what was actually burnt. This factor fades a cell's
    quality down the further its actual storage time/finesse sit from what it
    asked for, so a block of collided cells reads as a gradient rather than a
    flat stripe.
    """
    t_err = abs(1.0 / spacing_mhz - t_target) / t_target
    f_err = abs(spacing_mhz / width_mhz - f_target) / f_target
    return float(np.clip(1.0 - t_err, 0.0, 1.0) * np.clip(1.0 - f_err, 0.0, 1.0))


def _resolve_storage_sep(min_storage_sep, storage_min, storage_max, top_n):
    """
    The minimum gap (us) between the storage times a scan keeps spectra for.

    ``"auto"`` (or ``None``) spreads ``top_n`` picks across the scanned range:
    ``(storage_max - storage_min) / (2 * top_n)``. Each kept storage time
    excludes everything within one gap either side of it, so ``top_n`` picks
    together cover about the whole range - enough room for all of them
    without letting them cluster on one quality ridge. A number is taken as
    the gap itself; ``0`` turns the rule off (best storage times, however
    close together).
    """
    if min_storage_sep is None or min_storage_sep == "auto":
        return (float(storage_max) - float(storage_min)) / (2.0 * max(1, top_n))
    sep = float(min_storage_sep)
    if not sep >= 0.0:
        raise ValueError(f"min_storage_sep must be 'auto' or >= 0 us, "
                         f"not {min_storage_sep!r}")
    return sep


def _best_per_storage(scored, uniq, top_n, df=None, min_sep_us=0.0):
    """
    Pick the spectra a scan keeps: the best comb at each storage time, for the
    ``top_n`` best storage times that sit at least ``min_sep_us`` apart.

    ``scored`` is an iterable of ``(quality, n)`` pairs, ``n`` indexing
    ``uniq``'s ``(s_steps, w_steps)`` keys. "Storage time" here is the tooth
    spacing actually burned, ``s_steps`` - two requested storage times that
    snap to the same mesh spacing are the same comb - so the kept set never
    holds two spectra at one storage time, only the best finesse at each.
    Ranked on the raw comb quality (no :func:`_target_match`), since this
    ranks burned combs, not requested cells.

    Neighbouring storage times on a fine grid score almost alike, so the plain
    top ``top_n`` tends to be one cluster around the best. With
    ``min_sep_us > 0`` (``df`` then required, to turn ``s_steps`` back into
    us) the pick is greedy, best first: a storage time is skipped if it lies
    closer than ``min_sep_us`` to one already kept. That can leave fewer than
    ``top_n`` picks when the gap is wide for the range.

    Returns ``(quality, n)`` pairs, best first, at most ``top_n`` of them.
    Shared by :func:`run_scan` and :func:`collect_top` so the two always agree.
    """
    per_s = {}               # s_steps -> (quality, n)
    for key in scored:
        s = uniq[key[1]][0]
        if s not in per_s or key > per_s[s]:
            per_s[s] = key
    ranked = sorted(per_s.values(), reverse=True)
    if not min_sep_us:
        return ranked[:top_n]
    kept, kept_t = [], []
    for key in ranked:
        t = 1.0 / (uniq[key[1]][0] * df)
        # The small slack keeps a gap that is exactly one grid step from
        # being rejected by floating-point noise.
        if all(abs(t - tk) >= min_sep_us * (1.0 - 1e-9) for tk in kept_t):
            kept.append(key)
            kept_t.append(t)
            if len(kept) == top_n:
                break
    return kept


def run_scan(storage_min, storage_max, n_points,
             finesse_min, finesse_max, n_finesse,
             target_window, top_n, pump_passes,
             params=None, burn=burn_parallel,
             parallel=True, n_workers=None, maxtasksperchild=None,
             progress=None, log=None, estimate=None, on_done=None,
             sim_dir=None, resume=True, checkpoint_every=10,
             min_storage_sep="auto"):
    """
    Sweep AFC storage time and comb finesse, scoring each spectrum by its AFC
    comb-quality score (:func:`_score_quality`) over a window at the centre -
    per-period depth times tooth-to-tooth uniformity times periodicity times
    squareness times gap background ``exp(-d0)`` times the match between the
    written and the measured finesse, so a clean regular comb with empty gaps
    that holds its written finesse outranks a deep-but-messy one.

    The spectrum window and mesh come from ``params`` - fmin, fmax and df are
    not swept here.

    storage_min/max   range of storage times to try (us). The tooth spacing is
                      ``1 / storage_time`` (MHz), so this range corresponds to
                      spacings ``1/storage_max .. 1/storage_min``.
    n_points          number of storage times across that range, spaced linearly
                      in us - and burned at exactly those storage times: each
                      comb's mean tooth period is 1/t, as a fractional number
                      of df steps (see :func:`period_steps` / :func:`comb`), so
                      long storage times are sampled as evenly as short ones
                      and any n_points works
    finesse_min/max   range of comb finesse F = tooth_spacing / tooth_width
    n_finesse         number of finesse values across that range
    target_window     judge each spectrum over this width at the centre (MHz).
                      ``None`` -> 4 / storage_min, so the window always holds
                      several teeth however wide the spacing gets.
    top_n             how many storage times to keep a spectrum for, for
                      plotting: the best-finesse comb at each storage time,
                      for the ``top_n`` best storage times (see
                      :func:`_best_per_storage`) - never two spectra at the
                      same storage time. The sweep itself carries only scores;
                      these spectra are recomputed in a short collect pass once
                      the winners are known, which costs about
                      ``top_n / unique burns`` extra time and keeps the sweep's
                      memory flat. Clamped to the number of distinct (mesh-
                      snapped) storage times.
    min_storage_sep   minimum gap (us) between the kept storage times, so the
                      top spectra are distinct combs rather than one cluster
                      of near-identical neighbours. ``"auto"`` (default)
                      spreads them across the range -
                      ``(storage_max - storage_min) / (2 * top_n)``, see
                      :func:`_resolve_storage_sep`; ``0`` turns it off. A wide
                      gap can leave fewer than ``top_n`` spectra kept.
    parallel          run the sweep across a multiprocessing.Pool (default True)
    n_workers         processes to use when ``parallel`` (default one fewer than
                      os.cpu_count(), to leave the machine responsive)
    maxtasksperchild  recycle each worker process after this many tasks
                      (default: never). Not needed for memory (the per-worker
                      buffers are reused, not rebuilt - see burn_parallel), but
                      available for a very long run on a tight machine.
    progress          optional callback ``progress(done, total)``; when ``None``
                      a tqdm bar is used if stdout is a TTY
    log               optional callback ``log(str)`` for the status lines;
                      defaults to ``print``
    estimate          optional callback ``estimate(seconds)`` called once with
                      the projected total run time, right after the timing
                      probe and before the sweep starts
    on_done           optional callback ``on_done(state, out, error)``, called
                      once the run has a terminal state - "complete" (with the
                      full result dict as ``out``, ``error=None``) or "failed"
                      (``out=None``, ``error`` the message/traceback text),
                      including a pre-flight refusal that never reaches the
                      sweep at all. Not called on "cancelled" - a Cancel is not
                      a failure. This is the hook :func:`scan_teams_notifier`
                      plugs into; keep calling it at every terminal exit if you
                      touch this function's try/except/finally structure.
    sim_dir           a simulation folder to checkpoint into. ``None`` (the
                      default) reproduces the old in-memory-only behaviour
                      exactly. When set, every completed burn is appended to
                      ``sim_dir/progress.csv`` as it finishes (flushed and
                      fsynced every ``checkpoint_every`` points or 10s,
                      whichever comes first), so a crash, a cancel, or a power
                      loss loses at most a few seconds of work. See
                      :func:`load_run` to read a folder back, complete or not.
    resume            when ``sim_dir`` already holds a run: continue it,
                      skipping burns already in ``progress.csv`` (default
                      ``True``). ``False`` against an existing folder raises,
                      rather than overwriting it - pick a new folder instead.
                      A folder whose settings differ from this call (anything
                      that changes what ``quality`` means - df, the crystal,
                      the window, pump_passes, the burn mode, target_window, or
                      :data:`QUALITY_VERSION`) also raises, naming the fields
                      that changed.
    checkpoint_every  flush/fsync progress.csv after this many completed
                      burns, or after 10s, whichever comes first (default 10)

    Returns a dict with
        tag            short label for the run
        results        one dict per requested (storage, finesse) cell that was
                       actually burned: spacing_MHz, width_MHz, finesse,
                       storage_us, quality, measured_finesse. ``finesse`` is
                       the requested (written) value, the heatmap's axis;
                       ``measured_finesse`` is what the burned comb actually
                       came out with (:func:`_measured_finesse` - typically
                       far lower once broadening spreads the teeth; ``nan``
                       when unmeasurable, or for a cell resumed from a
                       checkpoint written before the column existed). A
                       cancelled or crashed run simply omits the cells it
                       never reached.
        best           up to top_n dicts, one per storage time (its best
                       finesse), best first; each also carrying spec, f, n
        params, finesse_vals, target_window, pump_passes, metric
        sim_dir, state, n_done, n_total, n_missing, resumed_from
                       state is "complete", "cancelled" or "failed"
    """
    params = DEFAULT_PARAMS if params is None else params
    log = print if log is None else log

    p = settings(params)
    df_user = p["df"]
    tag = f"{storage_min:g}-{storage_max:g}us"

    if target_window is None:
        target_window = 4.0 / storage_min
    _resolve_storage_sep(min_storage_sep, storage_min, storage_max, 1)

    # The scan grid is linear in storage time and in finesse, and so is what
    # gets burned: each (t, F) cell's comb has a mean tooth period of exactly
    # 1/t (period_steps - fractional, each tooth snapped to the mesh), not 1/t
    # rounded to a whole step, which would bunch the storage times actually
    # burned towards short t and leave ever wider gaps at long t.
    t_targets = np.linspace(storage_min, storage_max, max(2, int(n_points)))
    finesse_vals = np.linspace(finesse_min, finesse_max, max(1, int(n_finesse)))

    # A fractional period puts each tooth up to half a step off its ideal
    # position; refining df (_resolve_scan_df - see there for the bounds) keeps
    # that jitter below the spacing between neighbouring storage times.
    df, _capped_by_memory = _resolve_scan_df(t_targets, params, finesse_min)

    scan_params = params
    if df != df_user:
        scan_params = copy.deepcopy(params)
        scan_params["numerical_params"]["df"] = df
        p = settings(scan_params)

    requested = []          # (t_us, F, s_steps, w_steps)
    todo = {}               # (s_steps, w_steps) -> index
    distinct_spacings = set()
    for t in t_targets:
        s = period_steps(t, df)
        distinct_spacings.add(s)
        for F in finesse_vals:
            _, w = comb_steps(t, F, df)
            requested.append((float(t), float(F), s, w))
            todo.setdefault((s, w), len(todo))
    uniq = list(todo)       # unique (s_steps, w_steps), in first-seen order

    # requested and results (below) are both fully materialised lists, one
    # entry per (storage, finesse) cell - a caller driving run_scan directly
    # (bypassing the GUI's own n_points/n_finesse caps) could otherwise build
    # a many-million-entry list before the first burn even starts.
    if len(requested) > 4_000_000:
        raise ValueError(
            f"n_points x n_finesse = {len(requested)} cells is too large to "
            f"hold as a results table (limit 4,000,000) - reduce n_points "
            f"or n_finesse")

    # Open (or resume) the simulation folder, if one was given. cache is
    # prefilled from progress.csv when resuming - the only reuse mechanism,
    # and it is what shrinks items (below) to just the unburned cells.
    cache = {}               # (s_steps, w_steps) -> quality
    fmeas = {}               # (s_steps, w_steps) -> measured finesse (nan if
                             # unmeasurable, or from a pre-column checkpoint)
    ckpt = None
    if sim_dir is not None:
        burn_spec = _burn_spec(scan_params, burn, pump_passes, target_window)
        grid_spec = {"storage_min": storage_min, "storage_max": storage_max,
                    "n_points": int(n_points), "finesse_min": finesse_min,
                    "finesse_max": finesse_max, "n_finesse": int(n_finesse),
                    "top_n": int(top_n), "df_user": float(df_user),
                    "min_storage_sep": min_storage_sep}
        grid = {"requested": requested, "uniq": uniq,
                "finesse_vals": [float(v) for v in finesse_vals],
                "t_targets": [float(v) for v in t_targets]}
        ckpt, done = _open_run_dir(sim_dir, tag, burn_spec, grid_spec, grid,
                                   scan_params, resume, log,
                                   checkpoint_every=checkpoint_every)
        cache.update((k, q) for k, (q, _) in done.items())
        fmeas.update((k, fa) for k, (_, fa) in done.items())

    # One result row per requested (storage time, finesse) cell, from the
    # cache - built once at the top so both an early "already complete" return
    # and the normal path share the same reconstruction (see the end of this
    # function). Cells not yet burned (a resume, a cancel, a crash) are simply
    # omitted rather than filled with NaN: _scan_grids/_quality_by_storage
    # already treat a missing cell as a gap, so the scan plots work unchanged
    # on a partial run.
    def _build_results():
        results = []
        n_missing = 0
        for t, F, s, w in requested:
            q = cache.get((s, w))
            if q is None:
                n_missing += 1
                continue
            spacing_mhz, width_mhz = s * df, w * df
            q *= _target_match(t, F, spacing_mhz, width_mhz)
            results.append({"spacing_MHz": spacing_mhz, "width_MHz": width_mhz,
                            "finesse": F, "storage_us": t, "quality": q,
                            "measured_finesse": fmeas.get((s, w),
                                                          float("nan"))})
        return results, n_missing

    log(f"[{tag}]  {len(t_targets)} storage times x {len(finesse_vals)} "
        f"finesse = {len(requested)} cells ({len(uniq)} unique burns)   window "
        f"{p['fmin']:g} to {p['fmax']:g} MHz, df {df:g}, judged over "
        f"{target_window:g} MHz")
    if df != df_user:
        _refined_bytes, _ = _estimate_worker_bytes(scan_params,
                                                   min_finesse=finesse_min)
        log(f"    refined df {df_user:g} -> {df:g} MHz so neighbouring storage "
            f"times differ by at least one mesh step of tooth spacing (finer "
            f"mesh, slower burns) - ~{_fmt_bytes(_refined_bytes)}/worker now")
    # Past t_jitter, neighbouring storage times' periods differ by less than
    # one df step: still distinct combs on average (fractional periods), but
    # with each tooth up to df/2 off its ideal position.
    _dt = (storage_max - storage_min) / max(1, len(t_targets) - 1)
    t_jitter = float(np.sqrt(_dt / df)) if _dt > 0 else np.inf
    if t_jitter < storage_max:
        log(f"    note: above ~{t_jitter:.3g} us neighbouring storage times "
            f"differ by less than one df {df:g} MHz step of tooth spacing - "
            f"each still burns its own exact mean period, but with teeth up to "
            f"half a step off the ideal comb (df not refined further: "
            + ("worker memory" if _capped_by_memory else "mesh limits")
            + ")")

    # Built once: transition table, axes, initial rho, spectrum FFT setup.
    # Each worker process builds its own copy in _init_worker; this one is the
    # parent's, for the timing probe and the serial path.
    ctx = _scan_context(scan_params, burn, pump_passes, target_window)

    items = [(n, key) for n, key in enumerate(uniq) if key not in cache]
    n_done0 = len(uniq) - len(items)
    if n_done0:
        log(f"    resuming: {n_done0} of {len(uniq)} burns already done, "
            f"{len(items)} left")

    # One spectrum is kept per storage time (see _best_per_storage), so
    # keeping more than there are distinct storage times is meaningless.
    top_n = int(max(1, min(top_n, len(distinct_spacings))))
    if top_n >= len(distinct_spacings):
        log(f"    keeping the best spectrum at every one of the "
            f"{len(distinct_spacings)} storage times")

    # Pre-flight memory budget: fail fast with advice, rather than starting a
    # sweep that swaps or gets OOM-killed part way through. n_f comes from the
    # context already built above; the per-worker estimate uses finesse_min,
    # the worst case for pump-array size.
    n_f = len(ctx["f"])
    _per_worker, _breakdown = _estimate_worker_bytes(scan_params,
                                                     min_finesse=finesse_min)
    _avail, _avail_src = _available_bytes()
    _parent_estimate = _per_worker + top_n * n_f * 8 + len(requested) * 350
    _budget = 0.6 * _avail - _parent_estimate
    _nw = n_workers or max(1, (os.cpu_count() or 1) - 1)
    if parallel and items:
        _fit = max(0, int(_budget // _per_worker))
        if _fit < 1:
            _msg = (
                f"not enough memory for even one scan worker: need "
                f"~{_fmt_bytes(_per_worker)}/worker + "
                f"~{_fmt_bytes(_parent_estimate)} for the parent, only "
                f"~{_fmt_bytes(_avail)} available ({_avail_src}). Coarsen df, "
                f"narrow fmin/fmax, or raise the minimum finesse.")
            if on_done is not None:
                on_done("failed", None, _msg)
            raise ValueError(_msg)
        if _fit < _nw:
            log(f"    memory: ~{_fmt_bytes(_per_worker)}/worker "
                f"(context {_fmt_bytes(_breakdown['context'])}, burn "
                f"{_fmt_bytes(_breakdown['burn'])}, ffts "
                f"{_fmt_bytes(_breakdown['spectrum'])}) + "
                f"~{_fmt_bytes(_parent_estimate)} parent")
            log(f"    {_fmt_bytes(_avail)} available ({_avail_src}) - budget "
                f"{_fmt_bytes(max(0, _budget))} - workers {_nw} -> {_fit}")
            _nw = _fit
    _top_n_ceiling = max(1, int((0.25 * _avail) // (n_f * 8)))
    if top_n > _top_n_ceiling:
        log(f"    keep top N {top_n} -> {_top_n_ceiling}: retaining "
            f"{top_n} spectra at n_f={n_f} would need more memory than is "
            f"available ({_fmt_bytes(_avail)})")
        top_n = _top_n_ceiling
    storage_sep = _resolve_storage_sep(min_storage_sep, storage_min,
                                       storage_max, top_n)
    if storage_sep:
        log(f"    keeping storage times at least {storage_sep:.3g} us apart"
            + (" (auto)" if min_storage_sep in (None, "auto") else ""))

    # A single spectrum, timed, to estimate the whole run - the sweep plus the
    # collect pass that fetches the winners' spectra afterwards. Probed from
    # the remaining items (not the whole grid), and its result is kept rather
    # than thrown away, so resuming an almost-finished scan does not redo a
    # burn on every attempt.
    if items:
        _probe_item = items[len(items) // 2]
        _t0 = time.time()
        _, _probe_rec = compute_one(_probe_item, ctx=ctx, want_spec=False)
        _one = time.time() - _t0
        items = [it for it in items if it[0] != _probe_item[0]]
        _pq, _pn, _psmhz, _pwmhz, _pfa = _probe_rec
        cache[uniq[_pn]] = _pq
        fmeas[uniq[_pn]] = _pfa
        if ckpt is not None:
            _ps, _pw = uniq[_pn]
            ckpt.append(_pn, _ps, _pw, _psmhz, _pwmhz, _pq, _one, _pfa)
    else:
        _one = 0.0
    _n_jobs = len(items) + top_n
    _total = _one * _n_jobs
    if parallel:
        _total /= max(1, _nw)
    log("    about " + (f"{_total:.0f} seconds" if _total < 90
                        else f"{_total / 60:.0f} minutes"))
    if estimate is not None:
        estimate(_total)

    # The best (quality, n) seen so far at each storage time (s_steps) - keys
    # only, no arrays, and at most one entry per distinct spacing. The top_n
    # cut happens in _collect, via _best_per_storage.
    best = {}
    _key_of = {key: n for n, key in enumerate(uniq)}

    def _keep(key):
        s = uniq[key[1]][0]
        if s not in best or key > best[s]:
            best[s] = key

    # Seed the winners from whatever is already in the cache - the
    # resumed burns, and the probe above - so a resume's top spectra are
    # chosen from the whole grid, not just what this call re-burns. A resumed
    # cache can hold keys the *current* grid no longer requests at all (an
    # old checkpoint from before n_points/finesse changed, say) - _key_of
    # only knows this call's uniq, so anything else is simply not a
    # candidate for best and is skipped rather than looked up.
    for _key, _quality in cache.items():
        _n = _key_of.get(_key)
        if _n is not None:
            _keep((_quality, _n))

    # Cancel reaches here through the `progress` callback (the GUI's raises
    # from inside it); this flag is how the outer try/finally below tells a
    # cancel from a genuine failure without shb.py knowing anything about the
    # GUI's own exception type.
    _cancelled = [False]

    def _tick(done):
        if progress is None:
            return
        try:
            progress(done, _n_jobs)
        except BaseException:
            _cancelled[0] = True
            raise

    # Live solver status in the run log: a throttled line (roughly every 5 % of
    # the sweep, or every 15 s, whichever comes first) so a long sweep is not a
    # silent wait.
    _sweep_t0 = time.time()
    _log_step = max(1, len(items) // 20)
    _log_last = [_sweep_t0]
    log(f"[{tag}]  sweep start - {len(items)} burns"
        + (f" across {max(1, _nw)} workers" if parallel else " (serial)"))

    _prev_done_t = [_sweep_t0]      # for progress.csv's "seconds" column

    def _consume(i, record):
        quality, n, s_mhz, w_mhz, f_meas = record
        s_steps, w_steps = uniq[n]
        cache[(s_steps, w_steps)] = quality
        fmeas[(s_steps, w_steps)] = f_meas
        _keep((quality, n))
        now = time.time()
        if ckpt is not None:
            # Wall time since the previous completion - exact in the serial
            # path, an average per-item slice under parallelism (imap_unordered
            # gives no per-task timer). A diagnostic column only; nothing reads
            # it back.
            ckpt.append(n, s_steps, w_steps, s_mhz, w_mhz, quality,
                        now - _prev_done_t[0], f_meas)
            _prev_done_t[0] = now
        _tick(i)
        if (i == 1 or i == len(items) or i % _log_step == 0
                or now - _log_last[0] > 15.0):
            _log_last[0] = now
            el = now - _sweep_t0
            eta = el / i * (len(items) - i)
            t_s = 1.0 / s_mhz if s_mhz else float("nan")
            fin = s_mhz / w_mhz if w_mhz else float("nan")
            log(f"  scan {i}/{len(items)} ({100.0 * i / len(items):.0f}%) - "
                f"{_fmt_dur(el)} elapsed, ~{_fmt_dur(eta)} left - "
                f"last t_s {t_s:.3g}us F {fin:.1f} (measured {f_meas:.1f}) "
                f"Q {quality:.2f}")
            if ckpt is not None:
                ckpt.write_status(state="running")

    def _collect(pool):
        """
        The winners' spectra, once the sweep has picked them.

        The sweep carried only scores, so this is where the kept spectra are
        made: one contiguous (top_n, n_f) block, and one frequency axis shared
        by every record rather than a separate unpickled copy each.
        """
        keys = _best_per_storage(best.values(), uniq, top_n, df, storage_sep)
        if storage_sep and len(keys) < top_n:
            log(f"    only {len(keys)} storage times are {storage_sep:.3g} us "
                f"apart - keeping {len(keys)} of the {top_n} asked for")
        f_axis = ctx["f"]
        specs = np.empty((len(keys), len(f_axis)))
        jobs = [(row, uniq[n]) for row, (_, n) in enumerate(keys)]
        log(f"[{tag}]  collecting the best spectrum at the top {len(jobs)} "
            f"storage times")
        _t = time.time()
        stream = (pool.imap_unordered(_spectrum_for, jobs, chunksize=1)
                  if pool is not None
                  else (_spectrum_at(job, ctx) for job in jobs))
        for done, (row, spec) in enumerate(stream, start=1):
            specs[row] = spec
            _tick(len(items) + done)
        log(f"[{tag}]  collect done in {_fmt_dur(time.time() - _t)}")

        rows = []
        for row, (quality, n) in enumerate(keys):
            s_steps, w_steps = uniq[n]
            s_mhz, w_mhz = s_steps * df, w_steps * df
            f_meas = fmeas.get(uniq[n], float("nan"))
            if not np.isfinite(f_meas):
                # resumed from a checkpoint older than the column: the
                # spectrum is right here, so measure it rather than leave a gap
                f_meas = _score_quality_and_finesse(
                    f_axis, specs[row], target_window, s_mhz, w_mhz)[1]
            rows.append({"quality": quality, "n": n, "spec": specs[row],
                         "f": f_axis, "spacing_MHz": s_mhz,
                         "width_MHz": w_mhz, "finesse": s_mhz / w_mhz,
                         "measured_finesse": f_meas,
                         "storage_us": 1.0 / s_mhz})
        return rows

    _state = "failed"
    _error = None
    try:
        if not items:
            best_rows = _collect(None)
        elif parallel:
            import multiprocessing as mp
            n_workers = _nw
            # One item per task. The context no longer travels with the tasks,
            # so a task is a few dozen bytes - and burns differ ~10x in cost
            # across a finesse range, so chunking leaves whole minutes of idle
            # workers at the tail and delays a cancel by a chunk. Tiny points
            # still chunk, where per-task overhead would otherwise show.
            chunksize = 1 if _one > 0.05 else 8
            # Set, not defaulted: a GUI process has already imported numpy
            # with a threaded BLAS by the time it gets here, but its spawned
            # children inherit this environment and import theirs
            # single-threaded.
            for _var in _BLAS_THREAD_VARS:
                os.environ[_var] = "1"
            # Spawn, not fork: children must import numpy afresh to pick up
            # the single-thread BLAS variables, and forking a live Qt process
            # to copy the parent's whole heap into every worker is worth
            # avoiding.
            with mp.get_context("spawn").Pool(
                    processes=n_workers, initializer=_init_worker,
                    initargs=(scan_params, burn, pump_passes, target_window),
                    maxtasksperchild=maxtasksperchild) as pool:
                for i, record in enumerate(
                        pool.imap_unordered(_run_point, items,
                                            chunksize=chunksize), start=1):
                    _consume(i, record)
                log(f"[{tag}]  sweep done in "
                    f"{_fmt_dur(time.time() - _sweep_t0)}")
                best_rows = _collect(pool)
        else:
            iterator = items
            if progress is None and sys.stdout.isatty():
                try:
                    from tqdm import tqdm
                    iterator = tqdm(items, desc=f"scan {tag}", unit="spec")
                except ImportError:
                    pass
            for i, item in enumerate(iterator, start=1):
                _consume(i, compute_one(item, ctx=ctx, want_spec=False)[1])
            log(f"[{tag}]  sweep done in {_fmt_dur(time.time() - _sweep_t0)}")
            best_rows = _collect(None)
        _state = "complete"
    except BaseException as exc:
        _state = "cancelled" if _cancelled[0] else "failed"
        _error = None if _cancelled[0] else "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))
        if on_done is not None and _state == "failed":
            on_done("failed", None, _error)
        raise
    finally:
        results, n_missing = _build_results()
        if ckpt is not None:
            if _error:
                with open(os.path.join(sim_dir, "runlog.txt"), "a") as fh:
                    fh.write(_error)
            ckpt.close(state=_state, error=_error)

    if n_missing:
        log(f"    {n_missing} of {len(requested)} cells not burned "
            f"(partial run)")

    out = {"tag": tag,
          "results": results,
          "best": best_rows,
          "params": scan_params,
          "df_used": df,
          "finesse_vals": [float(v) for v in finesse_vals],
          "target_window": target_window,
          "pump_passes": pump_passes,
          "metric": "afc_quality",
          "sim_dir": sim_dir,
          "state": _state,
          # Counted against the *current* uniq, not len(cache): a resume
          # after the grid changed can leave stale keys sitting in cache that
          # the current grid no longer requests at all (see the seeding loop
          # above), and those should not inflate n_done past n_total.
          "n_done": sum(1 for k in uniq if k in cache),
          "n_total": len(uniq),
          "n_missing": n_missing,
          "resumed_from": n_done0}
    if sim_dir is not None and _state == "complete":
        try:
            write_run_outputs(sim_dir, out)
        except Exception as exc:                       # noqa: BLE001
            log(f"    could not write final outputs to {sim_dir}: {exc}")
    if on_done is not None:
        on_done("complete", out, None)
    return out


# ===========================================================================
# Notifications - optional webhook alert when a scan finishes or fails
# ===========================================================================

def resolve_teams_webhook():
    """
    The configured Teams (Power Automate) webhook URL, or None if none is set.

    Checked in order: the ``SHB_TEAMS_WEBHOOK`` environment variable, then a
    ``.teams_webhook`` file next to this module (one line, the URL). Neither
    is required; with neither set, :func:`notify_teams` is a silent no-op.

    The webhook URL is a bearer secret - anyone holding it can post to the
    channel - so ``.teams_webhook`` is gitignored and must never be committed;
    the environment variable lets it be overridden per-machine without
    touching that file at all.
    """
    url = os.environ.get("SHB_TEAMS_WEBHOOK")
    if url:
        return url.strip()
    path = Path(__file__).with_name(".teams_webhook")
    if path.exists():
        text = path.read_text().strip()
        return text or None
    return None


def notify_teams(title, text, *, webhook_url=None, log=None):
    """
    Best-effort POST of a simple Teams "MessageCard" to a Power Automate
    webhook.

    Never raises - a notification failure must not take down or mask the
    result of an actual scan. A problem is reported to ``log`` if given,
    otherwise printed to stderr, and otherwise dropped.

    ``webhook_url`` defaults to :func:`resolve_teams_webhook`; with neither
    that nor the environment/file configured, this is a silent no-op (no
    network attempt at all).
    """
    if webhook_url is None:
        webhook_url = resolve_teams_webhook()
    if not webhook_url:
        return
    payload = json.dumps({
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": title,
        "title": title,
        "text": text,
    }).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        msg = f"Teams notification failed: {exc}"
        if log is not None:
            log(msg)
        else:
            print(msg, file=sys.stderr)


def scan_teams_notifier(webhook_url=None, log=None):
    """
    An ``on_done`` callback for :func:`run_scan` that posts a finish/fail
    summary to Teams - pass as ``run_scan(..., on_done=scan_teams_notifier())``.

    The webhook is resolved once, here, rather than on every call: with
    nothing configured this returns ``None`` (not a callable), so a caller
    can pass the result straight through and ``run_scan``'s ``on_done is not
    None`` check disables the hook for free - no wasted HTTP attempt on the
    common case of nobody having set this up.

    Only "complete" and "failed" post a message; "cancelled" (the caller
    simply never invokes this for that state - see run_scan's on_done
    parameter) is a user's own choice, not a failure worth an alert.
    """
    if webhook_url is None:
        webhook_url = resolve_teams_webhook()
    if not webhook_url:
        return None

    def on_done(state, out, error):
        if state == "complete":
            best = out["best"][0] if out["best"] else None
            summary = (
                f"quality {best['quality']:.3f} at t_s {best['storage_us']:.3g}"
                f" us, F {best['finesse']:.1f}" if best else "no spectra kept")
            notify_teams(
                f"Scan finished: {out['tag']}",
                f"{out['n_done']}/{out['n_total']} burns done. Best: {summary}.",
                webhook_url=webhook_url, log=log)
        elif state == "failed":
            notify_teams(
                "Scan failed",
                error or "(no error message captured)",
                webhook_url=webhook_url, log=log)
    return on_done


# ===========================================================================
# Checkpointing - simulation folders, so a long scan survives a crash
# ===========================================================================
#
# A simulation folder holds:
#   manifest.json     written once, atomically - the resume contract (burn_spec)
#                      plus the requested grid (grid_spec) and full params
#   grid.json         the requested grid + unique burns, written once
#   progress.csv      APPEND-ONLY, one row per completed burn - the checkpoint
#   status.json       atomically replaced every ~15s - the live "how far in"
#                      view, and (pid + heartbeat) the run's lock
#   runlog.txt        every line the caller's `log` callback emits, timestamped
#   results.csv, best.json, top_spectra.npy, figures/*.png   final outputs
#   figures/top_spectra/*.png                                 one per kept spectrum
#
# progress.csv is authoritative; status.json is advisory and never consulted
# by a resume - a folder with no status.json at all (the writer died before
# its first write) still resumes correctly from progress.csv alone.

_HEARTBEAT_STALE_S = 120.0     # well above the ~15s status cadence


def _burn_spec(params, burn, pump_passes, target_window):
    """
    The part of a scan's settings that changes what a checkpointed ``quality``
    *means* for a given ``(spacing_steps, width_steps)`` key - so it is what a
    resume must match. Everything else (which cells are requested, how many
    workers, top_n) is bookkeeping, not physics, and lives in ``grid_spec``
    instead.
    """
    p = settings(params)
    return {
        "quality_version": QUALITY_VERSION,
        "df_used": float(p["df"]),
        "g": [float(x) for x in p["g"]],
        "e": [float(x) for x in p["e"]],
        "dm": np.asarray(p["dm"], dtype=float).tolist(),
        "h_width": float(p["h_width"]),
        "in_width": float(p["in_width"]),
        "fmin": float(p["fmin"]),
        "fmax": float(p["fmax"]),
        "alpha": float(p["alpha"]),
        "pump_passes": int(pump_passes),
        "burn": burn.__name__,
        "target_window": float(target_window),
    }


def _grid_hash(burn_spec):
    """A short, stable fingerprint of a burn_spec, for a fast resume check."""
    text = json.dumps(burn_spec, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _diff_burn_spec(old, new):
    """Field-by-field differences between two burn_specs, as (field, old, new)."""
    diffs = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            diffs.append((key, old.get(key), new.get(key)))
    return diffs


def _replace_with_retry(src, dst, attempts=20, delay=0.1):
    """
    ``os.replace(src, dst)`` that tolerates a cloud-synced folder (Dropbox,
    OneDrive, ...) transiently holding ``dst`` open.

    Those clients open a file for hashing/upload right after it changes, and
    on Windows that briefly makes ``os.replace`` fail with "access is
    denied" (WinError 5) or a sharing violation (WinError 32) even though
    nothing in this process is holding it. The lock is normally gone within
    a few hundred ms, so a short retry clears it instead of crashing a scan
    whose ``sim_dir`` happens to live in a synced folder - see
    :func:`_atomic_write_json`, the only caller.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 32
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay)


def _atomic_write_json(path, obj):
    """Write JSON to ``path`` via a temp file + ``os.replace``, so a reader
    never observes a half-written file. See :func:`_replace_with_retry` for
    why the final rename tolerates a momentary lock from a synced folder."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# measured_finesse was added after the first simulation folders were written,
# so it goes last: every older column keeps its index, and a folder started
# under _PROGRESS_HEADER_V1 resumes by appending the longer rows beneath its
# old header (see _read_progress_rows for how the two are told apart).
_PROGRESS_HEADER_V1 = ("n", "s_steps", "w_steps", "spacing_MHz", "width_MHz",
                       "quality", "seconds")
_PROGRESS_HEADER = _PROGRESS_HEADER_V1 + ("measured_finesse",)


def _parse_steps(text):
    """
    A tooth period as written to ``progress.csv``: an ``int`` for a whole
    number of df steps, else the exact ``float`` (``repr`` round-trips) - see
    :func:`period_steps`. ``int`` and ``float`` keys of equal value hash alike,
    so lookups against ``grid.json``'s ``uniq`` work either way.
    """
    try:
        return int(text)
    except ValueError:
        return float(text)


def _read_progress(path):
    """
    Completed burns from an append-only ``progress.csv``, as
    ``{(s_steps, w_steps): quality}`` - :func:`_read_progress_rows` without
    the measured finesse.
    """
    return {k: q for k, (q, _) in _read_progress_rows(path).items()}


def _read_progress_rows(path):
    """
    Completed burns from an append-only ``progress.csv``, as
    ``{(s_steps, w_steps): (quality, measured_finesse)}``.

    A crash between a write and its flush can leave the last line partial, or
    - when the file was extended but the write never reached disk - a run of
    NUL bytes. Both are handled: the final line is dropped outright unless the
    file ends with a newline, and any remaining row with a NUL byte, the wrong
    field count, or an unparseable number is skipped. On a duplicate key the
    last row wins, so a point re-run after a resume overwrites its own earlier,
    possibly-truncated line.

    A folder started before the ``measured_finesse`` column existed has the
    old 7-column header; its rows read with ``measured_finesse = nan``, and a
    resume appends full-width rows beneath them, so such a file holds both
    widths and both are accepted. A file with the current header only ever
    holds full-width rows, so a short row there is damage, not history.
    """
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace")
    ends_clean = text.endswith("\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if lines and not ends_clean:
        lines.pop()                              # torn final line
    if len(lines) < 2:                           # header only, or nothing
        return done
    widths = {len(_PROGRESS_HEADER)}
    if tuple(lines[0].strip().split(",")) == _PROGRESS_HEADER_V1:
        widths.add(len(_PROGRESS_HEADER_V1))
    for line in lines[1:]:
        if "\x00" in line:
            continue
        parts = line.split(",")
        if len(parts) not in widths:
            continue
        try:
            s_steps = _parse_steps(parts[1])
            w_steps = int(parts[2])
            quality = float(parts[5])
            f_meas = (float(parts[7]) if len(parts) > 7 else float("nan"))
        except ValueError:
            continue
        done[(s_steps, w_steps)] = (quality, f_meas)
    return done


def _trim_torn_tail(path):
    """
    Cut a torn final line (one with no closing newline - see
    :func:`_read_progress_rows`) off ``progress.csv`` before appending to it.

    Readers already ignore those bytes, so this changes nothing they see; it
    only stops the first appended row from being glued onto the torn one.
    That matters more now that a pre-``measured_finesse`` file accepts two row
    widths: a torn ``"5,1"`` glued to a full row reads as a plausible
    full-width row with the wrong ``s_steps``. Everything before the last
    newline is left untouched - this is not a rewrite.
    """
    if not os.path.exists(path):
        return
    with open(path, "rb+") as fh:
        data = fh.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        fh.seek(cut)
        fh.truncate()
        if cut == 0:                    # even the header was torn
            fh.write((",".join(_PROGRESS_HEADER) + "\n").encode())


def _is_live(status):
    """
    Whether ``status`` (a loaded ``status.json``) describes a scan that is
    genuinely still running.

    ``state`` is checked first and is the deciding factor: a folder marked
    complete/cancelled/failed is never live, even if the pid it names happens
    to still exist (it may belong to an unrelated later process, or - inside
    a single long-lived interpreter, such as a test driving several scans
    back to back - to the very code now asking the question). Only when the
    state still says "running" does the pid/heartbeat pair decide whether
    that is real or stale: both a live pid *and* a fresh heartbeat, on this
    host, are required, since the pid alone can be reused after a reboot and
    a heartbeat alone cannot tell a paused writer from a dead one. That is
    what recognises a stale "running" status - SIGKILL, an OOM kill, power
    loss - as safe to resume.
    """
    if status.get("state") != "running":
        return False
    if status.get("host") != socket.gethostname():
        return False
    try:
        pid = int(status.get("pid"))
        os.kill(pid, 0)                          # existence check only
    except (TypeError, ValueError, OSError):
        return False
    return (time.time() - float(status.get("heartbeat", 0))) < _HEARTBEAT_STALE_S


class _Checkpoint:
    """
    Owns one simulation folder's on-disk state while :func:`run_scan` runs.

    Call :meth:`append` after every completed burn and :meth:`close` exactly
    once, from a ``finally``, however the sweep ends.
    """

    def __init__(self, sim_dir, resumed_from, n_total, checkpoint_every=10):
        self.sim_dir = sim_dir
        self.resumed_from = resumed_from
        self.n_total = n_total
        self._checkpoint_every = max(1, int(checkpoint_every))
        _trim_torn_tail(os.path.join(sim_dir, "progress.csv"))
        self._fh = open(os.path.join(sim_dir, "progress.csv"), "a")
        self._since_sync = 0
        self._last_sync = time.time()
        self._t0 = time.time()
        self._n_done = resumed_from
        self._best_quality = None
        self._last_point = None

    def append(self, n, s_steps, w_steps, s_mhz, w_mhz, quality, seconds,
               measured_finesse=float("nan")):
        self._fh.write(f"{n},{s_steps},{w_steps},{s_mhz!r},{w_mhz!r},"
                       f"{quality!r},{seconds:.3f},{float(measured_finesse)!r}\n")
        self._since_sync += 1
        self._n_done += 1
        if self._best_quality is None or quality > self._best_quality:
            self._best_quality = quality
        self._last_point = {
            "storage_us": 1.0 / s_mhz if s_mhz else None,
            "finesse": s_mhz / w_mhz if w_mhz else None,
            "measured_finesse": (float(measured_finesse)
                                 if np.isfinite(measured_finesse) else None),
            "quality": quality}
        now = time.time()
        # Both a count and a time bound: at a fine mesh a burn takes seconds,
        # so an fsync per point is free and worth it; the count bound alone
        # would make a fast, coarse scan fsync tens of times a second, and the
        # time bound alone would lose up to 10s of a fast one. Together the
        # loss window on a crash is whichever bound is smaller.
        if (self._since_sync >= self._checkpoint_every
                or now - self._last_sync > 10.0):
            self._sync()
            self._last_sync = now

    def _sync(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_sync = 0

    def write_status(self, state="running", error=None):
        status = {
            "state": state, "pid": os.getpid(), "host": socket.gethostname(),
            "started": self._t0, "heartbeat": time.time(),
            "done": self._n_done, "total": self.n_total,
            "resumed_from": self.resumed_from,
            "elapsed_s": time.time() - self._t0,
            "best_quality": self._best_quality,
            "last": self._last_point, "error": error,
        }
        _atomic_write_json(os.path.join(self.sim_dir, "status.json"), status)

    def close(self, state="complete", error=None):
        self._sync()
        self._fh.close()
        self.write_status(state=state, error=error)


def _open_run_dir(sim_dir, tag, burn_spec, grid_spec, grid, params, resume,
                  log, checkpoint_every=10):
    """
    Open (or create) a simulation folder for :func:`run_scan`.

    Returns a :class:`_Checkpoint` and ``done``, the completed burns read from
    ``progress.csv`` when resuming (``{key: (quality, measured_finesse)}``, as
    :func:`_read_progress_rows` gives them; empty for a new folder).
    Refuses - raising ``ValueError`` - a folder that is live, that mismatches
    on ``burn_spec``, or (when ``resume`` is false) that already has a run:
    there is no forced resume, because a forced one would silently mix two
    physics settings into one results table and nothing downstream could tell.
    """
    os.makedirs(sim_dir, exist_ok=True)
    manifest_path = os.path.join(sim_dir, "manifest.json")
    grid_path = os.path.join(sim_dir, "grid.json")
    grid_hash = _grid_hash(burn_spec)

    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        if manifest.get("schema_version", 1) > 1:
            raise ValueError(
                f"{sim_dir} was written by a newer version of shb.py "
                f"(schema_version {manifest['schema_version']}) - cannot "
                f"resume it here")

        status_path = os.path.join(sim_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path) as fh:
                status = json.load(fh)
            if _is_live(status):
                raise ValueError(
                    f"{sim_dir} is being written by pid {status.get('pid')} "
                    f"on {status.get('host')} (heartbeat "
                    f"{time.time() - status.get('heartbeat', 0):.0f}s ago) - "
                    f"wait for it to finish, or use a different folder")

        if manifest.get("grid_hash") != grid_hash:
            diffs = _diff_burn_spec(manifest.get("burn_spec", {}), burn_spec)
            lines = "\n".join(f"    {k:<14} {o!r:>10} -> {n!r}"
                              for k, o, n in diffs)
            raise ValueError(
                f"cannot resume {sim_dir}: it was burned with different "
                f"settings -\n{lines}\nChange these back, or pick a new "
                f"simulation folder.")

        if not resume:
            raise ValueError(
                f"{sim_dir} already has a run. Pass resume=True to continue "
                f"it, or choose a different simulation folder - a fresh run "
                f"never overwrites an existing one.")

        if manifest.get("grid_spec") != grid_spec:
            history = manifest.setdefault("history", [])
            history.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "grid_spec": manifest.get("grid_spec")})
            manifest["grid_spec"] = grid_spec
            manifest["n_cells"] = len(grid["requested"])
            manifest["n_unique"] = len(grid["uniq"])
            _atomic_write_json(manifest_path, manifest)
            _atomic_write_json(grid_path, grid)

        done = _read_progress_rows(os.path.join(sim_dir, "progress.csv"))
        return (_Checkpoint(sim_dir, resumed_from=len(done),
                            n_total=len(grid["uniq"]),
                            checkpoint_every=checkpoint_every), done)

    manifest = {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "grid_hash": grid_hash,
        "burn_spec": burn_spec,
        "grid_spec": grid_spec,
        "run_spec": {"host": socket.gethostname(),
                    "python": sys.version.split()[0], "numpy": np.__version__},
        "n_cells": len(grid["requested"]),
        "n_unique": len(grid["uniq"]),
        "params": json.loads(params_to_json(params)),
        "history": [],
    }
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(grid_path, grid)
    with open(os.path.join(sim_dir, "progress.csv"), "w") as fh:
        fh.write(",".join(_PROGRESS_HEADER) + "\n")
    log(f"    new simulation folder: {sim_dir}")
    return (_Checkpoint(sim_dir, resumed_from=0, n_total=len(grid["uniq"]),
                       checkpoint_every=checkpoint_every), {})


def inspect_run(sim_dir, burn_spec=None):
    """
    A cheap summary of a simulation folder - state, how far in it is, whether
    it is live, and (when ``burn_spec`` is given) whether it matches this
    call's settings. Reads ``progress.csv`` but never the spectra, so it is
    fast enough to call on every keystroke of a GUI folder field.

    Returns ``{state, n_done, n_total, live, pid, mismatch}``. ``state`` is
    ``"new"`` when ``sim_dir`` has no manifest yet; otherwise the folder's own
    recorded state (``running``/``cancelled``/``failed``/``complete``), with a
    stale ``running`` corrected to whatever ``status.json`` last said before
    the writer died. ``mismatch`` is a list of ``(field, old, new)`` from
    :func:`_diff_burn_spec`, empty unless ``burn_spec`` was passed and differs.
    """
    manifest_path = os.path.join(sim_dir, "manifest.json")
    if not os.path.isdir(sim_dir) or not os.path.exists(manifest_path):
        return {"state": "new", "n_done": 0, "n_total": 0, "live": False,
               "pid": None, "mismatch": []}
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    live, pid, state = False, None, "complete"
    status_path = os.path.join(sim_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path) as fh:
            status = json.load(fh)
        live = _is_live(status)
        pid = status.get("pid")
        state = "running" if live else status.get("state", "unknown")

    done = _read_progress(os.path.join(sim_dir, "progress.csv"))
    # Counted against the current grid's uniq, not len(done): a folder whose
    # grid changed (n_points widened, say) can have progress.csv rows the
    # current grid no longer requests at all, and those should not inflate
    # n_done past n_total.
    n_total = manifest.get("n_unique", len(done))
    grid_path = os.path.join(sim_dir, "grid.json")
    if os.path.exists(grid_path):
        with open(grid_path) as fh:
            uniq = [tuple(k) for k in json.load(fh)["uniq"]]
        n_done = sum(1 for k in uniq if k in done)
    else:
        n_done = len(done)
    mismatch = ([] if burn_spec is None else
               _diff_burn_spec(manifest.get("burn_spec", {}), burn_spec))
    return {"state": state, "n_done": n_done, "n_total": n_total,
           "live": live, "pid": pid, "mismatch": mismatch}


def load_run(sim_dir):
    """
    Read a simulation folder back into the dict :func:`run_scan` returns.

    Works on a folder in any state - complete, cancelled, failed, or one whose
    writing process was killed outright: the completed burns are whatever
    reached ``progress.csv``. ``results`` is rebuilt from ``grid.json``'s
    requested grid and the completed keys, omitting any cell not yet burned -
    the same convention :func:`run_scan` itself uses, so the scan plots need
    no special-casing for a partial run. A folder checkpointed before
    ``progress.csv`` had a ``measured_finesse`` column reads that field as
    ``nan`` for every cell. ``best`` comes from ``best.json`` and
    ``top_spectra.npy`` when a collect pass has run (see :func:`collect_top`),
    and is empty otherwise.
    """
    with open(os.path.join(sim_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(sim_dir, "grid.json")) as fh:
        grid = json.load(fh)
    done = _read_progress_rows(os.path.join(sim_dir, "progress.csv"))

    params = params_from_json(json.dumps(manifest["params"]))
    df = float(manifest["burn_spec"]["df_used"])

    results = []
    n_missing = 0
    for t, F, s, w in grid["requested"]:
        row = done.get((s, int(w)))
        if row is None:
            n_missing += 1
            continue
        q, f_meas = row
        spacing_mhz, width_mhz = s * df, w * df
        q *= _target_match(t, F, spacing_mhz, width_mhz)
        results.append({"spacing_MHz": spacing_mhz, "width_MHz": width_mhz,
                        "finesse": F, "storage_us": t, "quality": q,
                        "measured_finesse": f_meas})

    best = []
    best_path = os.path.join(sim_dir, "best.json")
    spec_path = os.path.join(sim_dir, "top_spectra.npy")
    if os.path.exists(best_path) and os.path.exists(spec_path):
        with open(best_path) as fh:
            best_meta = json.load(fh)
        stack = np.load(spec_path)
        f_axis = stack[0]
        for i, meta in enumerate(best_meta, start=1):
            best.append({**meta, "f": f_axis, "spec": stack[i]})

    state = "unknown"
    status_path = os.path.join(sim_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path) as fh:
            state = json.load(fh).get("state", "unknown")

    # Counted against the current grid's uniq, not len(done): a folder whose
    # grid changed can have progress.csv rows the current grid no longer
    # requests at all (see the equivalent note in run_scan's own seeding
    # loop), and those should not inflate n_done past n_total.
    n_done = sum(1 for s, w in grid["uniq"] if (s, int(w)) in done)
    return {"tag": manifest["tag"], "results": results, "best": best,
           "params": params, "df_used": df,
           "finesse_vals": grid["finesse_vals"],
           "target_window": manifest["burn_spec"]["target_window"],
           "pump_passes": manifest["burn_spec"]["pump_passes"],
           "metric": "afc_quality", "sim_dir": sim_dir, "state": state,
           "n_done": n_done, "n_total": len(grid["uniq"]),
           "n_missing": n_missing, "resumed_from": n_done}


def _write_best(sim_dir, best):
    """
    ``best.json`` (metadata only) + ``top_spectra.npy`` (row 0 = the shared
    frequency axis, then one row per spectrum) - the same layout the GUI's
    export has always written, so :func:`load_run` reads either back.
    """
    meta = [{k: v for k, v in r.items() if k not in ("f", "spec")}
           for r in best]
    _atomic_write_json(os.path.join(sim_dir, "best.json"), meta)
    path = os.path.join(sim_dir, "top_spectra.npy")
    if not best:
        if os.path.exists(path):
            os.remove(path)
        return
    n_f = len(best[0]["f"])
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                   shape=(len(best) + 1, n_f))
    mm[0] = best[0]["f"]
    for i, r in enumerate(best, start=1):
        mm[i] = r["spec"]
    mm.flush()
    del mm


def collect_top(sim_dir, top_n=None, min_storage_sep=None):
    """
    (Re-)burn the best spectrum at each of the ``top_n`` best storage times
    for a simulation folder from its checkpointed scores (the same rule as
    :func:`run_scan`, via :func:`_best_per_storage`), and write ``best.json`` +
    ``top_spectra.npy`` - without re-running the sweep.

    Works whether the sweep finished, was cancelled, or crashed: it only needs
    ``progress.csv``'s scores. ``top_n`` and ``min_storage_sep`` (``"auto"``
    or a gap in us, as in :func:`run_scan`) default to the values recorded in
    the folder's manifest - ``"auto"`` for a folder from before the setting
    existed. Returns the ``best`` list, in the same shape :func:`run_scan`'s
    own ``best`` carries. Each kept comb's ``measured_finesse`` comes from
    ``progress.csv``, or - for a folder checkpointed before that column
    existed - is read off the re-burned spectrum itself.
    """
    with open(os.path.join(sim_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(sim_dir, "grid.json")) as fh:
        grid = json.load(fh)
    rows = _read_progress_rows(os.path.join(sim_dir, "progress.csv"))
    if not rows:
        raise ValueError(f"{sim_dir} has no completed burns to collect from")
    done = {k: q for k, (q, _) in rows.items()}

    uniq = [tuple(k) for k in grid["uniq"]]
    grid_spec = manifest["grid_spec"]
    top_n = int(grid_spec["top_n"] if top_n is None else top_n)
    # Clamped as run_scan clamps it, so an "auto" gap comes out the same.
    top_n = max(1, min(top_n, len({s for s, _ in uniq})))
    if min_storage_sep is None:
        min_storage_sep = grid_spec.get("min_storage_sep", "auto")
    storage_sep = _resolve_storage_sep(min_storage_sep,
                                       grid_spec["storage_min"],
                                       grid_spec["storage_max"], top_n)

    params = params_from_json(json.dumps(manifest["params"]))
    burn = (burn_series if manifest["burn_spec"]["burn"] == "burn_series"
           else burn_parallel)
    pump_passes = manifest["burn_spec"]["pump_passes"]
    target_window = manifest["burn_spec"]["target_window"]
    ctx = _scan_context(params, burn, pump_passes, target_window)
    df = float(manifest["burn_spec"]["df_used"])

    # Keys the current grid no longer requests (a checkpoint from before
    # n_points/finesse changed) are not candidates - as in run_scan's seeding.
    key_of = {key: n for n, key in enumerate(uniq)}
    ranked = _best_per_storage(
        ((q, key_of[k]) for k, q in done.items() if k in key_of), uniq, top_n,
        df, storage_sep)

    best = []
    for quality, n in ranked:
        s_steps, w_steps = uniq[n]
        _, spec = _spectrum_at((0, (s_steps, w_steps)), ctx)
        s_mhz, w_mhz = s_steps * df, w_steps * df
        f_meas = rows[(s_steps, w_steps)][1]
        if not np.isfinite(f_meas):
            f_meas = _score_quality_and_finesse(ctx["f"], spec, target_window,
                                                s_mhz, w_mhz)[1]
        best.append({"quality": quality, "n": n, "spec": spec, "f": ctx["f"],
                     "spacing_MHz": s_mhz, "width_MHz": w_mhz,
                     "finesse": s_mhz / w_mhz, "measured_finesse": f_meas,
                     "storage_us": 1.0 / s_mhz})

    _write_best(sim_dir, best)
    return best


#: The columns of a scan's ``results.csv`` (and the GUI's ``*_results.csv``
#: export), one per key of a :func:`run_scan` ``results`` row. ``finesse`` is
#: the written (requested) finesse; ``measured_finesse`` the one the burned
#: comb came out with - empty for a scan saved before that was recorded.
RESULTS_COLUMNS = ("spacing_MHz", "width_MHz", "finesse", "storage_us",
                   "quality", "measured_finesse")


def write_run_outputs(sim_dir, scan):
    """
    Write a completed (or partial) scan's final outputs into its simulation
    folder: ``results.csv`` (:data:`RESULTS_COLUMNS`), ``best.json`` +
    ``top_spectra.npy`` (see :func:`_write_best`), the scan figures
    (:data:`SCAN_FIGURES`) under ``figures/``, and one
    image per kept spectrum under ``figures/top_spectra/``
    (:func:`save_top_spectrum_figures`).

    Called by :func:`run_scan` itself once a sweep with a ``sim_dir``
    completes, and usable standalone to re-export a folder (a different
    ``top_n`` via :func:`collect_top` first, say) without recomputing anything.
    """
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        w.writerows(scan["results"])

    _write_best(sim_dir, scan["best"])

    fig_dir = os.path.join(sim_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 4), constrained_layout=True)
    for name, fn in SCAN_FIGURES:
        fig.clear()
        try:
            # A very partial run (a handful of cells, or none kept yet) can
            # leave too little for one of these to draw - skip that figure
            # rather than fail the whole export over it.
            fn(fig.add_subplot(111), scan)
            fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=200)
        except Exception:                              # noqa: BLE001
            pass
    del fig
    # One image per kept spectrum too - the overlay above gets unreadable past
    # a handful of them.
    try:
        save_top_spectrum_figures(scan, os.path.join(fig_dir, "top_spectra"))
    except Exception:                                  # noqa: BLE001
        pass


#: Everything of a burn result :func:`save_burn` keeps. ``rho``/``rho0`` (and
#: ``det``/``trans``) are left out on purpose: the populations are
#: ``n_g x n_det`` - far larger than the spectrum at a fine mesh - and nothing
#: that redraws a finished burn reads them.
_BURN_KEYS = ("f", "spec", "pump_freqs", "stages")


def save_burn(path, out, params, session=None):
    """
    Write a single-hole / comb / preparation-sequence result to ``path`` (a
    pickle) so :func:`load_result` can reopen it later.

    ``out`` is what :func:`run_burn` or :func:`run_sequence` returned;
    ``params`` the parameters it ran with, kept so a reopened burn plots on
    its own crystal and window rather than whatever is configured now.
    ``session`` is any extra caller state to carry along - the GUI stores its
    side-panel settings there so reopening also restores the mode, fields and
    sequence rows. ``f`` and ``spec`` are stored as-is, so a reloaded
    spectrum is bit-identical to the one that was saved.
    """
    rec = {k: out[k] for k in _BURN_KEYS if out.get(k) is not None}
    rec.update({"kind": "burn", "params": params, "session": session})
    with open(path, "wb") as fh:
        pickle.dump(rec, fh)


def upgrade_legacy_scan(scan):
    """
    Convert a scan pickled by the old notebooks' ``run_scan`` (the
    ``results_YYYYMMDD_HHMMSS.pkl`` files) into the dict :func:`run_scan`
    returns today, so the same plots and GUI views read it.

    The notebook scan swept tooth spacing x burn width and scored each point
    by its delta OD over the central ``target_window``, storing

        results   (spacing_MHz, width_MHz, delta_od) tuples
        best      (delta_od, n, spec, spacing_MHz, width_steps, f) tuples

    Those map onto today's axes directly - storage time is ``1 / spacing``
    and finesse ``spacing / width`` - but the score is a different quantity,
    so it is carried over as ``quality`` with ``metric = "delta_od"`` and the
    plots label it as such rather than as AFC comb quality.
    """
    params = scan["params"]
    df = settings(params)["df"]

    def cell(spacing, width):
        return {"spacing_MHz": float(spacing), "width_MHz": float(width),
                "finesse": float(spacing) / float(width),
                "storage_us": 1.0 / float(spacing)}

    results = [{**cell(s, w), "quality": float(d),
                "measured_finesse": float("nan")}
               for s, w, d in scan["results"]]

    # One shared frequency axis, as run_scan keeps it: every record of one
    # scan was burned on the same mesh, and the notebook pickled one copy per
    # record only because it held the arrays by value.
    best, f_axis = [], None
    for d, _, spec, spacing, width_steps, f in scan.get("best", []):
        if f_axis is None:
            f_axis = np.asarray(f)
        best.append({**cell(spacing, width_steps * df), "quality": float(d),
                     "f": f_axis, "spec": np.asarray(spec)})

    finesse_vals = sorted({round(r["finesse"], 9) for r in results})
    return {"tag": scan.get("tag", "legacy"), "results": results,
           "best": best, "params": params, "df_used": df,
           "finesse_vals": finesse_vals,
           "target_window": scan.get("target_window"),
           "pump_passes": scan.get("pump_passes"),
           "metric": "delta_od", "sim_dir": None, "state": "complete",
           "n_done": len(results), "n_total": len(results), "n_missing": 0}


def _sibling_config(npy_path):
    """
    The ``<prefix>_config.json`` an export wrote beside ``<prefix>_spectrum.npy``
    - its ``params`` (and GUI session), or ``(None, None)`` if there is none.
    """
    base = os.path.splitext(npy_path)[0]
    if base.endswith("_spectrum"):
        base = base[:-len("_spectrum")]
    cfg = base + "_config.json"
    if not os.path.exists(cfg):
        return None, None
    with open(cfg) as fh:
        data = json.load(fh)
    params = data["params"] if "params" in data else data
    params = params_from_json(json.dumps(params, default=_json_default))
    return params, (data if "gui" in data else None)


def load_result(path):
    """
    Open any saved simulation result. Returns ``(kind, data)``, with ``kind``
    either ``"scan"`` (``data`` shaped like :func:`run_scan`'s return) or
    ``"burn"`` (shaped like :func:`run_burn`'s, minus the populations, plus
    ``params`` and ``session`` - either may be None).

    Accepts

    - a simulation folder - read with :func:`load_run`;
    - a scan pickle, either a GUI export (``*_scan.pkl``) or an old
      notebook's ``results_*.pkl`` (converted by :func:`upgrade_legacy_scan`);
    - a burn pickle written by :func:`save_burn` (``*_burn.pkl``);
    - a ``(2, n)`` ``.npy`` of frequency and absorption - an older GUI
      ``*_spectrum.npy`` export or a notebook ``{tag}.npy``. These carry no
      pump pattern, so ``pump_freqs`` is empty; ``params`` comes from a
      sibling ``*_config.json`` when one exists.
    """
    path = os.fspath(path)
    if os.path.isdir(path):
        if not os.path.exists(os.path.join(path, "manifest.json")):
            raise ValueError("not a simulation folder (no manifest.json "
                             "in it)")
        return "scan", load_run(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(f"expected a (2, n) frequency/absorption array, "
                             f"got shape {arr.shape}")
        params, session = _sibling_config(path)
        return "burn", {"f": arr[0], "spec": arr[1],
                        "pump_freqs": np.array([], dtype=float),
                        "params": params, "session": session}

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError("not a saved simulation result")
    if data.get("kind") == "burn":
        return "burn", data
    if "results" in data and "params" in data:
        res = data["results"]
        if len(res) and not isinstance(res[0], dict):
            return "scan", upgrade_legacy_scan(data)
        if "best" in data:
            # sim_dir is whatever was live when this was exported - drop it
            # if that folder is gone, so nothing points at a missing path.
            if data.get("sim_dir") and not os.path.isdir(data["sim_dir"]):
                data["sim_dir"] = None
            return "scan", data
    raise ValueError("not a saved simulation result - expected a scan "
                     "('results'/'best') or a burn pickle")


# ===========================================================================
# Plotting - matplotlib, drawing onto a supplied Axes
# ===========================================================================

def plot_spectrum(ax, f, spec, params, *, title=None, lw=0.8,
                  xlim=None, ylim=None):
    """
    The simulated absorption spectrum: f (MHz) against normalised absorption.

    Styled to match the write-up figure in SHsimulation_v1_3test2multi.ipynb
    cell 18 - black line, dashed grey "unburned" reference.
    """
    p = settings(params)
    ax.axhline(p["alpha"], color="0.5", lw=0.8, ls="--", label="unburned")
    ax.plot(f, spec, color="black", lw=lw, label="spectrum")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Absorption (Normalised)")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    return ax


def plot_comb(ax, pump_freqs, params, *, title=None, show_xlabel=True):
    """
    The "PSD of burning pulse sequence" panel.

    The pump sequence in this model is just a set of frequencies - the comb
    teeth, each several df bins wide - so this draws that set directly as a
    filled step trace over the window (height 1 where a pump frequency sits,
    0 elsewhere). A true time-domain FFT power spectrum would need an assumed
    pulse duration, which the model does not carry, so this faithful
    representation of what is actually burned is used instead.

    ``show_xlabel=False`` drops the x-axis label, for when this sits on top of
    the spectrum panel sharing its Frequency axis.
    """
    p = settings(params)
    df = p["df"]
    f0 = int(to_steps(p["fmin"], df))
    f1 = int(to_steps(p["fmax"], df))
    axis = df * np.arange(f0, f1 + 1)

    pf = np.asarray(pump_freqs, dtype=float)
    mask = _pump_mask(pf, axis, p["fmin"], df)

    ax.fill_between(axis, mask, step="mid", color="C0", alpha=0.8, lw=0)
    if show_xlabel:
        ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Pump on")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    if title is None:
        title = f"Burning pulse sequence: {pf.size} pump frequencies"
    ax.set_title(title)
    return ax


def _pump_mask(pump_freqs, axis, fmin, df):
    """Height-1-where-pumped step trace for a set of pump frequencies."""
    mask = np.zeros(len(axis))
    pf = np.asarray(pump_freqs, dtype=float)
    if pf.size:
        idx = np.round((pf - fmin) / df).astype(int)
        idx = idx[(idx >= 0) & (idx < len(mask))]
        mask[idx] = 1.0
    return mask


def plot_sequence(ax, stages, params, *, title=None, show_xlabel=True):
    """
    The staged version of the :func:`plot_comb` panel, for a preparation
    sequence.

    One band per stage, stacked bottom-to-top in burn order and labelled on the
    y axis, so both *what* each stage burns and *when it burns it* are readable
    at a glance - a pit and the burn-back pulses sitting outside it would be
    indistinguishable if they were drawn on top of each other.

    ``show_xlabel=False`` drops the x-axis label, for when this sits on top of
    the spectrum panel sharing its Frequency axis.
    """
    p = settings(params)
    df = p["df"]
    f0 = int(to_steps(p["fmin"], df))
    f1 = int(to_steps(p["fmax"], df))
    axis = df * np.arange(f0, f1 + 1)

    n_total = 0
    ticks, labels = [], []
    for i, st in enumerate(stages):
        freqs = np.asarray(st["freqs"], dtype=float)
        n_total += freqs.size
        mask = _pump_mask(freqs, axis, p["fmin"], df)
        ax.fill_between(axis, i + 0.8 * mask, i, step="mid",
                        color=f"C{i % 10}", alpha=0.8, lw=0)
        ticks.append(i + 0.4)
        labels.append(str(st.get("name", f"stage {i + 1}")))

    if show_xlabel:
        ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Burn order")
    ax.set_ylim(-0.1, max(1, len(stages)) + 0.1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize="small")
    if title is None:
        title = (f"Preparation sequence: {len(stages)} stages, "
                 f"{n_total} pump frequencies")
    ax.set_title(title)
    return ax

def plot_populations(ax, out, *, title="Ion classes addressed by the burn",
                     xlim=None, ylim=None):
    """Ground-state populations against atomic detuning, after the burn."""
    for k in range(out["rho"].shape[0]):
        ax.plot(out["det"], out["rho"][k], lw=1.0, label=f"ground state {k}")
    ax.set_xlabel("Atomic detuning (MHz)")
    ax.set_ylabel("Population")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_level_diagram(ax, g, e, dm=None, *, title=None):
    """
    The hyperfine level structure: ground manifold below, excited above, in
    the style of the usual hole-burning level diagrams (e.g. Yang et al.,
    PRA 107, 022802, Fig. 6).

    The two manifolds are drawn on **separate scales** - each is stretched to
    fill its own band - because their spans differ by an order of magnitude
    (Pr:CaWO4: ~150 MHz ground, ~11 MHz excited) and the optical gap between
    them by another nine. Spacings are therefore to scale *within* a
    manifold only; the labelled arrows give the real gaps in MHz.

    Levels are labelled ``g0``, ``g1``, ... / ``e0``, ... by their index in
    ``g`` / ``e``, i.e. the row/column of ``dm``. When ``dm`` is given, each
    ``g_k -> e_m`` transition is a vertical line whose weight and opacity
    scale with ``dm[k, m]`` relative to the strongest; zero entries are not
    drawn, and a uniform ``dm`` gives equal lines.
    """
    g = np.asarray(g, dtype=float)
    e = np.asarray(e, dtype=float)

    def band(vals, base):
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return np.full(vals.shape, base + 0.5)
        return base + 0.1 + 0.8 * (vals - lo) / (hi - lo)

    def num(v):
        return f"{round(float(v), 6):g}"

    y_g, y_e = band(g, 0.0), band(e, 2.0)
    x0, x1 = 0.22, 0.80

    if dm is not None:
        dm = np.asarray(dm, dtype=float)
        top = dm.max() if dm.size and dm.max() > 0 else 1.0
        n = len(g) * len(e)
        xs = np.linspace(x0 + 0.04, x1 - 0.04, n) if n > 1 else [0.5 * (x0 + x1)]
        for i, (k, m) in enumerate((k, m) for k in range(len(g))
                                   for m in range(len(e))):
            w = dm[k, m] / top
            if w <= 0:
                continue
            ax.plot([xs[i], xs[i]], [y_g[k], y_e[m]], color="0.35",
                    lw=0.4 + 2.6 * w, alpha=0.2 + 0.7 * w,
                    solid_capstyle="butt", zorder=1)

    for vals, ys, name, colour, base in ((g, y_g, "g", "C0", 0.0),
                                         (e, y_e, "e", "C1", 2.0)):
        for i, (v, y) in enumerate(zip(vals, ys)):
            ax.plot([x0, x1], [y, y], color=colour, lw=2.0, zorder=3)
            ax.text(x0 - 0.02, y, f"{name}{i}  {num(v)}", ha="right",
                    va="center", fontsize=8)
        order = np.argsort(ys)
        for a, b in zip(order, order[1:]):
            if ys[b] == ys[a]:
                continue
            ax.annotate("", xy=(x1 + 0.06, ys[b]), xytext=(x1 + 0.06, ys[a]),
                        arrowprops=dict(arrowstyle="<->", lw=0.8,
                                        shrinkA=0, shrinkB=0))
            ax.text(x1 + 0.08, 0.5 * (ys[a] + ys[b]),
                    f"{num(abs(vals[b] - vals[a]))} MHz",
                    ha="left", va="center", fontsize=8)
        if name == "g":
            ax.text(0.5 * (x0 + x1), base + 0.03, "ground", ha="center",
                    va="top", fontsize=9, color=colour)
        else:
            ax.text(0.5 * (x0 + x1), base + 0.97, "excited", ha="center",
                    va="bottom", fontsize=9, color=colour)

    ax.text(0.5 * (x0 + x1), 1.5, "optical transition (not to scale)",
            ha="center", va="center", fontsize=8, color="0.4", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc=ax.get_facecolor(),
                      ec="none"))
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.12, 3.12)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def _metric_label(scan, *, tex=True):
    """
    What a scan's ``quality`` column measures, for axis labels and titles.
    AFC comb quality for any scan :func:`run_scan` produced; the delta OD an
    old notebook scored by for one :func:`upgrade_legacy_scan` converted.
    """
    if scan.get("metric") == "delta_od":
        return r"$\Delta$Absorption" if tex else "ΔAbsorption"
    return "AFC comb quality"


def _scan_grids(scan, value="quality"):
    """
    (storage_vals, finesse_vals, z) from a scan's ``results``.

    storage_vals   1 / tooth_spacing, in us, ascending
    finesse_vals   tooth_spacing / tooth_width (the written finesse), ascending
    z              z[finesse_row, storage_col] = that cell's ``value`` - AFC
                   comb quality by default, or e.g. ``"measured_finesse"``
                   (NaN where unset, or where a row does not carry ``value``)
    """
    res = scan["results"]
    storage_vals = np.array(sorted({r["storage_us"] for r in res}))
    finesse_vals = np.array(sorted({round(r["finesse"], 9) for r in res}))
    z = np.full((len(finesse_vals), len(storage_vals)), np.nan)
    for r in res:
        col = int(np.searchsorted(storage_vals, r["storage_us"]))
        row = int(np.searchsorted(finesse_vals, round(r["finesse"], 9)))
        z[row, col] = r.get(value, np.nan)
    return storage_vals, finesse_vals, z


def _measured_note(r):
    """`` (measured 2.1)`` for a result/best row with a finite measured
    finesse, else ``""`` - for labels that quote the written finesse."""
    fa = r.get("measured_finesse")
    if fa is None or not np.isfinite(fa):
        return ""
    return f" (measured {fa:.1f})"


def _require_measured_finesse(scan):
    """The ``results`` rows that carry a finite measured finesse; raises
    ``ValueError`` (shown in place of the plot) when there are none."""
    rows = [r for r in scan["results"]
            if np.isfinite(r.get("measured_finesse", np.nan))]
    if not rows:
        raise ValueError("No measured finesse recorded for this scan -\n"
                         "it was saved before shb.py recorded one.")
    return rows


def plot_scan_heatmap(ax, scan, *, save=None):
    """AFC comb quality over storage time (1/spacing) and finesse (spacing/width)."""
    storage_vals, finesse_vals, z = _scan_grids(scan)
    mesh = ax.pcolormesh(storage_vals, finesse_vals, z, shading="nearest",
                         vmin=0.0)
    label = _metric_label(scan)
    ax.figure.colorbar(mesh, ax=ax, label=label)
    ax.set_title(f"{label} over the central {scan['target_window']:g} MHz")
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel(r"Finesse  $F = \Delta / \gamma$")
    if save:
        ax.figure.savefig(save)
    return ax


def plot_scan_measured_finesse(ax, scan, *, save=None):
    """
    The finesse each burned comb actually came out with
    (:func:`_measured_finesse`), on the same storage time x *written* finesse
    grid as :func:`plot_scan_heatmap`, so the two read side by side.

    A comb written at ``F = 10`` routinely measures nearer ``F = 2`` once
    power broadening and side-hole refilling spread the teeth; a column that
    stays one colour all the way up the finesse axis is one where the written
    finesse made no real difference to the comb.
    """
    _require_measured_finesse(scan)
    storage_vals, finesse_vals, z = _scan_grids(scan, "measured_finesse")
    mesh = ax.pcolormesh(storage_vals, finesse_vals, z, shading="nearest",
                         vmin=0.0)
    ax.figure.colorbar(mesh, ax=ax, label=r"Measured finesse $F_a$")
    ax.set_title(f"Measured comb finesse over the central "
                 f"{scan['target_window']:g} MHz")
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel(r"Written finesse  $F_w = \Delta / \gamma$")
    if save:
        ax.figure.savefig(save)
    return ax


def plot_finesse_written_vs_measured(ax, scan, *, save=None):
    """
    Measured finesse against written finesse, one point per scan cell,
    coloured by its comb quality, with the ``F_a = F_w`` line a comb that kept
    its written finesse would sit on. How far the cloud falls below that line
    is how much of the scan's finesse axis the burned combs never reached.
    """
    rows = _require_measured_finesse(scan)
    f_w = np.array([r["finesse"] for r in rows])
    f_a = np.array([r["measured_finesse"] for r in rows])
    q = np.array([r["quality"] for r in rows])
    pts = ax.scatter(f_w, f_a, c=q, s=14, vmin=0.0)
    hi = float(max(f_w.max(), f_a.max()))
    ax.plot([0.0, hi], [0.0, hi], color="0.5", lw=0.8, ls="--",
            label="as written")
    ax.figure.colorbar(pts, ax=ax, label=_metric_label(scan))
    ax.set_title("Measured vs written comb finesse")
    ax.set_xlabel(r"Written finesse  $F_w = \Delta / \gamma$")
    ax.set_ylabel(r"Measured finesse  $F_a$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9, loc="upper left")
    if save:
        ax.figure.savefig(save)
    return ax


def _quality_by_storage(scan):
    """(storage_vals, q_min, q_mean, q_max, best_finesse) across all finesse."""
    by_st = {}
    for r in scan["results"]:
        by_st.setdefault(r["storage_us"], []).append((r["quality"],
                                                      r["finesse"]))
    storage_vals = np.array(sorted(by_st))
    q_min = np.array([min(v[0] for v in by_st[s]) for s in storage_vals])
    q_max = np.array([max(v[0] for v in by_st[s]) for s in storage_vals])
    q_mean = np.array([np.mean([v[0] for v in by_st[s]]) for s in storage_vals])
    best_fin = np.array([max(by_st[s])[1] for s in storage_vals])
    return storage_vals, q_min, q_mean, q_max, best_fin


def plot_scan_max(ax, scan, *, save=None):
    """Best / mean / worst AFC comb quality over finesse at each storage time."""
    storage_vals, q_min, q_mean, q_max, best_fin = _quality_by_storage(scan)

    ax.fill_between(storage_vals, q_min, q_max, color="C0", alpha=0.18,
                    label="finesse spread")
    ax.plot(storage_vals, q_mean, color="C0", lw=1.8, label="mean")
    ax.plot(storage_vals, q_max, color="C0", lw=1.0, ls="--", label="best")
    ax.plot(storage_vals, q_min, color="C0", lw=1.0, ls=":", label="worst")
    i = int(np.argmax(q_max))
    best_row = max((r for r in scan["results"]
                    if r["storage_us"] == storage_vals[i]),
                   key=lambda r: r["quality"])
    ax.plot(storage_vals[i], q_max[i], linestyle="none", marker="o",
            markersize=11, markerfacecolor="none", markeredgecolor="C1",
            markeredgewidth=1.8,
            label=f"best: {storage_vals[i]:.3g} $\\mu$s, F {best_fin[i]:.1f}"
                  + _measured_note(best_row))
    label = _metric_label(scan)
    ax.set_title(f"{label} vs storage time (best / mean / worst)")
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9, loc="lower right")
    if save:
        ax.figure.savefig(save)
    return ax


def plot_top_spectra(ax, scan, *, save=None, legend=True):
    """
    The kept top-N spectra overlaid, for visual inspection.

    Returns ``(ax, lines, labels)`` - ``lines`` are the per-spectrum
    ``Line2D`` handles (not the dashed unburned reference) and ``labels`` the
    matching legend strings, so a caller can build its own legend / toggle UI.
    Pass ``legend=False`` to skip the in-axes legend.
    """
    p = settings(scan["params"])
    ax.axhline(p["alpha"], color="0.5", lw=0.8, ls="--")
    lines, labels = [], []
    for r in scan["best"]:
        lab = (f"quality {r['quality']:.3f} | "
               f"t_s {r['storage_us']:.3g} $\\mu$s, F {r['finesse']:.1f}"
               + _measured_note(r))
        (ln,) = ax.plot(r["f"], r["spec"], lw=0.7, label=lab)
        lines.append(ln)
        labels.append(lab)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Absorption (Normalised)")
    ax.set_title(f"Best spectrum at the top {len(scan['best'])} storage times "
                 f"by {_metric_label(scan)} ({scan['tag']})")
    if legend:
        ax.legend(fontsize=7, loc="upper right")
    if save:
        ax.figure.savefig(save)
    return ax, lines, labels


def plot_top_spectrum(ax, scan, rank, *, save=None):
    """
    One kept spectrum on its own - ``scan["best"][rank]`` - with the unburned
    level dashed in for reference. The single-spectrum counterpart of
    :func:`plot_top_spectra`, whose overlay gets unreadable past a handful.
    """
    p = settings(scan["params"])
    r = scan["best"][rank]
    ax.axhline(p["alpha"], color="0.5", lw=0.8, ls="--")
    ax.plot(r["f"], r["spec"], lw=0.7, color="C0")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Absorption (Normalised)")
    ax.set_title(f"#{rank + 1}: t_s {r['storage_us']:.3g} $\\mu$s, "
                 f"F {r['finesse']:.1f}{_measured_note(r)}, "
                 f"{_metric_label(scan)} "
                 f"{r['quality']:.3f} ({scan['tag']})")
    if save:
        ax.figure.savefig(save)
    return ax


def save_top_spectrum_figures(scan, out_dir, *, dpi=200):
    """
    Write one PNG per kept spectrum (:func:`plot_top_spectrum`) into
    ``out_dir``, named by rank, storage time and finesse - e.g.
    ``01_ts2.5us_F8.0.png`` - and return the paths written.

    PNGs already in ``out_dir`` are removed first, so re-collecting at a
    smaller ``top_n`` does not leave stale images from the old ranking behind.
    One Figure is reused and cleared between spectra rather than a fresh one
    each: at a fine mesh the Agg backend's path buffers run to tens of MB per
    figure, and this keeps only one alive at a time.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(out_dir, name))
    best = scan.get("best") or []
    if not best:
        return []
    from matplotlib.figure import Figure
    width = len(str(len(best)))
    paths = []
    fig = Figure(figsize=(8, 4), constrained_layout=True)
    for rank, r in enumerate(best):
        fig.clear()
        plot_top_spectrum(fig.add_subplot(111), scan, rank)
        path = os.path.join(
            out_dir, f"{rank + 1:0{width}d}_ts{r['storage_us']:.4g}us_"
                     f"F{r['finesse']:.3g}.png")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    del fig
    return paths


#: The whole-scan figures, as ``(name, plot function)``: what
#: :func:`write_run_outputs` saves under ``figures/`` (and the GUI's export as
#: ``<prefix>_<name>.png``). Each draws onto a supplied ``Axes``.
SCAN_FIGURES = (("heatmap", plot_scan_heatmap),
                ("measured_finesse", plot_scan_measured_finesse),
                ("finesse_written_vs_measured", plot_finesse_written_vs_measured),
                ("quality_vs_storage", plot_scan_max),
                ("topspectra", plot_top_spectra))


# ===========================================================================
# Plotting - plotly, returning a Figure (kept for the notebooks' HTML exports)
# ===========================================================================

def _go():
    import plotly.graph_objects as go
    return go


def plot_spectrum_plotly(f, spec, params, *, title="Spectrum", colour="#6D2E0F"):
    go = _go()
    p = settings(params)
    fig = go.Figure()
    fig.add_hline(y=p["alpha"], line_dash="dash", line_color="#94A3B8",
                  line_width=1, annotation_text="unburned",
                  annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=f, y=spec, mode="lines",
                             line=dict(width=0.9, color=colour), name="spectrum",
                             hovertemplate="%{x:.3f} MHz<br>Abs %{y:.2f}"
                                           "<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Frequency (MHz)",
                      yaxis_title="Absorption (normalised)",
                      template="plotly_white")
    return fig


def plot_scan_heatmap_plotly(scan):
    go = _go()
    storage_vals, finesse_vals, z = _scan_grids(scan)
    fig = go.Figure(data=go.Heatmap(
        x=storage_vals, y=finesse_vals, z=z, zmin=0.0,
        colorscale="Inferno",
        colorbar=dict(title=_metric_label(scan, tex=False)),
        hovertemplate="storage %{x:.3g} us<br>finesse %{y:.1f}"
                      "<br>quality %{z:.3f}<extra></extra>"))
    fig.update_layout(
        title=f"{_metric_label(scan, tex=False)} over the central "
              f"{scan['target_window']:g} MHz  "
              f"({scan['tag']})",
        xaxis_title="Storage time (us)",
        yaxis_title="Finesse (tooth spacing / tooth width)",
        template="plotly_white")
    return fig


def plot_scan_max_plotly(scan):
    go = _go()
    storage_vals, q_min, q_mean, q_max, best_fin = _quality_by_storage(scan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([storage_vals, storage_vals[::-1]]),
        y=np.concatenate([q_max, q_min[::-1]]),
        fill="toself", fillcolor="rgba(192,86,33,0.15)",
        line=dict(width=0), name="finesse spread", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=storage_vals, y=q_mean, mode="lines+markers",
        line=dict(width=1.8, color="#C05621"), marker=dict(size=4),
        name="mean", customdata=best_fin,
        hovertemplate="storage %{x:.3g} us<br>mean quality %{y:.3f}"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=storage_vals, y=q_max, mode="lines", line=dict(width=1, dash="dash",
        color="#C05621"), name="best",
        hovertemplate="storage %{x:.3g} us<br>best quality %{y:.3f}"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=storage_vals, y=q_min, mode="lines", line=dict(width=1, dash="dot",
        color="#C05621"), name="worst",
        hovertemplate="storage %{x:.3g} us<br>worst quality %{y:.3f}"
                      "<extra></extra>"))
    i = int(np.argmax(q_max))
    fig.add_trace(go.Scatter(
        x=[storage_vals[i]], y=[q_max[i]], mode="markers",
        marker=dict(size=11, color="rgba(0,0,0,0)",
                    line=dict(color="#2C7A7B", width=2)),
        name=f"best: {storage_vals[i]:.3g} us", hoverinfo="skip"))
    fig.update_layout(
        title=f"{_metric_label(scan, tex=False)} vs storage time - "
              f"best / mean / worst ({scan['tag']})",
        xaxis_title="Storage time (us)",
        yaxis_title=_metric_label(scan, tex=False),
        yaxis=dict(rangemode="tozero"), template="plotly_white")
    return fig


def plot_top_spectra_plotly(scan):
    go = _go()
    p = settings(scan["params"])
    fig = go.Figure()
    fig.add_hline(y=p["alpha"], line_dash="dash", line_color="#94A3B8",
                  line_width=1)
    for r in scan["best"]:
        fig.add_trace(go.Scatter(
            x=r["f"], y=r["spec"], mode="lines", line=dict(width=0.7),
            name=f"quality {r['quality']:.3f} | t_s {r['storage_us']:.3g} us, "
                 f"F {r['finesse']:.1f}"))
    fig.update_layout(
        title=f"Best spectrum at the top {len(scan['best'])} storage times by "
              f"{_metric_label(scan, tex=False)} ({scan['tag']})",
        xaxis_title="Frequency (MHz)", yaxis_title="Absorption (normalised)",
        template="plotly_white")
    return fig


# ===========================================================================
# CLI
# ===========================================================================

def _smoke():
    """The original CLI check: burn one comb and print a short summary."""
    params = _default_params()
    validate_params(params)

    params["output_params"]["norm"] = normalisation_factor(params)
    print(f"normalisation factor: {params['output_params']['norm']:.6f}")

    tooth_spacing, burn_width, pump_passes = 2.35, 1.175, 10
    pump_freqs, actual_spacing, actual_width, n_teeth = comb_at(
        tooth_spacing, burn_width, params)

    out = run_burn(pump_freqs, params, burn=burn_parallel,
                   pump_passes=pump_passes)
    f, spec = out["f"], out["spec"]

    print(f"{n_teeth} teeth, {len(pump_freqs)} pump frequencies, "
          f"{pump_passes} pass parallel")
    print(f"spacing {actual_spacing:g} MHz, width {actual_width:g} MHz "
          f"({actual_width / actual_spacing:.0%} of the tooth filled)")
    print(f"Absorption {spec.min():.2f} to {spec.max():.2f}  "
          f"(dAbsorption {spec.max() - spec.min():.2f}, "
          f"unburned {settings(params)['alpha']:.2f})")

    # And a preparation sequence, on a coarser mesh so this stays a smoke test:
    # a GEM pit, its burn-back anti-hole and the cleaning sweeps, burned in
    # order. There is no test suite, so this is the check that the staged path
    # still runs end to end.
    seq_params = _default_params()
    seq_params["numerical_params"]["df"] = 0.01
    validate_params(seq_params)
    # Its own norm: the unburned peak depends on df, so reusing the fine-mesh
    # one above would scale this spectrum wrongly.
    seq_params["output_params"]["norm"] = normalisation_factor(seq_params)

    rows = prep_gem(seq_params, peak_width=1.0, pit_passes=4,
                    burn_back_passes=4, clean=True, clean_passes=4)
    stages = sequence_stages(rows, seq_params)
    seq = run_sequence(stages, seq_params, burn=burn_parallel)
    centre = int(np.argmin(np.abs(seq["f"])))
    print(f"GEM sequence: {len(stages)} stages "
          f"({', '.join(st['name'] for st in stages)}), "
          f"{seq['pump_freqs'].size} pump frequencies")
    print(f"Absorption {seq['spec'].min():.2f} to {seq['spec'].max():.2f}  "
          f"(at the pit centre {seq['spec'][centre]:.2f})")


def _cli_load_config(path):
    """
    A params dict from either a plain ``params_to_json`` file or a GUI
    ``..._config.json`` session export (``{"params": ..., "gui": ...}``).
    """
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "params" in data and "crystal_params" not in data:
        data = data["params"]
    return params_from_json(json.dumps(data))


def _storage_sep_arg(text):
    """``--min-storage-sep``: ``auto`` or a non-negative gap in us."""
    if text.strip().lower() == "auto":
        return "auto"
    import argparse
    try:
        return _resolve_storage_sep(float(text), 0.0, 0.0, 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected 'auto' or a gap >= 0 in us, not {text!r}") from exc


def _cli_scan(args):
    # Imported by name rather than used as the running __main__ module, so
    # that everything the Pool pickles (compute_one, _init_worker, ...)
    # carries the "shb" qualname a spawned child can always resolve, the same
    # as when this file is used as a library rather than run directly.
    import shb as _shb

    if args.config:
        params = _cli_load_config(args.config)
    elif args.preset:
        params = copy.deepcopy(_shb.PRESETS[args.preset])
    else:
        params = copy.deepcopy(_shb.DEFAULT_PARAMS)

    pump_passes = (args.pump_passes if args.pump_passes is not None
                  else _shb.settings(params)["pump_passes"])
    burn = _shb.burn_series if args.burn == "series" else _shb.burn_parallel

    scan = _shb.run_scan(
        args.storage_min, args.storage_max, args.n_points,
        args.finesse_min, args.finesse_max, args.n_finesse,
        args.target_window, args.top_n, pump_passes,
        params=params, burn=burn, parallel=not args.serial,
        n_workers=args.workers, sim_dir=args.sim_dir, resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        min_storage_sep=args.min_storage_sep,
        on_done=_shb.scan_teams_notifier())
    print(f"[{scan['tag']}] {scan['state']}: {scan['n_done']}/{scan['n_total']} "
         f"burns, {len(scan['results'])} results, {len(scan['best'])} kept "
         f"-> {args.sim_dir}")


def _cli_collect(args):
    import shb as _shb
    best = _shb.collect_top(args.sim_dir, top_n=args.top_n,
                            min_storage_sep=args.min_storage_sep)
    scan = _shb.load_run(args.sim_dir)
    _shb.write_run_outputs(args.sim_dir, scan)
    print(f"collected {len(best)} spectra, wrote results.csv / best.json / "
         f"top_spectra.npy / figures/ to {args.sim_dir}")


def _cli_status(args):
    import shb as _shb
    info = _shb.inspect_run(args.sim_dir)
    live = f" (live, pid {info['pid']})" if info["live"] else ""
    print(f"{args.sim_dir}: {info['state']}{live}  "
         f"{info['n_done']}/{info['n_total']} burns done")


def _main(argv=None):
    """
    Command line entry point.

    With no subcommand this runs the original single-burn smoke test. The
    subcommands exist so a long sweep can run outside the GUI, checkpointing
    to a simulation folder, and be resumed after a crash::

        python shb.py scan --sim-dir runs/afc1 --storage-min 2 --storage-max 10 \\
            --n-points 40 --finesse-min 2 --finesse-max 20 --n-finesse 20
        python shb.py scan --sim-dir runs/afc1          # resumes by default
        python shb.py collect --sim-dir runs/afc1
        python shb.py status  --sim-dir runs/afc1
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("scan", "collect", "status"):
        return _smoke()

    import argparse
    parser = argparse.ArgumentParser(prog="shb.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="run (or resume) a storage/finesse sweep")
    sc.add_argument("--sim-dir", required=True)
    sc.add_argument("--storage-min", type=float, required=True)
    sc.add_argument("--storage-max", type=float, required=True)
    sc.add_argument("--n-points", type=int, required=True)
    sc.add_argument("--finesse-min", type=float, required=True)
    sc.add_argument("--finesse-max", type=float, required=True)
    sc.add_argument("--n-finesse", type=int, required=True)
    sc.add_argument("--target-window", type=float, default=None)
    sc.add_argument("--top-n", type=int, default=50,
                    help="keep the best-finesse spectrum at this many of the "
                    "best storage times (default 50)")
    sc.add_argument("--min-storage-sep", type=_storage_sep_arg, default="auto",
                    help="minimum gap (us) between kept storage times, or "
                    "'auto' (default: spread across the range); 0 = off")
    sc.add_argument("--pump-passes", type=int, default=None,
                    help="default: whatever the params/preset carries")
    sc.add_argument("--burn", choices=("parallel", "series"), default="parallel")
    sc.add_argument("--serial", action="store_true",
                    help="disable multiprocessing")
    sc.add_argument("--workers", type=int, default=None)
    sc.add_argument("--checkpoint-every", type=int, default=10)
    sc.add_argument("--resume", dest="resume", action="store_true",
                    default=True, help="resume sim-dir if it already exists "
                    "(default)")
    sc.add_argument("--no-resume", dest="resume", action="store_false")
    sc.add_argument("--preset", default=None, choices=sorted(PRESETS))
    sc.add_argument("--config", default=None,
                    help="a params or GUI config.json file; overrides --preset")

    co = sub.add_parser("collect",
                        help="(re-)collect the top spectra from a run's "
                        "checkpoint, without re-sweeping")
    co.add_argument("--sim-dir", required=True)
    co.add_argument("--top-n", type=int, default=None,
                    help="how many storage times to collect the best spectrum "
                    "for (default: the run's own setting)")
    co.add_argument("--min-storage-sep", type=_storage_sep_arg, default=None,
                    help="minimum gap (us) between kept storage times, 'auto', "
                    "or 0 = off (default: the run's own setting)")

    st = sub.add_parser("status", help="how far a simulation folder has got")
    st.add_argument("--sim-dir", required=True)

    ns = parser.parse_args(argv)
    {"scan": _cli_scan, "collect": _cli_collect,
     "status": _cli_status}[ns.cmd](ns)


if __name__ == "__main__":
    _main()
