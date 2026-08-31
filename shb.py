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
    Combs       comb, comb_at, single_hole
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
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

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
    "normalisation_factor",
    # combs
    "comb", "comb_at", "single_hole",
    # scan
    "compute_one", "run_scan",
    # plotting (matplotlib, draw onto an Axes)
    "plot_spectrum", "plot_comb", "plot_populations",
    "plot_scan_heatmap", "plot_scan_max", "plot_top_spectra",
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
    "axes.prop_cycle": __import__("cycler").cycler(
        "color", ["#1f77b4", "#e05c2a", "#2ca02c", "#9467bd",
                  "#8c564b", "#d62728", "#17becf", "#bcbd22"]),
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
        mpl.rcParams.update(_STYLE_FALLBACK)


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

    for _ in range(pump_passes):

        # hits[k, d]        how many addressed transitions take ions out of
        #                   ground state k in ion class d
        # landing[j, k, d]  the branchings of those same transitions, summed -
        #                   so landing/hits is where the ions actually go
        # NOTE: carried over from the notebook - the author flagged this block
        # for a final correctness check. Behaviour is left exactly as it was;
        # only the per-pump-frequency Python loop is replaced by a bincount, so
        # a fine comb (tens of thousands of pump frequencies) is no longer slow.
        hits = np.zeros_like(rho)
        landing = np.zeros((n_g, n_g, n_det))

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
                landing[:, k, :] += dm[:, m][:, None] * counts[None, :]

        # The ions leave. A class addressed three times over still only has
        # one population to give, so this is a mask, not a count.
        excited = rho * (hits > 0)

        # And come back down, sharing equally between the branchings of
        # whichever transitions took them.
        share = np.divide(landing, hits,
                          out=np.zeros_like(landing), where=hits > 0)
        rho = rho - excited + (share * excited).sum(axis=1)

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
    ``width_steps`` bins wide. Arguments are in whole df steps.
    """
    p = settings(params)
    df = p["df"]
    f0 = int(to_steps(p["fmin"], df))
    f1 = int(to_steps(p["fmax"], df))

    n_teeth = (f1 - f0) // spacing_steps + 1
    starts = f0 + spacing_steps * np.arange(n_teeth)
    return df * (starts[:, None] + np.arange(width_steps)[None, :]).ravel()


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


# ===========================================================================
# Scan
# ===========================================================================

def _score_contrast(f, spec, target_window, a_ref):
    """
    AFC contrast over ``target_window`` MHz at the centre of ``f``.

    ``C = (A_max - A_min) / (2 * A_unburned)`` over the window - the Michelson
    visibility of the comb with the background pinned to the unburned absorption
    level (``a_ref``) instead of estimated per point as ``(A_max + A_min) / 2``.
    Because ``a_ref`` is fixed for the whole sweep, C is a strict rescaling of
    the raw delta absorption ``A_max - A_min`` and ranks the scan identically.
    """
    if a_ref <= 0.0:
        return 0.0
    centre = f[len(f) // 2]
    lo = np.searchsorted(f, centre - target_window / 2)
    hi = np.searchsorted(f, centre + target_window / 2)
    seg = spec[lo:hi]
    if seg.size == 0:
        return 0.0
    return float(seg.max() - seg.min()) / (2.0 * a_ref)


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

    # Unburned absorption over the judging window - the fixed reference the AFC
    # contrast is normalised by, computed once so every point uses the same one.
    spec0 = spectrum(rho0, trans, det, f, params, ctx=sctx)
    centre = f[len(f) // 2]
    lo = np.searchsorted(f, centre - target_window / 2)
    hi = np.searchsorted(f, centre + target_window / 2)
    a_ref = float(spec0[lo:hi].max()) if hi > lo else float(spec0.max())

    return {
        "params": params, "burn": burn, "pump_passes": pump_passes,
        "target_window": target_window, "df": settings(params)["df"],
        "trans": trans, "trans_steps": trans_steps,
        "f": f, "det": det, "det0": det0,
        "rho0": rho0, "sctx": sctx, "a_ref": a_ref,
    }


def compute_one(item, params=None, burn=None, pump_passes=None,
                df=None, target_window=None, ctx=None):
    """
    One scan point, pulled out of the loop so it can run in a worker process.

    ``item`` is ``(n, (spacing_steps, width_steps))``. Returns
    ``(result_row, record)`` where
        result_row = (spacing_MHz, width_MHz, contrast)
        record     = (contrast, n, spec, spacing_MHz, width_MHz, f)
    ``contrast`` is the AFC contrast (:func:`_score_contrast`).

    Pass ``ctx`` (from :func:`_scan_context`) for the fast path: the transition
    table, axes, initial populations and spectrum FFT setup are reused instead of
    rebuilt every call. Without ``ctx`` it falls back to a plain ``run_burn``.
    """
    n, (spacing_steps, width_steps) = item

    if ctx is not None:
        params, burn, df = ctx["params"], ctx["burn"], ctx["df"]
        pump_passes, target_window = ctx["pump_passes"], ctx["target_window"]
        a_ref = ctx["a_ref"]
        freqs = comb(spacing_steps, width_steps, params)
        rho = burn(ctx["rho0"], freqs, ctx["trans_steps"], ctx["det"],
                   ctx["det0"], params, pump_passes)
        f = ctx["f"]
        spec = spectrum(rho, ctx["trans"], ctx["det"], f, params,
                        ctx=ctx["sctx"])
    else:
        a_ref = settings(params)["alpha"]
        out = run_burn(comb(spacing_steps, width_steps, params), params,
                       burn=burn, pump_passes=pump_passes)
        f, spec = out["f"], out["spec"]

    contrast = _score_contrast(f, spec, target_window, a_ref)

    spacing_mhz = spacing_steps * df
    width_mhz = width_steps * df
    result_row = (spacing_mhz, width_mhz, contrast)
    record = (contrast, n, spec, spacing_mhz, width_mhz, f)
    return result_row, record


def run_scan(storage_min, storage_max, n_points,
             finesse_min, finesse_max, n_finesse,
             target_window, top_n, pump_passes,
             params=None, burn=burn_parallel,
             parallel=True, n_workers=None,
             progress=None, log=None, estimate=None):
    """
    Sweep AFC storage time and comb finesse, scoring each spectrum by its AFC
    contrast (A_max - A_min) / (2 * A_unburned) (:func:`_score_contrast`) over a
    window at the centre - a rescaling of the raw delta absorption, so the best
    point matches the delta-absorption maximum.

    The spectrum window and mesh come from ``params`` - fmin, fmax and df are
    not swept here.

    storage_min/max   range of storage times to try (us). The tooth spacing is
                      ``1 / storage_time`` (MHz), so this range corresponds to
                      spacings ``1/storage_max .. 1/storage_min``.
    n_points          number of storage times across that range, spaced linearly
                      in us; each is converted to a spacing and rounded to the df
                      mesh (so any n_points works - there is no step that has to
                      divide df)
    finesse_min/max   range of comb finesse F = tooth_spacing / tooth_width
    n_finesse         number of finesse values across that range
    target_window     judge each spectrum over this width at the centre (MHz).
                      ``None`` -> 4 / storage_min, so the window always holds
                      several teeth however wide the spacing gets.
    top_n             how many of the best spectra to keep for plotting
    parallel          run the sweep across a multiprocessing.Pool (default True)
    n_workers         processes to use when ``parallel`` (default os.cpu_count())
    progress          optional callback ``progress(done, total)``; when ``None``
                      a tqdm bar is used if stdout is a TTY
    log               optional callback ``log(str)`` for the status lines;
                      defaults to ``print``
    estimate          optional callback ``estimate(seconds)`` called once with
                      the projected total run time, right after the timing
                      probe and before the sweep starts

    Returns a dict with
        tag            short label for the run
        results        one dict per requested (storage, finesse) cell:
                       spacing_MHz, width_MHz, finesse, storage_us, contrast
        best           top_n dicts by contrast, each also carrying spec, f, n
        params, finesse_vals, target_window, pump_passes, metric
    """
    params = DEFAULT_PARAMS if params is None else params
    log = print if log is None else log

    p = settings(params)
    df_user = p["df"]
    tag = f"{storage_min:g}-{storage_max:g}us"

    if target_window is None:
        target_window = 4.0 / storage_min

    # The scan grid is linear in storage time and in finesse. Each (t, F) cell
    # keeps its requested t and F (so the heatmap axes are evenly spaced); the
    # burn behind it uses the tooth spacing 1/t rounded to the mesh.
    t_targets = np.linspace(storage_min, storage_max, max(2, int(n_points)))
    finesse_vals = np.linspace(finesse_min, finesse_max, max(1, int(n_finesse)))

    # Tooth spacing is 1/t, quantised to the df mesh, so two close storage times
    # can round to the same comb. Refine df by a whole factor (so it still
    # divides the crystal levels and the window edges) until consecutive
    # spacings are at least one step apart - bounded below by what is physically
    # meaningful (~h_width/5) and by a cap on the spectrum array size.
    df = df_user
    spacings = 1.0 / t_targets
    min_gap = (float(np.min(np.abs(np.diff(spacings)))) if len(spacings) > 1
               else df_user)
    if min_gap < df_user:
        floor = min(df_user, max(1e-4, 0.2 * float(p["h_width"])))
        floor = max(floor,
                    (float(p["fmax"]) - float(p["fmin"])) / 2_000_000)
        want = min(500, int(np.ceil(df_user / max(min_gap, floor))))
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
            df = cand
            break

    scan_params = params
    if df != df_user:
        scan_params = copy.deepcopy(params)
        scan_params["numerical_params"]["df"] = df
        p = settings(scan_params)

    requested = []          # (t_us, F, s_steps, w_steps)
    todo = {}               # (s_steps, w_steps) -> index
    distinct_spacings = set()
    for t in t_targets:
        s = int(max(1, round(1.0 / t / df)))
        distinct_spacings.add(s)
        for F in finesse_vals:
            w = int(min(s, max(1, round(s / F))))
            requested.append((float(t), float(F), s, w))
            todo.setdefault((s, w), len(todo))
    uniq = list(todo)       # unique (s_steps, w_steps), in first-seen order

    log(f"[{tag}]  {len(t_targets)} storage times x {len(finesse_vals)} "
        f"finesse = {len(requested)} cells ({len(uniq)} unique burns)   window "
        f"{p['fmin']:g} to {p['fmax']:g} MHz, df {df:g}, judged over "
        f"{target_window:g} MHz")
    if df != df_user:
        log(f"    refined df {df_user:g} -> {df:g} MHz so the {len(t_targets)} "
            f"requested storage times resolve to {len(distinct_spacings)} "
            f"distinct combs (finer mesh, slower burns)")
    if len(distinct_spacings) < len(t_targets):
        log(f"    note: even at df {df:g} MHz only {len(distinct_spacings)} of "
            f"{len(t_targets)} storage times land on distinct combs across "
            f"{storage_min:g}-{storage_max:g} us - the rest differ by less than "
            f"the mesh and repeat a burn")

    # Built once: transition table, axes, initial rho, spectrum FFT setup.
    ctx = _scan_context(scan_params, burn, pump_passes, target_window)

    # A single spectrum, timed, to estimate the whole sweep.
    _t0 = time.time()
    compute_one((0, uniq[len(uniq) // 2]), ctx=ctx)
    _one = time.time() - _t0
    _total = _one * len(uniq)
    if parallel:
        _total /= max(1, n_workers or os.cpu_count() or 1)
    log("    about " + (f"{_total:.0f} seconds" if _total < 90
                        else f"{_total / 60:.0f} minutes"))
    if estimate is not None:
        estimate(_total)

    worker = partial(compute_one, ctx=ctx)
    items = list(enumerate(uniq))

    cache = {}               # (s_steps, w_steps) -> contrast
    best = []                # heap of (contrast, n, record_dict)

    def _keep(contrast, n, spec, f, s_mhz, w_mhz):
        import heapq
        rec = {"contrast": contrast, "n": n, "spec": spec, "f": f,
               "spacing_MHz": s_mhz, "width_MHz": w_mhz,
               "finesse": s_mhz / w_mhz, "storage_us": 1.0 / s_mhz}
        key = (contrast, n)
        if len(best) < top_n:
            heapq.heappush(best, (key, rec))
        elif key > best[0][0]:
            heapq.heappushpop(best, (key, rec))

    def _consume(i, item, result_row, record):
        (s_steps, w_steps) = item[1]
        s_mhz, w_mhz, contrast = result_row
        cache[(s_steps, w_steps)] = contrast
        _keep(contrast, record[1], record[2], record[5], s_mhz, w_mhz)
        if progress is not None:
            progress(i, len(items))

    if parallel:
        from multiprocessing import Pool
        n_workers = n_workers or os.cpu_count() or 1
        chunksize = max(1, len(items) // (n_workers * 4))
        with Pool(processes=n_workers) as pool:
            for i, (item, (row, rec)) in enumerate(
                    zip(items, pool.imap(worker, items, chunksize=chunksize)),
                    start=1):
                _consume(i, item, row, rec)
    else:
        iterator = items
        if progress is None and sys.stdout.isatty():
            try:
                from tqdm import tqdm
                iterator = tqdm(items, desc=f"scan {tag}", unit="spec")
            except ImportError:
                pass
        for i, item in enumerate(iterator, start=1):
            row, rec = worker(item)
            _consume(i, item, row, rec)

    # One result row per requested (storage time, finesse) cell, from the cache.
    # storage_us and finesse are the requested grid values (evenly spaced);
    # spacing_MHz / width_MHz are the mesh-snapped values actually burned.
    results = []
    for t, F, s, w in requested:
        results.append({
            "spacing_MHz": s * df,
            "width_MHz": w * df,
            "finesse": F,
            "storage_us": t,
            "contrast": cache[(s, w)],
        })

    best_rows = [rec for _, rec in sorted(best, key=lambda t: t[0],
                                          reverse=True)]
    return {"tag": tag,
            "results": results,
            "best": best_rows,
            "params": scan_params,
            "df_used": df,
            "finesse_vals": [float(v) for v in finesse_vals],
            "target_window": target_window,
            "pump_passes": pump_passes,
            "metric": "afc_contrast"}


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

    mask = np.zeros(len(axis))
    pf = np.asarray(pump_freqs, dtype=float)
    if pf.size:
        idx = np.round((pf - p["fmin"]) / df).astype(int)
        idx = idx[(idx >= 0) & (idx < len(mask))]
        mask[idx] = 1.0

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


def _scan_grids(scan):
    """
    (storage_vals, finesse_vals, z) from a scan's ``results``.

    storage_vals   1 / tooth_spacing, in us, ascending
    finesse_vals   tooth_spacing / tooth_width, ascending
    z              z[finesse_row, storage_col] = AFC contrast (NaN where unset)
    """
    res = scan["results"]
    storage_vals = np.array(sorted({r["storage_us"] for r in res}))
    finesse_vals = np.array(sorted({round(r["finesse"], 9) for r in res}))
    z = np.full((len(finesse_vals), len(storage_vals)), np.nan)
    for r in res:
        col = int(np.searchsorted(storage_vals, r["storage_us"]))
        row = int(np.searchsorted(finesse_vals, round(r["finesse"], 9)))
        z[row, col] = r["contrast"]
    return storage_vals, finesse_vals, z


def plot_scan_heatmap(ax, scan, *, save=None):
    """AFC contrast over storage time (1/spacing) and finesse (spacing/width)."""
    storage_vals, finesse_vals, z = _scan_grids(scan)
    mesh = ax.pcolormesh(storage_vals, finesse_vals, z, shading="nearest",
                         vmin=0.0)
    ax.figure.colorbar(mesh, ax=ax, label="AFC contrast")
    ax.set_title(f"AFC contrast over the central {scan['target_window']:g} MHz")
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel(r"Finesse  $F = \Delta / \gamma$")
    if save:
        ax.figure.savefig(save)
    return ax


def plot_scan_max(ax, scan, *, save=None):
    """The best AFC contrast achievable at each storage time."""
    best_per = {}
    for r in scan["results"]:
        st = r["storage_us"]
        if st not in best_per or r["contrast"] > best_per[st][0]:
            best_per[st] = (r["contrast"], r["finesse"])

    storage_vals = np.array(sorted(best_per))
    max_c = np.array([best_per[s][0] for s in storage_vals])
    best_fin = np.array([best_per[s][1] for s in storage_vals])

    ax.plot(storage_vals, max_c, color="C0", lw=1.8,
            label="best AFC contrast")
    i = int(np.argmax(max_c))
    ax.plot(storage_vals[i], max_c[i], linestyle="none", marker="o",
            markersize=11, markerfacecolor="none", markeredgecolor="C1",
            markeredgewidth=1.8,
            label=f"best: {storage_vals[i]:.3g} $\\mu$s, F {best_fin[i]:.1f}")
    ax.set_title("Best AFC contrast at each storage time")
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel("Maximum AFC contrast")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9, loc="lower right")
    if save:
        ax.figure.savefig(save)
    return ax


def plot_top_spectra(ax, scan, *, save=None):
    """The kept top-N spectra overlaid, for visual inspection."""
    p = settings(scan["params"])
    ax.axhline(p["alpha"], color="0.5", lw=0.8, ls="--")
    for r in scan["best"]:
        ax.plot(r["f"], r["spec"], lw=0.7,
                label=f"contrast {r['contrast']:.3f} | "
                      f"t_s {r['storage_us']:.3g} $\\mu$s, F {r['finesse']:.1f}")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Absorption (Normalised)")
    ax.set_title(f"Top {len(scan['best'])} spectra by AFC contrast "
                 f"({scan['tag']})")
    ax.legend(fontsize=7, loc="upper right")
    if save:
        ax.figure.savefig(save)
    return ax


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
        colorscale="Inferno", colorbar=dict(title="AFC contrast"),
        hovertemplate="storage %{x:.3g} us<br>finesse %{y:.1f}"
                      "<br>contrast %{z:.3f}<extra></extra>"))
    fig.update_layout(
        title=f"AFC contrast over the central {scan['target_window']:g} MHz  "
              f"({scan['tag']})",
        xaxis_title="Storage time (us)",
        yaxis_title="Finesse (tooth spacing / tooth width)",
        template="plotly_white")
    return fig


def plot_scan_max_plotly(scan):
    go = _go()
    best_per = {}
    for r in scan["results"]:
        st = r["storage_us"]
        if st not in best_per or r["contrast"] > best_per[st][0]:
            best_per[st] = (r["contrast"], r["finesse"])
    storage_vals = np.array(sorted(best_per))
    max_c = np.array([best_per[s][0] for s in storage_vals])
    best_fin = np.array([best_per[s][1] for s in storage_vals])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=storage_vals, y=max_c, mode="lines+markers",
        line=dict(width=1.4, color="#C05621"), marker=dict(size=5),
        name="best AFC contrast", customdata=best_fin,
        hovertemplate="storage %{x:.3g} us<br>contrast %{y:.3f}"
                      "<br>finesse %{customdata:.1f}<extra></extra>"))
    i = int(np.argmax(max_c))
    fig.add_trace(go.Scatter(
        x=[storage_vals[i]], y=[max_c[i]], mode="markers",
        marker=dict(size=11, color="rgba(0,0,0,0)",
                    line=dict(color="#2C7A7B", width=2)),
        name=f"best: {storage_vals[i]:.3g} us", hoverinfo="skip"))
    fig.update_layout(
        title=f"Best AFC contrast at each storage time  ({scan['tag']})",
        xaxis_title="Storage time (us)", yaxis_title="Maximum AFC contrast",
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
            name=f"contrast {r['contrast']:.3f} | t_s {r['storage_us']:.3g} us, "
                 f"F {r['finesse']:.1f}"))
    fig.update_layout(
        title=f"Top {len(scan['best'])} spectra by AFC contrast ({scan['tag']})",
        xaxis_title="Frequency (MHz)", yaxis_title="Absorption (normalised)",
        template="plotly_white")
    return fig


# ===========================================================================
# CLI smoke test
# ===========================================================================

def _main():
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


if __name__ == "__main__":
    _main()
