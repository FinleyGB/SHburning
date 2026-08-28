import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


import numpy as np


def settings(params):
    """All the parameters as one flat dict, which is how the functions read them."""
    flat = {}
    for section in params.values():
        flat.update(section)
    return flat


def to_steps(values, df):
    """
    MHz to whole df steps. df must divide every frequency in the simulation
    exactly
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
    df = p['df']
    g = np.asarray(p['g'], dtype=float)
    e = np.asarray(p['e'], dtype=float)

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
    df = p['df']
    fmin = p['fmin'] if fmin is None else fmin
    fmax = p['fmax'] if fmax is None else fmax

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
    pump_passes = p['pump_passes'] if pump_passes is None else pump_passes
    df, dm = p['df'], p['dm']
    n_g, n_e = trans_steps.shape
    rho = rho.copy()

    # The pump frequencies as whole bins. This is where an off-mesh comb is
    # caught, before any of it is burned.
    pump_steps = to_steps(pump_freqs, df)

    for _ in range(pump_passes):

        # hits[k, d]        how many addressed transitions take ions out of
        #                   ground state k in ion class d
        # landing[j, k, d]  the branchings of those same transitions, summed -
        #                   so landing/hits is where the ions actually go
        # I still need to check this so that it is 100% correct
        hits = np.zeros_like(rho)
        landing = np.zeros((n_g, n_g, len(det)))

        for pump_step in pump_steps:
            for k in range(n_g):
                for m in range(n_e):

                    # The ion class whose (k, m) transition sits exactly on
                    # this pump frequency. Integer subtraction - the bin is
                    # computed, not rounded to.
                    d = int(pump_step - trans_steps[k, m] - det0)
                    if not (0 <= d < len(det)):
                        continue

                    hits[k, d] += 1
                    landing[:, k, d] += dm[:, m]

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
    """
    p = settings(params)
    pump_passes = p['pump_passes'] if pump_passes is None else pump_passes
    df, dm = p['df'], p['dm']
    n_g, n_e = trans_steps.shape
    rho = rho.copy()

    pump_steps = to_steps(pump_freqs, df)

    for _ in range(pump_passes):
        for pump_step in pump_steps:
            for k in range(n_g):
                for m in range(n_e):

                    d = int(pump_step - trans_steps[k, m] - det0)
                    if not (0 <= d < len(det)):
                        continue

                    removed = rho[k, d]
                    rho[k, d] = 0.0
                    rho[:, d] += removed * dm[:, m]

    return rho

def spectrum(rho, trans, det, f, params, norm=None, alpha=None):
    """
    Absorption spectrum from the populations.

    norm and alpha default to config. the spectrum is divided by norm and
    multiplied by alpha, which puts the unburned crystal at alpha.
    """
    p = settings(params)
    norm = (p['norm'] if p['norm'] is not None else 1.0) if norm is None else norm
    alpha = p['alpha'] if alpha is None else alpha
    h_width, in_width, dm = p['h_width'], p['in_width'], p['dm']

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


def run_burn(pump_freqs, params, burn=burn_parallel,
             pump_passes=None, fmin=None, fmax=None):
    """
    The whole thing: set up, burn, take the spectrum.

    Pass burn=burn_series for a series burn instead of a parallel one.

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
    spec = spectrum(rho, trans, det, f, params)

    return {'f': f, 'spec': spec, 'rho': rho, 'rho0': rho0,
            'det': det, 'trans': trans, 'trans_steps': trans_steps,
            'pump_freqs': np.asarray(pump_freqs)}


def normalisation_factor(params):
    """
    Peak absorption of the unburned crystal, measured over a 100 MHz window at
    the centre of the inhomogeneous line.

    Dividing a spectrum by this and multiplying by alpha puts the unburned
    crystal at alpha, with holes below it and antiholes above.

    It has to be computed rather than written down
    """
    g, e, trans, trans_steps = build_transitions(params)
    f, det, det0 = make_axes(params, trans_steps, fmin=-50.0, fmax=50.0)
    rho = initial_populations(g, det)
    spec = spectrum(rho, trans, det, f, params, norm=1.0, alpha=1.0)
    return float(spec.max())






def comb(spacing_steps, width_steps, params):
    """
    Teeth every spacing_steps bins across the spectrum window, each tooth
    width_steps bins wide.

    """
    p = settings(params)
    df = p['df']
    f0 = int(to_steps(p['fmin'], df))
    f1 = int(to_steps(p['fmax'], df))

    n_teeth = (f1 - f0) // spacing_steps + 1
    starts = f0 + spacing_steps * np.arange(n_teeth)
    return df * (starts[:, None] + np.arange(width_steps)[None, :]).ravel()




def compute_one(item, params=None, burn=None, pump_passes=None,
                 df=None, target_window=None):
    """
    everything removed from the loop from run_scan so we can calculate in parallel
    """
    n, (spacing_steps, width_steps) = item

    out = run_burn(comb(spacing_steps, width_steps, params), params,
                    burn=burn, pump_passes=pump_passes)
    f, spec = out['f'], out['spec']

    centre = f[len(f) // 2]
    lo = np.searchsorted(f, centre - target_window / 2)
    hi = np.searchsorted(f, centre + target_window / 2)
    delta_od = spec[lo:hi].max() - spec[lo:hi].min()

    tooth_spacing = spacing_steps * df
    result_row = (tooth_spacing, width_steps * df, delta_od)
    record = (delta_od, n, spec, tooth_spacing, width_steps, f)

    return result_row, record
