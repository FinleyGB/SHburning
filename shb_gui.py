"""
PyQt6 GUI for the spectral hole burning / AFC simulator in ``shb.py``.

Layout (matches the project mockup)

    +---------------------+-------------------------------------------+
    | Crystal parameters  |  PSD of burning pulse sequence            |
    |                     +-------------------------------------------+
    | Simulation          |                                           |
    | Parameters          |  Simulated spectral feature               |
    |                     |                                           |
    |                     +-------------------------------------------+
    |                     |  Simulation run log                       |
    +---------------------+-------------------------------------------+

The left column is a two-tab panel: "Simulation" (the controls above) and
"Export" (bundle the run log, config JSON, figures and data to a folder).

Run with::

    python shb_gui.py

All computation happens on a background QThread so the window stays responsive;
the run log streams the same lines the module would otherwise print.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

if sys.platform == "win32":  # noqa: E402
    # A conda/Anaconda install can ship its own Qt6Core.dll (e.g. under
    # <prefix>/Library/bin, alongside PyQt5/PySide6) that is binary-
    # incompatible with the one bundled inside the PyQt6 wheel. If that
    # DLL gets resolved instead, importing PyQt6.QtCore fails with
    # "DLL load failed ... The specified procedure could not be found."
    # Force PyQt6's own Qt6Core.dll to load first, from its own folder
    # (winmode=0 lets it find its sibling DLLs there), before anything
    # else on the system gets a chance to shadow it.
    import PyQt6
    _qt6_bin = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "bin")
    if os.path.isdir(_qt6_bin):
        os.add_dll_directory(_qt6_bin)
        import ctypes
        _qt6_core = os.path.join(_qt6_bin, "Qt6Core.dll")
        if os.path.isfile(_qt6_core):
            ctypes.WinDLL(_qt6_core, winmode=0)

# Import PyQt6 before matplotlib: matplotlib's QtAgg backend binds to
# whichever Qt binding is already in sys.modules (checking PyQt6, PySide6,
# PyQt5, PySide2 in that order), so if another binding is also installed
# (e.g. a conda env's own PyQt5) and PyQt6 isn't imported first, matplotlib
# can silently pick the wrong major version and crash on mismatched Qt
# instances.
from PyQt6.QtCore import (  # noqa: E402
    QLocale, QObject, Qt, QThread, QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import QAction  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QCheckBox, QPushButton, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QStackedWidget, QProgressBar, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QScrollArea, QSizePolicy, QInputDialog)

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qt import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import shb  # noqa: E402


# Default root for auto-named simulation folders (see MainWindow._auto_sim_dir /
# _pick_sim_dir): the lab's shared Dropbox scan storage. Falls back to the
# current directory if that Dropbox isn't mounted on this machine.
_DROPBOX_SIM_ROOT = (
    r"C:\Users\GEM love\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_"
    r"Photonics_Lab\Experiments\Current Experiments\Visible Quantum Memory"
    r"\SHB_simulations\Pr_YSO"
)
DEFAULT_SIM_ROOT = _DROPBOX_SIM_ROOT if os.path.isdir(_DROPBOX_SIM_ROOT) else None


def fmt_dur(secs):
    """Seconds as a short human string: '8s', '1m20s', '3m', '2h5m'."""
    secs = max(0, int(round(secs)))
    if secs < 60:
        return f"{secs}s"
    mins, s = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m{s}s" if s else f"{mins}m"
    hrs, m = divmod(mins, 60)
    return f"{hrs}h{m}m" if m else f"{hrs}h"


# ===========================================================================
# Background worker
# ===========================================================================

class SimWorker(QObject):
    """Runs norm / burn / scan jobs off the UI thread."""

    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)          # done, total  (0, 0 => indeterminate)
    estimate = pyqtSignal(float)             # projected total run time, seconds
    norm_ready = pyqtSignal(float)
    burn_done = pyqtSignal(dict)
    scan_done = pyqtSignal(dict)
    error = pyqtSignal(str)
    busy = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # -- jobs ---------------------------------------------------------------

    def compute_norm(self, params):
        self.busy.emit(True)
        try:
            shb.validate_params(params)
            n = shb.normalisation_factor(params)
            self.log.emit(f"normalisation factor: {n:.6f}")
            self.norm_ready.emit(n)
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.busy.emit(False)

    def run_burn_job(self, params, pump_freqs, burn_name, pump_passes, summary):
        self._cancel = False
        self.busy.emit(True)
        self.progress.emit(0, 0)
        try:
            shb.validate_params(params)
            if params["output_params"].get("norm") is None:
                self.log.emit("computing normalisation factor...")
                params["output_params"]["norm"] = shb.normalisation_factor(params)
                self.norm_ready.emit(params["output_params"]["norm"])
                self.log.emit(
                    f"normalisation factor: {params['output_params']['norm']:.6f}")

            burn = shb.burn_series if burn_name == "series" else shb.burn_parallel
            self.log.emit(summary)
            n_pf = int(np.atleast_1d(pump_freqs).size)
            self.log.emit(f"burning {n_pf} pump frequencies...")
            out = shb.run_burn(pump_freqs, params, burn=burn,
                               pump_passes=pump_passes)
            self.log.emit("spectrum done")
            spec = out["spec"]
            p = shb.settings(params)
            self.log.emit(
                f"Absorption {spec.min():.2f} to {spec.max():.2f}  "
                f"(dAbsorption {spec.max() - spec.min():.2f}, "
                f"unburned {p['alpha']:.2f})")
            self.burn_done.emit(out)
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.progress.emit(1, 1)
            self.busy.emit(False)

    def run_sequence_job(self, params, rows, burn_name, summary):
        """
        A preparation sequence: the rows become stages, and the stages are
        burned in order (shb.run_sequence). The result dict is the same shape
        run_burn returns - plus its stages - so it comes back on the existing
        burn_done signal and every plot / export path downstream is unchanged.
        """
        self._cancel = False
        self.busy.emit(True)
        self.progress.emit(0, 0)
        try:
            shb.validate_params(params)
            if params["output_params"].get("norm") is None:
                self.log.emit("computing normalisation factor...")
                params["output_params"]["norm"] = shb.normalisation_factor(params)
                self.norm_ready.emit(params["output_params"]["norm"])
                self.log.emit(
                    f"normalisation factor: {params['output_params']['norm']:.6f}")

            burn = shb.burn_series if burn_name == "series" else shb.burn_parallel
            stages = shb.sequence_stages(rows, params)
            self.log.emit(summary)
            out = shb.run_sequence(stages, params, burn=burn,
                                   log=self.log.emit,
                                   should_cancel=lambda: self._cancel)
            self.log.emit("spectrum done")
            spec = out["spec"]
            p = shb.settings(params)
            self.log.emit(
                f"Absorption {spec.min():.2f} to {spec.max():.2f}  "
                f"(dAbsorption {spec.max() - spec.min():.2f}, "
                f"unburned {p['alpha']:.2f})")
            self.burn_done.emit(out)
        except shb.Cancelled:
            self.log.emit("sequence cancelled")
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.progress.emit(1, 1)
            self.busy.emit(False)

    def run_scan_job(self, params, kwargs):
        self._cancel = False
        self.busy.emit(True)
        try:
            shb.validate_params(params)

            def progress(done, total):
                if self._cancel:
                    raise _Cancelled()
                self.progress.emit(done, total)

            scan = shb.run_scan(params=params, progress=progress,
                                log=self.log.emit,
                                estimate=self.estimate.emit,
                                on_done=shb.scan_teams_notifier(
                                    log=self.log.emit),
                                **kwargs)
            best = scan["best"]
            if best:
                b = best[0]
                self.log.emit(
                    f"[{scan['tag']}] best: AFC comb quality {b['quality']:.3f} at "
                    f"t_s {b['storage_us']:.3g} us, F {b['finesse']:.1f}"
                    f"{shb._measured_note(b)} "
                    f"(spacing {b['spacing_MHz']:.3f} MHz, "
                    f"width {b['width_MHz']:.3f} MHz)")
            self.scan_done.emit(scan)
        except _Cancelled:
            self.log.emit("scan cancelled")
            self._emit_partial(kwargs.get("sim_dir"))
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(str(exc))
            self._emit_partial(kwargs.get("sim_dir"))
        finally:
            self.busy.emit(False)

    def _emit_partial(self, sim_dir):
        """
        After a cancelled or failed scan, load back whatever the checkpoint
        kept - the same run_scan.close()/write_run_outputs path a resume
        starts from, so what the user sees here is exactly that, not a
        second, differently-built reconstruction.
        """
        if not sim_dir:
            return
        try:
            scan = shb.load_run(sim_dir)
        except Exception as exc:                       # noqa: BLE001
            self.log.emit(f"could not read partial run from {sim_dir}: {exc}")
            return
        self.log.emit(f"partial results kept: {scan['n_done']} of "
                      f"{scan['n_total']} burns in {sim_dir}")
        self.scan_done.emit(scan)


class _Cancelled(Exception):
    pass


# ===========================================================================
# Small widgets
# ===========================================================================

class StackedPlot(QWidget):
    """
    One figure, one toolbar, holding both right-column plots.

    Burn / preview: pump comb on top, spectrum below, sharing the Frequency
    axis (zoom or pan either and both move). Scan: a single full-height axis
    for the quality heatmap / quality-vs-storage / top-spectra views.

    Left-drag across the plot to zoom the x-axis (as in the plotly figures of
    SHsimulation_v1_3test2multi.ipynb); double-click or the Home button resets.
    """

    def __init__(self):
        super().__init__()
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._axes = []
        self._home = []
        self._span = None

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.toolbar)
        bar.addStretch(1)
        home_btn = QPushButton("Home")
        home_btn.setToolTip("Reset the plot to its original view")
        home_btn.clicked.connect(self.reset_view)
        bar.addWidget(home_btn)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(bar)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.addWidget(self.canvas, stretch=1)
        self._legend_panel = self._build_legend_panel()
        self._legend_panel.hide()
        content.addWidget(self._legend_panel)
        box.addLayout(content)

        self.canvas.setMinimumHeight(460)
        self.canvas.setMinimumWidth(360)
        self._top_lines = []
        self._top_checks = []

        self.canvas.mpl_connect("button_press_event", self._on_click)

    # -- detachable legend for the top-spectra view --------------------

    def _build_legend_panel(self):
        w = QWidget()
        w.setMaximumWidth(260)
        w.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Traces"))
        hdr.addStretch(1)
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        for b in (all_btn, none_btn):
            b.setMaximumWidth(48)
        all_btn.clicked.connect(lambda: self._set_all_traces(True))
        none_btn.clicked.connect(lambda: self._set_all_traces(False))
        hdr.addWidget(all_btn)
        hdr.addWidget(none_btn)
        v.addLayout(hdr)

        self._legend_scroll = QScrollArea()
        self._legend_scroll.setWidgetResizable(True)
        inner = QWidget()
        self._legend_box = QVBoxLayout(inner)
        self._legend_box.setContentsMargins(0, 0, 0, 0)
        self._legend_box.addStretch(1)          # trailing spacer, kept
        self._legend_scroll.setWidget(inner)
        v.addWidget(self._legend_scroll)
        return w

    def _clear_legend_panel(self):
        while self._legend_box.count() > 1:     # leave the trailing stretch
            item = self._legend_box.takeAt(0)
            if item is None:
                break
            wdg = item.widget()
            if wdg is not None:
                wdg.deleteLater()
        self._top_checks = []
        self._top_lines = []

    def _populate_legend_panel(self, lines, labels):
        self._clear_legend_panel()
        self._top_lines = list(lines)
        for ln, lab in zip(lines, labels):
            text = lab.replace("$\\mu$s", "us").replace("$\\mu$", "u")
            cb = QCheckBox(text)
            cb.setChecked(ln.get_visible())
            cb.setStyleSheet(f"color:{mcolors.to_hex(ln.get_color())}")
            cb.toggled.connect(
                lambda on, l=ln: (l.set_visible(on), self._autoscale_top_y()))
            self._legend_box.insertWidget(self._legend_box.count() - 1, cb)
            self._top_checks.append(cb)
        self._legend_panel.show()

    def _set_all_traces(self, on):
        for cb, ln in zip(self._top_checks, self._top_lines):
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
            ln.set_visible(on)
        self._autoscale_top_y()

    def _autoscale_top_y(self):
        """Rescale y to the visible top-spectra traces, keeping the x-range."""
        if not self._axes or not self._top_lines:
            return
        ax = self._axes[-1]
        x0, x1 = sorted(ax.get_xlim())
        # Every top-spectra line shares one frequency axis (shb.run_scan hands
        # out a single f array, not a copy per spectrum), so the x-range mask
        # is the same for all of them - compute it once instead of per line.
        xd = np.asarray(self._top_lines[0].get_xdata(), dtype=float)
        m = (xd >= x0) & (xd <= x1)
        lo, hi = np.inf, -np.inf
        if m.any():
            for ln in self._top_lines:
                if not ln.get_visible():
                    continue
                yd = np.asarray(ln.get_ydata(), dtype=float)
                lo = min(lo, float(np.nanmin(yd[m])))
                hi = max(hi, float(np.nanmax(yd[m])))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        self.canvas.draw_idle()

    # -- drawing --------------------------------------------------------

    def _stacked_axes(self, top_ratio=1.0):
        self._clear_legend_panel()
        self._legend_panel.hide()
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[top_ratio, 3],
                                      hspace=0.08)
        ax_top = self.figure.add_subplot(gs[0])
        ax_bot = self.figure.add_subplot(gs[1], sharex=ax_top)
        ax_top.tick_params(labelbottom=False)
        self._axes = [ax_top, ax_bot]
        return ax_top, ax_bot

    def _finish(self):
        """Save the home view and (re)attach the drag-zoom selector."""
        self._home = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in self._axes]
        try:
            self.toolbar.update()          # re-base the toolbar Home button
        except Exception:                  # noqa: BLE001
            pass
        self._install_span()
        self.canvas.draw_idle()

    def _install_span(self):
        from matplotlib.widgets import SpanSelector
        if not self._axes:
            self._span = None
            return
        self._span = SpanSelector(
            self._axes[-1], self._on_span, "horizontal", useblit=True,
            button=1, drag_from_anywhere=True,
            props=dict(alpha=0.15, facecolor="0.5"))

    @staticmethod
    def _draw_pulses(ax, pump_freqs, params, stages):
        """
        The top panel: one band per stage for a preparation sequence, the plain
        pump mask for a single hole or comb.
        """
        if stages:
            shb.plot_sequence(ax, stages, params, show_xlabel=False)
        else:
            shb.plot_comb(ax, pump_freqs, params, show_xlabel=False)

    @staticmethod
    def _top_ratio(stages):
        """A taller top panel when a sequence has several stages to stack."""
        return 1.0 if not stages else min(3.0, 0.5 + 0.35 * len(stages))

    def show_preview(self, pump_freqs, params, stages=None):
        ax_top, ax_bot = self._stacked_axes(self._top_ratio(stages))
        self._draw_pulses(ax_top, pump_freqs, params, stages)
        ax_bot.text(0.5, 0.5, "run to see the spectrum", ha="center",
                    va="center", color="0.5", transform=ax_bot.transAxes)
        ax_bot.set_xlabel("Frequency (MHz)")
        ax_bot.set_ylabel("Absorption (Normalised)")
        p = shb.settings(params)
        ax_top.set_xlim(p["fmin"], p["fmax"])
        self._finish()

    def show_burn(self, pump_freqs, params, out):
        ax_top, ax_bot = self._stacked_axes(self._top_ratio(out.get("stages")))
        self._draw_pulses(ax_top, pump_freqs, params, out.get("stages"))
        shb.plot_spectrum(ax_bot, out["f"], out["spec"], params)
        p = shb.settings(params)
        ax_top.set_xlim(p["fmin"], p["fmax"])
        self._finish()

    def show_scan(self, scan, view):
        self._clear_legend_panel()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._axes = [ax]
        try:
            if view == 0:
                shb.plot_scan_heatmap(ax, scan)
            elif view == 1:
                shb.plot_scan_max(ax, scan)
            elif view == 3:
                shb.plot_scan_measured_finesse(ax, scan)
            elif view == 4:
                shb.plot_finesse_written_vs_measured(ax, scan)
            else:
                _, lines, labels = shb.plot_top_spectra(ax, scan, legend=False)
                self._populate_legend_panel(lines, labels)
            if view != 2:
                self._legend_panel.hide()
        except Exception as exc:                        # noqa: BLE001
            self._legend_panel.hide()
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                    transform=ax.transAxes)
        self._finish()

    def show_message(self, text):
        self._clear_legend_panel()
        self._legend_panel.hide()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._axes = [ax]
        ax.text(0.5, 0.5, text, ha="center", va="center", color="0.5",
                transform=ax.transAxes)
        self._finish()

    # -- interaction ---------------------------------------------------

    def _on_span(self, xmin, xmax):
        if getattr(self.toolbar, "mode", "") or abs(xmax - xmin) < 1e-12:
            return
        for ax in self._axes:
            ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.dblclick:
            self.reset_view()

    def reset_view(self):
        for ax, xl, yl in self._home:
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        self.canvas.draw_idle()


class LevelsDialog(QDialog):
    """Editable g / e / dm tables, for departing from the crystal preset."""

    def __init__(self, crystal, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit crystal levels")
        g = list(crystal["g"])
        e = list(crystal["e"])
        dm = np.asarray(crystal["dm"], dtype=float)

        self.g_table = self._vector_table(g)
        self.e_table = self._vector_table(e)
        self.g_info = QLabel()
        self.e_info = QLabel()
        for lbl in (self.g_info, self.e_info):
            lbl.setStyleSheet("color: grey;")
            lbl.setWordWrap(True)
        self.g_table.itemChanged.connect(self._update_level_info)
        self.e_table.itemChanged.connect(self._update_level_info)
        self.dm_table = QTableWidget(dm.shape[0], dm.shape[1])
        self.dm_table.setHorizontalHeaderLabels(
            [f"e{m}" for m in range(dm.shape[1])])
        self.dm_table.setVerticalHeaderLabels(
            [f"g{k}" for k in range(dm.shape[0])])
        for k in range(dm.shape[0]):
            for m in range(dm.shape[1]):
                self.dm_table.setItem(k, m, QTableWidgetItem(_fmt_num(dm[k, m])))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Level diagram, redrawn live from the tables (shb.plot_level_diagram)
        self.figure = Figure(figsize=(4, 4.5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(340, 380)
        self.dm_table.itemChanged.connect(self._redraw_levels)

        tables = QVBoxLayout()
        tables.addWidget(QLabel("Ground levels g"))
        tables.addWidget(self.g_table)
        tables.addWidget(self.g_info)
        tables.addWidget(QLabel("Excited levels e"))
        tables.addWidget(self.e_table)
        tables.addWidget(self.e_info)
        tables.addWidget(QLabel("Branching / relative dipole matrix dm"))
        tables.addWidget(self.dm_table)

        columns = QHBoxLayout()
        columns.addLayout(tables, stretch=1)
        columns.addWidget(self.canvas, stretch=1)

        lay = QVBoxLayout(self)
        lay.addLayout(columns)
        lay.addWidget(buttons)

        self._update_level_info()

    @staticmethod
    def _describe_levels(table):
        """
        Level count, span and consecutive spacing for one row of a levels
        table - the numbers default_pit_width and anti_hole_pumps (shb.py)
        actually work from, so this is what matters when editing g or e, not
        just the raw energies.
        """
        vals = []
        for i in range(table.columnCount()):
            item = table.item(0, i)
            try:
                vals.append(float(item.text()))
            except (AttributeError, ValueError):
                return "-"
        vals.sort()
        text = f"{len(vals)} level(s)"
        if len(vals) > 1:
            spacing = ", ".join(_fmt_num(b - a) for a, b in zip(vals, vals[1:]))
            text += (f" - span {_fmt_num(vals[-1] - vals[0])} MHz"
                     f" - spacing {spacing} MHz")
        return text

    def _update_level_info(self, *_):
        self.g_info.setText(self._describe_levels(self.g_table))
        self.e_info.setText(self._describe_levels(self.e_table))
        self._redraw_levels()

    def _redraw_levels(self, *_):
        """Redraw the level diagram; a half-typed or empty cell keeps the
        last good drawing rather than raising mid-edit."""
        try:
            crystal = self.values()
        except ValueError:
            return
        self.figure.clear()
        shb.plot_level_diagram(self.figure.add_subplot(), crystal["g"],
                               crystal["e"], crystal["dm"])
        self.canvas.draw_idle()

    @staticmethod
    def _vector_table(values):
        t = QTableWidget(1, len(values))
        t.setVerticalHeaderLabels([""])
        for i, v in enumerate(values):
            t.setItem(0, i, QTableWidgetItem(_fmt_num(v)))
        t.setMaximumHeight(60)
        return t

    @staticmethod
    def _cell(table, r, c):
        item = table.item(r, c)
        if item is None or not item.text().strip():
            raise ValueError(f"empty cell at row {r}, column {c}")
        return float(item.text())

    def values(self):
        g = [self._cell(self.g_table, 0, i)
             for i in range(self.g_table.columnCount())]
        e = [self._cell(self.e_table, 0, i)
             for i in range(self.e_table.columnCount())]
        dm = np.array(
            [[self._cell(self.dm_table, k, m)
              for m in range(self.dm_table.columnCount())]
             for k in range(self.dm_table.rowCount())])
        return {"g": g, "e": e, "dm": dm}


class PulseSequenceDialog(QDialog):
    """
    The preparation-sequence editor: rows of burning pulses, burned in order.

    Each row is one pulse, or one swept pulse - a centre frequency, how many
    pulses it fires, and the frequency sweep they are spread over (see
    ``shb.pulse_group``). **Consecutive rows sharing a stage name burn
    together**; a change of name starts a new stage, burned afterwards on the
    populations the previous stage left. That is what lets a pit, its
    burn-back and the cleaning sweeps be one sequence rather than one
    indiscriminate burn, while the many teeth of a comb still go in at once.

    The presets (``shb.PREP_PRESETS``) fill the table from the loaded crystal's
    own level structure; everything they produce is then editable here.
    """

    COLS = ["On", "Stage", "Centre (MHz)", "Sweep (MHz)", "Passes"]

    def __init__(self, rows, params, parent=None, *, initial_preset=None,
                initial_field_values=None):
        super().__init__(parent)
        self.setWindowTitle("Edit preparation sequence")
        self.resize(720, 520)
        self._params = params
        self._initial_preset = initial_preset
        # {preset name: {kwarg: value}} - whatever was last typed into that
        # preset's fields, so reopening the dialog on the same preset shows
        # the values it was left on rather than resetting to the built-in
        # defaults every time.
        self._initial_field_values = initial_field_values or {}

        # -- presets ---------------------------------------------------
        # The combo lists the two built-in generators (GEM, Narrowband AFC),
        # then a separator, then every preset saved from this table ("Save as
        # preset..." below) - see _refresh_preset_combo, which populates it.
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        load_btn = QPushButton("Load preset")
        load_btn.setToolTip("Reload the selected preset's rows - the fields "
                            "above (for a built-in) already apply live as "
                            "you change them, so this is for an explicit "
                            "re-apply, e.g. after a value landed off the df "
                            "mesh, or to discard hand edits since it was "
                            "last loaded.")
        load_btn.clicked.connect(self._load_preset)
        save_btn = QPushButton("Save as preset...")
        save_btn.setToolTip("Save the rows currently in the table below as "
                            "a named preset you can pick again later. Using "
                            "the name of an existing saved preset updates "
                            "it.")
        save_btn.clicked.connect(self._save_as_preset)
        self.delete_preset_btn = QPushButton("Delete preset")
        self.delete_preset_btn.setToolTip(
            "Remove the selected saved preset. Built-in presets (GEM, "
            "Narrowband AFC) can't be deleted.")
        self.delete_preset_btn.clicked.connect(self._delete_preset)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preset"))
        top.addWidget(self.preset_combo, stretch=1)
        top.addWidget(load_btn)
        top.addWidget(save_btn)
        top.addWidget(self.delete_preset_btn)

        self._preset_form = QFormLayout()
        self._preset_fields = {}
        self._preset_field_labels = {}

        # -- table -----------------------------------------------------
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.itemChanged.connect(self._on_edited)

        add_btn = QPushButton("Add row")
        add_btn.clicked.connect(lambda: self._add_row(
            {"label": "stage", "centre": 0.0, "sweep": 0.0,
             "passes": 20, "on": True}))
        del_btn = QPushButton("Remove row")
        del_btn.clicked.connect(self._remove_row)
        up_btn = QPushButton("Move up")
        up_btn.clicked.connect(lambda: self._move_row(-1))
        down_btn = QPushButton("Move down")
        down_btn.clicked.connect(lambda: self._move_row(+1))
        for b in (up_btn, down_btn):
            b.setToolTip("Stages burn in table order, and the order is "
                         "physics: burn back before the pit and there is "
                         "nothing to burn back into.")

        btn_row = QHBoxLayout()
        for b in (add_btn, del_btn, up_btn, down_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)

        self.summary = QLabel("-")
        self.summary.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(self._preset_form)
        hint = QLabel(
            "Rows burn top to bottom, and rows sharing a stage name burn "
            "together. Passes 0 uses the pump passes set on the main window.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addWidget(self.table, stretch=1)
        lay.addLayout(btn_row)
        lay.addWidget(self.summary)
        lay.addWidget(buttons)

        self._refresh_preset_combo(select=self._initial_preset, apply=False)
        self._set_rows(rows)

    # -- preset fields --------------------------------------------------

    def _clear_preset_fields(self):
        while self._preset_form.rowCount():
            self._preset_form.removeRow(0)
        self._preset_fields = {}
        self._preset_field_labels = {}

    def _build_preset_fields(self, name):
        """
        Rebuild the preset's tunable controls from its field spec.

        A field tuple is normally ``(kwarg, label, default, decimals)``; a
        5th element names another (boolean) field this one only makes sense
        alongside - e.g. a shuffle pulse's width/passes next to its on/off
        checkbox - and starts the row hidden unless that field is already
        checked, toggling with it from then on.

        ``decimals`` chooses the *widget* ("bool" -> checkbox, 0 -> integer
        spin, anything else -> float spin) and sizes the spinner's step; it
        is no longer the typing precision, which is ``SPIN_DECIMALS`` for
        every float field.
        """
        self._clear_preset_fields()
        spec = shb.PREP_PRESETS.get(name)
        if spec is None:
            return
        df = shb.settings(self._params)["df"]
        remembered = self._initial_field_values.get(name, {})
        for field in spec["fields"]:
            kwarg, label, default, decimals = field[:4]
            depends_on = field[4] if len(field) > 4 else None
            if kwarg in remembered:
                value = remembered[kwarg]
            else:
                value = default(self._params) if callable(default) else default
            if decimals == "bool":
                w = QCheckBox()
                w.setChecked(bool(value))
                w.toggled.connect(lambda *_: self._apply_preset(quiet=True))
            else:
                if decimals == 0:
                    w = _ispin(0, 100000, int(value))
                else:
                    # Step by the mesh spacing for MHz fields, so nudging the
                    # spinner mostly stays on-mesh instead of landing off it
                    # on two ticks out of three. "finesse" (dimensionless)
                    # and "storage_time" (microseconds, and only ever an
                    # *approximate* mesh position once inverted to a tooth
                    # spacing - see prep_broadband_afc) aren't MHz fields, so
                    # a df-sized step would be the wrong unit entirely; they
                    # keep a plain decimal step instead. Either way the spec's
                    # `decimals` only sizes this step - the box itself takes
                    # SPIN_DECIMALS places whatever the step is.
                    step = (10.0 ** -min(decimals, 3)
                           if kwarg in ("finesse", "storage_time")
                           else max(df, 1e-6))
                    w = _dspin(-100000.0, 100000.0, step, float(value))
                w.valueChanged.connect(lambda *_: self._apply_preset(quiet=True))
            label_w = QLabel(label)
            self._preset_form.addRow(label_w, w)
            self._preset_fields[kwarg] = w
            self._preset_field_labels[kwarg] = label_w
            if depends_on:
                gate = self._preset_fields.get(depends_on)
                if gate is not None:
                    self._set_field_row_visible(kwarg, gate.isChecked())
                    gate.toggled.connect(
                        lambda checked, k=kwarg: self._set_field_row_visible(
                            k, checked))

    def _set_field_row_visible(self, kwarg, visible):
        """Show/hide one preset field's label and control together, e.g. a
        shuffle pulse's width/passes appearing only once its checkbox is on."""
        w = self._preset_fields.get(kwarg)
        label_w = self._preset_field_labels.get(kwarg)
        if w is not None:
            w.setVisible(visible)
        if label_w is not None:
            label_w.setVisible(visible)

    def current_field_values(self):
        """The current preset's field values as builder kwargs - what
        _apply_preset feeds the builder, and what a caller can stash away to
        restore into a later dialog's initial_field_values."""
        return {k: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
               for k, w in self._preset_fields.items()}

    def _sync_preset_fields(self, name):
        """
        Show the right fields panel for whichever kind of preset is selected:
        tunable fields for a built-in generator, a plain note for a saved
        row-snapshot (it has nothing to tune - edit the rows directly).

        Also updates delete_preset_btn's enabled state, since built-in
        presets can't be deleted.
        """
        if name in shb.PREP_PRESETS:
            self._build_preset_fields(name)
        else:
            self._clear_preset_fields()
            if name:
                self._preset_form.addRow(QLabel(
                    "(saved preset - edit rows directly below)"))
        self.delete_preset_btn.setEnabled(
            bool(name) and name not in shb.PREP_PRESETS)

    def _apply_current_preset_selection(self):
        """Apply whatever the combo currently shows - a built-in's fields, or
        a saved preset's row snapshot."""
        name = self.preset_combo.currentText()
        if name in shb.PREP_PRESETS:
            self._apply_preset(quiet=True)
        elif name:
            self._load_user_preset(name)

    def _on_preset_changed(self, name):
        """The combo box changed under user interaction - sync the fields
        panel for the new selection and apply it immediately. (__init__
        populates the combo via _refresh_preset_combo(apply=False), not
        through here, so opening the dialog keeps showing the rows it was
        given rather than jumping to a freshly generated preset.)"""
        self._sync_preset_fields(name)
        self._apply_current_preset_selection()

    def _apply_preset(self, *, quiet):
        """
        Regenerate the table from the preset builder and the current field
        values. Built-in presets only - see _load_user_preset for a saved
        row-snapshot.

        quiet=True (every live field edit) reports a bad combination inline,
        in the summary label - the spin-box step is aligned to df, but a
        typed value or an unlucky combination of fields can still land off
        mesh, and that must not pop a modal on every arrow click. quiet=False
        (the explicit "Load preset" button) still pops one, since that click
        is a deliberate action worth a clear stop.
        """
        name = self.preset_combo.currentText()
        spec = shb.PREP_PRESETS.get(name)
        if spec is None:
            return
        kwargs = self.current_field_values()
        try:
            rows = spec["builder"](self._params, **kwargs)
        except Exception as exc:                        # noqa: BLE001
            if quiet:
                self.summary.setText(f"! {exc}")
                self.summary.setStyleSheet("color: #a00;")
            else:
                QMessageBox.critical(self, "Preset", str(exc))
            return
        self._set_rows(rows)

    def _load_user_preset(self, name):
        """Load a saved preset's row snapshot (shb.load_user_presets())."""
        rows = shb.load_user_presets().get(name)
        if rows is not None:
            self._set_rows(rows)

    def _load_preset(self):
        name = self.preset_combo.currentText()
        if name in shb.PREP_PRESETS:
            self._apply_preset(quiet=False)
        elif name in shb.load_user_presets():
            self._load_user_preset(name)

    def _refresh_preset_combo(self, select=None, *, apply=True):
        """
        Rebuild preset_combo's items - the two built-ins, a separator, then
        every saved user preset (alphabetical) - and resync the fields panel/
        delete button for whatever ends up selected.

        select picks which item to land on (default: the first). apply=False
        skips reloading rows for that selection - used when constructing the
        dialog, so the rows it was given are shown rather than immediately
        overwritten by whatever preset happens to be first in the list; used
        again after deleting a preset, so that doesn't blow away whatever is
        currently in the table either.
        """
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(shb.PREP_PRESETS))
        user_names = sorted(shb.load_user_presets())
        if user_names:
            self.preset_combo.insertSeparator(self.preset_combo.count())
            self.preset_combo.addItems(user_names)
        target = select or self.preset_combo.itemText(0)
        idx = self.preset_combo.findText(target)
        self.preset_combo.setCurrentIndex(max(0, idx))
        self.preset_combo.blockSignals(False)
        self._sync_preset_fields(self.preset_combo.currentText())
        if apply:
            self._apply_current_preset_selection()

    def _save_as_preset(self):
        current = self.preset_combo.currentText()
        default_name = current if current not in shb.PREP_PRESETS else ""
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:",
                                        text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in shb.PREP_PRESETS:
            QMessageBox.critical(
                self, "Save preset",
                f'"{name}" is a built-in preset name and can\'t be '
                f'overwritten - pick a different name.')
            return
        if name in shb.load_user_presets():
            resp = QMessageBox.question(
                self, "Save preset",
                f'A preset named "{name}" already exists - overwrite it '
                f'with the rows currently in the table?')
            if resp != QMessageBox.StandardButton.Yes:
                return
        try:
            shb.save_user_preset(name, self.values())
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.critical(self, "Save preset", str(exc))
            return
        self._refresh_preset_combo(select=name)

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if name not in shb.load_user_presets():
            return
        resp = QMessageBox.question(
            self, "Delete preset", f'Delete the saved preset "{name}"?')
        if resp != QMessageBox.StandardButton.Yes:
            return
        shb.delete_user_preset(name)
        self._refresh_preset_combo(apply=False)

    # -- table ----------------------------------------------------------

    def _set_rows(self, rows):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in rows:
            self._add_row(row, quiet=True)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._on_edited()

    def _add_row(self, row, quiet=False):
        r = self.table.rowCount()
        self.table.insertRow(r)

        on = QTableWidgetItem()
        on.setFlags(on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        on.setCheckState(Qt.CheckState.Checked if row.get("on", True)
                         else Qt.CheckState.Unchecked)
        self.table.setItem(r, 0, on)
        for c, text in enumerate((
                str(row.get("label", "stage")),
                _fmt_num(row.get("centre", 0.0)),
                _fmt_num(row.get("sweep", 0.0)),
                f"{int(row.get('passes') or 0)}"), start=1):
            self.table.setItem(r, c, QTableWidgetItem(text))
        if not quiet:
            self._on_edited()

    def _remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._on_edited()

    def _move_row(self, delta):
        r = self.table.currentRow()
        t = r + delta
        if r < 0 or not (0 <= t < self.table.rowCount()):
            return
        self.table.blockSignals(True)
        taken = [self.table.takeItem(r, c) for c in range(len(self.COLS))]
        self.table.removeRow(r)
        self.table.insertRow(t)
        for c, item in enumerate(taken):
            self.table.setItem(t, c, item)
        self.table.blockSignals(False)
        self.table.setCurrentCell(t, 1)
        self._on_edited()

    def _tint_groups(self):
        """Shade alternate stages, so the row grouping is visible."""
        from PyQt6.QtGui import QBrush, QColor
        shades = (QColor(0, 0, 0, 0), QColor(0, 0, 0, 18))
        last, band = None, 0
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            label = item.text().strip() if item is not None else ""
            if last is not None and label != last:
                band ^= 1
            last = label
            for c in range(len(self.COLS)):
                cell = self.table.item(r, c)
                if cell is not None:
                    cell.setBackground(QBrush(shades[band]))

    def _on_edited(self, *_):
        self.table.blockSignals(True)
        self._tint_groups()
        self.table.blockSignals(False)
        try:
            stages = shb.sequence_stages(self.values(), self._params)
        except ValueError as exc:
            self.summary.setText(f"! {exc}")
            self.summary.setStyleSheet("color: #a00;")
            return
        # Per-stage counts are summed, not deduped: a cleaning sweep
        # deliberately revisits frequencies the pit already burned, and both
        # burns are real work.
        n = sum(int(st["freqs"].size) for st in stages)
        distinct = len(np.unique(np.concatenate(
            [st["freqs"] for st in stages]))) if stages else 0
        names = ", ".join(st["name"] for st in stages) or "nothing enabled"
        self.summary.setText(
            f"{len(stages)} stage(s) in burn order: {names} - "
            f"{n} pulses over {distinct} distinct frequencies")
        self.summary.setStyleSheet("color: grey;")

    # -- read back ------------------------------------------------------

    @staticmethod
    def _cell(table, r, c):
        item = table.item(r, c)
        if item is None or not item.text().strip():
            raise ValueError(f"empty cell at row {r + 1}, "
                             f"{PulseSequenceDialog.COLS[c]}")
        return item.text().strip()

    def values(self):
        """The rows, as the plain dicts shb.sequence_stages consumes."""
        rows = []
        for r in range(self.table.rowCount()):
            on_item = self.table.item(r, 0)
            try:
                centre = float(self._cell(self.table, r, 2))
                sweep = float(self._cell(self.table, r, 3))
                passes = int(float(self._cell(self.table, r, 4)))
            except ValueError as exc:
                raise ValueError(f"row {r + 1}: {exc}") from None
            rows.append({
                "label": self._cell(self.table, r, 1),
                "centre": centre,
                "sweep": abs(sweep),
                "passes": passes if passes > 0 else None,
                "on": (on_item is None
                       or on_item.checkState() == Qt.CheckState.Checked),
            })
        return rows


#: Decimal places every float field accepts. 1e-9 MHz is finer than any mesh
#: the simulator can build (df bottoms out at 1 kHz), so the limit is
#: effectively "whatever you can type"; 9 is also what shb._scan_grids rounds
#: finesse values to when it groups them into heatmap rows, so a box can never
#: distinguish two values the heatmap would silently merge. Don't raise it:
#: the widest field range is 1e6, and 7 integer digits + 9 places is already
#: the last cleanly representable double.
SPIN_DECIMALS = 9


class _PreciseSpinBox(QDoubleSpinBox):
    """
    A ``QDoubleSpinBox`` that accepts real precision without displaying it.

    Qt's ``decimals`` is both the input limit and the display format, so the
    3-decimal fields this replaces could not be given a finer value at all -
    the extra digits were dropped at the keystroke. Raising ``decimals``
    alone would pad every field out to its full width ("2.350000000"), so
    ``textFromValue`` trims the padding back off. The trimmed text is still
    exactly what Qt's own validator accepts, so typing, stepping and
    ``interpretText`` are unaffected.

    Nothing here relaxes the mesh invariant: an off-mesh frequency is still
    rejected by ``shb.to_steps`` when the run starts.
    """

    def textFromValue(self, v):
        loc = self.locale()
        if not self.isGroupSeparatorShown():        # Qt's own default
            loc.setNumberOptions(loc.numberOptions()
                                 | QLocale.NumberOption.OmitGroupSeparator)
        text = loc.toString(float(v), "f", self.decimals())
        point = loc.decimalPoint()                  # a string, not a char
        if point in text:
            text = text.rstrip("0")
            if text.endswith(point):
                text = text[:-len(point)]
        return "0" if text in ("", "-", "-0") else text

    def _hint_pad(self):
        # Both hints are measured from textFromValue of the range ends,
        # which now trim to something *narrower* than the 3-decimal boxes
        # these replace - leave room for a few typed digits instead, or a
        # squeezed form layout can shrink a field below the number in it.
        return self.fontMetrics().horizontalAdvance("0000")

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(hint.width() + self._hint_pad())
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(hint.width() + self._hint_pad())
        return hint


def _fmt_num(v, decimals=SPIN_DECIMALS):
    """
    A number as table-cell text, to the precision a spin box would hold.

    ``:g`` stops at 6 significant figures, which silently rounded a
    hand-typed value every time a table was repopulated - a preset
    re-apply, a dialog reopen - and ``_save_as_preset`` then wrote the
    rounding to disk. Fixed-point to the same places the boxes accept, with
    the padding trimmed off, so what a cell shows is exactly what a box
    could hold and ``float()`` reads it back unchanged. It also keeps the
    float noise of a computed level (-27.490000000000002) out of the cell,
    which a shortest-round-trip ``repr`` would not. Anything smaller than
    the last place - never a frequency, but a dm entry could be - falls
    back to ``repr``, which round-trips exactly.
    """
    v = float(v)
    text = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        text = "0"
    return repr(v) if v and float(text) == 0.0 else text


def _dspin(lo, hi, step, value, *, decimals=SPIN_DECIMALS):
    s = _PreciseSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


def _ispin(lo, hi, value):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _dropbox_info():
    """
    Parsed contents of the Dropbox desktop client's own info.json - the file
    Dropbox itself uses to know where each linked account syncs to, so this
    follows wherever the user has it configured (or a business + personal
    account both linked) rather than assuming ``~/Dropbox``.

    Returns ``{}`` if Dropbox is not installed or the file can't be read.
    """
    candidates = [
        os.path.join(os.path.expanduser("~"), ".dropbox", "info.json"),
        os.path.join(os.environ.get("APPDATA", ""), "Dropbox", "info.json"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Dropbox", "info.json"),
    ]
    for info_path in candidates:
        if not info_path or not os.path.isfile(info_path):
            continue
        try:
            with open(info_path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def _dropbox_folders():
    """
    Each linked Dropbox account's own synced folder (personal namespace, for
    a business/team account - see _dropbox_info).

    Returns a list of ``(label, path)`` pairs, existing directories only.
    """
    info = _dropbox_info()
    found = []
    for kind, entry in info.items():
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and os.path.isdir(path):
            label = f"Dropbox ({kind})" if len(info) > 1 else "Dropbox"
            found.append((label, path))
    return found


# Project folders worth a one-click bookmark in every save/open dialog,
# looked up relative to each account's Dropbox *team* root (a business
# account's shared team folders, like this lab's, are siblings of the
# personal namespace _dropbox_folders returns - not nested under it) rather
# than as absolute paths, so this keeps working if the team folder ever moves
# or is re-synced under a different account.
_PINNED_SUBFOLDERS = [
    "RES_EPS_Quantum_Photonics_Lab/Experiments/Current Experiments/BB Telecom QM",
    "RES_EPS_Quantum_Photonics_Lab/Experiments/Current Experiments"
    "/Visible Quantum Memory",
]


def _sidebar_folders():
    """
    Locations pinned into every file/folder dialog's sidebar: each linked
    Dropbox account's own folder (see _dropbox_folders) plus the
    _PINNED_SUBFOLDERS project folders under each account's team root.
    Existing directories only, so a folder missing on a given machine (or a
    personal-only Dropbox account, which has no team root) just doesn't
    appear.
    """
    info = _dropbox_info()
    folders = list(_dropbox_folders())
    roots = {entry.get("root_path") or entry.get("path")
             for entry in info.values() if isinstance(entry, dict)}
    for root in roots:
        if not root:
            continue
        for rel in _PINNED_SUBFOLDERS:
            path = os.path.join(root, rel)
            if os.path.isdir(path):
                folders.append((os.path.basename(rel), path))
    return folders


# ===========================================================================
# Main window
# ===========================================================================

class MainWindow(QMainWindow):

    request_norm = pyqtSignal(dict)
    request_burn = pyqtSignal(dict, object, str, int, str)
    request_sequence = pyqtSignal(dict, object, str, str)
    request_scan = pyqtSignal(dict, dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral hole burning - Pr:YSO AFC simulator")
        self.resize(1300, 850)

        self._crystal = {k: (np.array(v, dtype=float) if k == "dm" else list(v))
                         for k, v in shb.DEFAULT_PARAMS["crystal_params"].items()}
        self._linewidth = dict(shb.DEFAULT_PARAMS["linewidth_params"])
        self._norm = None
        self._last_burn = None
        self._scan = None
        # The params and side-panel session a burn was started with, attached
        # to its result when it lands (_on_burn_done) - so a later export
        # saves what actually ran, not whatever the fields say by then.
        self._burn_params = None
        self._burn_session = None

        self._job_t0 = 0.0
        self._job_estimate = None       # seconds, from run_scan's probe
        self._job_kind = None           # "burn" | "scan" | "norm" | None
        self._progress_base = None      # first "done" seen this job, for a
                                        # resumed scan's ETA (see _on_progress)
        self._sim_root = DEFAULT_SIM_ROOT  # remembered parent for auto-named
                                        # simulation folders

        self._sequence_rows = []        # preparation-sequence editor rows
        self._last_prep_preset = None   # preset shown last time the
                                        # sequence editor was open, so
                                        # reopening it lands on the same one
                                        # instead of always the first built-in
        self._last_prep_fields = {}     # {preset name: {kwarg: value}} - the
                                        # field values it was left on, so
                                        # reopening on the same preset shows
                                        # them instead of resetting to defaults

        self._build_menu()
        self._build_body()
        self._build_worker()

        # A GEM preparation to start from, derived from the loaded crystal.
        # Best effort: a preset whose defaults do not fit the current window
        # must not stop the window opening.
        try:
            self._sequence_rows = shb.prep_gem(self._collect_params())
        except Exception as exc:                        # noqa: BLE001
            self._sequence_rows = []
            print(f"could not build the default sequence: {exc}",
                  file=sys.stderr)

        self._norm_timer = QTimer(self)
        self._norm_timer.setSingleShot(True)
        self._norm_timer.setInterval(600)
        self._norm_timer.timeout.connect(self._recompute_norm)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._tick_progress_text)

        self._refresh_comb_preview()
        self._refresh_export_enabled()
        self._mark_norm_stale()

    # -- construction -----------------------------------------------------

    def _build_menu(self):
        bar = self.menuBar()
        assert bar is not None
        file_menu = bar.addMenu("&File")
        assert file_menu is not None
        for text, slot in [
            ("Import config (.json)...", self._import_config),
            ("Open run folder...", self._open_run_folder),
            ("Open saved result (.pkl / .npy)...", self._open_saved_result),
            ("Export...", self._open_export),
        ]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = bar.addMenu("&Help")
        assert help_menu is not None
        about = QAction("About", self)
        about.triggered.connect(lambda: QMessageBox.information(
            self, "About",
            "Spectral hole burning in Pr:YSO.\n"
            "GUI front end for shb.py, ported from the MATLAB "
            "Bloch-Simulator (zenodo.org/records/16962167)."))
        help_menu.addAction(about)

    def _build_body(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left column: Simulation / Export tabs --------------------
        sim_tab = QWidget()
        sim_lay = QVBoxLayout(sim_tab)
        import_btn = QPushButton("Import config (.json)...")
        import_btn.clicked.connect(self._import_config)
        sim_lay.addWidget(import_btn)
        sim_lay.addWidget(self._build_crystal_group())
        sim_lay.addWidget(self._build_sim_group(), stretch=1)

        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self._scrollable(sim_tab), "Simulation")
        self.left_tabs.addTab(self._scrollable(self._build_export_tab()),
                              "Export")
        self.left_tabs.setFixedWidth(380)
        root.addWidget(self.left_tabs)

        # ---- right column ---------------------------------------------
        right = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Scan view:"))
        self.scan_view = QComboBox()
        # Indices are what StackedPlot.show_scan switches on - append new
        # views rather than inserting, or every `== 2` (top spectra) shifts.
        self.scan_view.addItems(["Comb-quality heatmap",
                                 "Quality vs storage time (best/mean/worst)",
                                 "Top spectra",
                                 "Measured finesse heatmap",
                                 "Measured vs written finesse"])
        self.scan_view.setEnabled(False)
        self.scan_view.currentIndexChanged.connect(self._draw_feature)
        row.addWidget(self.scan_view)
        self.collect_btn = QPushButton("Collect top spectra")
        self.collect_btn.setToolTip(
            "Re-burn the best spectrum at each of the top storage times from "
            "this run's checkpoint - for a cancelled run that never reached "
            "the collect pass, or to redo it at a different 'keep top N "
            "storage times' without resweeping.")
        self.collect_btn.setEnabled(False)
        self.collect_btn.clicked.connect(self._on_collect_top_clicked)
        row.addWidget(self.collect_btn)
        row.addStretch(1)
        row.addWidget(QLabel("(drag to zoom the x-axis · "
                             "double-click or Home to reset)"))
        right.addLayout(row)

        self.plot = StackedPlot()
        right.addWidget(self.plot, stretch=8)

        log_box = QVBoxLayout()
        log_label = QLabel("Simulation run log")
        log_label.setStyleSheet("font-weight: bold; padding: 2px;")
        log_box.addWidget(log_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMinimumHeight(120)
        log_box.addWidget(self.log_view)
        lw = QWidget()
        lw.setLayout(log_box)
        right.addWidget(lw, stretch=2)

        root.addLayout(right, stretch=1)

    @staticmethod
    def _scrollable(widget):
        """
        Put a tab's controls in a scroll area.

        The left column is a fixed width and the Scan and Preparation-sequence
        pages are tall; on a short window the Run button and the progress bar
        used to be simply unreachable. Scrolling vertically (never
        horizontally - the column is already as wide as it gets) keeps every
        control reachable.
        """
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _build_crystal_group(self):
        box = QGroupBox("Crystal parameters")
        lay = QVBoxLayout(box)

        self.crystal_combo = QComboBox()
        self.crystal_combo.addItems(list(shb.PRESETS))
        self.crystal_combo.currentTextChanged.connect(self._apply_preset)
        lay.addWidget(self.crystal_combo)

        self.dipole_check = QCheckBox("Relative dipole strength")
        self.dipole_check.setChecked(True)
        self.dipole_check.setToolTip(
            "On: use the measured branching-ratio matrix.\n"
            "Off: every transition equally strong.")
        self.dipole_check.stateChanged.connect(self._on_params_changed)
        lay.addWidget(self.dipole_check)

        edit_btn = QPushButton("Edit levels...")
        edit_btn.clicked.connect(self._edit_levels)
        lay.addWidget(edit_btn)

        return box

    def _build_sim_group(self):
        box = QGroupBox("Simulation Parameters")
        outer = QVBoxLayout(box)
        form = QFormLayout()

        self.fmin = _dspin(-20000, 0, 1.0, -50.0)
        self.fmax = _dspin(0, 20000, 1.0, 50.0)
        self.df = _dspin(0.001, 10.0, 0.001, 0.01)
        self.df.setToolTip("Mesh step. Must divide the level energies and the "
                           "window edges exactly (0.01, 0.005, 0.001 for the "
                           "default crystal).")
        self.h_width = _dspin(0.1, 100000.0, 1.0, 1.0)
        self.h_width.setToolTip("Homogeneous FWHM (Lorentzian), kHz.")
        self.in_width = _dspin(1.0, 1e6, 100.0, 10000.0)
        self.in_width.setToolTip("Inhomogeneous FWHM (Gaussian), MHz.")
        self.pump_passes = _ispin(1, 100000, 10)
        self.burn_mode = QComboBox()
        self.burn_mode.addItems(["parallel", "series"])
        self.alpha = _dspin(0.0, 1e6, 0.1, 1.0)

        for w in (self.fmin, self.fmax, self.df, self.h_width, self.in_width):
            w.valueChanged.connect(self._on_params_changed)
        for w in (self.df, self.fmin, self.fmax):
            w.valueChanged.connect(self._refresh_comb_preview)

        form.addRow("fmin (MHz)", self.fmin)
        form.addRow("fmax (MHz)", self.fmax)
        form.addRow("df (MHz)", self.df)
        form.addRow("h_width (kHz)", self.h_width)
        form.addRow("in_width (MHz)", self.in_width)
        form.addRow("pump passes", self.pump_passes)
        form.addRow("burn mode", self.burn_mode)
        form.addRow("alpha", self.alpha)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single hole", "One comb", "Scan",
                                  "Preparation sequence"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("mode", self.mode_combo)
        outer.addLayout(form)

        # -- mode-specific stack ---------------------------------------
        self.stack = QStackedWidget()

        # single hole
        sh = QWidget()
        sh_form = QFormLayout(sh)
        self.centre = _dspin(-10000, 10000, 1.0, 0.0)
        self.centre.valueChanged.connect(self._refresh_comb_preview)
        sh_form.addRow("centre (MHz)", self.centre)
        self.stack.addWidget(sh)

        # one comb
        oc = QWidget()
        oc_form = QFormLayout(oc)
        # Storage time and finesse rather than tooth spacing and burn width:
        # neither needs to sit on the df mesh - shb.comb_for maps them the
        # same way a scan cell is, so a scan winner re-burns identically here.
        self.comb_storage = _dspin(0.001, 100000, 0.1, 2.0)
        self.comb_storage.setToolTip(
            "AFC storage time, us - teeth 1 / storage time MHz apart.")
        self.comb_finesse = _dspin(1.0, 1000, 0.5, 2.0)
        self.comb_finesse.setToolTip(
            "Comb finesse: tooth spacing / tooth width.")
        self.comb_storage.valueChanged.connect(self._refresh_comb_preview)
        self.comb_finesse.valueChanged.connect(self._refresh_comb_preview)
        self.comb_info = QLabel("-")
        self.comb_info.setWordWrap(True)
        oc_form.addRow("storage time (us)", self.comb_storage)
        oc_form.addRow("finesse", self.comb_finesse)
        oc_form.addRow("on mesh", self.comb_info)
        self.stack.addWidget(oc)

        # scan
        sc = QWidget()
        sc_form = QFormLayout(sc)
        self.storage_min = _dspin(0.001, 100000, 0.5, 2.0)
        self.storage_max = _dspin(0.001, 100000, 0.5, 10.0)
        # Capped well below the field's old 1,000,000: `requested` and
        # `results` (shb.run_scan) are both fully materialised lists sized
        # n_points x n_finesse, and a heatmap beyond a couple of thousand
        # cells per axis already exceeds screen resolution.
        self.n_points = _ispin(2, 2000, 40)
        self.finesse_min = _dspin(1.0, 1000, 0.5, 2.0)
        self.finesse_max = _dspin(1.0, 1000, 0.5, 20.0)
        self.n_finesse = _ispin(1, 2000, 20)
        self.target_window = _dspin(0.1, 10000, 1.0, 20.0)
        self.top_n = _ispin(1, 5000, 50)
        self.parallel_check = QCheckBox("multiprocessing")
        self.parallel_check.setChecked(True)
        _cpu = os.cpu_count() or 1
        self.workers = _ispin(1, _cpu * 2, max(1, _cpu - 1))
        self.parallel_check.stateChanged.connect(
            lambda s: self.workers.setEnabled(bool(s)))
        for w in (self.storage_min, self.storage_max,
                  self.finesse_min, self.finesse_max):
            w.valueChanged.connect(self._refresh_comb_preview)
        sc_form.addRow("storage time min (us)", self.storage_min)
        sc_form.addRow("storage time max (us)", self.storage_max)
        sc_form.addRow("storage time points", self.n_points)
        sc_form.addRow("finesse min", self.finesse_min)
        sc_form.addRow("finesse max", self.finesse_max)
        sc_form.addRow("finesse points", self.n_finesse)
        sc_form.addRow("target window (MHz)", self.target_window)
        self.top_n.setToolTip(
            "Keep the best-finesse spectrum at each of this many of the best "
            "storage times - never two spectra at the same storage time.")
        sc_form.addRow("keep top N storage times", self.top_n)
        # Minimum gap between kept storage times: "auto" spreads the N picks
        # across the scanned range (shb._resolve_storage_sep); 0 turns it off.
        self.storage_sep_auto = QCheckBox("auto")
        self.storage_sep_auto.setChecked(True)
        self.storage_sep = _dspin(0.0, 10000.0, 0.1, 0.0)
        self.storage_sep.setEnabled(False)
        self.storage_sep_auto.toggled.connect(
            lambda on: self.storage_sep.setEnabled(not on))
        _sep_tip = ("Kept storage times are at least this far apart, so the "
                    "top spectra are distinct combs rather than near-identical "
                    "neighbours. 'auto' = (max - min) / (2 N); 0 = off.")
        self.storage_sep.setToolTip(_sep_tip)
        self.storage_sep_auto.setToolTip(_sep_tip)
        sep_row = QHBoxLayout()
        sep_row.setContentsMargins(0, 0, 0, 0)
        sep_row.addWidget(self.storage_sep_auto)
        sep_row.addWidget(self.storage_sep, 1)
        sep_box = QWidget()
        sep_box.setLayout(sep_row)
        sc_form.addRow("min storage separation (us)", sep_box)
        sc_form.addRow(self.parallel_check)
        sc_form.addRow(f"workers (of {_cpu})", self.workers)

        self.sim_dir = QLineEdit()
        self.sim_dir.setPlaceholderText(
            "(auto-named under a folder you choose)")
        self.sim_dir.textChanged.connect(self._refresh_sim_dir_state)
        sim_browse = QPushButton("Browse...")
        sim_browse.clicked.connect(self._pick_sim_dir)
        sim_row = QHBoxLayout()
        sim_row.addWidget(self.sim_dir, stretch=1)
        sim_row.addWidget(sim_browse)
        sc_form.addRow("simulation folder", sim_row)
        self.sim_resume = QCheckBox("resume if the folder already has a run")
        self.sim_resume.setChecked(True)
        sc_form.addRow(self.sim_resume)
        self.sim_dir_info = QLabel("new run - continuously checkpointed, so "
                                   "a crash or cancel keeps what finished")
        self.sim_dir_info.setWordWrap(True)
        self.sim_dir_info.setStyleSheet("color: gray")
        sc_form.addRow(self.sim_dir_info)
        self.stack.addWidget(sc)

        # preparation sequence
        ps = QWidget()
        ps_lay = QVBoxLayout(ps)
        ps_intro = QLabel(
            "A pit, burn-back and cleaning pulses - burned in order, each on "
            "the populations the last one left.")
        ps_intro.setWordWrap(True)
        ps_lay.addWidget(ps_intro)
        edit_seq = QPushButton("Edit sequence...")
        edit_seq.clicked.connect(self._edit_sequence)
        ps_lay.addWidget(edit_seq)
        self.seq_info = QLabel("-")
        self.seq_info.setWordWrap(True)
        self.seq_info.setStyleSheet("color: grey;")
        ps_lay.addWidget(self.seq_info)
        ps_lay.addStretch(1)
        self.stack.addWidget(ps)

        # QStackedWidget sizes itself to its TALLEST page by default (here,
        # the Scan page), not the one actually showing - so a short page like
        # Single hole or Preparation sequence would otherwise leave a large
        # dead gap between its own controls and the Run button below. Simply
        # overriding sizeHint()/minimumSizeHint() to the current page's own
        # is not enough here - the enclosing QGroupBox's own sizeHint() does
        # not reliably track that override across a page switch (measured:
        # it disagrees with its own layout's sizeHint() immediately after
        # setCurrentIndex(), even with every ancestor layout explicitly
        # invalidated). Capping the stack's actual maximumHeight to the
        # current page's sizeHint is a harder constraint the layout engine
        # has to honour regardless, and does track correctly.
        self.stack.currentChanged.connect(self._size_stack_to_current)
        self._size_stack_to_current()

        outer.addWidget(self.stack)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run_clicked)
        outer.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        outer.addWidget(self.progress)

        outer.addStretch(1)
        return box

    def _build_export_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        lay.addWidget(QLabel(
            "Bundle the current run to a folder. Each ticked item is written\n"
            "with the filename prefix below."))

        self.exp_log = QCheckBox("Run log (.txt)")
        self.exp_config = QCheckBox("Simulation config (.json)")
        self.exp_figures = QCheckBox("Figures (.png)")
        self.exp_data = QCheckBox("Data (.npy + .csv)")
        self.exp_pickle = QCheckBox("Pickle (.pkl, for reopening via "
                                    "File > Open saved result...)")
        for cb in (self.exp_log, self.exp_config, self.exp_figures,
                   self.exp_data, self.exp_pickle):
            cb.setChecked(True)
            lay.addWidget(cb)

        form = QFormLayout()
        self.exp_prefix = QLineEdit("shb_run")
        form.addRow("filename prefix", self.exp_prefix)
        lay.addLayout(form)

        self.exp_btn = QPushButton("Export...")
        self.exp_btn.clicked.connect(self._open_export)
        lay.addWidget(self.exp_btn)

        self.exp_hint = QLabel("Run a simulation first.")
        self.exp_hint.setStyleSheet("color: grey;")
        lay.addWidget(self.exp_hint)

        imp_btn = QPushButton("Import config (.json)...")
        imp_btn.clicked.connect(self._import_config)
        lay.addWidget(imp_btn)

        lay.addStretch(1)
        return tab

    def _build_worker(self):
        self._sim_thread = QThread(self)
        self.worker = SimWorker()
        self.worker.moveToThread(self._sim_thread)

        self.request_norm.connect(self.worker.compute_norm)
        self.request_burn.connect(self.worker.run_burn_job)
        self.request_sequence.connect(self.worker.run_sequence_job)
        self.request_scan.connect(self.worker.run_scan_job)

        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.estimate.connect(self._on_estimate)
        self.worker.norm_ready.connect(self._on_norm_ready)
        self.worker.burn_done.connect(self._on_burn_done)
        self.worker.scan_done.connect(self._on_scan_done)
        self.worker.error.connect(self._on_error)
        self.worker.busy.connect(self._on_busy)

        self._sim_thread.start()

    # -- params ---------------------------------------------------------

    def _collect_params(self):
        params = {
            "crystal_params": {
                "g": list(self._crystal["g"]),
                "e": list(self._crystal["e"]),
                "dm": np.asarray(self._crystal["dm"], dtype=float),
            },
            "linewidth_params": {
                "h_width": self.h_width.value() / 1000.0,
                "in_width": self.in_width.value(),
            },
            "numerical_params": {
                "fmin": self.fmin.value(),
                "fmax": self.fmax.value(),
                "df": self.df.value(),
            },
            "burn_params": {"pump_passes": self.pump_passes.value()},
            "output_params": {"alpha": self.alpha.value(), "norm": self._norm},
        }
        if not self.dipole_check.isChecked():
            params = shb.uniform_dipole_params(params)
            params["output_params"]["norm"] = self._norm
        return params

    def _on_params_changed(self, *_):
        self._mark_norm_stale()

    def _apply_preset(self, name):
        pr = shb.PRESETS.get(name)
        if pr is None:
            return
        self._crystal = {
            "g": list(pr["crystal_params"]["g"]),
            "e": list(pr["crystal_params"]["e"]),
            "dm": np.asarray(pr["crystal_params"]["dm"], dtype=float),
        }
        self._linewidth = dict(pr["linewidth_params"])
        n = pr["numerical_params"]
        fields = {self.fmin: n["fmin"], self.fmax: n["fmax"], self.df: n["df"],
                  self.h_width: pr["linewidth_params"]["h_width"] * 1000.0,
                  self.in_width: pr["linewidth_params"]["in_width"]}
        for w, v in fields.items():
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._append_log(f"preset: {name}")
        self._mark_norm_stale()
        self._refresh_comb_preview()

    def _mark_norm_stale(self):
        self._norm = None
        self.setWindowTitle(
            "Spectral hole burning - Pr:YSO AFC simulator   [norm: pending]")
        self._norm_timer.start()

    def _recompute_norm(self):
        if self._sim_thread.isRunning():
            try:
                params = self._collect_params()
                shb.validate_params(params)
            except Exception as exc:                   # noqa: BLE001
                self._append_log(f"! {exc}")
                return
            self._append_log("recomputing normalisation factor...")
            self.request_norm.emit(params)

    def _on_norm_ready(self, n):
        self._norm = float(n)
        self.setWindowTitle(
            f"Spectral hole burning - Pr:YSO AFC simulator   [norm: {n:.4f}]")

    # -- mode / preview ----------------------------------------------

    def _size_stack_to_current(self, *_):
        page = self.stack.currentWidget()
        if page is not None:
            self.stack.setMaximumHeight(page.sizeHint().height())

    def _on_mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        self._refresh_comb_preview()
        if idx == 2:
            self._refresh_sim_dir_state()

    def _current_pump_freqs(self, params):
        """
        The pump frequencies the current mode would burn.

        Returns ``(freqs, summary, stages)``. ``stages`` is None for the modes
        that burn one flat set of frequencies, and the ordered stage list for a
        preparation sequence - which is what the top panel needs to draw the
        stages separately, and what tells the run path which worker job to use.
        """
        mode = self.mode_combo.currentIndex()
        if mode == 0:
            return shb.single_hole(self.centre.value()), "single hole", None
        if mode == 1:
            freqs, t_s, fin, nt = shb.comb_for(
                self.comb_storage.value(), self.comb_finesse.value(), params)
            sp = 1.0 / t_s
            self.comb_info.setText(
                f"t_s {t_s:.4g} us, F {fin:.3g} - spacing {sp:.4g} MHz, "
                f"width {sp / fin:.4g} MHz, {nt} teeth")
            return (freqs,
                    f"comb: t_s {t_s:.4g} us, F {fin:.3g} "
                    f"({sp:.4g} MHz spacing), {nt} teeth",
                    None)
        if mode == 3:
            stages = shb.sequence_stages(self._sequence_rows, params)
            freqs = (np.unique(np.concatenate([st["freqs"] for st in stages]))
                     if stages else np.array([]))
            names = ", ".join(st["name"] for st in stages) or "empty"
            summary = (f"sequence: {len(stages)} stages ({names}), "
                       f"{freqs.size} pump frequencies")
            self.seq_info.setText(summary)
            return freqs, summary, stages
        # scan preview: comb at the middle of the storage-time / finesse grid
        t_mid = 0.5 * (self.storage_min.value() + self.storage_max.value())
        mid = 1.0 / max(t_mid, 1e-9)                 # tooth spacing, MHz
        f_mid = max(1.0, 0.5 * (self.finesse_min.value()
                                + self.finesse_max.value()))
        df = params["numerical_params"]["df"]
        mid = max(df, round(mid / df) * df)
        width = max(df, round((mid / f_mid) / df) * df)
        freqs, sp, wd, nt = shb.comb_at(mid, width, params)
        return freqs, (f"scan preview: t_s {1 / sp:.3g} us, comb at "
                       f"{sp:g} MHz, F {sp / wd:.1f}"), None

    def _refresh_comb_preview(self, *_):
        """Live pump-comb preview; the plot returns to preview when inputs change."""
        try:
            params = self._collect_params()
            shb.validate_params(params)
            freqs, _, stages = self._current_pump_freqs(params)
        except Exception as exc:                       # noqa: BLE001
            self.plot.show_message(str(exc))
            return
        self.plot.show_preview(freqs, params, stages)

    # -- run ----------------------------------------------------------

    def _on_run_clicked(self):
        # Gated on job state, not the button's text: once the button can also
        # read "Resume (612/800)" or "Open", comparing against a literal
        # "Cancel" string breaks.
        if self._job_kind is not None:
            self.worker.cancel()
            return

        if self.mode_combo.currentIndex() == 2 and self.run_btn.text() == "Open":
            # A completed folder: load it back for plotting rather than
            # recomputing anything - the button reads "Open" only when
            # inspect_run() has already confirmed every burn is done.
            sim_dir = self.sim_dir.text().strip() or self._auto_sim_dir()
            try:
                scan = shb.load_run(sim_dir)
            except Exception as exc:                    # noqa: BLE001
                self._on_error(str(exc))
                return
            self._append_log(f"opened {sim_dir}: {scan['state']}, "
                             f"{scan['n_done']}/{scan['n_total']} burns")
            self._on_scan_done(scan)
            return

        try:
            params = self._collect_params()
            shb.validate_params(params)
        except Exception as exc:                       # noqa: BLE001
            self._on_error(str(exc))
            return

        self._job_t0 = time.monotonic()
        self._job_estimate = None
        self._progress_base = None

        # Drop the previous run's spectra before starting a new one, rather
        # than waiting for the new scan_done/burn_done to overwrite them: the
        # old top-N spectra otherwise stay pinned in the live axes (and in
        # self._scan) for the whole new run, doubling the retained set.
        self._scan = None
        self._last_burn = None
        self.plot.show_message("running...")
        self._refresh_export_enabled()

        if self.mode_combo.currentIndex() == 2:
            sim_dir = self._resolved_sim_dir()
            kwargs = dict(
                storage_min=self.storage_min.value(),
                storage_max=self.storage_max.value(),
                n_points=self.n_points.value(),
                finesse_min=self.finesse_min.value(),
                finesse_max=self.finesse_max.value(),
                n_finesse=self.n_finesse.value(),
                target_window=self.target_window.value(),
                top_n=self.top_n.value(),
                min_storage_sep=self._min_storage_sep(),
                pump_passes=self.pump_passes.value(),
                burn=(shb.burn_series if self.burn_mode.currentText() == "series"
                      else shb.burn_parallel),
                parallel=self.parallel_check.isChecked(),
                n_workers=self.workers.value(),
                sim_dir=sim_dir,
                resume=self.sim_resume.isChecked(),
            )
            self._job_kind = "scan"
            self._append_log(f"--- scan --- {sim_dir}")
            self.sim_dir.setText(sim_dir)
            self._write_sim_dir_config(sim_dir, params)
            self.request_scan.emit(params, kwargs)
        else:
            try:
                freqs, summary, stages = self._current_pump_freqs(params)
            except Exception as exc:                       # noqa: BLE001
                self.plot.show_message(str(exc))
                self._on_error(str(exc))
                return
            self._job_kind = "burn"
            self._burn_params = params
            self._burn_session = self._collect_session()
            if stages is not None:
                self._append_log("--- preparation sequence ---")
                self.request_sequence.emit(params, list(self._sequence_rows),
                                           self.burn_mode.currentText(),
                                           summary)
            else:
                self._append_log("--- burn ---")
                self.request_burn.emit(params, np.asarray(freqs),
                                       self.burn_mode.currentText(),
                                       self.pump_passes.value(), summary)

    # -- worker callbacks -------------------------------------------

    def _on_busy(self, busy):
        if busy:
            self.run_btn.setText("Cancel")
        for w in (self.mode_combo, self.df, self.fmin, self.fmax):
            w.setEnabled(not busy)
        if busy:
            if self._job_kind in ("burn", "scan"):
                self.progress.setRange(0, 0)         # indeterminate until a tick
                self.progress.setFormat(
                    "starting..." if self._job_kind == "burn" else "probing...")
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.stop()
            self._job_kind = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat("done")
            # Restores the Run/Resume/Open label from the folder's own state,
            # rather than a blanket "Run" - a completed scan sitting in a
            # simulation folder should offer Resume/Open on the next look,
            # not silently revert to a plain Run.
            self._refresh_sim_dir_state()

    def _on_estimate(self, secs):
        self._job_estimate = secs
        self._append_log(f"estimated ~{fmt_dur(secs)} total")
        if self.progress.maximum() == 0:            # not yet ticking
            self.progress.setFormat(f"~{fmt_dur(secs)} estimated")

    def _on_progress(self, done, total):
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        elapsed = time.monotonic() - self._job_t0
        # A resumed scan starts this job's progress at done = (say) 612 while
        # elapsed is still ~0, so elapsed/done*(total-done) reports an ETA of
        # nearly nothing for the whole run. Track where *this* job's progress
        # started and use only the work done since then.
        if self._progress_base is None:
            self._progress_base = done
        did = done - self._progress_base
        if did > 0:
            eta = elapsed / did * (total - done)
            self.progress.setFormat(
                f"%p%  ·  {done}/{total}  ·  ~{fmt_dur(eta)} left")
        else:
            self.progress.setFormat(f"{done}/{total}")

    def _tick_progress_text(self):
        """Live elapsed/ETA text while a job runs (driven by a 0.5 s timer)."""
        if self._job_kind is None:
            return
        elapsed = time.monotonic() - self._job_t0
        if self._job_kind == "burn":
            self.progress.setFormat(f"running... {fmt_dur(elapsed)}")
        elif self.progress.maximum() == 0 and self._job_estimate:
            left = max(0.0, self._job_estimate - elapsed)
            self.progress.setFormat(
                f"probing... ~{fmt_dur(left)} left of ~{fmt_dur(self._job_estimate)}")

    def _on_burn_done(self, out):
        # A fresh burn from the worker carries neither; a reopened one
        # (_open_saved_result) brings its own, possibly None for an old
        # spectrum-only export.
        out.setdefault("params", self._burn_params)
        out.setdefault("session", self._burn_session)
        self._last_burn = out
        self._scan = None
        self.scan_view.setEnabled(False)
        self.collect_btn.setEnabled(False)
        self._draw_feature()
        self._refresh_export_enabled()

    def _on_scan_done(self, scan):
        self._scan = scan
        self._last_burn = None
        self.scan_view.setEnabled(True)
        # A cancelled or crashed run can have completed burns but no top
        # spectra yet (the collect pass never ran) - offer to fetch them from
        # the checkpoint instead of resweeping.
        self.collect_btn.setEnabled(
            bool(scan.get("sim_dir")) and not scan["best"]
            and scan.get("n_done", 0) > 0)
        if not scan["best"] and self.scan_view.currentIndex() == 2:
            self.scan_view.setCurrentIndex(0)
        self._draw_feature()
        self._refresh_export_enabled()
        self._refresh_sim_dir_state()

    def _on_error(self, msg):
        self._append_log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Simulation error", msg)

    def _append_log(self, line):
        stamp = time.strftime("[%d:%m:%Y %H:%M:%S]")
        if self._job_kind is not None and self._job_t0:
            el = max(0.0, time.monotonic() - self._job_t0)
            pre = f"{stamp} +{int(el) // 60:02d}:{int(el) % 60:02d}  "
        else:
            pre = f"{stamp} "
        text = "" if line is None else str(line)
        for phys in text.split("\n"):
            self.log_view.appendPlainText(pre + phys.rstrip())

    # -- plot --------------------------------------------------

    def _draw_feature(self, *_):
        if self._scan is not None:
            self.plot.show_scan(self._scan, self.scan_view.currentIndex())
        elif self._last_burn is not None:
            # The params the burn ran with, so a reopened result plots on its
            # own crystal and window; the side panel only for an old
            # spectrum-only export that saved none.
            params = self._last_burn.get("params") or self._collect_params()
            self.plot.show_burn(self._last_burn["pump_freqs"], params,
                                self._last_burn)
        else:
            self._refresh_comb_preview()

    # -- session import / export --------------------------------

    _GUI_FIELDS = ("centre", "comb_storage", "comb_finesse", "storage_min",
                   "storage_max", "finesse_min", "finesse_max", "target_window")
    _GUI_INTS = ("n_points", "n_finesse", "top_n", "workers")

    def _min_storage_sep(self):
        """The scan's ``min_storage_sep``: ``"auto"`` or the gap in us."""
        if self.storage_sep_auto.isChecked():
            return "auto"
        return self.storage_sep.value()

    def _collect_session(self):
        gui = {
            "mode": self.mode_combo.currentText(),
            "crystal_preset": self.crystal_combo.currentText(),
            "relative_dipole": self.dipole_check.isChecked(),
            "burn_mode": self.burn_mode.currentText(),
            "pump_passes": self.pump_passes.value(),
            "parallel": self.parallel_check.isChecked(),
            "h_width": self.h_width.value(),
            "in_width": self.in_width.value(),
            "sequence": [dict(r) for r in self._sequence_rows],
            "prep_preset": self._last_prep_preset,
            "prep_fields": self._last_prep_fields,
            "min_storage_sep": self._min_storage_sep(),
        }
        for name in self._GUI_FIELDS:
            gui[name] = getattr(self, name).value()
        for name in self._GUI_INTS:
            gui[name] = getattr(self, name).value()
        return {"params": self._collect_params(), "gui": gui}

    def _apply_session(self, data):
        params = data["params"] if "params" in data else data
        params = shb.params_from_json(json.dumps(params, default=shb._json_default))
        p = shb.settings(params)
        gui = data.get("gui", {}) if isinstance(data, dict) else {}

        # Preset first (it moves several fields); the explicit values below win.
        if gui.get("crystal_preset") in shb.PRESETS:
            self.crystal_combo.setCurrentText(gui["crystal_preset"])

        self._crystal = {
            "g": list(params["crystal_params"]["g"]),
            "e": list(params["crystal_params"]["e"]),
            "dm": np.asarray(params["crystal_params"]["dm"], dtype=float),
        }
        self._linewidth = dict(params["linewidth_params"])
        self.fmin.setValue(p["fmin"])
        self.fmax.setValue(p["fmax"])
        self.df.setValue(p["df"])
        self.h_width.setValue(p["h_width"] * 1000.0)
        self.in_width.setValue(p["in_width"])
        self.pump_passes.setValue(int(p["pump_passes"]))
        self.alpha.setValue(p["alpha"])

        if "relative_dipole" in gui:
            self.dipole_check.setChecked(bool(gui["relative_dipole"]))
        if gui.get("burn_mode") in ("parallel", "series"):
            self.burn_mode.setCurrentText(gui["burn_mode"])
        if "parallel" in gui:
            self.parallel_check.setChecked(bool(gui["parallel"]))
        # Sessions saved before the One comb inputs became storage time and
        # finesse carry tooth_spacing/burn_width (MHz) instead.
        if ("comb_storage" not in gui and gui.get("tooth_spacing")
                and gui.get("burn_width")):
            sp, wd = float(gui["tooth_spacing"]), float(gui["burn_width"])
            gui = {**gui, "comb_storage": 1.0 / sp, "comb_finesse": sp / wd}
        for name in self._GUI_FIELDS:
            if name in gui:
                getattr(self, name).setValue(float(gui[name]))
        for name in self._GUI_INTS:
            if name in gui:
                getattr(self, name).setValue(int(gui[name]))
        if "min_storage_sep" in gui:
            sep = gui["min_storage_sep"]
            auto = sep in (None, "auto")
            self.storage_sep_auto.setChecked(auto)
            if not auto:
                self.storage_sep.setValue(float(sep))
        if isinstance(gui.get("sequence"), list):
            self._sequence_rows = [dict(r) for r in gui["sequence"]]
        if gui.get("prep_preset"):
            self._last_prep_preset = gui["prep_preset"]
        if isinstance(gui.get("prep_fields"), dict):
            self._last_prep_fields = {
                name: dict(values)
                for name, values in gui["prep_fields"].items()
                if isinstance(values, dict)
            }
        if gui.get("mode") in ("Single hole", "One comb", "Scan",
                               "Preparation sequence"):
            self.mode_combo.setCurrentText(gui["mode"])

        self._mark_norm_stale()
        self._refresh_comb_preview()

    def _import_config(self):
        path = self._file_dialog("Import config", "", directory=False,
                                 name_filter="JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self._apply_session(data)
            self._append_log(f"imported config from {path}")
        except Exception as exc:                       # noqa: BLE001
            self._on_error(str(exc))

    # -- simulation folder -------------------------------------------

    def _auto_sim_dir(self):
        """A fresh run-folder name under the remembered root: <timestamp>_sim_<tag>_df<df>."""
        root = self._sim_root or os.getcwd()
        tag = f"{self.storage_min.value():g}-{self.storage_max.value():g}us"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(root, f"{stamp}_sim_{tag}_df{self.df.value():g}")

    def _resolved_sim_dir(self):
        """
        The simulation folder for the next Run/Resume click - the field's
        text if the user set one, otherwise a fresh auto-named folder. The
        timestamp is resolved here, at Run, not whenever the field happened
        to last be blank.
        """
        text = self.sim_dir.text().strip()
        return text if text else self._auto_sim_dir()

    def _pick_sim_dir(self):
        start = self._sim_root or os.getcwd()
        folder = self._file_dialog("Simulation folder", start)
        if not folder:
            return
        if os.path.exists(os.path.join(folder, "manifest.json")):
            # Picked an existing run directly - use it as-is, to resume it.
            self.sim_dir.setText(folder)
            self._sim_root = os.path.dirname(folder) or folder
        else:
            # Picked a plain folder - treat it as the root, auto-name inside.
            self._sim_root = folder
            self.sim_dir.setText(self._auto_sim_dir())

    def _write_sim_dir_config(self, sim_dir, params):
        """Best-effort config.json alongside the checkpoint, so File > Open
        run folder can restore the GUI session, not just the numbers."""
        try:
            os.makedirs(sim_dir, exist_ok=True)
            with open(os.path.join(sim_dir, "config.json"), "w") as fh:
                json.dump(self._collect_session(), fh, indent=2,
                          default=shb._json_default)
        except OSError as exc:                          # noqa: BLE001
            self._append_log(f"! could not write config.json: {exc}")

    def _refresh_sim_dir_state(self, *_):
        """
        Update the folder hint label and the Run button's label from what is
        actually on disk - new / resumable / complete / live / mismatched -
        so the button never has to be taken on faith.

        Only the Scan mode has a folder to inspect; the other modes just get
        a plain "Run" whenever this fires while idle (it is only ever called
        from _on_busy(False), after self._job_kind is already None, or from a
        UI signal firing while idle - never while a job is actually running).
        """
        if self.mode_combo.currentIndex() != 2:
            self.run_btn.setText("Run")
            self.run_btn.setEnabled(True)
            return
        busy = self._job_kind is not None
        path = self.sim_dir.text().strip() or self._auto_sim_dir()
        try:
            params = self._collect_params()
            burn = (shb.burn_series if self.burn_mode.currentText() == "series"
                   else shb.burn_parallel)
            # df can auto-refine inside run_scan (shb._resolve_scan_df) when
            # storage times land close together on the mesh; the manifest
            # records that *refined* df_used, so comparing against the raw
            # df spinbox value here would report a mismatch against a folder
            # burned with these exact settings the moment refinement kicks
            # in. Resolve it the same way run_scan will before comparing.
            t_targets = np.linspace(self.storage_min.value(),
                                    self.storage_max.value(),
                                    max(2, self.n_points.value()))
            df, _ = shb._resolve_scan_df(t_targets, params,
                                         self.finesse_min.value())
            if df != shb.settings(params)["df"]:
                params = copy.deepcopy(params)
                params["numerical_params"]["df"] = df
            burn_spec = shb._burn_spec(params, burn, self.pump_passes.value(),
                                       self.target_window.value())
            info = shb.inspect_run(path, burn_spec=burn_spec)
        except Exception:                               # noqa: BLE001
            info = {"state": "new", "n_done": 0, "n_total": 0, "live": False,
                    "pid": None, "mismatch": []}

        label, colour = "Run", "gray"
        if info["state"] == "new":
            text = ("new run - continuously checkpointed, so a crash or "
                    "cancel keeps what finished")
        elif info["live"]:
            text = (f"in use by pid {info['pid']} - wait for it to finish, "
                    f"or choose another folder")
            colour = "#b00"
        elif info["mismatch"]:
            fields = ", ".join(f for f, _, _ in info["mismatch"])
            text = (f"different settings ({fields}) - Run will refuse to "
                    f"resume this folder; pick a new one")
            colour = "#b00"
        elif info["n_total"] and info["n_done"] >= info["n_total"]:
            text = f"complete - {info['n_done']}/{info['n_total']} burns done"
            label = "Open"
        elif info["n_done"] > 0:
            text = (f"resume: {info['n_done']} of {info['n_total']} burns "
                    f"already done")
            label = f"Resume ({info['n_done']}/{info['n_total']})"
        else:
            text = "new run"

        self.sim_dir_info.setText(text)
        self.sim_dir_info.setStyleSheet(f"color: {colour}")
        if not busy:
            self.run_btn.setText(label)
            self.run_btn.setEnabled(not info["live"])

    def _on_collect_top_clicked(self):
        if self._scan is None or not self._scan.get("sim_dir"):
            return
        sim_dir = self._scan["sim_dir"]
        try:
            shb.collect_top(sim_dir, top_n=self.top_n.value(),
                            min_storage_sep=self._min_storage_sep())
            scan = shb.load_run(sim_dir)
        except Exception as exc:                        # noqa: BLE001
            self._on_error(str(exc))
            return
        self._append_log(f"collected {len(scan['best'])} top spectra from "
                         f"{sim_dir}")
        self._on_scan_done(scan)

    def _open_run_folder(self):
        path = self._file_dialog("Open a simulation folder",
                                 self._sim_root or os.getcwd())
        if path:
            self._open_result(path)

    def _open_saved_result(self):
        """
        Reopen a saved result file - a scan or burn pickle from the Export
        tab, an old notebook ``results_*.pkl``, or an older ``*_spectrum.npy``
        export (see shb.load_result) - and plot it as if it had just run.
        """
        path = self._file_dialog("Open a saved result",
                                 self._sim_root or os.getcwd(),
                                 directory=False,
                                 name_filter="Saved results (*.pkl *.npy)")
        if path:
            self._open_result(path)

    def _open_result(self, path):
        try:
            kind, data = shb.load_result(path)
        except Exception as exc:                        # noqa: BLE001
            self._on_error(f"could not read {path}: {exc}")
            return
        if kind == "burn":
            self._open_burn(path, data)
        elif os.path.isdir(path):
            self._open_scan_folder(path, data)
        else:
            self._append_log(f"opened {path}: {len(data['best'])} top "
                             f"spectra, {len(data['results'])} grid cells")
            self._on_scan_done(data)

    def _open_scan_folder(self, path, scan):
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    self._apply_session(json.load(fh))
            except Exception as exc:                     # noqa: BLE001
                self._append_log(f"! could not restore session: {exc}")
        self.mode_combo.setCurrentIndex(2)
        self.sim_dir.setText(path)
        self._sim_root = os.path.dirname(path) or path
        self._append_log(f"opened {path}: {scan['state']}, "
                         f"{scan['n_done']}/{scan['n_total']} burns")
        self._on_scan_done(scan)

    def _open_burn(self, path, out):
        # Restore the side panel first: _apply_session redraws the preview,
        # which _on_burn_done then replaces with the reopened spectrum.
        if out.get("session"):
            try:
                self._apply_session(out["session"])
            except Exception as exc:                     # noqa: BLE001
                self._append_log(f"! could not restore session: {exc}")
        elif out.get("params"):
            try:
                self._apply_session({"params": out["params"]})
            except Exception as exc:                     # noqa: BLE001
                self._append_log(f"! could not restore params: {exc}")
        self._sim_root = os.path.dirname(path) or self._sim_root
        stages = out.get("stages")
        what = (f"{len(stages)} stage(s)" if stages
                else f"{len(out['pump_freqs'])} pump frequencies"
                if len(out["pump_freqs"]) else "spectrum only")
        self._append_log(f"opened {path}: {what}")
        self._on_burn_done(out)

    def _refresh_export_enabled(self):
        have = self._last_burn is not None or self._scan is not None
        self.exp_btn.setEnabled(have)
        sim_dir = self._scan.get("sim_dir") if self._scan else None
        if sim_dir:
            # A checkpointed scan already has results.csv, best.json,
            # top_spectra.npy and figures/ written into sim_dir as it ran (or
            # completed) - export here is for a differently-named copy
            # elsewhere, not the only way to get the data out.
            self.exp_hint.setText(f"already saved in {sim_dir}")
        else:
            self.exp_hint.setText("" if have else "Run a simulation first.")
        if have:
            self.exp_prefix.setText(self._auto_prefix())

    def _auto_prefix(self):
        df = self.df.value()
        if self._scan is not None:
            return f"df{df:g}_scan_{self._scan['tag']}"
        mode = self.mode_combo.currentIndex()
        if mode == 0:
            return f"df{df:g}_hole_{self.centre.value():g}MHz"
        if mode == 3:
            names = []
            for row in self._sequence_rows:
                label = str(row.get("label", "")).strip().replace(" ", "")
                if label and label not in names:
                    names.append(label)
            return f"df{df:g}_seq_{'-'.join(names) or 'empty'}"
        return (f"df{df:g}_comb_{self.comb_storage.value():g}us_"
                f"F{self.comb_finesse.value():g}")

    def _open_export(self):
        if self._last_burn is None and self._scan is None:
            QMessageBox.information(self, "Export",
                                   "Run a simulation before exporting.")
            return
        start = (self._scan.get("sim_dir") if self._scan else None) or ""
        folder = self._file_dialog("Export to folder", start)
        if not folder:
            return
        try:
            written = self._do_export(folder)
        except Exception as exc:                       # noqa: BLE001
            self._on_error(str(exc))
            return
        self._append_log(f"exported {len(written)} file(s) to {folder}")
        QMessageBox.information(
            self, "Export complete",
            "Wrote:\n" + "\n".join(os.path.basename(w) for w in written))

    def _do_export(self, folder):
        prefix = (self.exp_prefix.text().strip() or "shb_run")
        base = os.path.join(folder, prefix)
        written = []

        def note(path):
            written.append(path)
            self._append_log(f"  wrote {os.path.basename(path)}")

        if self.exp_log.isChecked():
            path = f"{base}_runlog.txt"
            with open(path, "w") as fh:
                fh.write(self.log_view.toPlainText())
            note(path)

        if self.exp_config.isChecked():
            path = f"{base}_config.json"
            with open(path, "w") as fh:
                json.dump(self._collect_session(), fh, indent=2,
                          default=shb._json_default)
            note(path)

        if self.exp_figures.isChecked():
            self.plot.figure.savefig(f"{base}_plot.png", dpi=300)
            note(f"{base}_plot.png")
            if self._scan is not None:
                # One Figure, cleared between views, rather than a fresh one
                # per view: at 300 dpi the Agg backend builds transformed path
                # arrays proportional to n_f for every line, ~80 MB per figure
                # at df 0.001 (and this scan already has its own top-N traces
                # live in self.plot's canvas). Reusing the Figure means only
                # one of those buffers exists at a time, and dropping it with
                # `del` at the end - rather than letting it go out of scope
                # at the end of the loop's next iteration - releases it
                # immediately rather than at the next figure's construction.
                fig = Figure(figsize=(8, 4), constrained_layout=True)
                for name, fn in shb.SCAN_FIGURES:
                    fig.clear()
                    try:
                        fn(fig.add_subplot(111), self._scan)
                    except ValueError as exc:
                        # A scan saved before measured finesse was recorded
                        # has nothing to draw for those two views - skip
                        # them rather than abort the rest of the export.
                        self._append_log(f"  skipped {name}.png: "
                                         f"{' '.join(str(exc).split())}")
                        continue
                    fig.savefig(f"{base}_{name}.png", dpi=300)
                    note(f"{base}_{name}.png")
                del fig
                # And one image per kept spectrum, in its own subfolder -
                # listed once as the folder, not once per image, so the
                # "Export complete" box stays readable at a large top N.
                top_dir = f"{base}_top_spectra"
                n_img = len(shb.save_top_spectrum_figures(
                    self._scan, top_dir, dpi=300))
                if n_img:
                    written.append(top_dir)
                    self._append_log(f"  wrote {n_img} spectrum image(s) to "
                                     f"{os.path.basename(top_dir)}/")

        if self.exp_data.isChecked():
            if self._scan is not None:
                path = f"{base}_results.csv"
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=shb.RESULTS_COLUMNS)
                    w.writeheader()
                    w.writerows(self._scan["results"])
                note(path)
                best = self._scan["best"]
                if best:
                    # Written row by row into a memory-mapped .npy rather than
                    # np.vstack-ed first: vstack would allocate a second
                    # (top_n + 1, n_f) block on top of the spectra already
                    # retained in self._scan - 8 GB on top of 8 GB at the GUI's
                    # top_n maximum of 5000. The saved file is byte-identical;
                    # only the peak while writing it changes.
                    n_f = len(best[0]["f"])
                    mm = np.lib.format.open_memmap(
                        f"{base}_top_spectra.npy", mode="w+",
                        dtype=np.float64, shape=(len(best) + 1, n_f))
                    mm[0] = best[0]["f"]
                    for i, r in enumerate(best, start=1):
                        mm[i] = r["spec"]
                    mm.flush()
                    del mm
                    note(f"{base}_top_spectra.npy")
                if self.exp_pickle.isChecked():
                    with open(f"{base}_scan.pkl", "wb") as fh:
                        pickle.dump(self._scan, fh)
                    note(f"{base}_scan.pkl")
            elif self._last_burn is not None:
                out = self._last_burn
                np.save(f"{base}_spectrum.npy",
                        np.vstack([out["f"], out["spec"]]))
                note(f"{base}_spectrum.npy")
                path = f"{base}_spectrum.csv"
                with open(path, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["frequency_MHz", "absorption"])
                    w.writerows(zip(out["f"].tolist(), out["spec"].tolist()))
                note(path)
                if self.exp_pickle.isChecked():
                    path = f"{base}_burn.pkl"
                    shb.save_burn(
                        path, out,
                        out.get("params") or self._collect_params(),
                        session=(out.get("session")
                                 or self._collect_session()))
                    note(path)

        return written

    def _file_dialog(self, title, start="", *, directory=True, name_filter=""):
        """
        A file / folder picker with a usable size.

        The one-line QFileDialog.getExistingDirectory / getOpenFileName
        helpers take whatever geometry the platform hands back, which on some
        desktops is a dialog too small to read - the folder picker for saving a
        run was unusable. Building the dialog gives it an explicit size.
        """
        dlg = QFileDialog(self, title, start)
        if directory:
            dlg.setFileMode(QFileDialog.FileMode.Directory)
            dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        else:
            dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        if name_filter:
            dlg.setNameFilter(name_filter)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        # Pin the Dropbox root(s) and a couple of project folders into the
        # sidebar, so saving a scan there is a one-click location rather than
        # typing/navigating to it by hand.
        dlg.setSidebarUrls(dlg.sidebarUrls()
                           + [QUrl.fromLocalFile(path)
                              for _, path in _sidebar_folders()])
        dlg.resize(900, 600)
        if not dlg.exec():
            return ""
        chosen = dlg.selectedFiles()
        return chosen[0] if chosen else ""

    def _edit_sequence(self):
        try:
            params = self._collect_params()
            shb.validate_params(params)
        except Exception as exc:                        # noqa: BLE001
            self._on_error(str(exc))
            return
        dlg = PulseSequenceDialog(self._sequence_rows, params, self,
                                  initial_preset=self._last_prep_preset,
                                  initial_field_values=self._last_prep_fields)
        if dlg.exec():
            try:
                self._sequence_rows = dlg.values()
            except ValueError as exc:
                self._on_error(f"bad sequence row: {exc}")
                return
            self._last_prep_preset = dlg.preset_combo.currentText()
            if dlg.preset_combo.currentText() in shb.PREP_PRESETS:
                self._last_prep_fields[self._last_prep_preset] = \
                    dlg.current_field_values()
            self._append_log(f"sequence updated: {len(self._sequence_rows)} rows")
            self._refresh_comb_preview()

    def _edit_levels(self):
        dlg = LevelsDialog(self._crystal, self)
        if dlg.exec():
            try:
                self._crystal = dlg.values()
                self._append_log("crystal levels updated")
                self._mark_norm_stale()
                self._refresh_comb_preview()
            except ValueError as exc:
                self._on_error(f"bad level value: {exc}")

    # -- shutdown --------------------------------------------------

    def closeEvent(self, a0):
        self.worker.cancel()
        self._sim_thread.quit()
        self._sim_thread.wait(2000)
        super().closeEvent(a0)


def main():
    shb.apply_style()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:                                  # noqa: BLE001
        traceback.print_exc()
        raise
