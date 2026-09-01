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

import csv
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qt import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QAction  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QCheckBox, QPushButton, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QStackedWidget, QProgressBar, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QScrollArea, QSizePolicy)

import shb


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
                                estimate=self.estimate.emit, **kwargs)
            best = scan["best"]
            if best:
                b = best[0]
                self.log.emit(
                    f"[{scan['tag']}] best: AFC comb quality {b['quality']:.3f} at "
                    f"t_s {b['storage_us']:.3g} us, F {b['finesse']:.1f} "
                    f"(spacing {b['spacing_MHz']:.3f} MHz, "
                    f"width {b['width_MHz']:.3f} MHz)")
            self.scan_done.emit(scan)
        except _Cancelled:
            self.log.emit("scan cancelled")
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.busy.emit(False)


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
        lo, hi = np.inf, -np.inf
        for ln in self._top_lines:
            if not ln.get_visible():
                continue
            xd = np.asarray(ln.get_xdata(), dtype=float)
            yd = np.asarray(ln.get_ydata(), dtype=float)
            m = (xd >= x0) & (xd <= x1)
            if m.any():
                lo = min(lo, float(np.nanmin(yd[m])))
                hi = max(hi, float(np.nanmax(yd[m])))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        self.canvas.draw_idle()

    # -- drawing --------------------------------------------------------

    def _stacked_axes(self):
        self._clear_legend_panel()
        self._legend_panel.hide()
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.08)
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

    def show_preview(self, pump_freqs, params):
        ax_top, ax_bot = self._stacked_axes()
        shb.plot_comb(ax_top, pump_freqs, params, show_xlabel=False)
        ax_bot.text(0.5, 0.5, "run to see the spectrum", ha="center",
                    va="center", color="0.5", transform=ax_bot.transAxes)
        ax_bot.set_xlabel("Frequency (MHz)")
        ax_bot.set_ylabel("Absorption (Normalised)")
        p = shb.settings(params)
        ax_top.set_xlim(p["fmin"], p["fmax"])
        self._finish()

    def show_burn(self, pump_freqs, params, out):
        ax_top, ax_bot = self._stacked_axes()
        shb.plot_comb(ax_top, pump_freqs, params, show_xlabel=False)
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
        self.dm_table = QTableWidget(dm.shape[0], dm.shape[1])
        self.dm_table.setHorizontalHeaderLabels(
            [f"e{m}" for m in range(dm.shape[1])])
        self.dm_table.setVerticalHeaderLabels(
            [f"g{k}" for k in range(dm.shape[0])])
        for k in range(dm.shape[0]):
            for m in range(dm.shape[1]):
                self.dm_table.setItem(k, m, QTableWidgetItem(f"{dm[k, m]:g}"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Ground levels g"))
        lay.addWidget(self.g_table)
        lay.addWidget(QLabel("Excited levels e"))
        lay.addWidget(self.e_table)
        lay.addWidget(QLabel("Branching / relative dipole matrix dm"))
        lay.addWidget(self.dm_table)
        lay.addWidget(buttons)

    @staticmethod
    def _vector_table(values):
        t = QTableWidget(1, len(values))
        t.setVerticalHeaderLabels([""])
        for i, v in enumerate(values):
            t.setItem(0, i, QTableWidgetItem(f"{v:g}"))
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


def _dspin(lo, hi, step, decimals, value):
    s = QDoubleSpinBox()
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


# ===========================================================================
# Main window
# ===========================================================================

class MainWindow(QMainWindow):

    request_norm = pyqtSignal(dict)
    request_burn = pyqtSignal(dict, object, str, int, str)
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

        self._job_t0 = 0.0
        self._job_estimate = None       # seconds, from run_scan's probe
        self._job_kind = None           # "burn" | "scan" | "norm" | None

        self._build_menu()
        self._build_body()
        self._build_worker()

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
        self.left_tabs.addTab(sim_tab, "Simulation")
        self.left_tabs.addTab(self._build_export_tab(), "Export")
        self.left_tabs.setFixedWidth(360)
        root.addWidget(self.left_tabs)

        # ---- right column ---------------------------------------------
        right = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Scan view:"))
        self.scan_view = QComboBox()
        self.scan_view.addItems(["Comb-quality heatmap",
                                 "Quality vs storage time (best/mean/worst)",
                                 "Top spectra"])
        self.scan_view.setEnabled(False)
        self.scan_view.currentIndexChanged.connect(self._draw_feature)
        row.addWidget(self.scan_view)
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

        self.fmin = _dspin(-20000, 0, 1.0, 3, -50.0)
        self.fmax = _dspin(0, 20000, 1.0, 3, 50.0)
        self.df = _dspin(0.001, 10.0, 0.001, 3, 0.01)
        self.df.setToolTip("Mesh step. Must divide the level energies and the "
                           "window edges exactly (0.01, 0.005, 0.001 for the "
                           "default crystal).")
        self.h_width = _dspin(0.0001, 100.0, 0.001, 4, 0.001)
        self.h_width.setToolTip("Homogeneous FWHM (Lorentzian), MHz.")
        self.in_width = _dspin(1.0, 1e6, 100.0, 1, 10000.0)
        self.in_width.setToolTip("Inhomogeneous FWHM (Gaussian), MHz.")
        self.pump_passes = _ispin(1, 100000, 10)
        self.burn_mode = QComboBox()
        self.burn_mode.addItems(["parallel", "series"])
        self.alpha = _dspin(0.0, 1e6, 0.1, 3, 1.0)

        for w in (self.fmin, self.fmax, self.df, self.h_width, self.in_width):
            w.valueChanged.connect(self._on_params_changed)
        for w in (self.df, self.fmin, self.fmax):
            w.valueChanged.connect(self._refresh_comb_preview)

        form.addRow("fmin (MHz)", self.fmin)
        form.addRow("fmax (MHz)", self.fmax)
        form.addRow("df (MHz)", self.df)
        form.addRow("h_width (MHz)", self.h_width)
        form.addRow("in_width (MHz)", self.in_width)
        form.addRow("pump passes", self.pump_passes)
        form.addRow("burn mode", self.burn_mode)
        form.addRow("alpha", self.alpha)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single hole", "One comb", "Scan"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("mode", self.mode_combo)
        outer.addLayout(form)

        # -- mode-specific stack ---------------------------------------
        self.stack = QStackedWidget()

        # single hole
        sh = QWidget()
        sh_form = QFormLayout(sh)
        self.centre = _dspin(-10000, 10000, 1.0, 3, 0.0)
        self.centre.valueChanged.connect(self._refresh_comb_preview)
        sh_form.addRow("centre (MHz)", self.centre)
        self.stack.addWidget(sh)

        # one comb
        oc = QWidget()
        oc_form = QFormLayout(oc)
        self.tooth_spacing = _dspin(0.001, 1000, 0.01, 3, 2.35)
        self.burn_width = _dspin(0.001, 1000, 0.01, 3, 1.18)
        self.tooth_spacing.valueChanged.connect(self._refresh_comb_preview)
        self.burn_width.valueChanged.connect(self._refresh_comb_preview)
        self.comb_info = QLabel("-")
        self.comb_info.setWordWrap(True)
        oc_form.addRow("tooth spacing (MHz)", self.tooth_spacing)
        oc_form.addRow("burn width (MHz)", self.burn_width)
        oc_form.addRow("on mesh", self.comb_info)
        self.stack.addWidget(oc)

        # scan
        sc = QWidget()
        sc_form = QFormLayout(sc)
        self.storage_min = _dspin(0.001, 100000, 0.5, 3, 2.0)
        self.storage_max = _dspin(0.001, 100000, 0.5, 3, 10.0)
        self.n_points = _ispin(2, 1000000, 40)
        self.finesse_min = _dspin(1.0, 1000, 0.5, 2, 2.0)
        self.finesse_max = _dspin(1.0, 1000, 0.5, 2, 20.0)
        self.n_finesse = _ispin(1, 1000000, 20)
        self.target_window = _dspin(0.1, 10000, 1.0, 3, 20.0)
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
        sc_form.addRow("keep top N", self.top_n)
        sc_form.addRow(self.parallel_check)
        sc_form.addRow(f"workers (of {_cpu})", self.workers)
        self.stack.addWidget(sc)

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
        for cb in (self.exp_log, self.exp_config, self.exp_figures,
                   self.exp_data):
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
                "h_width": self.h_width.value(),
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
                  self.h_width: pr["linewidth_params"]["h_width"],
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

    def _on_mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        self._refresh_comb_preview()

    def _current_pump_freqs(self, params):
        mode = self.mode_combo.currentIndex()
        if mode == 0:
            return shb.single_hole(self.centre.value()), "single hole"
        if mode == 1:
            freqs, sp, wd, nt = shb.comb_at(
                self.tooth_spacing.value(), self.burn_width.value(), params)
            self.comb_info.setText(
                f"spacing {sp:g} MHz, width {wd:g} MHz "
                f"({wd / sp:.0%}), {nt} teeth")
            return freqs, f"comb: {sp:g} MHz spacing, {wd:g} MHz width, {nt} teeth"
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
                       f"{sp:g} MHz, F {sp / wd:.1f}")

    def _refresh_comb_preview(self, *_):
        """Live pump-comb preview; the plot returns to preview when inputs change."""
        try:
            params = self._collect_params()
            shb.validate_params(params)
            freqs, _ = self._current_pump_freqs(params)
        except Exception as exc:                       # noqa: BLE001
            self.plot.show_message(str(exc))
            return
        self.plot.show_preview(freqs, params)

    # -- run ----------------------------------------------------------

    def _on_run_clicked(self):
        if self.run_btn.text() == "Cancel":
            self.worker.cancel()
            return
        try:
            params = self._collect_params()
            shb.validate_params(params)
        except Exception as exc:                       # noqa: BLE001
            self._on_error(str(exc))
            return

        self._job_t0 = time.monotonic()
        self._job_estimate = None

        if self.mode_combo.currentIndex() == 2:
            kwargs = dict(
                storage_min=self.storage_min.value(),
                storage_max=self.storage_max.value(),
                n_points=self.n_points.value(),
                finesse_min=self.finesse_min.value(),
                finesse_max=self.finesse_max.value(),
                n_finesse=self.n_finesse.value(),
                target_window=self.target_window.value(),
                top_n=self.top_n.value(),
                pump_passes=self.pump_passes.value(),
                burn=(shb.burn_series if self.burn_mode.currentText() == "series"
                      else shb.burn_parallel),
                parallel=self.parallel_check.isChecked(),
                n_workers=self.workers.value(),
            )
            self._job_kind = "scan"
            self._append_log("--- scan ---")
            self.request_scan.emit(params, kwargs)
        else:
            freqs, summary = self._current_pump_freqs(params)
            self._job_kind = "burn"
            self._append_log("--- burn ---")
            self.request_burn.emit(params, np.asarray(freqs),
                                   self.burn_mode.currentText(),
                                   self.pump_passes.value(), summary)

    # -- worker callbacks -------------------------------------------

    def _on_busy(self, busy):
        self.run_btn.setText("Cancel" if busy else "Run")
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
        if done:
            eta = elapsed / done * (total - done)
            self.progress.setFormat(
                f"%p%  ·  {done}/{total}  ·  ~{fmt_dur(eta)} left")
        else:
            self.progress.setFormat(f"0/{total}")

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
        self._last_burn = out
        self._scan = None
        self.scan_view.setEnabled(False)
        self._draw_feature()
        self._refresh_export_enabled()

    def _on_scan_done(self, scan):
        self._scan = scan
        self._last_burn = None
        self.scan_view.setEnabled(True)
        self._draw_feature()
        self._refresh_export_enabled()

    def _on_error(self, msg):
        self._append_log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Simulation error", msg)

    def _append_log(self, line):
        stamp = time.strftime("%H:%M:%S")
        if self._job_kind is not None and self._job_t0:
            el = max(0.0, time.monotonic() - self._job_t0)
            pre = f"{stamp} +{int(el) // 60:02d}:{int(el) % 60:02d}  "
        else:
            pre = f"{stamp}  "
        text = "" if line is None else str(line)
        for phys in text.split("\n"):
            self.log_view.appendPlainText(pre + phys.rstrip())

    # -- plot --------------------------------------------------

    def _draw_feature(self, *_):
        if self._scan is not None:
            self.plot.show_scan(self._scan, self.scan_view.currentIndex())
        elif self._last_burn is not None:
            self.plot.show_burn(self._last_burn["pump_freqs"],
                                self._collect_params(), self._last_burn)
        else:
            self._refresh_comb_preview()

    # -- session import / export --------------------------------

    _GUI_FIELDS = ("centre", "tooth_spacing", "burn_width", "storage_min",
                   "storage_max", "finesse_min", "finesse_max", "target_window")
    _GUI_INTS = ("n_points", "n_finesse", "top_n", "workers")

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
        self.h_width.setValue(p["h_width"])
        self.in_width.setValue(p["in_width"])
        self.pump_passes.setValue(int(p["pump_passes"]))
        self.alpha.setValue(p["alpha"])

        if "relative_dipole" in gui:
            self.dipole_check.setChecked(bool(gui["relative_dipole"]))
        if gui.get("burn_mode") in ("parallel", "series"):
            self.burn_mode.setCurrentText(gui["burn_mode"])
        if "parallel" in gui:
            self.parallel_check.setChecked(bool(gui["parallel"]))
        for name in self._GUI_FIELDS:
            if name in gui:
                getattr(self, name).setValue(float(gui[name]))
        for name in self._GUI_INTS:
            if name in gui:
                getattr(self, name).setValue(int(gui[name]))
        if gui.get("mode") in ("Single hole", "One comb", "Scan"):
            self.mode_combo.setCurrentText(gui["mode"])

        self._mark_norm_stale()
        self._refresh_comb_preview()

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import config", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self._apply_session(data)
            self._append_log(f"imported config from {path}")
        except Exception as exc:                       # noqa: BLE001
            self._on_error(str(exc))

    def _refresh_export_enabled(self):
        have = self._last_burn is not None or self._scan is not None
        self.exp_btn.setEnabled(have)
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
        return (f"df{df:g}_comb_{self.tooth_spacing.value():g}MHz_"
                f"{self.burn_width.value():g}MHz")

    def _open_export(self):
        if self._last_burn is None and self._scan is None:
            QMessageBox.information(self, "Export",
                                   "Run a simulation before exporting.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export to folder")
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
                for name, fn in (("heatmap", shb.plot_scan_heatmap),
                                 ("quality_vs_storage", shb.plot_scan_max),
                                 ("topspectra", shb.plot_top_spectra)):
                    fig = Figure(figsize=(8, 4), constrained_layout=True)
                    fn(fig.add_subplot(111), self._scan)
                    fig.savefig(f"{base}_{name}.png", dpi=300)
                    note(f"{base}_{name}.png")

        if self.exp_data.isChecked():
            if self._scan is not None:
                path = f"{base}_results.csv"
                cols = ["spacing_MHz", "width_MHz", "finesse",
                        "storage_us", "quality"]
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(self._scan["results"])
                note(path)
                best = self._scan["best"]
                if best:
                    stack = np.vstack([best[0]["f"]]
                                      + [r["spec"] for r in best])
                    np.save(f"{base}_top_spectra.npy", stack)
                    note(f"{base}_top_spectra.npy")
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

        return written

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
