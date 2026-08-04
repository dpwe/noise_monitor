"""PyQtGraph front end: scrolling spectrogram, big dB(A) readout, level history.

The spectrogram is a fixed-size image scrolled in place (np.roll on the column
axis) rather than a growing array, so memory and redraw cost stay constant.
Rows are log-spaced in frequency, which makes the image's own y-axis linear in
"band index"; the axis labels are remapped to the real frequencies.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .config import Config
from .engine import MonitorEngine

pg.setConfigOptions(antialias=False, imageAxisOrder="col-major")


class LogFreqAxis(pg.AxisItem):
    """Labels a band-index axis with the frequencies those bands represent.

    The image's y axis is a band index, so ticks have to be placed at the
    fractional index each round frequency maps to. Labels come from a lookup
    keyed on that index rather than by interpolating back, which would turn
    1000 Hz into "1.00009k".

    Ticks are added in tiers -- decades first, then 5s, then the rest -- and any
    that would collide with one already placed is dropped, so the axis thins out
    gracefully as the window shrinks instead of overprinting.
    """

    #: Preferred frequencies, most important first.
    TICK_TIERS = (
        (10, 100, 1000, 10000),
        (50, 500, 5000, 20000),
        (20, 200, 2000, 20, 2000),
        (31.5, 63, 125, 250, 630, 1250, 2500, 4000, 8000, 16000),
    )
    #: Minimum pixel gap between two labels.
    MIN_TICK_SPACING_PX = 24

    def __init__(self, band_centers: np.ndarray, **kwargs):
        super().__init__(orientation="left", **kwargs)
        self.band_centers = np.asarray(band_centers, dtype=float)
        self._log_centers = np.log(self.band_centers)
        self._indices = np.arange(self.band_centers.size, dtype=float)
        self._labels: dict[float, float] = {}

    def _index_of(self, freq: float) -> float:
        return float(np.interp(np.log(freq), self._log_centers, self._indices))

    def tickValues(self, minVal, maxVal, size):
        lo, hi = self.band_centers[0], self.band_centers[-1]
        if maxVal <= minVal or size <= 0:
            return []
        px_per_unit = size / (maxVal - minVal)

        placed: list[tuple[float, float]] = []  # (index, frequency)
        for tier in self.TICK_TIERS:
            for freq in tier:
                if not (lo <= freq <= hi):
                    continue
                index = self._index_of(freq)
                if not (minVal <= index <= maxVal):
                    continue
                px = (index - minVal) * px_per_unit
                if any(
                    abs(px - (other - minVal) * px_per_unit) < self.MIN_TICK_SPACING_PX
                    for other, _ in placed
                ):
                    continue
                placed.append((index, freq))

        placed.sort()
        self._labels = {round(index, 6): freq for index, freq in placed}
        return [(1.0, [index for index, _ in placed])]

    def tickStrings(self, values, scale, spacing):
        out = []
        for value in values:
            freq = self._labels.get(round(float(value), 6))
            if freq is None:  # pyqtgraph asked about a tick we did not place
                freq = float(np.interp(value, self._indices, self.band_centers))
            out.append(f"{freq / 1000:g}k" if freq >= 1000 else f"{freq:g}")
        return out


class MonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: MonitorEngine, config: Config):
        super().__init__()
        self.engine = engine
        self.config = config
        ui = config.ui

        hops_per_s = config.audio.samplerate / config.analysis.hop
        self.n_cols = max(64, int(ui.history_s * hops_per_s))
        self.n_bands = config.analysis.n_bands
        self.unit = "dB SPL" if engine.calibrated else "dBFS"

        # image[time, freq] with col-major image order.
        self.image_data = np.full((self.n_cols, self.n_bands), ui.db_min, dtype=np.float32)

        self.setWindowTitle("Noise Monitor")
        self._build()
        self._start_timer()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ui = self.config.ui
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.setCentralWidget(central)

        layout.addLayout(self._build_header())

        # --- spectrogram --------------------------------------------------
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, stretch=3)

        self.spec_plot = self.glw.addPlot(
            row=0, col=0, axisItems={"left": LogFreqAxis(self.engine.band_centers)}
        )
        self.spec_plot.setLabel("left", "Frequency", units="Hz")
        self.spec_plot.setLabel("bottom", "Time", units="s")
        self.spec_plot.setMouseEnabled(x=False, y=False)
        self.spec_plot.hideButtons()

        self.image = pg.ImageItem(self.image_data)
        self.spec_plot.addItem(self.image)
        # Scale the x axis so it reads in seconds, with 0 at "now".
        seconds = self.n_cols * self.config.analysis.hop / self.config.audio.samplerate
        self.image.setRect(QtCore.QRectF(-seconds, 0, seconds, self.n_bands))
        self.spec_plot.setXRange(-seconds, 0, padding=0)
        self.spec_plot.setYRange(0, self.n_bands, padding=0)

        colormap = pg.colormap.get(ui.colormap)
        self.bar = pg.ColorBarItem(
            values=(ui.db_min, ui.db_max), colorMap=colormap, label=f"Band {self.unit}"
        )
        self.bar.setImageItem(self.image, insert_in=self.spec_plot)

        # --- level history -------------------------------------------------
        self.level_plot = self.glw.addPlot(row=1, col=0)
        self.level_plot.setLabel("left", f"L{self.config.analysis.weighting}F", units=self.unit)
        self.level_plot.setLabel("bottom", "Time", units="s")
        self.level_plot.showGrid(x=True, y=True, alpha=0.3)
        self.level_plot.setMouseEnabled(x=False, y=True)
        self.level_plot.setYRange(ui.db_min, ui.db_max, padding=0)
        self.level_plot.setXRange(-ui.level_history_s, 0, padding=0)
        self.level_curve = self.level_plot.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)

        layout.addWidget(self._build_status())

        if ui.fullscreen:
            self.showFullScreen()
        else:
            self.resize(1200, 800)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()

        self.level_label = QtWidgets.QLabel("--.-")
        font = QtGui.QFont()
        font.setPointSize(64)
        font.setBold(True)
        self.level_label.setFont(font)
        self.level_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.level_label)

        weighting = self.config.analysis.weighting.upper()
        # Uncalibrated, the number is a full-scale ratio, not a sound pressure.
        suffix = f"dB{weighting}" if self.engine.calibrated else f"dBFS({weighting})"
        self.unit_label = QtWidgets.QLabel(suffix)
        unit_font = QtGui.QFont()
        unit_font.setPointSize(22)
        self.unit_label.setFont(unit_font)
        self.unit_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        row.addWidget(self.unit_label)
        row.addStretch(1)

        self.stats_label = QtWidgets.QLabel("")
        stats_font = QtGui.QFont()
        stats_font.setPointSize(13)
        stats_font.setFamily("monospace")
        self.stats_label.setFont(stats_font)
        self.stats_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.stats_label)

        self.clip_label = QtWidgets.QLabel("CLIP")
        clip_font = QtGui.QFont()
        clip_font.setPointSize(16)
        clip_font.setBold(True)
        self.clip_label.setFont(clip_font)
        self.clip_label.setStyleSheet("color: #b00020;")
        self.clip_label.setVisible(False)
        row.addWidget(self.clip_label)
        return row

    def _build_status(self) -> QtWidgets.QLabel:
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: #888;")
        note = self.engine.calibration_note
        text = f"{self.engine.source.description}  |  calibration: {note}"
        if not self.engine.calibrated:
            text += "  |  showing dBFS, NOT absolute SPL"
            self.status_label.setStyleSheet("color: #b06000; font-weight: bold;")
        self.status_label.setText(text)
        return self.status_label

    # ------------------------------------------------------------------
    def _start_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(int(1000 / self.config.ui.refresh_hz))

    def _refresh(self) -> None:
        columns, state = self.engine.drain()

        if columns:
            new = np.asarray(columns, dtype=np.float32)  # (n_new, n_bands)
            if new.shape[0] >= self.n_cols:
                self.image_data[:] = new[-self.n_cols :]
            else:
                k = new.shape[0]
                self.image_data[:-k] = self.image_data[k:]
                self.image_data[-k:] = new
            self.image.setImage(
                self.image_data,
                autoLevels=False,
                levels=(self.config.ui.db_min, self.config.ui.db_max),
            )

        if np.isfinite(state.level_db):
            self.level_label.setText(f"{state.level_db:.1f}")

        if state.recent_levels:
            times = np.array([t for t, _ in state.recent_levels])
            levels = np.array([v for _, v in state.recent_levels])
            self.level_curve.setData(times - times[-1], levels)

        self.clip_label.setVisible(state.clip_hold)
        self._update_stats(state)

        if state.error:
            self.status_label.setText(f"ERROR: {state.error}")
            self.status_label.setStyleSheet("color: #b00020; font-weight: bold;")
            self.timer.stop()

    def _update_stats(self, state) -> None:
        w = self.config.analysis.weighting.upper()
        lines = []
        if np.isfinite(state.leq_1s):
            lines.append(f"L{w}eq,1s  {state.leq_1s:6.1f}")
        iv = state.last_interval
        if iv is not None:
            lines.append(f"L{w}eq,{iv.duration_s:g}s {iv.leq:6.1f}")
            lines.append(f"L{w}max    {iv.lmax:6.1f}")
            lines.append(f"L{w}90     {iv.l90:6.1f}")
        if state.dropped_blocks or state.overflows:
            lines.append(f"dropped {state.dropped_blocks} / xrun {state.overflows}")
        self.stats_label.setText("\n".join(lines))

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
        elif key == QtCore.Qt.Key.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(event)


def run_ui(engine: MonitorEngine, config: Config) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MonitorWindow(engine, config)
    window.show()
    engine.start()
    return app.exec()
