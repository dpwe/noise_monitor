"""PyQtGraph front end: two spectrograms, one live and one averaged over a day.

They get half the window each. The top panel is live -- tens of seconds, one
column per FFT hop. The bottom one covers the past day at one column per
`ui.long_column_s`, with the dB(A) Leq over the same windows drawn on top of
it. Both are fixed-size images scrolled in place rather than growing arrays, so
memory and redraw cost stay constant however long the run is.

The readout and the statistics are drawn *over* the live panel rather than in a
strip above it, so nothing is spent on background. Rows are log-spaced in
frequency, which makes the images' own y-axis linear in "band index"; the axis
labels are remapped to the real frequencies.
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


def _duration_label(seconds: float) -> str:
    """A compact name for an averaging window: '1s', '3min', '24h'.

    Deliberately imprecise: a column is a whole number of FFT hops, so a
    nominal 3 minutes is really 180.011 s, and nobody wants to read that.
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.3g}h"
    if seconds >= 60:
        return f"{seconds / 60:.3g}min"
    return f"{seconds:.3g}s"


def _grid_levels(low: float, high: float, step: float = 10.0) -> list[float]:
    """Round dB values strictly inside (low, high), for the overlay's grid."""
    first = np.ceil(low / step) * step
    return [float(v) for v in np.arange(first, high, step) if low < v < high]


class MonitorWindow(QtWidgets.QMainWindow):
    #: Relative heights of the two panels: live spectrogram, day-long average.
    ROW_STRETCH = (1, 1)
    #: Gap between the overlaid readout and the edge of the live image.
    OVERLAY_MARGIN_PX = 10

    def __init__(self, engine: MonitorEngine, config: Config):
        super().__init__()
        self.engine = engine
        self.config = config
        ui = config.ui

        hops_per_s = config.audio.samplerate / config.analysis.hop
        self.n_cols = max(64, int(ui.history_s * hops_per_s))
        self.n_bands = config.analysis.n_bands
        self.unit = "dB SPL" if engine.calibrated else "dBFS"
        self._level_number = ""
        self._stats_text = ""
        self.weighting = config.analysis.weighting.upper()

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

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, stretch=1)

        self._build_live_spectrogram()
        self._build_long_term()
        self._build_colorbar()
        for row, stretch in enumerate(self.ROW_STRETCH):
            self.glw.ci.layout.setRowStretchFactor(row, stretch)
        self._build_overlay()

        layout.addWidget(self._build_status())

        if ui.fullscreen:
            self.showFullScreen()
        else:
            self.resize(1200, 900)

    def _build_live_spectrogram(self) -> None:
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

    def _build_colorbar(self) -> None:
        """One colour bar spanning both rows, driving both images.

        Sharing it keeps the two plots the same width -- a bar on one row only
        would push that panel in and misalign the two time axes -- and makes it
        obvious that the panels are on the same scale.
        """
        ui = self.config.ui
        self.bar = pg.ColorBarItem(
            values=(ui.db_min, ui.db_max),
            colorMap=pg.colormap.get(ui.colormap),
            label=f"Band {self.unit}",
        )
        self.bar.setImageItem([self.image, self.long_image])
        self.glw.addItem(self.bar, row=0, col=1, rowspan=len(self.ROW_STRETCH))

    def _build_long_term(self) -> None:
        """The day-long panel: an averaged spectrogram with the Leq on top.

        The level gets its own ViewBox rather than sharing the plot's, so it
        can keep a narrow dB scale (`ui.level_min`..`level_max`) while the
        image underneath keeps the 90 dB colour scale. The two are linked in x
        and their geometries are kept in step by `_sync_long_overlay`.
        """
        ui = self.config.ui
        snapshot = self.engine.long_term()

        plot = self.glw.addPlot(
            row=1, col=0, axisItems={"left": LogFreqAxis(self.engine.band_centers)}
        )
        self.long_plot = plot
        plot.setLabel("left", "Frequency", units="Hz")
        # Hours are not an SI unit; without this the axis reads "mh" whenever
        # the span is short enough for pyqtgraph to reach for a prefix.
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Time (hours ago)")
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()

        self.long_image = pg.ImageItem(snapshot.image)
        plot.addItem(self.long_image)
        hours = snapshot.span_s / 3600.0
        self.long_image.setRect(QtCore.QRectF(-hours, 0, hours, self.n_bands))
        plot.setXRange(-hours, 0, padding=0)
        plot.setYRange(0, self.n_bands, padding=0)

        self.long_level_vb = pg.ViewBox(enableMouse=False)
        # A sibling of the PlotItem in the scene, so it needs lifting explicitly
        # or the image paints over the trace.
        self.long_level_vb.setZValue(plot.zValue() + 100)
        plot.showAxis("right")
        plot.scene().addItem(self.long_level_vb)
        right_axis = plot.getAxis("right")
        right_axis.linkToView(self.long_level_vb)
        right_axis.setLabel(
            f"L{self.weighting}eq,{_duration_label(snapshot.column_s)}", units=self.unit
        )
        self.long_level_vb.setXLink(plot)
        self.long_level_vb.setYRange(ui.level_min, ui.level_max, padding=0)

        dashed = pg.mkPen("#ffffff", width=1, style=QtCore.Qt.PenStyle.DashLine)
        dashed.setColor(QtGui.QColor(255, 255, 255, 90))
        for db in _grid_levels(ui.level_min, ui.level_max):
            self.long_level_vb.addItem(
                pg.InfiniteLine(pos=db, angle=0, pen=dashed), ignoreBounds=True
            )

        # Red: viridis contains none of it, so the trace never disappears into
        # the image whatever the level underneath.
        self.long_level_curve = pg.PlotDataItem(pen=pg.mkPen("#ff3b30", width=2))
        self.long_level_vb.addItem(self.long_level_curve)

        plot.vb.sigResized.connect(self._sync_long_overlay)
        self._sync_long_overlay()

    def _sync_long_overlay(self) -> None:
        """Hold the level overlay exactly on top of the long-term image."""
        self.long_level_vb.setGeometry(self.long_plot.vb.sceneBoundingRect())
        self.long_level_vb.linkedViewChanged(self.long_plot.vb, self.long_level_vb.XAxis)

    def _build_overlay(self) -> None:
        """The readout, drawn over the live spectrogram instead of above it.

        Plain QLabels parented to the graphics widget, not TextItems: a
        TextItem is laid out in scene units and would change size with the
        view, and these need to stay put at a fixed point size. Each carries a
        translucent backing so it stays legible over a bright band.
        """
        # Number and unit share one label -- and so one backing box -- because
        # rich text lines them up on a baseline better than two boxes can.
        self.level_label = self._overlay_label()
        self._set_level_text("--.-")

        self.stats_label = self._overlay_label(family="monospace", point_size=13)
        self.stats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.stats_label.setVisible(False)

        self.clip_label = self._overlay_label(point_size=15, color="#ff5252")
        self.clip_label.setText("CLIP")
        self.clip_label.setVisible(False)

        # Follow the image, not the window: the plot area moves when the axis
        # labels change width.
        self.spec_plot.vb.sigResized.connect(self._place_overlay)
        self._place_overlay()

    def _set_level_text(self, number: str) -> None:
        suffix = f"dB{self.weighting}"
        if not self.engine.calibrated:
            # Uncalibrated, the number is a full-scale ratio, not a pressure.
            suffix = f"dBFS({self.weighting})"
        note = ""
        if self.engine.average_s > 0:
            note = (
                f'<br><span style="font-size:11pt; color:#cfcfcf">'
                f"{_duration_label(self.engine.average_s)} average</span>"
            )
        self.level_label.setText(
            f'<span style="font-size:44pt; font-weight:bold">{number}</span>'
            f'<span style="font-size:15pt">&nbsp;{suffix}</span>{note}'
        )

    def _overlay_label(
        self, point_size: int | None = None, family: str | None = None,
        color: str = "#ffffff",
    ) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(self.glw)
        if point_size is not None or family is not None:
            font = QtGui.QFont()
            if point_size is not None:
                font.setPointSize(point_size)
            if family is not None:
                font.setFamily(family)
            label.setFont(font)
        label.setStyleSheet(
            f"color: {color}; background-color: rgba(0, 0, 0, 120);"
            "border-radius: 4px; padding: 2px 8px;"
        )
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.raise_()
        return label

    def _place_overlay(self) -> None:
        """Pin the overlay labels to the corners of the live image."""
        area = self.glw.mapFromScene(self.spec_plot.vb.sceneBoundingRect()).boundingRect()
        margin = self.OVERLAY_MARGIN_PX
        left, top = area.left() + margin, area.top() + margin

        for label in (self.level_label, self.stats_label, self.clip_label):
            label.adjustSize()
        self.level_label.move(left, top)
        self.clip_label.move(left, top + self.level_label.height() + 4)
        self.stats_label.move(area.right() - self.stats_label.width() - margin, top)

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

        # The long-term panel moves once per averaging window; redrawing it at
        # the live rate would copy a day of history 30 times a second for
        # nothing.
        self.long_timer = QtCore.QTimer(self)
        self.long_timer.timeout.connect(self._refresh_long_term)
        self.long_timer.start(max(100, int(1000 * self.config.ui.long_refresh_s)))
        self._refresh_long_term()

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

        level = state.leq_avg if np.isfinite(state.leq_avg) else state.level_db
        if np.isfinite(level):
            number = f"{level:.1f}"
            if number != self._level_number:
                self._level_number = number
                self._set_level_text(number)
                self._place_overlay()  # the box grows and shrinks with the digits

        self.clip_label.setVisible(state.clip_hold)
        self._update_stats(state)

        if state.error:
            self.status_label.setText(f"ERROR: {state.error}")
            self.status_label.setStyleSheet("color: #b00020; font-weight: bold;")
            self.timer.stop()
            self.long_timer.stop()

    def _refresh_long_term(self) -> None:
        ui = self.config.ui
        snapshot = self.engine.long_term()
        self.long_image.setImage(
            snapshot.image, autoLevels=False, levels=(ui.db_min, ui.db_max)
        )
        if snapshot.n_valid:
            hours_ago = snapshot.ages_s() / 3600.0
            self.long_level_curve.setData(-hours_ago, snapshot.levels[-snapshot.n_valid :])

    def _update_stats(self, state) -> None:
        w = self.weighting
        lines = []
        iv = state.last_interval
        if iv is not None:
            lines.append(f"L{w}eq,{iv.duration_s:g}s {iv.leq:6.1f}")
            lines.append(f"L{w}max    {iv.lmax:6.1f}")
            lines.append(f"L{w}90     {iv.l90:6.1f}")
        if state.dropped_blocks or state.overflows:
            lines.append(f"dropped {state.dropped_blocks} / xrun {state.overflows}")
        text = "\n".join(lines)
        if text != self._stats_text:
            self._stats_text = text
            self.stats_label.setText(text)
            self.stats_label.setVisible(bool(text))
            self._place_overlay()  # right-anchored, so its width matters

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
        self.long_timer.stop()
        self.engine.stop()
        super().closeEvent(event)


def run_ui(engine: MonitorEngine, config: Config) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MonitorWindow(engine, config)
    window.show()
    engine.start()
    return app.exec()
