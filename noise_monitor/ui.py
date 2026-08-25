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

import time
from pathlib import Path

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


class ClockAxis(pg.AxisItem):
    """Labels an "hours before now" axis with the local 24 hour clock.

    The panel scrolls, so the mapping from position to clock time moves with
    it; `set_now` is called on every redraw to keep it current.

    Ticks are placed on round local hours rather than at round offsets from
    now, so they stay put between redraws instead of sliding. Each tick's
    instant is a fixed number of seconds before now, and its label is whatever
    the local clock read then -- so an autumn DST change prints 01:00 twice,
    which is what the clock actually did.
    """

    #: Candidate tick spacings in minutes, finest first. Each divides an hour
    #: or a day, so ticks land on the same clock times every cycle.
    TICK_MINUTES = (1, 5, 15, 30, 60, 120, 180, 360, 720)
    #: Minimum pixel gap between two labels.
    MIN_TICK_SPACING_PX = 70
    #: Room for the tick marks and their offset, on top of the two text lines.
    LABEL_PADDING_PX = 14

    def __init__(self, **kwargs):
        super().__init__(orientation="bottom", **kwargs)
        self.enableAutoSIPrefix(False)
        self._now = time.time()

    def reserve_date_line(self) -> None:
        """Claim height for the second line of the dated labels.

        The axis sizes itself from one line of tick text, so the date
        underneath is drawn past its bottom edge and clipped. Ask for the room
        up front instead.
        """
        line = QtGui.QFontMetrics(self.font()).height()
        self.setHeight(2 * line + self.LABEL_PADDING_PX)

    def set_now(self, now: float) -> None:
        self._now = float(now)
        self.picture = None  # the labels changed; force a repaint
        self.update()

    def tickValues(self, minVal, maxVal, size):
        if maxVal <= minVal or size <= 0:
            return []
        px_per_hour = size / (maxVal - minVal)
        step_min = next(
            (
                m for m in self.TICK_MINUTES
                if (m / 60.0) * px_per_hour >= self.MIN_TICK_SPACING_PX
            ),
            self.TICK_MINUTES[-1],
        )
        step_hours = step_min / 60.0

        # The most recent instant whose local clock is a round multiple of the
        # step since local midnight. Sub-hour steps matter: a `--long-span` of
        # an hour or less would otherwise get no ticks at all. Reading the
        # fields off localtime rather than flooring the timestamp keeps the
        # half-hour time zones honest.
        local = time.localtime(self._now)
        since_midnight = local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec
        back = since_midnight % (step_min * 60)

        values = []
        position = -back / 3600.0
        while position >= minVal:
            if position <= maxVal:
                values.append(position)
            position -= step_hours
        values.reverse()
        if not values:
            # A span shorter than the finest step -- no round clock time falls
            # inside it. Label the middle rather than handing back a bare axis:
            # a tick on the boundary gets culled for overflowing it.
            values = [(minVal + maxVal) / 2.0]
        return [(step_hours, values)]

    def tickStrings(self, values, scale, spacing):
        """Clock times, with the date on the tick that opens a new day.

        Over a day that is the midnight tick, and one date is enough: what is
        right of the mark is that date, what is left is the day before. A span
        that never crosses midnight has no such tick, so the oldest one is
        dated instead -- otherwise the date would never appear at all.
        """
        when = [time.localtime(self._now + float(v) * 3600.0) for v in values]
        times = [time.strftime("%H:%M", w) for w in when]
        dates = [time.strftime("%Y-%m-%d", w) for w in when]

        opens = {i for i in range(1, len(dates)) if dates[i] != dates[i - 1]}
        if not opens and dates:
            opens = {0}
        return [
            f"{clock}\n{dates[i]}" if i in opens else clock
            for i, clock in enumerate(times)
        ]


class MonitorWindow(QtWidgets.QMainWindow):
    #: Relative heights of the two panels: live spectrogram, day-long average.
    ROW_STRETCH = (1, 1)
    #: Gap between the overlaid readout and the edge of the live image.
    OVERLAY_MARGIN_PX = 10
    #: Point sizes in the overlaid readout: number, unit, averaging note.
    READOUT_POINT_SIZES = (22, 8, 7)
    #: How long a transient status message (a saved screenshot) stays up.
    STATUS_FLASH_MS = 5000
    #: Tallest the colour bar gets. It is a legend, not a plot; at full window
    #: height it dominated a column it shares with nothing else.
    COLORBAR_MAX_HEIGHT_PX = 240

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
        self._status_extra = ""
        self._status_style = "color: #888;"
        self._status_flashing = False
        self._fatal = False
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
        self.bar.setMaximumHeight(self.COLORBAR_MAX_HEIGHT_PX)
        self.glw.addItem(self.bar, row=0, col=1, rowspan=len(self.ROW_STRETCH))
        # Capped and top-aligned, so the space below it is free for the button.
        self.glw.ci.layout.setAlignment(self.bar, QtCore.Qt.AlignmentFlag.AlignTop)

    def _build_long_term(self) -> None:
        """The day-long panel: an averaged spectrogram with the Leq on top.

        The level gets its own ViewBox rather than sharing the plot's, so it
        can keep a narrow dB scale (`ui.level_min`..`level_max`) while the
        image underneath keeps the 90 dB colour scale. The two are linked in x
        and their geometries are kept in step by `_sync_long_overlay`.
        """
        ui = self.config.ui
        snapshot = self.engine.long_term()

        self.clock_axis = ClockAxis()
        plot = self.glw.addPlot(
            row=1,
            col=0,
            axisItems={
                "left": LogFreqAxis(self.engine.band_centers),
                "bottom": self.clock_axis,
            },
        )
        self.long_plot = plot
        plot.setLabel("left", "Frequency", units="Hz")
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        self.clock_axis.reserve_date_line()

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

        self.clip_label = self._overlay_label(point_size=10, color="#ff5252")
        self.clip_label.setText("CLIP")
        self.clip_label.setVisible(False)

        self.shot_button = self._build_shot_button()

        # Follow the image, not the window: the plot area moves when the axis
        # labels change width.
        self.spec_plot.vb.sigResized.connect(self._place_overlay)
        self._place_overlay()

    def _build_shot_button(self) -> QtWidgets.QPushButton:
        """A mouse-reachable twin of the S key, tucked under the colour bar."""
        button = QtWidgets.QPushButton("Shot", self.glw)
        font = QtGui.QFont()
        font.setPointSize(10)
        button.setFont(font)
        button.setToolTip("Save a PNG of the window (S)")
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # keep S, F, Q live
        button.setStyleSheet(
            "QPushButton { color: #ddd; background-color: #3a3a3a;"
            " border: 1px solid #555; border-radius: 4px; padding: 3px 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:pressed { background-color: #2a2a2a; }"
        )
        button.clicked.connect(lambda: self.save_screenshot())
        button.raise_()
        return button

    def _set_level_text(self, number: str) -> None:
        big, small, tiny = self.READOUT_POINT_SIZES
        suffix = f"dB{self.weighting}"
        if not self.engine.calibrated:
            # Uncalibrated, the number is a full-scale ratio, not a pressure.
            suffix = f"dBFS({self.weighting})"
        note = ""
        if self.engine.average_s > 0:
            note = (
                f'<br><span style="font-size:{tiny}pt; color:#cfcfcf">'
                f"{_duration_label(self.engine.average_s)} average</span>"
            )
        self.level_label.setText(
            f'<span style="font-size:{big}pt; font-weight:bold">{number}</span>'
            f'<span style="font-size:{small}pt">&nbsp;{suffix}</span>{note}'
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
            "border-radius: 3px; padding: 1px 5px;"
        )
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.raise_()
        return label

    def _place_overlay(self) -> None:
        """Pin the overlay labels to the corners of the live image."""
        area = self.glw.mapFromScene(self.spec_plot.vb.sceneBoundingRect()).boundingRect()
        margin = self.OVERLAY_MARGIN_PX
        left, top = area.left() + margin, area.top() + margin

        for label in (self.level_label, self.clip_label):
            label.adjustSize()
        self.level_label.move(left, top)
        self.clip_label.move(left, top + self.level_label.height() + 4)

        bar = self.glw.mapFromScene(self.bar.sceneBoundingRect()).boundingRect()
        self.shot_button.adjustSize()
        self.shot_button.move(
            bar.center().x() - self.shot_button.width() // 2,
            bar.bottom() + self.OVERLAY_MARGIN_PX,
        )

    def _build_status(self) -> QtWidgets.QLabel:
        self.status_label = QtWidgets.QLabel()
        note = self.engine.calibration_note
        text = f"{self.engine.source.description}  |  calibration: {note}"
        if not self.engine.calibrated:
            text += "  |  showing dBFS, NOT absolute SPL"
            self._status_style = "color: #b06000; font-weight: bold;"
        self._status_text = text
        self.status_label.setStyleSheet(self._status_style)
        self.status_label.setText(text)
        return self.status_label

    def _flash_status(self, message: str, error: bool = False) -> None:
        """Say something in the status line, then put it back as it was."""
        self._status_flashing = True
        self.status_label.setStyleSheet(
            f"color: {'#b00020' if error else '#2e7d32'}; font-weight: bold;"
        )
        self.status_label.setText(message)
        QtCore.QTimer.singleShot(self.STATUS_FLASH_MS, self._restore_status)

    def _restore_status(self) -> None:
        self._status_flashing = False
        if self._fatal:  # an error arrived while the message was up; leave it
            return
        self.status_label.setStyleSheet(self._status_style)
        self.status_label.setText(self._status_text + self._status_extra)

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
        self._update_status(state)

        if state.error:
            self._fatal = True
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
        # The right-hand edge of the image is now, so the clock labels move
        # with every redraw even though the tick positions do not.
        self.clock_axis.set_now(time.time())
        if snapshot.n_valid:
            hours_ago = snapshot.ages_s() / 3600.0
            self.long_level_curve.setData(-hours_ago, snapshot.levels[-snapshot.n_valid :])

    def _update_status(self, state) -> None:
        """Keep lost audio visible somewhere now that the stats overlay is gone.

        Dropped blocks and xruns mean the log is missing sound, which is worth
        knowing about; it goes in the status line rather than over the image.
        """
        extra = ""
        if state.dropped_blocks or state.overflows:
            extra = f"  |  dropped {state.dropped_blocks} / xrun {state.overflows}"
        if extra != self._status_extra:
            self._status_extra = extra
            if not self._status_flashing:  # it will be picked up on restore
                self.status_label.setText(self._status_text + extra)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
        elif key == QtCore.Qt.Key.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif key == QtCore.Qt.Key.Key_S:
            self.save_screenshot()
        else:
            super().keyPressEvent(event)

    def save_screenshot(self) -> Path | None:
        """Write a PNG of the window and report where it went, or why not."""
        directory = Path(self.config.ui.screenshot_dir).expanduser()
        path = directory / time.strftime("noise-monitor-%Y%m%d-%H%M%S.png")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # The button is chrome, not measurement; keep it out of the image.
            self.shot_button.setVisible(False)
            try:
                image = self.grab()
            finally:
                self.shot_button.setVisible(True)
            if not image.save(str(path)):
                raise OSError("Qt could not write the image")
        except OSError as exc:
            # A full or read-only disk must not take the meter down with it.
            self._flash_status(f"screenshot failed: {exc}", error=True)
            return None
        self._flash_status(f"screenshot saved to {path}")
        return path

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
