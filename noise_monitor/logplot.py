"""Reading the CSV logs back, and plotting them against a calendar axis.

The live UI shows the last day. This is for the months behind it: load whatever
`noise-YYYYMMDD.csv` files exist, concatenate them onto one time axis and let
the mouse do the rest.

Two things matter for data this size. A year of ten-second rows is about three
million points, so the curve is downsampled in *peak* mode -- averaging would
flatten exactly the short loud events a noise log exists to record, whereas
peak keeps the envelope honest at every zoom level. And time the monitor was
not running is broken with NaN rather than joined, for the same reason the
long-term panel leaves gaps: a straight line across an outage is a measurement
nobody made.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Columns that are levels in dB, and so can share a y axis.
_LEVEL_RE = re.compile(r"^(L[ACZ](eq|max|min|\d+)|Lpeak)$")

#: A run longer than this many times the median interval is an outage, not a
#: sample. Two would trip on ordinary scheduling jitter.
GAP_TOLERANCE = 3.0


@dataclass
class LogSeries:
    """Logged rows from one or more CSV files, on one time axis."""

    #: Unix timestamps of the interval ends, ascending.
    timestamps: np.ndarray
    #: Column name -> values, NaN where a file did not carry that column.
    columns: dict[str, np.ndarray]
    #: The files this came from.
    sources: list[Path]

    def __len__(self) -> int:
        return int(self.timestamps.size)

    @property
    def level_columns(self) -> list[str]:
        return [name for name in self.columns if _LEVEL_RE.match(name)]

    @property
    def interval_s(self) -> float:
        """The typical spacing between rows, for deciding what is a gap."""
        if self.timestamps.size < 2:
            return 0.0
        return float(np.median(np.diff(self.timestamps)))

    def resolve(self, metric: str) -> str | None:
        """Find a column by name, tolerating the weighting letter.

        A log written with C-weighting has LCeq where an A-weighted one has
        LAeq, and asking for the wrong letter should not be a silent miss.
        """
        if metric in self.columns:
            return metric
        if metric.startswith("L"):
            for letter in "ACZ":
                candidate = f"L{letter}{metric[1:]}"
                if candidate in self.columns:
                    return candidate
        return None

    def broken_at_gaps(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """(times, values) with NaN inserted wherever the log has a hole."""
        values = self.columns[metric]
        if self.timestamps.size < 2:
            return self.timestamps, values
        limit = self.interval_s * GAP_TOLERANCE
        holes = np.flatnonzero(np.diff(self.timestamps) > limit)
        if holes.size == 0:
            return self.timestamps, values
        # One NaN sample just after each hole starts is enough to break the
        # line without moving any real point.
        times = np.insert(self.timestamps, holes + 1, self.timestamps[holes] + limit / 2)
        broken = np.insert(values, holes + 1, np.nan)
        return times, broken


# ----------------------------------------------------------------------
def find_logs(directory: Path | str, prefix: str = "noise", days: int | None = None) -> list[Path]:
    """The log files in `directory`, oldest first, optionally the newest `days`.

    Daily rotation names them so a lexical sort is a chronological one.
    """
    paths = sorted(Path(directory).glob(f"{prefix}*.csv"))
    if days is not None and days > 0:
        paths = paths[-days:]
    return paths


def load_logs(paths: list[Path] | Path | str, prefix: str = "noise") -> LogSeries:
    """Read one or more log CSVs onto a single ascending time axis."""
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        paths = find_logs(path, prefix) if path.is_dir() else [path]
    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError("no log files to read")

    stamps: list[float] = []
    gathered: dict[str, dict[int, float]] = {}
    row_index = 0
    used: list[Path] = []

    for path in paths:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "timestamp" not in reader.fieldnames:
                continue  # not one of ours
            used.append(path)
            for row in reader:
                try:
                    when = float(row["timestamp"])
                except (TypeError, ValueError):
                    continue  # a torn final line, most likely
                stamps.append(when)
                for name, text in row.items():
                    if name in ("time", "timestamp") or text in (None, ""):
                        continue
                    try:
                        gathered.setdefault(name, {})[row_index] = float(text)
                    except ValueError:
                        continue
                row_index += 1

    if not stamps:
        raise ValueError(f"no readable rows in {', '.join(str(p) for p in paths)}")

    timestamps = np.asarray(stamps, dtype=float)
    order = np.argsort(timestamps, kind="stable")
    columns = {}
    for name, values in gathered.items():
        column = np.full(timestamps.size, np.nan)
        column[list(values.keys())] = list(values.values())
        columns[name] = column[order]
    return LogSeries(timestamps[order], columns, used)


# ----------------------------------------------------------------------
#: Distinct at a glance and colourblind-safe; first is LAeq, the default.
CURVE_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd")


def plot_logs(series: LogSeries, metrics: list[str], unit: str = "dB SPL") -> int:
    """Open a window showing `metrics` against a calendar axis. Blocks."""
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = build_window(series, metrics, unit)
    window.show()
    return app.exec()


def build_window(series: LogSeries, metrics: list[str], unit: str = "dB SPL"):
    """The plot window itself, so it can be built without running an event loop."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    pg.setConfigOptions(antialias=False)
    window = pg.GraphicsLayoutWidget()
    span = (series.timestamps[0], series.timestamps[-1])
    window.setWindowTitle(
        f"Noise Monitor logs — {len(series)} rows, "
        f"{_stamp(span[0])} to {_stamp(span[1])}"
    )
    window.resize(1200, 600)

    # DateAxisItem draws the month only where a month boundary falls in view,
    # so a week inside one month gets bare day numbers. Rather than mutate its
    # shared zoom-level objects, say what is on screen in a label that tracks
    # the view -- which also answers "how much am I looking at" at every zoom.
    range_label = window.addLabel("", row=0, col=0, justify="left", size="10pt")
    plot = window.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem()})
    plot.setLabel("left", "Level", units=unit)
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.addLegend(offset=(-10, 10))
    # Mouse: drag to pan, wheel to zoom. Unlike the live UI, this is the point.
    plot.setMouseEnabled(x=True, y=True)

    for depth, (colour, metric) in enumerate(zip(CURVE_COLOURS, metrics)):
        times, values = series.broken_at_gaps(metric)
        curve = plot.plot(
            times, values, pen=pg.mkPen(colour, width=1), name=metric,
            connect="finite", skipFiniteCheck=False,
        )
        # Peak-mode downsampling keeps short loud events visible when zoomed
        # out; clipping to view keeps redraws cheap on a long log.
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        # First metric on top: with dense data a later curve simply hides an
        # earlier one, and the first is the one that was asked for.
        curve.setZValue(-depth)

    # Set x from the data rather than leaving it to auto-range, which has not
    # been applied yet when the readout below first reads the view -- it would
    # open showing 1969. y stays automatic and follows whatever is on screen.
    plot.enableAutoRange(axis="y")
    plot.setAutoVisible(y=True)
    plot.setXRange(series.timestamps[0], series.timestamps[-1], padding=0.02)

    def show_range():
        (start, end), _ = plot.viewRange()
        range_label.setText(_range_text(start, end))

    plot.sigXRangeChanged.connect(show_range)
    show_range()

    window.plot = plot
    window.range_label = range_label

    def key_press(event):
        if event.key() in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            window.close()
        elif event.key() == QtCore.Qt.Key.Key_A:
            plot.enableAutoRange()  # after zooming somewhere unhelpful
        else:
            pg.GraphicsLayoutWidget.keyPressEvent(window, event)

    window.keyPressEvent = key_press
    return window


def _stamp(when: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(when).strftime(fmt)


def _range_text(start: float, end: float) -> str:
    """What is currently on screen, spelled out, with how much of it there is."""
    span = max(end - start, 0.0)
    fmt = "%a %d %b %Y" if span > 2 * 86400 else "%a %d %b %Y %H:%M"
    return f"{_stamp(start, fmt)}  to  {_stamp(end, fmt)}   ({_span_text(span)})"


def _span_text(seconds: float) -> str:
    for size, unit in ((86400.0, "days"), (3600.0, "hours"), (60.0, "minutes")):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.0f} seconds"
