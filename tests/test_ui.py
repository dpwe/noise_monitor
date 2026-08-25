"""Axis label mapping. Skipped unless a Qt binding is installed."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

from noise_monitor.spectrum import BandMapper, StreamingSTFT  # noqa: E402
from noise_monitor.ui import ClockAxis, LogFreqAxis, _duration_label  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """AxisItem is a QGraphicsItem, so it needs a live QApplication."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def axis(qapp):
    stft = StreamingSTFT(4096, 1024, 48000)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 256, 48000)
    return LogFreqAxis(mapper.band_centers)


def _ticks(axis, size=600.0):
    values = axis.tickValues(0, axis.band_centers.size, size)[0][1]
    return values, axis.tickStrings(values, 1.0, 1.0)


def test_labels_are_round_numbers_not_round_trips(axis):
    """Interpolating index->frequency would print '1.00009k'; the map must not."""
    _, labels = _ticks(axis)
    assert "1k" in labels
    assert not any("00009" in text or "0001" in text for text in labels)


def test_ticks_land_at_the_right_band_index(axis):
    values, labels = _ticks(axis)
    for index, label in zip(values, labels):
        freq = float(label[:-1]) * 1000 if label.endswith("k") else float(label)
        # The band at that index must be the frequency the label claims.
        assert axis.band_centers[int(round(index))] == pytest.approx(freq, rel=0.02)


def test_ticks_are_ordered_and_unique(axis):
    values, labels = _ticks(axis)
    assert list(values) == sorted(values)
    assert len(set(labels)) == len(labels)


def test_ticks_thin_out_on_a_short_axis(axis):
    many, _ = _ticks(axis, size=800.0)
    few, _ = _ticks(axis, size=120.0)
    assert len(few) < len(many)
    assert len(few) >= 2


def test_no_ticks_collide(axis):
    size = 400.0
    values, _ = _ticks(axis, size=size)
    px = np.asarray(values) * size / axis.band_centers.size
    assert np.all(np.diff(px) >= LogFreqAxis.MIN_TICK_SPACING_PX - 1e-6)


def test_ticks_stay_inside_the_band_range(axis):
    values, _ = _ticks(axis)
    assert np.all(np.asarray(values) >= 0)
    assert np.all(np.asarray(values) <= axis.band_centers.size)


def test_degenerate_axis_returns_no_ticks(axis):
    assert axis.tickValues(0, 0, 100.0) == []
    assert axis.tickValues(0, 100, 0.0) == []


@pytest.mark.parametrize(
    "seconds,expected",
    [(1.0, "1s"), (0.125, "0.125s"), (10.0, "10s"), (60.0, "1min"),
     # A column is a whole number of hops, so the nominal 3 min is really
     # 180.011 s -- the label must not say so.
     (180.011, "3min"), (3600.0, "1h"), (86400.5, "24h")],
)
def test_duration_labels_are_readable(seconds, expected):
    assert _duration_label(seconds) == expected


# ----------------------------------------------------------------------
# The long-term panel's clock axis.

#: 2026-08-21 14:37:23 local time, an arbitrary "now" with an awkward offset
#: into the hour so alignment has something to do.
NOW = time.mktime((2026, 8, 21, 14, 37, 23, 0, 0, -1))


@pytest.fixture
def clock(qapp):
    axis = ClockAxis()
    axis.set_now(NOW)
    return axis


def _clock_ticks(axis, lo=-24.0, hi=0.0, size=1000.0):
    spacing, values = axis.tickValues(lo, hi, size)[0]
    return spacing, values, axis.tickStrings(values, 1.0, spacing)


def _times(labels):
    """The clock part of each label, dropping any date suffix."""
    return [label.split("\n")[0] for label in labels]


def test_ticks_land_on_round_local_hours(clock):
    """Not on round offsets from now, which would slide as the panel scrolls."""
    _, _, labels = _clock_ticks(clock)
    assert all(t.endswith(":00") for t in _times(labels))


def test_labels_are_the_clock_time_at_that_point(clock):
    _, values, labels = _clock_ticks(clock)
    for value, label in zip(values, _times(labels)):
        expected = time.localtime(NOW + value * 3600.0)
        assert label == time.strftime("%H:%M", expected)
    # 14:37:23 now, and a 2 h step over a day, so the newest tick is the last
    # even hour -- 14:00, offset back by the 37m23s since it, not by 0.
    assert _times(labels)[-1] == "14:00"
    assert values[-1] == pytest.approx(-(37 * 60 + 23) / 3600.0, abs=1e-9)


def test_ticks_run_oldest_first_and_stay_in_range(clock):
    _, values, _ = _clock_ticks(clock)
    assert list(values) == sorted(values)
    assert all(-24.0 <= v <= 0.0 for v in values)


def test_ticks_thin_out_rather_than_overprint(clock):
    wide_spacing, wide, _ = _clock_ticks(clock, size=1600.0)
    narrow_spacing, narrow, _ = _clock_ticks(clock, size=200.0)
    assert narrow_spacing > wide_spacing
    assert len(narrow) < len(wide)
    assert len(narrow) >= 2


def test_no_ticks_collide(clock):
    size = 1000.0
    _, values, _ = _clock_ticks(clock, size=size)
    px = np.diff(np.asarray(values)) * size / 24.0
    assert np.all(px >= ClockAxis.MIN_TICK_SPACING_PX - 1e-6)


def test_a_short_span_gets_sub_hour_ticks(clock):
    """An hour is the wrong grain for a --long-span of a few hours."""
    spacing, values, labels = _clock_ticks(clock, lo=-3.0, hi=0.0, size=1000.0)
    assert spacing == pytest.approx(0.25)  # 15 minutes
    assert len(values) >= 8
    assert labels[-1] == "14:30"


# --- the date, carried by whichever tick opens a day ------------------

def _dated(labels):
    return [label for label in labels if "\n" in label]


def test_the_midnight_tick_carries_the_date(clock):
    _, _, labels = _clock_ticks(clock)
    midnight = [label for label in labels if label.startswith("00:00")]
    assert midnight == ["00:00\n2026-08-21"]


def test_the_oldest_tick_is_dated_so_its_day_is_named_too(clock):
    _, _, labels = _clock_ticks(clock)
    assert labels[0].endswith("\n2026-08-20")  # the day before


def test_only_day_openings_are_dated(clock):
    """A date on every tick would be unreadable; one per day is the point."""
    _, _, labels = _clock_ticks(clock)
    assert len(_dated(labels)) == 2  # yesterday at the left edge, then midnight
    assert len(labels) > 6


def test_a_span_inside_one_day_is_still_dated_once(clock):
    """Miss midnight and the date would otherwise never appear at all."""
    _, _, labels = _clock_ticks(clock, lo=-3.0, hi=0.0, size=1000.0)
    assert _dated(labels) == [labels[0]]
    assert labels[0].endswith("\n2026-08-21")


def test_a_span_shorter_than_an_hour_is_still_labelled(clock):
    """Regression: hour-only steps left such a span with no ticks at all."""
    _, values, labels = _clock_ticks(clock, lo=-0.25, hi=0.0, size=1000.0)
    assert len(values) >= 2
    assert [t[:3] for t in _times(labels)] == ["14:"] * len(labels)


def test_moving_now_moves_the_labels(clock):
    spacing, _, before = _clock_ticks(clock)
    clock.set_now(NOW + spacing * 3600.0)
    _, _, after = _clock_ticks(clock)
    assert before != after
    # Advance by exactly one spacing and the grid shifts along by one tick:
    # what was the newest label is now the second newest.
    assert after[-2] == before[-1]


def test_a_span_with_no_round_time_in_it_labels_the_middle(clock):
    """Never a bare axis -- and not on the boundary, which gets culled."""
    lo, hi = -0.004, 0.0
    _, values, labels = _clock_ticks(clock, lo=lo, hi=hi, size=1000.0)
    assert values == [pytest.approx((lo + hi) / 2)]
    assert lo < values[0] < hi
    assert labels == ["14:37\n2026-08-21"]  # the only tick, so it carries the date


def test_degenerate_axis_returns_no_ticks(clock):
    assert clock.tickValues(0, 0, 100.0) == []
    assert clock.tickValues(-24, 0, 0.0) == []


# ----------------------------------------------------------------------
# The window itself: construction, and the screenshot key.


@pytest.fixture
def window(qapp, tmp_path):
    from noise_monitor.capture import ArraySource
    from noise_monitor.config import Config
    from noise_monitor.engine import MonitorEngine
    from noise_monitor.ui import MonitorWindow

    cfg = Config()
    cfg.logging.enabled = False
    cfg.ui.screenshot_dir = tmp_path / "shots"
    cfg.ui.long_span_s = 3600.0  # 20 columns, not 480
    source = ArraySource(np.zeros(1024), cfg.audio.samplerate, cfg.audio.blocksize)
    win = MonitorWindow(MonitorEngine(cfg, source, cal=None), cfg)
    win.resize(600, 400)
    yield win
    win.timer.stop()
    win.long_timer.stop()
    win.close()


def test_the_window_builds(window):
    """Cheap insurance: most of the UI has no other automated exercise."""
    assert window.long_plot is not None
    assert window.clock_axis is not None


def test_saving_a_screenshot_writes_a_png(window):
    path = window.save_screenshot()
    assert path is not None
    assert path.exists() and path.suffix == ".png"
    assert path.stat().st_size > 0
    assert "screenshot saved" in window.status_label.text()


def test_the_s_key_is_wired_to_it(window):
    event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_S,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    window.keyPressEvent(event)
    assert list(Path(window.config.ui.screenshot_dir).glob("*.png"))


def test_a_screenshot_that_cannot_be_written_is_reported_not_raised(window, tmp_path):
    """A full or read-only disk must not take the meter down with it."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    window.config.ui.screenshot_dir = blocker / "shots"
    assert window.save_screenshot() is None
    assert "failed" in window.status_label.text()


def test_the_status_line_goes_back_to_normal_afterwards(window):
    original = window.status_label.text()
    window.save_screenshot()
    assert window.status_label.text() != original
    window._restore_status()
    assert window.status_label.text() == original
