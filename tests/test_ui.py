"""Axis label mapping. Skipped unless a Qt binding is installed."""

import os
import time

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets  # noqa: E402

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


def test_ticks_land_on_round_local_hours(clock):
    """Not on round offsets from now, which would slide as the panel scrolls."""
    _, _, labels = _clock_ticks(clock)
    assert all(label.endswith(":00") for label in labels)


def test_labels_are_the_clock_time_at_that_point(clock):
    _, values, labels = _clock_ticks(clock)
    for value, label in zip(values, labels):
        expected = time.localtime(NOW + value * 3600.0)
        assert label == time.strftime("%H:%M", expected)
    # 14:37:23 now, and a 2 h step over a day, so the newest tick is the last
    # even hour -- 14:00, offset back by the 37m23s since it, not by 0.
    assert labels[-1] == "14:00"
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


def test_a_short_span_still_gets_ticks(clock):
    _, values, labels = _clock_ticks(clock, lo=-3.0, hi=0.0, size=1000.0)
    assert len(values) >= 2
    assert labels[-1] == "14:00"


def test_moving_now_moves_the_labels(clock):
    spacing, _, before = _clock_ticks(clock)
    clock.set_now(NOW + spacing * 3600.0)
    _, _, after = _clock_ticks(clock)
    assert before != after
    # Advance by exactly one spacing and the grid shifts along by one tick:
    # what was the newest label is now the second newest.
    assert after[-2] == before[-1]


def test_degenerate_axis_returns_no_ticks(clock):
    assert clock.tickValues(0, 0, 100.0) == []
    assert clock.tickValues(-24, 0, 0.0) == []
