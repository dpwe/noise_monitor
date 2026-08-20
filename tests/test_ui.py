"""Axis label mapping. Skipped unless a Qt binding is installed."""

import os

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets  # noqa: E402

from noise_monitor.spectrum import BandMapper, StreamingSTFT  # noqa: E402
from noise_monitor.ui import LogFreqAxis, _duration_label, _grid_levels  # noqa: E402


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


def test_grid_levels_stay_strictly_inside_the_range():
    assert _grid_levels(30.0, 60.0) == [40.0, 50.0]
    # The ends are the axis itself; a line drawn on them is just noise.
    assert _grid_levels(30.0, 50.0) == [40.0]
    assert _grid_levels(41.0, 49.0) == []
