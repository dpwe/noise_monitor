"""Axis label mapping. Skipped unless a Qt binding is installed."""

import os

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets  # noqa: E402

from noise_monitor.spectrum import BandMapper, StreamingSTFT  # noqa: E402
from noise_monitor.ui import LogFreqAxis  # noqa: E402


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
