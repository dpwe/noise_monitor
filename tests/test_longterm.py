"""The hours-scale averaging behind the second spectrogram."""

import numpy as np
import pytest

from noise_monitor.config import Config
from noise_monitor.longterm import LongTermAverage


def make(hops_per_column=4, n_columns=5, n_bands=3, hop_s=0.5):
    return LongTermAverage(n_bands, hops_per_column, n_columns, hop_s, fill_db=-10.0)


def feed(acc, level_db, count, band_db=None):
    band = np.full(acc.n_bands, level_db if band_db is None else band_db, dtype=float)
    return [acc.add(band, level_db) for _ in range(count)]


def test_a_column_appears_only_when_its_window_is_full():
    acc = make(hops_per_column=4)
    assert feed(acc, 50.0, 3) == [False, False, False]
    assert acc.snapshot().n_valid == 1  # the partial column, shown early
    assert acc.add(np.full(3, 50.0), 50.0) is True


def test_a_steady_level_averages_to_itself():
    acc = make(hops_per_column=4)
    feed(acc, 55.0, 4)
    snap = acc.snapshot()
    assert snap.levels[-1] == pytest.approx(55.0)
    assert snap.image[-1] == pytest.approx(55.0)


def test_averaging_is_in_power_not_in_decibels():
    """Half a window at 40 dB and half at 80 dB is 77 dB, not 60."""
    acc = make(hops_per_column=4)
    feed(acc, 40.0, 2)
    feed(acc, 80.0, 2)
    snap = acc.snapshot()
    expected = 10 * np.log10((2 * 10**4.0 + 2 * 10**8.0) / 4)
    assert snap.levels[-1] == pytest.approx(expected)
    assert snap.image[-1] == pytest.approx(expected, abs=1e-3)


def test_the_newest_column_is_on_the_right_and_old_ones_scroll_off():
    acc = make(hops_per_column=1, n_columns=3)
    for level in (10.0, 20.0, 30.0, 40.0):
        feed(acc, level, 1)
    snap = acc.snapshot()
    assert snap.levels == pytest.approx([20.0, 30.0, 40.0])
    assert snap.n_valid == 3  # 10 dB fell off the left, and it stays full


def test_empty_columns_are_filled_not_holes():
    acc = make(hops_per_column=1, n_columns=4)
    feed(acc, 50.0, 1)
    snap = acc.snapshot()
    assert snap.n_valid == 1
    assert np.all(snap.image[:-1] == -10.0)
    assert np.all(np.isnan(snap.levels[:-1]))


def test_the_partial_column_does_not_survive_into_the_history():
    """It is a view, not state: it must not double-count when it completes."""
    acc = make(hops_per_column=2, n_columns=4)
    feed(acc, 60.0, 1)
    acc.snapshot()
    acc.snapshot()
    feed(acc, 60.0, 1)
    snap = acc.snapshot()
    assert snap.n_valid == 1
    assert snap.levels[-1] == pytest.approx(60.0)


def test_snapshots_are_copies():
    acc = make(hops_per_column=1)
    feed(acc, 50.0, 1)
    snap = acc.snapshot()
    snap.image[:] = 0.0
    snap.levels[:] = 0.0
    assert acc.snapshot().levels[-1] == pytest.approx(50.0)


def test_ages_run_oldest_first_and_end_near_now():
    acc = make(hops_per_column=4, n_columns=5, hop_s=0.5)  # 2 s columns
    for _ in range(3):
        feed(acc, 50.0, 4)
    snap = acc.snapshot()
    assert snap.column_s == pytest.approx(2.0)
    assert snap.span_s == pytest.approx(10.0)
    ages = snap.ages_s()
    assert ages.size == snap.n_valid == 3
    assert list(ages) == pytest.approx([5.0, 3.0, 1.0])


def test_a_level_that_never_arrives_leaves_the_trace_empty():
    acc = make(hops_per_column=2)
    feed(acc, float("nan"), 2, band_db=50.0)
    snap = acc.snapshot()
    assert np.isnan(snap.levels[-1])
    assert snap.image[-1] == pytest.approx(50.0)


def test_silence_gives_a_floor_not_negative_infinity():
    acc = LongTermAverage(3, 1, 2)
    acc.add(np.full(3, -np.inf), -np.inf)
    assert np.all(np.isfinite(acc.snapshot().image[-1]))


def test_geometry_comes_from_the_config():
    cfg = Config()
    cfg.audio.samplerate = 48000
    cfg.analysis.hop = 1024
    cfg.analysis.n_bands = 64
    cfg.ui.long_column_s = 180.0
    cfg.ui.long_span_s = 86400.0
    acc = LongTermAverage.from_config(cfg)
    assert acc.n_bands == 64
    assert acc.column_s == pytest.approx(180.0, rel=1e-3)
    assert acc.span_s == pytest.approx(86400.0, rel=1e-3)
    assert acc.n_columns == 480


def test_a_column_shorter_than_a_hop_still_makes_progress():
    cfg = Config()
    cfg.ui.long_column_s = 0.0
    acc = LongTermAverage.from_config(cfg)
    assert acc.hops_per_column == 1


def test_degenerate_sizes_are_rejected():
    with pytest.raises(ValueError):
        LongTermAverage(4, 0, 10)
