"""The hours-scale averaging behind the second spectrogram, and its store."""

import numpy as np
import pytest

from noise_monitor.config import Config
from noise_monitor.longterm import (
    LongTermAverage,
    load_history,
    save_history,
)

#: An arbitrary fixed instant that lands exactly on a 10 s slot boundary.
BASE = 1_800_000_000.0


def make(column_s=10.0, n_columns=5, n_bands=3, fill_db=-10.0):
    return LongTermAverage(n_bands, column_s, n_columns, fill_db)


def feed(acc, level_db, when, band_db=None):
    band = np.full(acc.n_bands, level_db if band_db is None else band_db, dtype=float)
    return acc.add(band, level_db, when=when)


def levels_at(acc, *levels, start=BASE, column_s=10.0):
    """One column per consecutive slot, starting at `start`."""
    for i, level in enumerate(levels):
        feed(acc, level, start + i * column_s)


# --- averaging --------------------------------------------------------

def test_a_column_is_banked_when_the_clock_leaves_its_slot():
    acc = make()
    assert feed(acc, 50.0, BASE) is False
    assert feed(acc, 50.0, BASE + 5) is False  # same slot
    assert acc.snapshot().levels[-1] == pytest.approx(50.0)  # shown while partial
    assert feed(acc, 50.0, BASE + 10) is True  # next slot: the first is banked


def test_a_steady_level_averages_to_itself():
    acc = make()
    for offset in (0, 2, 4, 6, 8):
        feed(acc, 55.0, BASE + offset)
    snap = acc.snapshot()
    assert snap.levels[-1] == pytest.approx(55.0)
    assert snap.image[-1] == pytest.approx(55.0)


def test_averaging_is_in_power_not_in_decibels():
    """Half a slot at 40 dB and half at 80 dB is 77 dB, not 60."""
    acc = make()
    for offset in (0, 2):
        feed(acc, 40.0, BASE + offset)
    for offset in (4, 6):
        feed(acc, 80.0, BASE + offset)
    snap = acc.snapshot()
    expected = 10 * np.log10((2 * 10**4.0 + 2 * 10**8.0) / 4)
    assert snap.levels[-1] == pytest.approx(expected)
    assert snap.image[-1] == pytest.approx(expected, abs=1e-3)


def test_a_level_that_never_arrives_leaves_the_trace_empty():
    acc = make()
    feed(acc, float("nan"), BASE, band_db=50.0)
    snap = acc.snapshot()
    assert np.isnan(snap.levels[-1])
    assert snap.image[-1] == pytest.approx(50.0)


def test_silence_gives_a_floor_not_negative_infinity():
    acc = LongTermAverage(3, 10.0, 2)
    acc.add(np.full(3, -np.inf), -np.inf, when=BASE)
    assert np.all(np.isfinite(acc.snapshot().image[-1]))


# --- the slot grid ----------------------------------------------------

def test_columns_scroll_left_as_slots_pass():
    acc = make()
    levels_at(acc, 10.0, 20.0, 30.0, 40.0, 50.0)
    assert acc.snapshot().levels == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0])


def test_the_oldest_columns_fall_off_the_left():
    acc = make(n_columns=3)
    levels_at(acc, 10.0, 20.0, 30.0, 40.0)
    assert acc.snapshot().levels == pytest.approx([20.0, 30.0, 40.0])


def test_a_silent_stretch_leaves_a_gap_not_a_join():
    """The whole point of the slot grid: time not recorded stays empty."""
    acc = make()
    feed(acc, 40.0, BASE)
    feed(acc, 50.0, BASE + 30)  # three slots later
    levels = acc.snapshot().levels
    assert levels[1] == pytest.approx(40.0)
    assert levels[-1] == pytest.approx(50.0)
    assert np.isnan(levels[2]) and np.isnan(levels[3])


def test_a_long_silence_clears_the_window():
    acc = make(n_columns=5)
    levels_at(acc, 10.0, 20.0, 30.0)
    feed(acc, 60.0, BASE + 10 * 20)  # far beyond the window
    levels = acc.snapshot().levels
    assert levels[-1] == pytest.approx(60.0)
    assert np.all(np.isnan(levels[:-1]))


def test_a_backwards_clock_step_is_ignored():
    """An NTP correction must not reopen a slot that is already banked."""
    acc = make()
    levels_at(acc, 10.0, 20.0)
    assert feed(acc, 99.0, BASE - 100) is False
    assert acc.snapshot().levels[-1] == pytest.approx(20.0)


def test_empty_columns_are_filled_not_holes():
    acc = make(n_columns=4)
    feed(acc, 50.0, BASE)
    snap = acc.snapshot()
    assert np.all(snap.image[:-1] == -10.0)
    assert np.all(np.isnan(snap.levels[:-1]))


def test_the_partial_column_does_not_survive_into_the_history():
    """It is a view, not state: it must not double-count when it completes."""
    acc = make()
    feed(acc, 60.0, BASE)
    acc.snapshot()
    acc.snapshot()
    feed(acc, 60.0, BASE + 5)
    snap = acc.snapshot()
    assert snap.levels[-1] == pytest.approx(60.0)
    assert np.all(np.isnan(snap.levels[:-1]))


def test_snapshots_are_copies():
    acc = make()
    feed(acc, 50.0, BASE)
    snap = acc.snapshot()
    snap.image[:] = 0.0
    snap.levels[:] = 0.0
    assert acc.snapshot().levels[-1] == pytest.approx(50.0)


# --- geometry ---------------------------------------------------------

def test_centers_run_oldest_first_and_end_half_a_column_back():
    acc = make(column_s=2.0, n_columns=5)
    snap = acc.snapshot()
    assert snap.span_s == pytest.approx(10.0)
    assert list(snap.centers_s()) == pytest.approx([-9.0, -7.0, -5.0, -3.0, -1.0])


def test_the_edge_is_the_end_of_the_newest_slot():
    acc = make(column_s=10.0)
    assert acc.snapshot().end_time is None  # nothing recorded yet
    feed(acc, 50.0, BASE + 3)
    assert acc.snapshot().end_time == pytest.approx(BASE + 10)


def test_geometry_comes_from_the_config():
    cfg = Config()
    cfg.analysis.n_bands = 64
    cfg.ui.long_column_s = 180.0
    cfg.ui.long_span_s = 86400.0
    acc = LongTermAverage.from_config(cfg)
    assert acc.n_bands == 64
    assert acc.column_s == pytest.approx(180.0)
    assert acc.span_s == pytest.approx(86400.0)
    assert acc.n_columns == 480


def test_degenerate_sizes_are_rejected():
    with pytest.raises(ValueError):
        LongTermAverage(4, 10.0, 0)
    with pytest.raises(ValueError):
        LongTermAverage(4, 0.0, 10)


# --- the store --------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return tmp_path / "history" / "long_term.npz"


def saved(store, *levels, column_s=10.0, n_columns=5, n_bands=3):
    acc = make(column_s=column_s, n_columns=n_columns, n_bands=n_bands)
    levels_at(acc, *levels, column_s=column_s)
    save_history(store, acc.state())
    return acc


def test_nothing_is_written_before_any_audio_arrives():
    assert make().state() is None


def test_a_saved_history_reloads_onto_the_same_slots(store):
    saved(store, 41.0, 42.0, 43.0)
    fresh = make()
    note = fresh.restore(load_history(store), now=BASE + 20)
    assert "restored 3 columns" in note
    levels = fresh.snapshot().levels
    assert levels[-3:] == pytest.approx([41.0, 42.0, 43.0], abs=0.05)
    assert np.all(np.isnan(levels[:-3]))


def test_the_bands_come_back_too(store):
    saved(store, 44.0)
    fresh = make()
    fresh.restore(load_history(store), now=BASE)
    assert fresh.snapshot().image[-1] == pytest.approx(44.0, abs=0.05)


def test_time_it_was_down_comes_back_as_a_gap(store):
    """Restarting after two slots off must not close the gap up."""
    saved(store, 41.0, 42.0, 43.0)
    fresh = make()
    note = fresh.restore(load_history(store), now=BASE + 40)
    assert "gap" in note
    levels = fresh.snapshot().levels
    assert levels[:3] == pytest.approx([41.0, 42.0, 43.0], abs=0.05)
    assert np.all(np.isnan(levels[3:]))  # the two slots it was down


def test_a_smaller_window_keeps_the_newest_columns(store):
    saved(store, *[40.0 + i for i in range(10)], n_columns=10)
    fresh = make(n_columns=4)
    fresh.restore(load_history(store), now=BASE + 90)
    assert fresh.snapshot().levels == pytest.approx([46.0, 47.0, 48.0, 49.0], abs=0.05)


def test_a_larger_window_leaves_the_extra_columns_empty(store):
    saved(store, 41.0, 42.0, n_columns=3)
    fresh = make(n_columns=6)
    fresh.restore(load_history(store), now=BASE + 10)
    levels = fresh.snapshot().levels
    assert levels[-2:] == pytest.approx([41.0, 42.0], abs=0.05)
    assert np.all(np.isnan(levels[:-2]))


def test_history_older_than_the_window_is_refused(store):
    saved(store, 41.0)
    fresh = make(n_columns=5)
    note = fresh.restore(load_history(store), now=BASE + 10 * 50)
    assert "older than the window" in note
    assert np.all(np.isnan(fresh.snapshot().levels))


def test_history_from_the_future_is_refused(store):
    """A clock that was wrong last run must not put columns ahead of now."""
    saved(store, 41.0, 42.0)
    fresh = make()
    note = fresh.restore(load_history(store), now=BASE - 1000)
    assert "future" in note
    assert np.all(np.isnan(fresh.snapshot().levels))


def test_a_different_column_length_is_refused(store):
    saved(store, 41.0, 42.0, column_s=10.0)
    fresh = make(column_s=20.0)
    note = fresh.restore(load_history(store), now=BASE + 10)
    assert "columns" in note and "ignored" in note
    assert np.all(np.isnan(fresh.snapshot().levels))


def test_a_different_band_count_is_refused(store):
    saved(store, 41.0, n_bands=3)
    fresh = make(n_bands=8)
    note = fresh.restore(load_history(store), now=BASE)
    assert "bands" in note and "ignored" in note


def test_a_missing_store_is_not_fatal(tmp_path):
    note = make().restore(load_history(tmp_path / "nope.npz"), now=BASE)
    assert "no stored history" in note


def test_a_corrupt_store_is_not_fatal(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"this is not an npz file at all")
    assert load_history(store) is None
    assert "no stored history" in make().restore(None, now=BASE)


def test_a_store_from_another_format_is_refused(store):
    saved(store, 41.0)
    with np.load(store) as data:
        fields = {name: data[name] for name in data.files}
    fields["format"] = np.int32(999)
    save_history(store, fields)
    assert load_history(store) is None


def test_the_write_leaves_no_temporary_behind(store):
    saved(store, 41.0)
    assert store.exists()
    assert list(store.parent.iterdir()) == [store]


def test_a_half_written_file_cannot_replace_a_good_one(store, monkeypatch):
    """The rename is the point: the old history survives a failed write."""
    acc = saved(store, 41.0, 42.0)
    good = store.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", explode)
    with pytest.raises(OSError):
        save_history(store, acc.state())
    assert store.read_bytes() == good
