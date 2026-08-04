import numpy as np
import pytest

from noise_monitor.metrics import IntervalAccumulator


def _feed(acc, signal, block=1024):
    """Push a signal through, using the signal itself as its own time weighting.

    Returns every interval the accumulator completed.
    """
    out = []
    for i in range(0, signal.size, block):
        chunk = signal[i : i + block]
        out.extend(acc.add(chunk, chunk**2, chunk))
    return out


def test_leq_of_a_known_level():
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0, spl_offset_db=100.0)
    x = np.full(fs, 0.1)  # mean square 0.01 -> -20 dBFS
    (stats,) = _feed(acc, x)
    assert stats.leq == pytest.approx(80.0, abs=0.01)  # -20 dBFS + 100 dB offset


def test_leq_is_energy_averaged_not_level_averaged():
    """Half a second at -20 dBFS and half at -40 must give ~-23 dB, not -30."""
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0)
    x = np.concatenate([np.full(fs // 2, 0.1), np.full(fs // 2, 0.01)])
    (stats,) = _feed(acc, x)
    expected = 10 * np.log10((0.01 + 0.0001) / 2)
    assert stats.leq == pytest.approx(expected, abs=0.01)


def test_max_min_and_peak():
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0)
    x = np.concatenate([np.full(fs // 2, 0.1), np.full(fs // 2, 0.5)])
    (stats,) = _feed(acc, x)
    assert stats.lmax == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert stats.lmin == pytest.approx(20 * np.log10(0.1), abs=0.01)
    assert stats.lpeak == pytest.approx(20 * np.log10(0.5), abs=0.01)


def test_percentiles_are_exceedance_levels():
    """Ln is the level exceeded n% of the time, so L10 > L50 > L90."""
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=10.0)
    # 10 s: 8 s quiet at -40 dBFS, 2 s loud at -20 dBFS. The loud fraction (20%)
    # straddles L10 but not L50 or L90, so every percentile is unambiguous.
    x = np.concatenate([np.full(8 * fs, 0.01), np.full(2 * fs, 0.1)])
    (stats,) = _feed(acc, x)
    assert stats.l10 > stats.l50
    assert stats.l50 == pytest.approx(stats.l90)
    assert stats.l10 == pytest.approx(-20.0, abs=0.1)
    assert stats.l90 == pytest.approx(-40.0, abs=0.1)


def test_interval_resets_between_emissions():
    fs = 4800
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0)
    x = np.concatenate([np.full(fs, 0.1), np.full(fs, 0.01)])
    first, second = _feed(acc, x)
    assert first.leq == pytest.approx(20 * np.log10(0.1), abs=0.01)
    assert second.leq == pytest.approx(20 * np.log10(0.01), abs=0.01)
    assert second.clipped_samples == 0


def test_boundaries_are_sample_accurate_not_block_aligned():
    """An interval that is not a whole number of blocks must not drift."""
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=0.5)
    # 2 s of audio in 1024-sample blocks: 0.5 s is 23.4 blocks, so every
    # boundary but the last falls inside a block.
    stats = _feed(acc, np.full(2 * fs, 0.1), block=1024)
    assert len(stats) == 4
    assert all(s.duration_s == pytest.approx(0.5) for s in stats)
    # Timestamps are derived from the sample count, so spacing is exact.
    spacing = np.diff([s.timestamp for s in stats])
    assert spacing == pytest.approx([0.5, 0.5, 0.5])


def test_timestamps_are_anchored_to_start_time():
    fs = 48000
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0, start_time=1000.0)
    stats = _feed(acc, np.full(3 * fs, 0.1))
    assert [s.timestamp for s in stats] == pytest.approx([1001.0, 1002.0, 1003.0])


def test_clipped_samples_are_counted_into_the_right_interval():
    fs = 4800
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0)
    x = np.full(2 * fs, 0.1)
    mask = np.zeros(2 * fs, dtype=bool)
    mask[: 100] = True          # first interval
    mask[fs : fs + 7] = True    # second interval
    out = []
    for i in range(0, x.size, 1024):
        chunk = x[i : i + 1024]
        out.extend(acc.add(chunk, chunk**2, chunk, mask[i : i + 1024]))
    assert [s.clipped_samples for s in out] == [100, 7]


def test_stat_sampling_is_block_size_independent():
    """Odd block sizes must not change the statistical levels."""
    fs = 48000
    x = np.linspace(0.01, 0.1, fs)
    a = _feed(IntervalAccumulator(samplerate=fs, interval_s=1.0), x, block=4800)
    b = _feed(IntervalAccumulator(samplerate=fs, interval_s=1.0), x, block=997)
    assert a[0].l10 == pytest.approx(b[0].l10)
    assert a[0].l50 == pytest.approx(b[0].l50)
    assert a[0].l90 == pytest.approx(b[0].l90)


def test_row_keys_carry_the_weighting():
    fs = 4800
    acc = IntervalAccumulator(samplerate=fs, interval_s=1.0, weighting="A")
    (stats,) = _feed(acc, np.full(fs, 0.1))
    row = stats.as_row()
    assert "LAeq" in row and "LAmax" in row and "LA90" in row
    assert "timestamp" in row and "clipped_samples" in row
