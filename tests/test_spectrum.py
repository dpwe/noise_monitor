"""The spectrum path must conserve power, or band levels are not real levels."""

import numpy as np
import pytest

from noise_monitor.spectrum import (
    BandMapper,
    StreamingSTFT,
    band_levels_db,
    power_preserving_window,
)


def test_window_is_power_preserving():
    w = power_preserving_window(4096)
    assert np.mean(w**2) == pytest.approx(1.0)


def test_stft_obeys_parseval_for_noise():
    """Summed bin power equals the signal's mean square."""
    fs, nfft = 48000, 4096
    rng = np.random.default_rng(0)
    x = rng.standard_normal(nfft * 20) * 0.1
    stft = StreamingSTFT(nfft, nfft, fs)  # no overlap: frames are independent
    frames = stft.push(x)
    total = np.mean([f.sum() for f in frames])
    assert total == pytest.approx(np.mean(x**2), rel=0.05)


def test_stft_recovers_a_sine_level():
    """A unit-RMS 1 kHz sine must land at 0 dB of total power."""
    fs, nfft = 48000, 4096
    t = np.arange(nfft * 8) / fs
    x = np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)
    frames = StreamingSTFT(nfft, nfft, fs).push(x)
    for frame in frames:
        assert 10 * np.log10(frame.sum()) == pytest.approx(0.0, abs=0.05)


def test_stft_hop_produces_expected_frame_count():
    fs, nfft, hop = 48000, 1024, 256
    stft = StreamingSTFT(nfft, hop, fs)
    frames = stft.push(np.zeros(nfft + hop * 5))
    assert len(frames) == 6


def test_band_mapper_conserves_power():
    fs, nfft = 48000, 4096
    stft = StreamingSTFT(nfft, nfft, fs)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 256, fs)
    rng = np.random.default_rng(1)
    power = rng.random(stft.freqs.size)

    in_band = (stft.freqs >= 20.0) & (stft.freqs <= 20000.0)
    # Bins wholly inside [fmin, fmax] must be fully accounted for; edge bins are
    # partly outside, so compare against the interior only.
    interior = (stft.freqs > 20.0 + (stft.freqs[1] - stft.freqs[0])) & (
        stft.freqs < 20000.0 - (stft.freqs[1] - stft.freqs[0])
    )
    assert mapper(power).sum() >= power[interior].sum() - 1e-9
    assert mapper(power).sum() <= power[in_band].sum() + 1e-9


def test_band_mapper_never_duplicates_a_bin():
    """Each bin's total contribution across bands is at most 1.0 of itself."""
    fs, nfft = 48000, 4096
    stft = StreamingSTFT(nfft, nfft, fs)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 256, fs)
    column_sums = mapper.matrix.sum(axis=0)
    assert np.all(column_sums <= 1.0 + 1e-9)


def test_band_edges_and_centers_are_consistent():
    fs = 48000
    stft = StreamingSTFT(4096, 4096, fs)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 128, fs)
    assert mapper.band_edges[0] == pytest.approx(20.0)
    assert mapper.band_edges[-1] == pytest.approx(20000.0)
    assert mapper.band_centers.size == 128
    assert np.all(np.diff(mapper.band_centers) > 0)


def test_band_mapper_clamps_to_nyquist():
    fs = 8000
    stft = StreamingSTFT(1024, 1024, fs)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 64, fs)
    assert mapper.band_edges[-1] == pytest.approx(4000.0)


def test_tone_lands_in_the_right_band():
    fs, nfft = 48000, 8192
    t = np.arange(nfft * 4) / fs
    x = np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)
    stft = StreamingSTFT(nfft, nfft, fs)
    mapper = BandMapper(stft.freqs, 20.0, 20000.0, 256, fs)
    band_power = mapper(stft.push(x)[0])
    peak = mapper.band_centers[np.argmax(band_power)]
    assert peak == pytest.approx(1000.0, rel=0.03)


def test_band_levels_apply_spl_offset():
    power = np.array([1.0, 0.01])
    widths = np.array([10.0, 10.0])
    levels = band_levels_db(power, spl_offset_db=100.0, bandwidths=widths, scale="band")
    assert levels == pytest.approx([100.0, 80.0])


def test_density_scaling_divides_by_bandwidth():
    power = np.array([1.0])
    widths = np.array([10.0])
    band = band_levels_db(power, 0.0, widths, scale="band")
    density = band_levels_db(power, 0.0, widths, scale="density")
    assert band[0] - density[0] == pytest.approx(10.0)  # 10*log10(10)


def test_density_scaling_is_flat_for_white_noise():
    fs, nfft = 48000, 8192
    rng = np.random.default_rng(3)
    x = rng.standard_normal(nfft * 40) * 0.05
    stft = StreamingSTFT(nfft, nfft, fs)
    mapper = BandMapper(stft.freqs, 100.0, 15000.0, 128, fs)

    frames = stft.push(x)
    mean_power = np.mean([mapper(f) for f in frames], axis=0)
    density = band_levels_db(mean_power, 0.0, mapper.bandwidths, scale="density")
    # White noise has constant density, so the trend across bands must be flat.
    slope = np.polyfit(np.log10(mapper.band_centers), density, 1)[0]
    assert abs(slope) < 0.5
