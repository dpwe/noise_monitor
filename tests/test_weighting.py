"""Check the weighting filters against the IEC 61672-1 tabulated responses."""

import numpy as np
import pytest
from scipy.signal import sosfreqz

from noise_monitor.weighting import (
    ExponentialLevel,
    StreamingFilter,
    weighting_response_db,
    weighting_sos,
)

# Exact third-octave frequencies, 1000 * 10**(n/10), with the standard's
# tabulated weightings (dB re 1 kHz).
_N = np.arange(-20, 14)
EXACT_FREQS = 1000.0 * 10.0 ** (_N / 10.0)

A_TABLE = np.array([
    -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
    -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8,
    0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1,
    -2.5, -4.3, -6.6, -9.3,
])

C_TABLE = np.array([
    -14.3, -11.2, -8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5,
    -0.3, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0,
    -4.4, -6.2, -8.5, -11.2,
])


def test_analog_a_weighting_matches_iec_table():
    got = weighting_response_db("A", EXACT_FREQS)
    assert np.max(np.abs(got - A_TABLE)) < 0.15


def test_analog_c_weighting_matches_iec_table():
    got = weighting_response_db("C", EXACT_FREQS)
    assert np.max(np.abs(got - C_TABLE)) < 0.15


@pytest.mark.parametrize("kind", ["A", "C"])
def test_weighting_is_unity_at_1khz(kind):
    assert weighting_response_db(kind, np.array([1000.0]))[0] == pytest.approx(0.0, abs=0.02)


def test_z_weighting_is_flat():
    assert np.all(weighting_response_db("Z", EXACT_FREQS) == 0.0)


# IEC 61672-1 class 1 acceptance limits (dB) at the exact third-octave
# frequencies above. They widen at the extremes, and are unbounded below at
# 16 kHz and up -- which is exactly where a bilinear-transformed filter
# under-responds, and why plain bilinear remains conformant.
CLASS1_UPPER = np.array([
    3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5,
    1.1, 1.1, 1.1, 1.1, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.1, 1.4, 1.6, 1.6, 1.6, 1.6, 1.6, 2.1, 2.1, 2.1,
    2.6, 3.0, 3.5, 4.0,
])
CLASS1_LOWER = np.array([
    -np.inf, -np.inf, -4.5, -2.5, -2.0, -2.0, -1.5, -1.5, -1.5, -1.5,
    -1.1, -1.1, -1.1, -1.1, -1.4, -1.4, -1.4, -1.4, -1.4, -1.4,
    -1.1, -1.4, -1.6, -1.6, -1.6, -1.6, -1.6, -2.1, -2.1, -3.1,
    -3.6, -6.0, -17.0, -np.inf,
])


@pytest.mark.parametrize("kind", ["A", "C"])
@pytest.mark.parametrize("fs", [44100, 48000])
def test_digital_filter_is_accurate_below_8khz(kind, fs):
    """Where the A-weighted energy actually lives, the filter must be tight."""
    sos = weighting_sos(kind, fs)
    freqs = EXACT_FREQS[EXACT_FREQS <= 8000]
    _, h = sosfreqz(sos, worN=2 * np.pi * freqs / fs)
    digital = 20 * np.log10(np.abs(h))
    analog = weighting_response_db(kind, freqs)
    assert np.max(np.abs(digital - analog)) < 0.7


@pytest.mark.parametrize("kind", ["A", "C"])
@pytest.mark.parametrize("fs", [44100, 48000])
def test_digital_filter_meets_class1_tolerance(kind, fs):
    """The bilinear filter must stay inside IEC 61672-1 class 1 across the band.

    Its deviation is one-sided (it under-responds near Nyquist), which the
    standard's asymmetric high-frequency limits permit.
    """
    sos = weighting_sos(kind, fs)
    freqs = EXACT_FREQS[EXACT_FREQS < fs / 2]
    _, h = sosfreqz(sos, worN=2 * np.pi * freqs / fs)
    error = 20 * np.log10(np.abs(h)) - weighting_response_db(kind, freqs)
    n = freqs.size
    assert np.all(error <= CLASS1_UPPER[:n]), f"{kind}@{fs}: over tolerance"
    assert np.all(error >= CLASS1_LOWER[:n]), f"{kind}@{fs}: under tolerance"


@pytest.mark.parametrize("slope,limit", [(-1.0, 0.3), (-2.0, 0.1)])
def test_bilinear_error_costs_little_on_realistic_spectra(slope, limit):
    """LAeq error for pink (-1) and traffic-like (-2) spectra stays small."""
    fs, n = 48000, 1 << 16
    freqs = np.fft.rfftfreq(n, 1 / fs)
    freqs[0] = 1e-6
    _, h = sosfreqz(weighting_sos("A", fs), worN=2 * np.pi * freqs / fs)
    digital = 20 * np.log10(np.abs(h))
    analog = weighting_response_db("A", freqs)

    power = freqs**slope
    power[(freqs < 10) | (freqs > 20000)] = 0
    exact = 10 * np.log10(np.sum(power * 10 ** (analog / 10)))
    got = 10 * np.log10(np.sum(power * 10 ** (digital / 10)))
    assert abs(got - exact) < limit


def test_z_weighting_sos_passes_signal_through():
    sos = weighting_sos("Z", 48000)
    x = np.random.default_rng(0).standard_normal(1000)
    assert StreamingFilter(sos)(x) == pytest.approx(x)


def test_streaming_filter_matches_one_shot():
    """Block boundaries must be invisible -- state has to carry across calls."""
    fs = 48000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10000)
    sos = weighting_sos("A", fs)

    one_shot = StreamingFilter(sos)(x)
    streaming = StreamingFilter(sos)
    chunked = np.concatenate([streaming(x[i : i + 512]) for i in range(0, x.size, 512)])
    assert chunked == pytest.approx(one_shot, abs=1e-12)


def test_a_weighted_level_of_a_tone():
    """A 1 kHz tone is unweighted; a 100 Hz tone loses 19.1 dB."""
    fs = 48000
    t = np.arange(4 * fs) / fs
    for freq, expected_drop in ((1000.0, 0.0), (100.0, 19.1)):
        x = np.sqrt(2) * np.sin(2 * np.pi * freq * t)  # unit RMS
        y = StreamingFilter(weighting_sos("A", fs))(x)
        settled = y[fs:]  # discard the filter's start-up transient
        level = 20 * np.log10(np.sqrt(np.mean(settled**2)))
        assert level == pytest.approx(-expected_drop, abs=0.2)


def test_exponential_level_reaches_steady_state():
    fs = 48000
    x = np.full(fs * 2, 0.5)  # constant amplitude -> mean square 0.25
    level = ExponentialLevel(fs, tau=0.125)
    out = level(x)
    assert out[-1] == pytest.approx(0.25, rel=1e-6)


def test_exponential_level_time_constant():
    """After one tau, a step reaches 1 - 1/e of its final value."""
    fs = 48000
    tau = 0.125
    level = ExponentialLevel(fs, tau)
    level(np.zeros(fs))  # prime at zero
    out = level(np.ones(fs))
    assert out[int(tau * fs) - 1] == pytest.approx(1 - np.exp(-1.0), abs=0.01)


def test_exponential_level_is_block_size_independent():
    fs = 48000
    rng = np.random.default_rng(2)
    x = rng.standard_normal(20000)
    a = ExponentialLevel(fs, 0.125)(x)
    streaming = ExponentialLevel(fs, 0.125)
    b = np.concatenate([streaming(x[i : i + 977]) for i in range(0, x.size, 977)])
    assert b == pytest.approx(a, abs=1e-12)
