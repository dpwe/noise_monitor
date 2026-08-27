import numpy as np
import pytest
from scipy.signal import freqz

from noise_monitor.calibration import (
    MIN_PLAUSIBLE_CLIP_SPL,
    REFERENCE_CALIBRATOR_SPL,
    bin_correction_db,
    clipping_spl,
    design_correction_fir,
    headroom_warning,
    parse_cal_file,
    resolve_spl_offset,
)

from .conftest import SENS_FACTOR, SERIAL


def test_parses_header_and_points(cal_file):
    cal = parse_cal_file(cal_file)
    assert cal.sens_factor_db == pytest.approx(SENS_FACTOR)
    assert cal.serial == SERIAL
    assert len(cal.frequencies) == 616
    assert cal.frequencies[0] == pytest.approx(10.054, abs=1e-3)
    assert cal.frequencies[-1] == pytest.approx(20016.816, abs=1e-2)
    assert np.all(np.diff(cal.frequencies) > 0)


def test_parses_90deg_variant_with_comment_line(cal_file_90deg):
    cal = parse_cal_file(cal_file_90deg)
    assert cal.sens_factor_db == pytest.approx(SENS_FACTOR)
    assert len(cal.frequencies) == 616


def test_bare_leading_dot_sensitivity(tmp_path):
    """Real files write '.58440' and '-.6162' rather than '0.58440'."""
    for text, expected in ((".58440", 0.5844), ("-.6162", -0.6162), ("1.2345", 1.2345)):
        path = tmp_path / f"cal{text}.txt"
        path.write_text(f"Sens Factor ={text}dB, SERNO: 1234\n100\t0.0\n1000\t0.0\n")
        assert parse_cal_file(path).sens_factor_db == pytest.approx(expected)


def test_response_is_zero_at_1khz(cal_file):
    cal = parse_cal_file(cal_file)
    assert cal.response_at(np.array([1000.0]))[0] == pytest.approx(0.0, abs=0.01)


def test_response_holds_endpoints_outside_range(cal_file):
    cal = parse_cal_file(cal_file)
    below = cal.response_at(np.array([1.0]))[0]
    above = cal.response_at(np.array([40000.0]))[0]
    assert below == pytest.approx(cal.response_db[0])
    assert above == pytest.approx(cal.response_db[-1])


def test_spl_offset_follows_rew_definition(cal_file):
    """94 dB SPL must map back onto the Sens Factor's own dBFS reading."""
    cal = parse_cal_file(cal_file)
    offset = cal.spl_offset_db()
    assert offset == pytest.approx(REFERENCE_CALIBRATOR_SPL - SENS_FACTOR)
    # A signal measuring `sens_factor` dBFS is by definition the 94 dB reference.
    assert SENS_FACTOR + offset == pytest.approx(REFERENCE_CALIBRATOR_SPL)


def test_input_gain_shifts_offset(cal_file):
    cal = parse_cal_file(cal_file)
    assert cal.spl_offset_db(6.0) - cal.spl_offset_db(0.0) == pytest.approx(6.0)


def test_explicit_offset_overrides_cal_file(cal_file):
    cal = parse_cal_file(cal_file)
    offset, note = resolve_spl_offset(cal, 123.4, 0.0)
    assert offset == pytest.approx(123.4)
    assert "calibrator" in note


def test_uncalibrated_is_flagged_not_faked():
    offset, note = resolve_spl_offset(None, None, 0.0)
    assert offset == 0.0
    assert "UNCALIBRATED" in note


def test_bin_correction_negates_response(cal_file):
    cal = parse_cal_file(cal_file)
    freqs = np.array([50.0, 200.0, 1000.0, 8000.0])
    corr = bin_correction_db(cal, freqs, max_boost_db=30.0)
    assert corr == pytest.approx(-cal.response_at(freqs))


def test_bin_correction_is_clamped(cal_file):
    cal = parse_cal_file(cal_file)
    corr = bin_correction_db(cal, np.array([10.0]), max_boost_db=2.0)
    assert abs(corr[0]) <= 2.0


def _fir_residual_db(cal, fs, taps, test_freqs):
    _, h = freqz(taps, worN=2 * np.pi * test_freqs / fs)
    return 20 * np.log10(np.abs(h)) + cal.response_at(test_freqs)


def test_correction_fir_flattens_the_response(cal_file):
    """The FIR's magnitude must be the inverse of the mic's, in-band."""
    cal = parse_cal_file(cal_file)
    fs = 48000
    taps = design_correction_fir(cal, fs, numtaps=2047, max_boost_db=20.0)
    residual = _fir_residual_db(cal, fs, taps, np.geomspace(63, 16000, 60))
    assert np.max(np.abs(residual)) < 0.3, f"worst residual {np.max(np.abs(residual)):.2f} dB"


def test_correction_fir_is_limited_at_low_frequency(cal_file):
    """An FIR cannot resolve fine detail below roughly fs/numtaps.

    At 48 kHz with 2047 taps that floor is about 23 Hz, so the correction is
    approximate in the bottom octave. A-weighting is already 40 dB down there,
    so this costs nothing in dB(A) -- but it should not silently get worse.
    """
    cal = parse_cal_file(cal_file)
    fs = 48000
    taps = design_correction_fir(cal, fs, numtaps=2047, max_boost_db=20.0)
    residual = _fir_residual_db(cal, fs, taps, np.geomspace(25, 63, 20))
    assert np.max(np.abs(residual)) < 1.5


def test_correction_fir_respects_the_boost_limit(cal_file):
    cal = parse_cal_file(cal_file)
    fs = 48000
    taps = design_correction_fir(cal, fs, numtaps=1023, max_boost_db=3.0)
    _, h = freqz(taps, worN=2 * np.pi * np.geomspace(20, 20000, 200) / fs)
    assert np.max(20 * np.log10(np.abs(h))) < 3.0 + 0.5


# --- the headroom sanity check ----------------------------------------
# A Sens-Factor-only offset makes a measurement mic look like it overloads
# at conversation level. That is the cheapest, loudest symptom of an offset
# roughly 30 dB too small, so the tools check for it.

def test_clipping_level_is_the_offset_less_a_full_scale_sine():
    assert clipping_spl(125.06) == pytest.approx(122.05, abs=0.01)


def test_an_impossible_overload_point_is_flagged():
    """94 - (-1.055) with no nominal sensitivity: clips at 92 dB SPL."""
    warning = headroom_warning(95.06)
    assert warning is not None
    assert "92 dB SPL" in warning
    assert "30 dB" in warning
    assert "calibrate" in warning


def test_a_plausible_offset_is_not_flagged():
    """The same microphone with its nominal sensitivity restored."""
    assert headroom_warning(125.06) is None


def test_the_threshold_sits_between_the_two():
    """The offset that puts clipping exactly on the threshold is MIN + 3.01,
    because a full-scale sine is 3.01 dB below the offset, not above it."""
    boundary = MIN_PLAUSIBLE_CLIP_SPL + 3.01
    assert headroom_warning(boundary - 0.1) is not None
    assert headroom_warning(boundary + 0.1) is None


def test_an_uncalibrated_zero_offset_is_flagged_too():
    """dBFS reported as SPL is the most wrong of all."""
    assert headroom_warning(0.0) is not None
