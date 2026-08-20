"""Whole-chain checks: does a known acoustic input produce the right number?"""

import csv
import time

import numpy as np
import pytest

from noise_monitor.analysis import SETTLE_TIME_CONSTANTS, Analyzer
from noise_monitor.calibration import parse_cal_file
from noise_monitor.capture import ArraySource, synthetic_signal
from noise_monitor.config import Config
from noise_monitor.engine import MonitorEngine
from noise_monitor.logsink import CsvLogger
from noise_monitor.weighting import TIME_CONSTANTS

from .conftest import SENS_FACTOR


def make_config(**overrides) -> Config:
    cfg = Config()
    cfg.audio.samplerate = 48000
    cfg.audio.blocksize = 1024
    cfg.analysis.nfft = 2048
    cfg.analysis.hop = 512
    cfg.logging.enabled = False
    for path, value in overrides.items():
        section, _, key = path.partition(".")
        setattr(getattr(cfg, section), key, value)
    return cfg


def sine(freq, seconds, fs, rms=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return np.sqrt(2) * rms * np.sin(2 * np.pi * freq * t)


def run(analyzer, signal, block=1024):
    frames = [analyzer.process(signal[i : i + block]) for i in range(0, signal.size, block)]
    return frames


def test_calibrator_tone_reads_94_db(cal_file):
    """The defining case: a tone at the Sens Factor's dBFS level is 94 dB SPL.

    A 1 kHz tone is unweighted by A-weighting and uncorrected by the cal file
    (which is 0 dB at 1 kHz), so this isolates the absolute reference.
    """
    cal = parse_cal_file(cal_file)
    cfg = make_config()
    analyzer = Analyzer(cfg, cal)

    target_rms = 10 ** (SENS_FACTOR / 20.0)
    signal = sine(1000.0, 3.0, cfg.audio.samplerate, rms=target_rms)
    frames = run(analyzer, signal)
    assert frames[-1].level_db == pytest.approx(94.0, abs=0.1)
    assert frames[-1].leq_avg == pytest.approx(94.0, abs=0.1)


def test_a_weighting_applies_to_the_headline_number(cal_file):
    """The same pressure at 100 Hz must read 19.1 dB lower in dB(A)."""
    cal = parse_cal_file(cal_file)
    cfg = make_config()
    target_rms = 10 ** (SENS_FACTOR / 20.0)

    levels = {}
    for freq in (1000.0, 100.0):
        analyzer = Analyzer(cfg, cal)
        # Undo the mic's own response at 100 Hz so this tests weighting alone.
        correction = cal.response_at(np.array([freq]))[0]
        signal = sine(freq, 3.0, cfg.audio.samplerate, rms=target_rms * 10 ** (correction / 20))
        levels[freq] = run(analyzer, signal)[-1].level_db

    assert levels[1000.0] - levels[100.0] == pytest.approx(19.1, abs=0.3)


def test_uncalibrated_analyzer_reports_dbfs():
    cfg = make_config()
    analyzer = Analyzer(cfg, cal=None)
    assert not analyzer.calibrated
    signal = sine(1000.0, 2.0, cfg.audio.samplerate, rms=0.1)
    assert run(analyzer, signal)[-1].level_db == pytest.approx(-20.0, abs=0.1)


def test_band_levels_sum_to_the_broadband_level(cal_file):
    """Power conservation end to end: Z-weighted bands must sum to Leq(Z)."""
    cal = parse_cal_file(cal_file)
    cfg = make_config()
    cfg.analysis.weighting = "Z"
    cfg.analysis.nfft = 4096
    cfg.analysis.hop = 4096
    analyzer = Analyzer(cfg, cal)

    rng = np.random.default_rng(0)
    fs = cfg.audio.samplerate
    # Bandlimit well inside [fmin, fmax] so no power falls off the ends.
    noise = rng.standard_normal(fs * 4)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(noise.size, 1 / fs)
    spec[(freqs < 100) | (freqs > 10000)] = 0
    signal = np.fft.irfft(spec, n=noise.size)
    signal *= 0.05 / np.sqrt(np.mean(signal**2))

    frames = run(analyzer, signal)
    columns = [c for f in frames for c in f.columns]
    assert columns, "expected spectrogram columns"

    band_total = np.mean([10 ** (c / 10) for c in columns], axis=0).sum()
    band_db = 10 * np.log10(band_total)
    assert band_db == pytest.approx(frames[-1].leq_avg, abs=0.5)


def test_spectrogram_column_shape_and_rate(cal_file):
    cal = parse_cal_file(cal_file)
    cfg = make_config()
    analyzer = Analyzer(cfg, cal)
    signal = sine(1000.0, 1.0, cfg.audio.samplerate, rms=0.01)
    columns = [c for f in run(analyzer, signal) for c in f.columns]
    assert all(c.shape == (cfg.analysis.n_bands,) for c in columns)
    # One column per hop, less the initial fill of the FFT window.
    expected = (signal.size - cfg.analysis.nfft) // cfg.analysis.hop + 1
    assert abs(len(columns) - expected) <= 1


def test_clipping_is_detected():
    cfg = make_config()
    analyzer = Analyzer(cfg, cal=None)
    quiet = sine(1000.0, 0.1, cfg.audio.samplerate, rms=0.01)
    assert not any(f.clipped for f in run(analyzer, quiet))

    loud = np.clip(sine(1000.0, 0.1, cfg.audio.samplerate, rms=1.0), -1.0, 1.0)
    assert any(f.clipped for f in run(analyzer, loud))


def test_intervals_are_emitted_at_the_configured_rate():
    cfg = make_config()
    cfg.logging.interval_s = 0.5
    analyzer = Analyzer(cfg, cal=None)
    signal = sine(1000.0, 4.0, cfg.audio.samplerate, rms=0.1)
    run(analyzer, signal)
    intervals = analyzer.pop_intervals()

    # 4 s of audio, less the detector warm-up, in 0.5 s intervals.
    settle = SETTLE_TIME_CONSTANTS * TIME_CONSTANTS[cfg.analysis.time_weighting]
    assert len(intervals) == int((4.0 - settle) / 0.5)
    assert all(iv.duration_s == pytest.approx(0.5) for iv in intervals)
    spacing = np.diff([iv.timestamp for iv in intervals])
    assert spacing == pytest.approx(np.full(len(intervals) - 1, 0.5))
    assert analyzer.pop_intervals() == []


def test_startup_transient_is_excluded_from_statistics():
    """Without a warm-up, the first interval's Lmin reports the meter settling."""
    cfg = make_config()
    cfg.logging.interval_s = 1.0
    analyzer = Analyzer(cfg, cal=None)
    # A steady tone: every statistic should agree to within a fraction of a dB.
    signal = sine(1000.0, 3.0, cfg.audio.samplerate, rms=0.1)
    run(analyzer, signal)
    first = analyzer.pop_intervals()[0]
    assert first.lmin == pytest.approx(first.leq, abs=0.1)
    assert first.l90 == pytest.approx(first.leq, abs=0.1)
    assert first.leq == pytest.approx(-20.0, abs=0.1)


def test_engine_runs_and_logs(tmp_path, cal_file):
    cal = parse_cal_file(cal_file)
    cfg = make_config()
    cfg.logging.enabled = True
    cfg.logging.interval_s = 0.2
    cfg.ui.history_s = 2.0

    fs = cfg.audio.samplerate
    signal = synthetic_signal(fs, duration_s=2.0, target_dbfs=-30.0)
    source = ArraySource(signal, fs, cfg.audio.blocksize, realtime=False, loop=False)
    logger = CsvLogger(tmp_path, rotate_daily=False)
    engine = MonitorEngine(cfg, source, cal, logger)

    engine.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
        _, state = engine.drain()
        if state.error:
            pytest.fail(state.error)
        if state.last_interval is not None and not source.queue.qsize():
            break
    engine.stop()

    path = logger.path_for(__import__("datetime").datetime.now())
    rows = list(csv.DictReader(open(path)))
    assert len(rows) >= 5
    assert {"time", "LAeq", "LAmax", "LA90", "Lpeak"} <= set(rows[0])
    # -30 dBFS pink noise through A-weighting, plus the ~94.6 dB offset.
    leq = float(rows[-1]["LAeq"])
    assert 40.0 < leq < 90.0, leq


def test_engine_rejects_a_samplerate_mismatch():
    cfg = make_config()
    source = ArraySource(np.zeros(1000), 44100, cfg.audio.blocksize)
    engine = MonitorEngine(cfg, source, cal=None)
    with pytest.raises(RuntimeError, match="44100"):
        engine.start()
