"""The analysis chain: raw blocks in, calibrated spectrogram columns and levels out."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    MicCalibration,
    bin_correction_db,
    design_correction_fir,
    resolve_spl_offset,
)
from .config import Config
from .metrics import IntervalAccumulator, IntervalStats
from .spectrum import BandMapper, StreamingSTFT, band_levels_db
from .weighting import (
    TIME_CONSTANTS,
    ExponentialLevel,
    StreamingFilter,
    StreamingFIR,
    weighting_response_db,
    weighting_sos,
)

#: Time constants of exponential-detector settling to discard at start-up.
#: Five gets within 1% of the steady-state value.
SETTLE_TIME_CONSTANTS = 5.0


@dataclass
class AnalysisFrame:
    """Everything the UI needs from one processed audio block."""

    #: Zero or more spectrogram columns (band levels in dB SPL), oldest first.
    columns: list[np.ndarray] = field(default_factory=list)
    #: Current exponentially time-weighted level, e.g. LAF, in dB SPL.
    level_db: float = float("nan")
    #: Linear (rectangular) Leq over the last second.
    leq_1s: float = float("nan")
    #: True if any sample in this block reached the clipping threshold.
    clipped: bool = False


class Analyzer:
    """Owns all DSP state. Not thread safe -- run it on one thread."""

    def __init__(self, config: Config, cal: MicCalibration | None = None):
        self.config = config
        self.cal = cal
        acfg, ccfg, ancfg = config.audio, config.calibration, config.analysis
        self.samplerate = acfg.samplerate

        self.spl_offset_db, self.calibration_note = resolve_spl_offset(
            cal, ccfg.spl_offset_db, ccfg.input_gain_db
        )
        self.calibrated = "UNCALIBRATED" not in self.calibration_note

        # --- microphone frequency response correction -------------------
        # Either correct the time-domain signal with an FIR (so the meter and
        # the spectrogram share one correction), or correct the spectrum bins
        # only. Doing both would double-count.
        self._fir: StreamingFIR | None = None
        want_fr = ccfg.apply_frequency_correction and cal is not None
        if want_fr and ccfg.correct_broadband:
            taps = design_correction_fir(
                cal, self.samplerate, ccfg.fir_taps, ccfg.max_boost_db
            )
            self._fir = StreamingFIR(taps)

        # --- spectral analysis -------------------------------------------
        self.stft = StreamingSTFT(ancfg.nfft, ancfg.hop, self.samplerate)
        self.bands = BandMapper(
            self.stft.freqs, ancfg.fmin, ancfg.fmax, ancfg.n_bands, self.samplerate
        )
        if want_fr and self._fir is None:
            corr_db = bin_correction_db(cal, self.stft.freqs, ccfg.max_boost_db)
            self._bin_power_gain = 10 ** (corr_db / 10.0)
        else:
            self._bin_power_gain = None

        # Optional weighting applied to the displayed bands (Z = as measured).
        self._band_weight_db = weighting_response_db(
            ancfg.spectrogram_weighting, self.bands.band_centers
        )

        # --- broadband meter ----------------------------------------------
        self.weighting = ancfg.weighting.upper()
        self._weight = StreamingFilter(weighting_sos(self.weighting, self.samplerate))
        tau = TIME_CONSTANTS.get(ancfg.time_weighting.lower())
        if tau is None:
            raise ValueError(
                f"unknown time_weighting {ancfg.time_weighting!r}; expected 'fast' or 'slow'"
            )
        self._time_weight = ExponentialLevel(self.samplerate, tau)

        self._interval = IntervalAccumulator(
            samplerate=self.samplerate,
            interval_s=config.logging.interval_s,
            weighting=self.weighting,
            spl_offset_db=self.spl_offset_db,
        )
        self._pending: list[IntervalStats] = []
        self._started = False
        # The exponential detector needs a few time constants before its output
        # means anything. Samples inside that window are excluded from the
        # logged statistics, otherwise the first interval's Lmin and L90 report
        # the meter warming up rather than the sound field.
        self._settle_remaining = int(round(SETTLE_TIME_CONSTANTS * tau * self.samplerate))

        # Rolling one-second Leq, as (sum of squares, sample count) per block.
        self._recent: deque[tuple[float, int]] = deque()
        self._recent_samples = 0

        self._clip_amplitude = 10 ** (acfg.clip_threshold_dbfs / 20.0)

    # ------------------------------------------------------------------
    @property
    def band_centers(self) -> np.ndarray:
        return self.bands.band_centers

    def note_dropped_blocks(self, count: int) -> None:
        """Record blocks lost to a queue overflow, for the log."""
        self._interval.dropped_blocks += count

    def process(self, block: np.ndarray) -> AnalysisFrame:
        raw = np.asarray(block, dtype=np.float64).reshape(-1)
        if raw.size == 0:
            return AnalysisFrame()

        clipped_mask = np.abs(raw) >= self._clip_amplitude
        clipped = bool(clipped_mask.any())

        signal = self._fir(raw) if self._fir is not None else raw

        # Broadband: frequency weighting, then exponential time weighting.
        weighted = self._weight(signal)
        time_ms = self._time_weight(weighted)

        skip = min(self._settle_remaining, raw.size)
        self._settle_remaining -= skip
        if skip < raw.size:
            if not self._started:
                # Anchor the interval timeline to the first sample that counts.
                self._interval.start_time = time.time() - (raw.size - skip) / self.samplerate
                self._started = True
            self._pending.extend(
                self._interval.add(
                    weighted[skip:], time_ms[skip:], raw[skip:], clipped_mask[skip:]
                )
            )

        level_db = 10 * np.log10(max(float(time_ms[-1]), 1e-20)) + self.spl_offset_db
        leq_1s = self._rolling_leq(weighted)

        # Spectrogram columns.
        columns = []
        for power in self.stft.push(signal):
            if self._bin_power_gain is not None:
                power = power * self._bin_power_gain
            band_power = self.bands(power)
            db = band_levels_db(
                band_power,
                self.spl_offset_db,
                self.bands.bandwidths,
                self.config.analysis.scale,
            )
            columns.append(db + self._band_weight_db)

        return AnalysisFrame(
            columns=columns, level_db=level_db, leq_1s=leq_1s, clipped=clipped
        )

    def pop_intervals(self) -> list[IntervalStats]:
        """Return and clear any completed logging intervals."""
        out, self._pending = self._pending, []
        return out

    # ------------------------------------------------------------------
    def _rolling_leq(self, weighted: np.ndarray) -> float:
        self._recent.append((float(np.dot(weighted, weighted)), weighted.size))
        self._recent_samples += weighted.size
        while self._recent_samples - self._recent[0][1] >= self.samplerate:
            _, n = self._recent.popleft()
            self._recent_samples -= n
        total = sum(s for s, _ in self._recent)
        return 10 * np.log10(max(total / max(self._recent_samples, 1), 1e-20)) + self.spl_offset_db
