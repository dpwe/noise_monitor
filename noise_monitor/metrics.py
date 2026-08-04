"""Accumulation of standard sound level statistics over a logging interval."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Fast levels are sampled this often for the statistical (Ln) percentiles.
#: 100 ms is the usual practice and keeps the interval buffers small.
STAT_SAMPLE_INTERVAL_S = 0.1


@dataclass
class IntervalStats:
    """One logged row: equivalent, extreme and statistical levels."""

    #: Unix timestamp of the END of the interval.
    timestamp: float
    duration_s: float
    weighting: str
    leq: float  # equivalent continuous level over the interval
    lmax: float  # max time-weighted level
    lmin: float  # min time-weighted level
    l10: float  # level exceeded 10% of the interval
    l50: float
    l90: float  # level exceeded 90% -- the usual background/residual indicator
    lpeak: float  # true (unweighted-in-time) peak sample level
    clipped_samples: int
    dropped_blocks: int

    def as_row(self) -> dict[str, object]:
        w = self.weighting.upper()
        return {
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 3),
            f"L{w}eq": round(self.leq, 2),
            f"L{w}max": round(self.lmax, 2),
            f"L{w}min": round(self.lmin, 2),
            f"L{w}10": round(self.l10, 2),
            f"L{w}50": round(self.l50, 2),
            f"L{w}90": round(self.l90, 2),
            "Lpeak": round(self.lpeak, 2),
            "clipped_samples": self.clipped_samples,
            "dropped_blocks": self.dropped_blocks,
        }


@dataclass
class IntervalAccumulator:
    """Accumulates weighted samples and emits one IntervalStats per interval.

    Interval boundaries are *sample accurate*: a block straddling a boundary is
    split, so intervals never drift with the audio block size. Timestamps are
    derived from the sample count rather than the wall clock, so a run of
    intervals is exactly evenly spaced regardless of scheduling jitter.

    All inputs are linear (mean-square / amplitude) quantities; dB conversion
    happens once at emit time, with `spl_offset_db` turning dBFS into dB SPL.
    """

    samplerate: int
    interval_s: float
    weighting: str = "A"
    spl_offset_db: float = 0.0
    #: Wall-clock time corresponding to sample 0 of the stream.
    start_time: float = 0.0

    _sum_sq: float = 0.0
    _n: int = 0
    _max_ms: float = 0.0
    _min_ms: float = float("inf")
    _peak: float = 0.0
    _stat_samples: list[float] = field(default_factory=list)
    _stat_phase: int = 0
    _total_samples: int = 0
    clipped_samples: int = 0
    dropped_blocks: int = 0

    def __post_init__(self) -> None:
        self._stat_stride = max(1, int(round(STAT_SAMPLE_INTERVAL_S * self.samplerate)))
        self._target_n = max(1, int(round(self.interval_s * self.samplerate)))

    def add(
        self,
        weighted: np.ndarray,
        time_weighted_ms: np.ndarray,
        raw: np.ndarray,
        clipped_mask: np.ndarray | None = None,
    ) -> list[IntervalStats]:
        """Add one block, returning any intervals it completed.

        `weighted` is the frequency-weighted signal, `time_weighted_ms` its
        exponentially time-weighted mean square (same length), and `raw` the
        uncorrected input used for peak and clipping detection.
        """
        completed: list[IntervalStats] = []
        pos = 0
        total = weighted.size
        while pos < total:
            take = min(self._target_n - self._n, total - pos)
            end = pos + take
            self._ingest(
                weighted[pos:end],
                time_weighted_ms[pos:end],
                raw[pos:end],
                None if clipped_mask is None else clipped_mask[pos:end],
            )
            pos = end
            if self._n >= self._target_n:
                completed.append(self._emit())
        return completed

    def _ingest(
        self,
        weighted: np.ndarray,
        time_weighted_ms: np.ndarray,
        raw: np.ndarray,
        clipped_mask: np.ndarray | None,
    ) -> None:
        self._sum_sq += float(np.dot(weighted, weighted))
        self._n += weighted.size
        self._total_samples += weighted.size

        if time_weighted_ms.size:
            self._max_ms = max(self._max_ms, float(time_weighted_ms.max()))
            self._min_ms = min(self._min_ms, float(time_weighted_ms.min()))
            # Decimate onto a stream-global sample grid, so neither block nor
            # interval boundaries shift the sampling instants.
            start = (-self._stat_phase) % self._stat_stride
            if start < time_weighted_ms.size:
                self._stat_samples.extend(time_weighted_ms[start :: self._stat_stride].tolist())
            self._stat_phase = (self._stat_phase + time_weighted_ms.size) % self._stat_stride

        if raw.size:
            self._peak = max(self._peak, float(np.abs(raw).max()))
        if clipped_mask is not None:
            self.clipped_samples += int(np.count_nonzero(clipped_mask))

    def ready(self) -> bool:
        return self._n >= self._target_n

    def _emit(self) -> IntervalStats:
        return self.emit(self.start_time + self._total_samples / self.samplerate)

    def emit(self, timestamp: float) -> IntervalStats:
        """Produce the interval's statistics and reset for the next one."""
        offset = self.spl_offset_db
        n = max(self._n, 1)
        leq = 10 * np.log10(max(self._sum_sq / n, 1e-20)) + offset

        if self._stat_samples:
            levels = 10 * np.log10(np.maximum(np.asarray(self._stat_samples), 1e-20)) + offset
            # Ln = level exceeded n% of the time = the (100-n)th percentile.
            l10, l50, l90 = np.percentile(levels, [90, 50, 10])
        else:
            l10 = l50 = l90 = leq

        lmax = 10 * np.log10(max(self._max_ms, 1e-20)) + offset
        lmin = 10 * np.log10(max(self._min_ms if self._min_ms != float("inf") else 0.0, 1e-20)) + offset
        # Peak is an amplitude, hence 20*log10, and is unweighted in time.
        lpeak = 20 * np.log10(max(self._peak, 1e-10)) + offset

        stats = IntervalStats(
            timestamp=timestamp,
            duration_s=self._n / self.samplerate,
            weighting=self.weighting,
            leq=float(leq),
            lmax=float(lmax),
            lmin=float(lmin),
            l10=float(l10),
            l50=float(l50),
            l90=float(l90),
            lpeak=float(lpeak),
            clipped_samples=self.clipped_samples,
            dropped_blocks=self.dropped_blocks,
        )
        self._reset()
        return stats

    def _reset(self) -> None:
        self._sum_sq = 0.0
        self._n = 0
        self._max_ms = 0.0
        self._min_ms = float("inf")
        self._peak = 0.0
        self._stat_samples.clear()
        # _stat_phase deliberately survives: it tracks a stream-global grid.
        self.clipped_samples = 0
        self.dropped_blocks = 0
