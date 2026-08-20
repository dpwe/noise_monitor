"""Runs capture -> analysis -> logging on a worker thread, feeding the UI."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .analysis import Analyzer
from .calibration import MicCalibration
from .capture import AudioSource
from .config import Config
from .logsink import CsvLogger
from .longterm import LongTermAverage, LongTermSnapshot
from .metrics import IntervalStats


@dataclass
class EngineState:
    """A snapshot of the meter, safe to read from the GUI thread."""

    level_db: float = float("nan")
    leq_avg: float = float("nan")
    clipped: bool = False
    clip_hold: bool = False
    overflows: int = 0
    dropped_blocks: int = 0
    last_interval: IntervalStats | None = None
    running: bool = False
    error: str | None = None


class MonitorEngine:
    """Owns the DSP thread. The UI only ever calls `drain()` and the properties."""

    def __init__(
        self,
        config: Config,
        source: AudioSource,
        cal: MicCalibration | None = None,
        logger: CsvLogger | None = None,
    ):
        self.config = config
        self.source = source
        self.analyzer = Analyzer(config, cal)
        self.logger = logger

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Enough columns for the whole visible spectrogram, so the UI can miss
        # a few refreshes without losing data.
        hops_per_s = config.audio.samplerate / config.analysis.hop
        self._max_columns = int(config.ui.history_s * hops_per_s) + 64
        self._columns: deque[np.ndarray] = deque(maxlen=self._max_columns)
        # The long-term panel is fed here rather than from drain(), so it keeps
        # accumulating whatever the UI does -- including not existing at all.
        self._long_term = LongTermAverage.from_config(config)
        self._state = EngineState()
        self._clip_hold_blocks = 0

    # ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.analyzer.calibrated

    @property
    def calibration_note(self) -> str:
        return self.analyzer.calibration_note

    @property
    def band_centers(self) -> np.ndarray:
        return self.analyzer.band_centers

    @property
    def average_s(self) -> float:
        """Window of the rolling Leq behind `EngineState.leq_avg`."""
        return self.analyzer.average_s

    def long_term(self) -> LongTermSnapshot:
        """A copy of the long-term average history."""
        with self._lock:
            return self._long_term.snapshot()

    def start(self) -> None:
        self._stop.clear()
        self.source.start()
        # PortAudio may hand back a different rate than requested; the DSP was
        # built for the configured one, so refuse rather than mis-measure.
        if self.source.samplerate != self.analyzer.samplerate:
            self.source.stop()
            raise RuntimeError(
                f"device opened at {self.source.samplerate} Hz but the analysis "
                f"chain was built for {self.analyzer.samplerate} Hz; set "
                f"audio.samplerate to match"
            )
        with self._lock:
            self._state.running = True
        self._thread = threading.Thread(target=self._run, name="dsp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.source.stop()
        if self.logger is not None:
            self.logger.close()
        with self._lock:
            self._state.running = False

    def drain(self) -> tuple[list[np.ndarray], EngineState]:
        """Take pending spectrogram columns and the current state."""
        with self._lock:
            columns = list(self._columns)
            self._columns.clear()
            state = EngineState(
                level_db=self._state.level_db,
                leq_avg=self._state.leq_avg,
                clipped=self._state.clipped,
                clip_hold=self._clip_hold_blocks > 0,
                overflows=self._state.overflows,
                dropped_blocks=self._state.dropped_blocks,
                last_interval=self._state.last_interval,
                running=self._state.running,
                error=self._state.error,
            )
        return columns, state

    # ------------------------------------------------------------------
    def _run(self) -> None:
        blocks_per_clip_hold = max(1, int(0.75 * self.config.audio.samplerate / self.config.audio.blocksize))
        try:
            while not self._stop.is_set():
                block = self.source.read(timeout=0.5)
                if block is None:
                    continue

                dropped = self.source.take_dropped()
                if dropped:
                    self.analyzer.note_dropped_blocks(dropped)

                frame = self.analyzer.process(block)

                intervals = self.analyzer.pop_intervals()
                if self.logger is not None:
                    for stats in intervals:
                        self.logger.write(stats)

                if frame.clipped:
                    self._clip_hold_blocks = blocks_per_clip_hold
                elif self._clip_hold_blocks > 0:
                    self._clip_hold_blocks -= 1

                with self._lock:
                    self._columns.extend(frame.columns)
                    for column in frame.columns:
                        self._long_term.add(column, frame.leq_avg)
                    self._state.level_db = frame.level_db
                    self._state.leq_avg = frame.leq_avg
                    self._state.clipped = frame.clipped
                    self._state.dropped_blocks += dropped
                    self._state.overflows = getattr(self.source, "overflows", 0)
                    if intervals:
                        self._state.last_interval = intervals[-1]
        except Exception as exc:  # surface it in the UI rather than dying silently
            with self._lock:
                self._state.error = f"{type(exc).__name__}: {exc}"
                self._state.running = False
            raise
        finally:
            with self._lock:
                self._state.running = False
