"""Hours-scale averaging of the spectrogram and the broadband level.

The live spectrogram covers a few tens of seconds. This keeps a second, much
slower one alongside it: every `hops_per_column` live columns are averaged into
one column of a fixed-width image spanning the past day.

Averaging happens in the **power** domain, not in dB. Averaging logarithms
would under-report a day dominated by short loud events -- the mean of 40 and
80 dB is 60 dB, but their energy mean is 77 dB. Each output column is therefore
a genuine Leq per band over its window, and the level trace is the broadband
Leq over the same window.

Storage is a fixed array scrolled in place, so a 24 hour history costs the same
as a 24 second one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

#: Floor applied before taking a logarithm, so silence gives -200 dB not -inf.
_POWER_FLOOR = 1e-20


@dataclass(frozen=True)
class LongTermSnapshot:
    """A copy of the long-term history, safe to read from the GUI thread."""

    #: (n_columns, n_bands) band levels in dB, oldest column first.
    image: np.ndarray
    #: (n_columns,) broadband level in dB, NaN where no audio has arrived yet.
    levels: np.ndarray
    #: Seconds of audio behind each column.
    column_s: float
    #: How many of the trailing columns hold data. The rest are still empty.
    n_valid: int

    @property
    def n_columns(self) -> int:
        return int(self.image.shape[0])

    @property
    def span_s(self) -> float:
        return self.n_columns * self.column_s

    def ages_s(self) -> np.ndarray:
        """Seconds before now at the centre of each valid column, oldest first.

        Positive numbers: 0 is now. Only the valid (trailing) columns are
        described, so this lines up with `levels[-n_valid:]`.
        """
        newest_first = (np.arange(self.n_valid, dtype=float) + 0.5) * self.column_s
        return newest_first[::-1]


class LongTermAverage:
    """Accumulates live spectrogram columns into a scrolling long-term average.

    Not thread safe: `add` is called from the DSP thread, and `snapshot` must be
    called under the same lock.
    """

    def __init__(
        self,
        n_bands: int,
        hops_per_column: int,
        n_columns: int,
        hop_s: float = 1.0,
        fill_db: float = 0.0,
    ):
        if min(n_bands, hops_per_column, n_columns) < 1:
            raise ValueError("n_bands, hops_per_column and n_columns must be >= 1")
        self.n_bands = int(n_bands)
        self.hops_per_column = int(hops_per_column)
        self.n_columns = int(n_columns)
        self.hop_s = float(hop_s)
        self.fill_db = float(fill_db)

        # Empty columns are filled with the colour scale's floor rather than
        # NaN, so the image renders as background instead of as holes.
        self._image = np.full((self.n_columns, self.n_bands), self.fill_db, dtype=np.float32)
        self._levels = np.full(self.n_columns, np.nan, dtype=np.float64)
        self._n_valid = 0

        self._band_power = np.zeros(self.n_bands, dtype=np.float64)
        self._level_power = 0.0
        self._level_count = 0
        self._count = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config) -> "LongTermAverage":
        ancfg, ui = config.analysis, config.ui
        hop_s = ancfg.hop / config.audio.samplerate
        hops_per_column = max(1, int(round(ui.long_column_s / hop_s)))
        n_columns = max(2, int(round(ui.long_span_s / (hops_per_column * hop_s))))
        return cls(ancfg.n_bands, hops_per_column, n_columns, hop_s, ui.db_min)

    @property
    def column_s(self) -> float:
        return self.hops_per_column * self.hop_s

    @property
    def span_s(self) -> float:
        return self.n_columns * self.column_s

    # ------------------------------------------------------------------
    def add(self, column_db: np.ndarray, level_db: float = float("nan")) -> bool:
        """Add one live column and its broadband level.

        Returns True if that completed a long-term column.
        """
        self._band_power += np.power(10.0, np.asarray(column_db, dtype=np.float64) / 10.0)
        if np.isfinite(level_db):
            self._level_power += 10.0 ** (float(level_db) / 10.0)
            self._level_count += 1
        self._count += 1
        if self._count < self.hops_per_column:
            return False
        image_column, level = self._average()
        self._image[:-1] = self._image[1:]
        self._image[-1] = image_column
        self._levels[:-1] = self._levels[1:]
        self._levels[-1] = level
        self._n_valid = min(self._n_valid + 1, self.n_columns)
        self._reset()
        return True

    def snapshot(self) -> LongTermSnapshot:
        """Copy the history out, with the column in progress at the right edge.

        Including the partial column means the newest few minutes are visible
        immediately rather than after a whole averaging window; it is an average
        over less audio than the others, which only matters for the first one.
        """
        image = self._image.copy()
        levels = self._levels.copy()
        n_valid = self._n_valid
        if self._count:
            image[:-1] = image[1:]
            levels[:-1] = levels[1:]
            image[-1], levels[-1] = self._average()
            n_valid = min(n_valid + 1, self.n_columns)
        return LongTermSnapshot(image, levels, self.column_s, n_valid)

    # ------------------------------------------------------------------
    def _average(self) -> tuple[np.ndarray, float]:
        bands = _to_db(self._band_power / max(self._count, 1))
        if self._level_count:
            level = float(_to_db(self._level_power / self._level_count))
        else:
            level = float("nan")
        return bands.astype(np.float32), level

    def _reset(self) -> None:
        self._band_power[:] = 0.0
        self._level_power = 0.0
        self._level_count = 0
        self._count = 0


def _to_db(power):
    return 10.0 * np.log10(np.maximum(power, _POWER_FLOOR))
