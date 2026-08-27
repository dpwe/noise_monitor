"""Hours-scale averaging of the spectrogram and the broadband level.

The live spectrogram covers a few tens of seconds. This keeps a second, much
slower one alongside it: `column_s` of live columns are averaged into one
column of a fixed-width image spanning the past day.

Averaging happens in the **power** domain, not in dB. Averaging logarithms
would under-report a day shaped by short loud events -- the mean of 40 and
80 dB is 60 dB, but their energy mean is 77 dB. Each output column is therefore
a genuine Leq per band over its window, and the level trace is the broadband
Leq over the same window.

Columns sit on **absolute wall-clock slots**: column N covers
`[N*column_s, (N+1)*column_s)` in Unix time. That is what makes the history
resumable. A relaunch drops back onto the same grid, so stored columns land at
the times they actually happened, and hours the monitor was not running stay
empty rather than being quietly closed up.

Storage is a fixed array scrolled in place, so a 24 hour history costs the same
as a 24 second one.
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

#: Floor applied before taking a logarithm, so silence gives -200 dB not -inf.
_POWER_FLOOR = 1e-20

#: Bumped when the stored layout changes in a way an old file cannot satisfy.
HISTORY_FORMAT = 1


@dataclass(frozen=True)
class LongTermSnapshot:
    """A copy of the long-term history, safe to read from the GUI thread."""

    #: (n_columns, n_bands) band levels in dB, oldest column first.
    image: np.ndarray
    #: (n_columns,) broadband level in dB, NaN where no audio was recorded.
    levels: np.ndarray
    #: Seconds of audio behind each column.
    column_s: float
    #: Absolute slot index of the newest column, or None before any audio.
    slot: int | None

    @property
    def n_columns(self) -> int:
        return int(self.image.shape[0])

    @property
    def span_s(self) -> float:
        return self.n_columns * self.column_s

    @property
    def end_time(self) -> float | None:
        """Unix time of the right-hand edge, or None before any audio.

        The edge is the *end* of the newest slot, so it can be up to one
        column into the future. That is what keeps the columns aligned to
        round clock times, which is the point of the slot grid.
        """
        return None if self.slot is None else (self.slot + 1) * self.column_s

    @property
    def has_levels(self) -> bool:
        return bool(np.isfinite(self.levels).any())

    def centers_s(self) -> np.ndarray:
        """Seconds before `end_time` at each column's centre, oldest first.

        Negative, so they plot straight onto a "time before now" axis.
        """
        back = np.arange(self.n_columns, dtype=float)[::-1] + 0.5
        return -(back * self.column_s)


class LongTermAverage:
    """Accumulates live spectrogram columns into a scrolling long-term average.

    Not thread safe: `add` is called from the DSP thread, and `snapshot` and
    `state` must be called under the same lock.
    """

    def __init__(
        self,
        n_bands: int,
        column_s: float,
        n_columns: int,
        fill_db: float = 0.0,
    ):
        if min(n_bands, n_columns) < 1:
            raise ValueError("n_bands and n_columns must be >= 1")
        if column_s <= 0:
            raise ValueError("column_s must be positive")
        self.n_bands = int(n_bands)
        self.column_s = float(column_s)
        self.n_columns = int(n_columns)
        self.fill_db = float(fill_db)

        # Empty columns are filled with the colour scale's floor rather than
        # NaN, so the image renders as background instead of as holes.
        self._image = np.full((self.n_columns, self.n_bands), self.fill_db, dtype=np.float32)
        self._levels = np.full(self.n_columns, np.nan, dtype=np.float64)
        self._slot: int | None = None
        self._reset()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config) -> "LongTermAverage":
        ui = config.ui
        column_s = max(1.0, float(ui.long_column_s))
        n_columns = max(2, int(round(ui.long_span_s / column_s)))
        return cls(config.analysis.n_bands, column_s, n_columns, ui.db_min)

    @property
    def span_s(self) -> float:
        return self.n_columns * self.column_s

    def slot_of(self, when: float) -> int:
        return int(when // self.column_s)

    # ------------------------------------------------------------------
    def add(
        self,
        column_db: np.ndarray,
        level_db: float = float("nan"),
        when: float | None = None,
    ) -> bool:
        """Add one live column and its broadband level.

        Returns True if that closed a column, which is the cue to save.
        """
        when = time.time() if when is None else float(when)
        slot = self.slot_of(when)
        closed = False
        if self._slot is None:
            self._slot = slot
        elif slot < self._slot:
            # The clock went backwards (an NTP step). Dropping the sample is
            # safer than reopening a slot that has already been banked.
            return False
        elif slot > self._slot:
            self._close(slot)
            closed = True

        self._band_power += np.power(10.0, np.asarray(column_db, dtype=np.float64) / 10.0)
        if np.isfinite(level_db):
            self._level_power += 10.0 ** (float(level_db) / 10.0)
            self._level_count += 1
        self._count += 1
        return closed

    def snapshot(self) -> LongTermSnapshot:
        """Copy the history out, with the column in progress at the right edge.

        Including the partial column means the newest few minutes are visible
        immediately rather than after a whole averaging window; it is an average
        over less audio than the others, which only matters for the first one.
        """
        image = self._image.copy()
        levels = self._levels.copy()
        if self._count:
            image[-1], levels[-1] = self._average()
        return LongTermSnapshot(image, levels, self.column_s, self._slot)

    # ------------------------------------------------------------------
    def state(self) -> dict | None:
        """What `save_history` writes, or None if there is nothing to write."""
        if self._slot is None:
            return None
        snapshot = self.snapshot()
        return {
            "format": np.int32(HISTORY_FORMAT),
            # float16 holds a dB to 0.06, far finer than the colour scale can
            # show, and halves what a 24/7 monitor writes to its SD card.
            "image": snapshot.image.astype(np.float16),
            "levels": snapshot.levels.astype(np.float32),
            "column_s": np.float64(self.column_s),
            "n_bands": np.int32(self.n_bands),
            "slot": np.int64(self._slot),
        }

    def restore(self, state: dict | None, now: float | None = None) -> str:
        """Lay a stored history onto the current grid. Returns what happened.

        Columns are placed by the slot they were recorded in, so a gap while
        the monitor was down stays a gap. A store written under a different
        column length or band count describes a different picture and is
        refused rather than stretched to fit.
        """
        if state is None:
            return "no stored history"
        try:
            stored_column_s = float(state["column_s"])
            stored_bands = int(state["n_bands"])
            stored_slot = int(state["slot"])
            image = np.asarray(state["image"], dtype=np.float32)
            levels = np.asarray(state["levels"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return "stored history unreadable, starting empty"

        if abs(stored_column_s - self.column_s) > 1e-6:
            return (
                f"stored history ignored: {stored_column_s:g} s columns, "
                f"now {self.column_s:g} s"
            )
        if stored_bands != self.n_bands or image.shape[1] != self.n_bands:
            return f"stored history ignored: {stored_bands} bands, now {self.n_bands}"

        now = time.time() if now is None else float(now)
        current = self.slot_of(now)
        shift = current - stored_slot
        if shift < 0:
            return "stored history ignored: it is in the future"
        if shift >= self.n_columns:
            return "stored history is older than the window"

        # Where the stored newest column lands on our grid, and how much of
        # the stored array reaches back into the window from there.
        newest = self.n_columns - 1 - shift
        first = max(0, image.shape[0] - 1 - newest)
        count = image.shape[0] - first
        self._image[newest - count + 1 : newest + 1] = image[first:]
        self._levels[newest - count + 1 : newest + 1] = levels[first:]
        self._slot = current

        recovered = int(np.isfinite(self._levels).sum())
        gap = shift * self.column_s
        note = f"restored {recovered} columns"
        if gap >= self.column_s:
            note += f", {_gap_label(gap)} gap"
        return note

    # ------------------------------------------------------------------
    def _close(self, new_slot: int) -> None:
        """Bank the slot being accumulated, then scroll on to `new_slot`."""
        if self._count:
            self._image[-1], self._levels[-1] = self._average()

        shift = new_slot - self._slot
        if shift >= self.n_columns:
            self._image[:] = self.fill_db
            self._levels[:] = np.nan
        else:
            self._image[:-shift] = self._image[shift:]
            self._image[-shift:] = self.fill_db
            self._levels[:-shift] = self._levels[shift:]
            self._levels[-shift:] = np.nan
        self._slot = new_slot
        self._reset()

    def _average(self) -> tuple[np.ndarray, float]:
        bands = _to_db(self._band_power / max(self._count, 1))
        if self._level_count:
            level = float(_to_db(self._level_power / self._level_count))
        else:
            level = float("nan")
        return bands.astype(np.float32), level

    def _reset(self) -> None:
        self._band_power = np.zeros(self.n_bands, dtype=np.float64)
        self._level_power = 0.0
        self._level_count = 0
        self._count = 0


# ----------------------------------------------------------------------
def save_history(path: Path | str, state: dict) -> None:
    """Write the history, atomically.

    A power cut during the write must not leave a half-file where a good one
    was, so it goes to a temporary beside it and is renamed into place -- the
    rename is what the filesystem makes atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # A file object, not a name: np.savez would otherwise append its own .npz
    # to the temporary and rename the wrong path.
    with open(tmp, "wb") as handle:
        np.savez(handle, **state)
    tmp.replace(path)


def load_history(path: Path | str) -> dict | None:
    """Read a stored history, or None if there is not a usable one.

    A missing, truncated or alien file is not a reason to refuse to start; the
    worst case is an empty panel.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            state = {name: data[name] for name in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None
    if int(state.get("format", -1)) != HISTORY_FORMAT:
        return None
    return state


def _gap_label(seconds: float) -> str:
    """How long the monitor was down, in whatever unit reads best."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _to_db(power):
    return 10.0 * np.log10(np.maximum(power, _POWER_FLOOR))
