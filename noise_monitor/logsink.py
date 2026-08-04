"""CSV logging of interval statistics, with optional daily rotation."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from .metrics import IntervalStats


class CsvLogger:
    """Appends one row per logging interval.

    Rows are flushed immediately so a power cut costs at most one interval, and
    an existing file is appended to (header written only when creating).
    """

    def __init__(self, directory: Path, rotate_daily: bool = True, prefix: str = "noise"):
        self.directory = Path(directory)
        self.rotate_daily = rotate_daily
        self.prefix = prefix
        self._path: Path | None = None
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._day: str | None = None

    def path_for(self, when: dt.datetime) -> Path:
        if self.rotate_daily:
            return self.directory / f"{self.prefix}-{when:%Y%m%d}.csv"
        return self.directory / f"{self.prefix}.csv"

    def write(self, stats: IntervalStats) -> None:
        when = dt.datetime.fromtimestamp(stats.timestamp)
        row = stats.as_row()
        # Store a human-readable local timestamp; keep the epoch value too so
        # downstream analysis never has to guess the timezone.
        row = {"time": when.isoformat(timespec="milliseconds"), **row}

        self._ensure_open(when, list(row.keys()))
        assert self._writer is not None and self._fh is not None
        self._writer.writerow(row)
        self._fh.flush()

    def _ensure_open(self, when: dt.datetime, fieldnames: list[str]) -> None:
        day = f"{when:%Y%m%d}"
        if self._fh is not None and (not self.rotate_daily or day == self._day):
            return
        self.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(when)
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
        self._path = path
        self._day = day

    @property
    def current_path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = None
        self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
