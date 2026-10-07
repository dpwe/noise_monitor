"""Where the analysis looks for the monitor's output.

The scripts read the live monitor's `logs/` and `screenshots/` directories,
which are not in this repository -- they are a running instrument's output,
and `.gitignore` keeps them out. By default they are looked for beside this
directory, which is how a monitor run in place lays itself out:

    noise_monitor/
      logs/                 <- CSV written by the running monitor
      screenshots/          <- PNGs from the S key and the daily capture
      fan_analysis/         <- this directory

Point `NOISE_MONITOR_DATA` somewhere else to analyse a different collection::

    NOISE_MONITOR_DATA=~/Downloads/noise_monitor python3 build_log.py
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Root holding `logs/` and `screenshots/`.
DATA_DIR = Path(os.environ.get("NOISE_MONITOR_DATA", HERE.parent)).expanduser()
LOG_DIR = DATA_DIR / "logs"
SCREENSHOT_DIR = DATA_DIR / "screenshots"

#: Written by build_log.py, read by chart.py.
EVENTS_CSV = HERE / "fan_events.csv"


def log_files() -> list[Path]:
    return sorted(LOG_DIR.glob("noise-*.csv"))


def screenshots() -> list[Path]:
    return sorted(SCREENSHOT_DIR.glob("*.png"))


def require_data() -> None:
    """Fail with something actionable rather than an empty result set."""
    missing = [d for d in (LOG_DIR, SCREENSHOT_DIR) if not d.is_dir()]
    if missing:
        raise SystemExit(
            "cannot find " + ", ".join(str(d) for d in missing)
            + f"\nSet NOISE_MONITOR_DATA to the directory holding logs/ and "
            f"screenshots/ (currently {DATA_DIR})."
        )
