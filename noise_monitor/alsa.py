"""Reading the UMIK-1's capture gain from ALSA (Linux / Raspberry Pi only).

Why this matters: the ``Sens Factor`` in a miniDSP calibration file is defined
with the mic's input volume at *maximum*. The UMIK-1's USB gain control is a
real gain, so at any lower setting the same sound pressure produces a lower
dBFS reading and the SPL offset must be raised by the difference::

    input_gain_db = max_gain_dB - current_gain_dB

That is what ``CalibrationConfig.input_gain_db`` holds, and this module derives
it from ALSA rather than making you guess.

The numbers come from the control's TLV dB scale, which ``amixer contents``
prints as a line like ``| dBscale-min=0.00dB,step=1.00dB,mute=0``. That gives
an exact dB value for every raw step, including the maximum -- which ``amixer
sget`` cannot, since it only reports the dB value of the *current* setting.

This is text-scraping a tool whose output format is not an API. Every function
returns ``None`` rather than guessing, and a returned value is worth
cross-checking once against ``amixer`` by hand. An acoustic calibrator
measurement (``noise-monitor calibrate``) beats all of this and makes the
question moot.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

# "numid=2,iface=MIXER,name='Mic Capture Volume'"
_NUMID_RE = re.compile(r"numid=(\d+),iface=MIXER,name='([^']*)'")
# "  ; type=INTEGER,access=rw---R--,values=1,min=0,max=24,step=0"
_TYPE_RE = re.compile(r"type=(\w+).*?min=(-?\d+),max=(-?\d+)")
# "  : values=18"
_VALUES_RE = re.compile(r":\s*values=(-?\d+)")
# "  | dBscale-min=0.00dB,step=1.00dB,mute=0"
_DBSCALE_RE = re.compile(r"dBscale-min=(-?\d+\.?\d*)dB,step=(-?\d+\.?\d*)dB")
# "  Mono: Capture 18 [75%] [18.00dB] [on]"
_SGET_DB_RE = re.compile(r"\[(-?\d+\.\d+)dB\]")


@dataclass
class CaptureGain:
    card: str
    control: str
    current_db: float
    max_db: float
    current_raw: int
    max_raw: int

    @property
    def input_gain_db(self) -> float:
        """How far below the calibration reference (maximum) this sits."""
        return self.max_db - self.current_db

    @property
    def at_reference(self) -> bool:
        return abs(self.input_gain_db) < 0.05


def available() -> bool:
    return sys.platform.startswith("linux") and shutil.which("amixer") is not None


def _run(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def find_card(name_hint: str = "umik") -> str | None:
    """ALSA card index for the first card whose description contains `name_hint`."""
    out = _run(["arecord", "-l"])
    if not out:
        return None
    for line in out.splitlines():
        # "card 1: U18 [Umik-1  Gain: 18dB], device 0: USB Audio [USB Audio]"
        match = re.match(r"card (\d+):\s*(.*)", line)
        if match and name_hint.lower() in match.group(2).lower():
            return match.group(1)
    return None


def parse_contents(text: str) -> list[dict]:
    """Split `amixer contents` output into one dict per control."""
    controls: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        header = _NUMID_RE.search(line)
        if header:
            current = {"numid": int(header.group(1)), "name": header.group(2)}
            controls.append(current)
            continue
        if current is None:
            continue
        if (m := _TYPE_RE.search(line)) and line.lstrip().startswith(";"):
            current["type"] = m.group(1)
            current["min"] = int(m.group(2))
            current["max"] = int(m.group(3))
        elif m := _VALUES_RE.search(line):
            current.setdefault("value", int(m.group(1)))
        elif m := _DBSCALE_RE.search(line):
            current["db_min"] = float(m.group(1))
            current["db_step"] = float(m.group(2))
    return controls


def read_capture_gain(card: str) -> CaptureGain | None:
    """Current and maximum capture gain in dB, or None if it can't be read."""
    out = _run(["amixer", "-c", str(card), "contents"])
    if not out:
        return None

    for control in parse_contents(out):
        name = control.get("name", "")
        if "capture volume" not in name.lower():
            continue
        needed = ("value", "max", "db_min", "db_step")
        if not all(k in control for k in needed):
            continue
        if control["max"] == control["min"]:
            continue
        db_min, step = control["db_min"], control["db_step"]
        return CaptureGain(
            card=str(card),
            control=name,
            current_db=db_min + control["value"] * step,
            max_db=db_min + control["max"] * step,
            current_raw=control["value"],
            max_raw=control["max"],
        )
    return None


def cross_check_current_db(card: str, control_name: str) -> float | None:
    """The current dB value as `amixer sget` reports it, for verification.

    `sget` takes a *simple* control name ("Mic"), whereas `contents` reports the
    full name ("Mic Capture Volume"), so trim the suffix.
    """
    simple = re.sub(r"\s*Capture Volume$", "", control_name, flags=re.IGNORECASE)
    out = _run(["amixer", "-c", str(card), "sget", simple])
    if not out:
        return None
    for line in out.splitlines():
        if "Capture" in line and (match := _SGET_DB_RE.search(line)):
            return float(match.group(1))
    return None


def describe(card_hint: str = "umik") -> str:
    """A human-readable summary for `noise-monitor gain`."""
    if not available():
        return (
            "ALSA gain readback is Linux-only and needs `amixer` (package "
            "alsa-utils).\nOn other platforms, set calibration.input_gain_db by "
            "hand, or use `noise-monitor calibrate` with an acoustic calibrator."
        )
    card = find_card(card_hint)
    if card is None:
        return (
            f"No ALSA card whose name contains {card_hint!r} was found.\n"
            "Run `arecord -l` to see what is connected."
        )
    gain = read_capture_gain(card)
    if gain is None:
        return (
            f"Found the microphone on card {card}, but could not read a dB "
            f"capture gain from ALSA.\nRun `amixer -c {card} contents` and set "
            "calibration.input_gain_db by hand (max dB minus current dB)."
        )

    lines = [
        f"card           : {gain.card}",
        f"control        : {gain.control}",
        f"current gain   : {gain.current_db:+.2f} dB  (step {gain.current_raw} of {gain.max_raw})",
        f"maximum gain   : {gain.max_db:+.2f} dB",
        "",
        f"input_gain_db  = {gain.input_gain_db:+.2f}    <- put this in your config",
    ]

    check = cross_check_current_db(card, gain.control)
    if check is not None and abs(check - gain.current_db) > 0.05:
        lines += [
            "",
            f"WARNING: `amixer sget` reports {check:+.2f} dB for the current "
            f"setting but the TLV scale gives {gain.current_db:+.2f} dB.",
            "Trust amixer and set input_gain_db by hand.",
        ]

    if gain.at_reference:
        lines += [
            "",
            "Capture gain is at maximum, which is the calibration file's own "
            "reference condition, so input_gain_db = 0.",
            "This is also the setting with the least headroom -- see the README "
            "on choosing a gain.",
        ]
    return "\n".join(lines)
