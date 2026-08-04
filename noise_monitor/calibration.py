"""miniDSP UMIK-1 calibration files: parsing, SPL offset and response correction.

File format (as shipped by miniDSP, one file per serial number)::

    Sens Factor =-.6162dB, SERNO: 7005701
    10.054	-3.7260
    10.179	-3.5710
    ...
    20016.816	-0.4338

Some files quote the header line, and the auto-generated ``*_90deg.txt`` variants
carry an extra quoted comment line. Both are tolerated.

Two independent pieces of calibration live in that file:

1. **Frequency response** -- the freq/dB pairs are the microphone's own
   deviation, normalised to 0 dB at 1 kHz. Following the REW convention they are
   *subtracted* from measurements.

2. **Absolute sensitivity** -- the ``Sens Factor`` is, per the REW
   documentation, "the input dBFS reading the mic will produce when driven by a
   94 dB calibrator with the input volume set to maximum". So, at that reference
   gain::

       dB SPL = dBFS_rms + (94 - sens_factor)

   where dBFS_rms is a plain RMS relative to a full-scale amplitude of 1.0 (a
   full-scale sine therefore reads -3.01 dBFS, which is the convention REW
   states it uses).

   The "input volume at maximum" condition matters: the UMIK-1's USB gain
   control is a real, driver-visible gain. If the capture level is not at the
   reference setting, the difference has to be added back -- see
   ``CalibrationConfig.input_gain_db`` and the README. An acoustic calibrator
   measurement (``noise-monitor calibrate``) bypasses the whole question and is
   the ground truth when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import firwin2

#: SPL of the reference acoustic calibrator the Sens Factor is defined against.
REFERENCE_CALIBRATOR_SPL = 94.0

_SENS_RE = re.compile(r"Sens\s*Factor\s*=\s*(-?\d*\.?\d+)\s*dB", re.IGNORECASE)
_SERNO_RE = re.compile(r"SERNO:\s*(\S+)", re.IGNORECASE)


@dataclass
class MicCalibration:
    """A parsed calibration file."""

    frequencies: np.ndarray  # Hz, ascending
    response_db: np.ndarray  # mic deviation in dB, subtract from measurements
    sens_factor_db: float | None
    serial: str | None
    source: Path | None

    @property
    def has_absolute_reference(self) -> bool:
        return self.sens_factor_db is not None

    def spl_offset_db(self, input_gain_db: float = 0.0) -> float:
        """dB to add to a dBFS RMS reading to obtain dB SPL."""
        if self.sens_factor_db is None:
            raise ValueError(
                f"{self.source} has no 'Sens Factor' header, so it cannot give an "
                "absolute level. Use an acoustic calibrator or set "
                "calibration.spl_offset_db explicitly."
            )
        return REFERENCE_CALIBRATOR_SPL - self.sens_factor_db + input_gain_db

    def response_at(self, freqs: np.ndarray) -> np.ndarray:
        """Interpolate the mic response (dB) at arbitrary frequencies.

        Interpolation is linear in log-frequency; outside the file's range the
        endpoint values are held rather than extrapolated.
        """
        freqs = np.asarray(freqs, dtype=float)
        safe = np.clip(freqs, self.frequencies[0], self.frequencies[-1])
        return np.interp(np.log(safe), np.log(self.frequencies), self.response_db)


def parse_cal_file(path: str | Path) -> MicCalibration:
    """Parse a miniDSP UMIK-1 / UMIK-2 calibration text file."""
    path = Path(path)
    sens: float | None = None
    serial: str | None = None
    freqs: list[float] = []
    resp: list[float] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip().strip('"').strip()
            if not line:
                continue
            if sens is None and "sens" in line.lower():
                match = _SENS_RE.search(line)
                if match:
                    # Files use a bare leading dot, e.g. "=-.6162dB".
                    sens = float(match.group(1))
                    serno = _SERNO_RE.search(line)
                    if serno:
                        serial = serno.group(1)
                    continue
            if line.startswith(("*", "#", ";")) or line[0].isalpha():
                continue  # comment / free-text line
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                f, db = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if f <= 0:
                continue
            freqs.append(f)
            resp.append(db)

    if len(freqs) < 2:
        raise ValueError(f"{path}: found {len(freqs)} calibration points, need at least 2")

    f_arr = np.asarray(freqs, dtype=float)
    r_arr = np.asarray(resp, dtype=float)
    order = np.argsort(f_arr)
    f_arr, r_arr = f_arr[order], r_arr[order]
    # Collapse any duplicate frequencies so np.interp stays well defined.
    keep = np.concatenate(([True], np.diff(f_arr) > 0))
    return MicCalibration(
        frequencies=f_arr[keep],
        response_db=r_arr[keep],
        sens_factor_db=sens,
        serial=serial,
        source=path,
    )


def bin_correction_db(
    cal: MicCalibration | None,
    freqs: np.ndarray,
    max_boost_db: float = 12.0,
) -> np.ndarray:
    """Per-frequency correction in dB to *add* to a measured spectrum.

    This is the negation of the mic's response, limited so that a large
    low-frequency boost cannot dominate.
    """
    if cal is None:
        return np.zeros_like(np.asarray(freqs, dtype=float))
    return np.clip(-cal.response_at(freqs), -max_boost_db, max_boost_db)


def design_correction_fir(
    cal: MicCalibration,
    samplerate: int,
    numtaps: int = 1023,
    max_boost_db: float = 12.0,
) -> np.ndarray:
    """Linear-phase FIR that flattens the microphone's frequency response.

    Used on the time-domain signal feeding the broadband dB(A) meter, so that
    the headline number carries the same correction as the spectrogram. The
    filter is linear phase, so it only delays the signal by ``numtaps // 2``
    samples -- irrelevant for level metrics.
    """
    if numtaps % 2 == 0:
        numtaps += 1  # firwin2 needs odd taps for a type-I (nonzero-Nyquist) filter
    nyquist = samplerate / 2.0

    # A dense log-spaced grid through the audio band, plus explicit DC and
    # Nyquist endpoints, which firwin2 requires.
    grid = np.geomspace(max(cal.frequencies[0], 5.0), min(cal.frequencies[-1], nyquist * 0.999), 512)
    gain_db = np.clip(-cal.response_at(grid), -max_boost_db, max_boost_db)

    freqs = np.concatenate(([0.0], grid, [nyquist]))
    gains = np.concatenate(([10 ** (gain_db[0] / 20)], 10 ** (gain_db / 20), [10 ** (gain_db[-1] / 20)]))

    # Strictly increasing frequencies are required.
    keep = np.concatenate(([True], np.diff(freqs) > 0))
    return firwin2(numtaps, freqs[keep], gains[keep], fs=samplerate)


def resolve_spl_offset(
    cal: MicCalibration | None,
    explicit_offset_db: float | None,
    input_gain_db: float = 0.0,
) -> tuple[float, str]:
    """Decide the dBFS->dB SPL offset and explain where it came from.

    Returns ``(offset_db, description)``. Falls back to 0 dB (i.e. the display
    shows dBFS, not SPL) when nothing absolute is available -- callers should
    surface that to the user rather than silently reporting bogus SPL.
    """
    if explicit_offset_db is not None:
        return explicit_offset_db, "acoustic calibrator / explicit spl_offset_db"
    if cal is not None and cal.has_absolute_reference:
        offset = cal.spl_offset_db(input_gain_db)
        detail = f"Sens Factor {cal.sens_factor_db:+.4f} dB"
        if cal.serial:
            detail += f", serial {cal.serial}"
        if input_gain_db:
            detail += f", input gain {input_gain_db:+.2f} dB"
        return offset, detail
    return 0.0, "UNCALIBRATED (levels are dBFS, not dB SPL)"
