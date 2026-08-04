"""Configuration objects, TOML loading and CLI overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class AudioConfig:
    #: Substring matched (case-insensitively) against input device names, or an
    #: integer index as a string, or None for the system default input.
    device: str | None = "UMIK"
    samplerate: int = 48000
    #: Frames per PortAudio callback. Smaller = lower latency, more CPU.
    blocksize: int = 1024
    #: Which channel to take if the device opens with more than one.
    channel: int = 0
    #: dBFS above which a block is flagged as clipped.
    clip_threshold_dbfs: float = -0.5


@dataclass
class CalibrationConfig:
    #: Path to the miniDSP UMIK-1 calibration .txt (serial-numbered).
    cal_file: Path | None = None
    #: Apply the mic's frequency response correction (spectrogram always;
    #: broadband levels via an FIR when `correct_broadband` is set).
    apply_frequency_correction: bool = True
    #: Also correct the time-domain signal feeding the dB(A) meter. Costs one
    #: FIR convolution per block; changes LAeq by a few tenths of a dB.
    correct_broadband: bool = True
    fir_taps: int = 1023
    #: Clamp the correction so a big low-frequency boost cannot run away.
    max_boost_db: float = 12.0
    #: How far the capture gain sits BELOW the cal file's reference condition
    #: (USB input volume at maximum). Positive numbers mean "quieter than
    #: reference" and are added to the measured SPL. See README.
    input_gain_db: float = 0.0
    #: Absolute override, in dB, added to a 0 dBFS RMS reading to get dB SPL.
    #: Set by `noise-monitor calibrate` with an acoustic calibrator. When set,
    #: this wins over the cal file's Sens Factor.
    spl_offset_db: float | None = None


@dataclass
class AnalysisConfig:
    nfft: int = 4096
    hop: int = 1024
    fmin: float = 20.0
    fmax: float = 20000.0
    #: Number of log-spaced rows in the spectrogram.
    n_bands: int = 256
    #: "band" = total SPL in each row's frequency range.
    #: "density" = SPL per Hz, which renders white noise as a flat field.
    scale: str = "band"
    #: Frequency weighting for the headline number: "A", "C" or "Z".
    weighting: str = "A"
    #: Weighting applied to the spectrogram itself. "Z" (none) shows the actual
    #: band sound pressure, which is normally what you want to look at.
    spectrogram_weighting: str = "Z"
    #: Exponential time weighting for the live readout: "fast" or "slow".
    time_weighting: str = "fast"


@dataclass
class LoggingConfig:
    enabled: bool = True
    directory: Path = Path("logs")
    #: Seconds per logged row (each row is an LAeq over this interval).
    interval_s: float = 10.0
    rotate_daily: bool = True


@dataclass
class UIConfig:
    #: Seconds of spectrogram history on screen.
    history_s: float = 30.0
    #: Seconds of level history in the bottom trace.
    level_history_s: float = 300.0
    db_min: float = 0.0
    db_max: float = 90.0
    colormap: str = "viridis"
    refresh_hz: float = 30.0
    fullscreen: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        for section, values in data.items():
            if not hasattr(cfg, section):
                raise ValueError(f"{path}: unknown config section [{section}]")
            _apply(getattr(cfg, section), values, f"{path}: [{section}]")
        return cfg


def _apply(obj: Any, values: dict, where: str) -> None:
    """Assign dict entries onto a dataclass, coercing to the declared type."""
    if not is_dataclass(obj):  # pragma: no cover - guarded by caller
        raise TypeError(f"{where} is not a config section")
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"{where}: unknown key '{key}'")
        setattr(obj, key, _coerce(known[key].type, value))


def _coerce(declared: Any, value: Any) -> Any:
    """Coerce a TOML scalar to the field's declared type.

    Annotations arrive as strings (`from __future__ import annotations`), so
    match on the text. TOML already yields correct str/bool/int types; the only
    real work is Path fields and int-literals landing in float fields.
    """
    text = declared if isinstance(declared, str) else getattr(declared, "__name__", "")
    if value is None:
        return None
    if "Path" in text:
        return Path(value).expanduser()
    if "float" in text and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value
