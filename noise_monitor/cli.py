"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .calibration import REFERENCE_CALIBRATOR_SPL, parse_cal_file, resolve_spl_offset
from .capture import (
    ArraySource,
    FileSource,
    MicrophoneSource,
    list_input_devices,
    synthetic_signal,
)
from .config import Config
from .engine import MonitorEngine
from .logsink import CsvLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noise-monitor",
        description="Real-time calibrated spectrogram and dB(A) sound level meter.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the live monitor (default)")
    _add_run_args(run)

    sub.add_parser("devices", help="list audio input devices")

    cal = sub.add_parser(
        "calibrate",
        help="measure the dBFS->dB SPL offset using an acoustic calibrator",
    )
    _add_common_args(cal)
    cal.add_argument("--seconds", type=float, default=5.0, help="measurement duration")
    cal.add_argument(
        "--spl", type=float, default=REFERENCE_CALIBRATOR_SPL,
        help="the calibrator's output level in dB SPL (default 94)",
    )
    cal.add_argument(
        "--frequency", type=float, default=1000.0, help="calibrator tone frequency in Hz"
    )

    check = sub.add_parser(
        "check", help="parse the cal file and report the resulting calibration"
    )
    _add_common_args(check)

    sub.add_parser(
        "gain", help="read the ALSA capture gain and derive input_gain_db (Linux)"
    )

    _add_run_args(parser)  # allow `noise-monitor --synthetic` with no subcommand
    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, help="TOML config file")
    p.add_argument("--cal-file", type=Path, help="miniDSP UMIK-1 calibration .txt")
    p.add_argument("--device", help="input device name substring or index")
    p.add_argument("--samplerate", type=int, help="sample rate in Hz")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    _add_common_args(p)
    p.add_argument("--blocksize", type=int, help="frames per audio callback")
    p.add_argument("--nfft", type=int, help="FFT size")
    p.add_argument("--hop", type=int, help="FFT hop in samples")
    p.add_argument(
        "--weighting", choices=["A", "C", "Z", "a", "c", "z"],
        help="frequency weighting for the level readout (default A)",
    )
    p.add_argument(
        "--time-weighting", choices=["fast", "slow"], help="exponential time weighting"
    )
    p.add_argument("--history", type=float, help="seconds of spectrogram on screen")
    p.add_argument(
        "--average", type=float, metavar="SECONDS",
        help="averaging window for the level readout (default 5)",
    )
    p.add_argument(
        "--long-column", type=float, metavar="MINUTES",
        help="minutes of audio per column of the long-term panel (default 3)",
    )
    p.add_argument(
        "--long-span", type=float, metavar="HOURS",
        help="hours covered by the long-term panel (default 24)",
    )
    p.add_argument("--db-min", type=float, help="colour scale minimum")
    p.add_argument("--db-max", type=float, help="colour scale maximum")
    p.add_argument("--log-dir", type=Path, help="directory for CSV logs")
    p.add_argument(
        "--screenshot-dir", type=Path, help="where the S key writes PNGs"
    )
    p.add_argument("--log-interval", type=float, help="seconds per logged row")
    p.add_argument("--no-log", action="store_true", help="disable CSV logging")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument(
        "--synthetic", action="store_true",
        help="use a generated test signal instead of the microphone",
    )
    p.add_argument("--wav", type=Path, help="replay a WAV file instead of the microphone")
    p.add_argument(
        "--headless", type=float, metavar="SECONDS",
        help="run without a GUI for this many seconds (logging only)",
    )


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    a = lambda name: getattr(args, name, None)  # noqa: E731

    if a("cal_file") is not None:
        cfg.calibration.cal_file = args.cal_file
    if a("device") is not None:
        cfg.audio.device = args.device
    if a("samplerate") is not None:
        cfg.audio.samplerate = args.samplerate
    if a("blocksize") is not None:
        cfg.audio.blocksize = args.blocksize
    if a("nfft") is not None:
        cfg.analysis.nfft = args.nfft
    if a("hop") is not None:
        cfg.analysis.hop = args.hop
    if a("weighting") is not None:
        cfg.analysis.weighting = args.weighting.upper()
    if a("time_weighting") is not None:
        cfg.analysis.time_weighting = args.time_weighting
    if a("history") is not None:
        cfg.ui.history_s = args.history
    if a("average") is not None:
        cfg.analysis.display_average_s = args.average
    if a("long_column") is not None:
        cfg.ui.long_column_s = args.long_column * 60.0
    if a("long_span") is not None:
        cfg.ui.long_span_s = args.long_span * 3600.0
    if a("db_min") is not None:
        cfg.ui.db_min = args.db_min
    if a("db_max") is not None:
        cfg.ui.db_max = args.db_max
    if a("log_dir") is not None:
        cfg.logging.directory = args.log_dir
    if a("screenshot_dir") is not None:
        cfg.ui.screenshot_dir = args.screenshot_dir
    if a("log_interval") is not None:
        cfg.logging.interval_s = args.log_interval
    if a("no_log"):
        cfg.logging.enabled = False
    if a("fullscreen"):
        cfg.ui.fullscreen = True
    return cfg


def load_calibration(cfg: Config):
    if cfg.calibration.cal_file is None:
        return None
    path = Path(cfg.calibration.cal_file).expanduser()
    if not path.exists():
        raise SystemExit(f"calibration file not found: {path}")
    return parse_cal_file(path)


def make_source(cfg: Config, args: argparse.Namespace):
    if getattr(args, "synthetic", False):
        signal = synthetic_signal(cfg.audio.samplerate, duration_s=20.0)
        return ArraySource(
            signal, cfg.audio.samplerate, cfg.audio.blocksize, realtime=True, loop=True
        )
    if getattr(args, "wav", None) is not None:
        return FileSource(args.wav, cfg.audio.blocksize, realtime=True, loop=True)
    return MicrophoneSource(
        cfg.audio.samplerate, cfg.audio.blocksize, cfg.audio.device, cfg.audio.channel
    )


# ----------------------------------------------------------------------
def cmd_devices(_args) -> int:
    devices = list_input_devices()
    if not devices:
        print("no input devices found")
        return 1
    print(f"{'idx':>4}  {'ch':>3}  {'default sr':>10}  name")
    for d in devices:
        print(f"{d['index']:>4}  {d['max_input_channels']:>3}  "
              f"{d['default_samplerate']:>10.0f}  {d['name']}")
    return 0


def cmd_gain(_args) -> int:
    from . import alsa

    print(alsa.describe())
    return 0


def cmd_check(args) -> int:
    cfg = apply_overrides(Config.load(args.config), args)
    cal = load_calibration(cfg)

    if cal is None:
        print("No calibration file configured (--cal-file or [calibration] cal_file).")
    else:
        print(f"Calibration file : {cal.source}")
        print(f"  serial         : {cal.serial or 'unknown'}")
        print(f"  points         : {len(cal.frequencies)} "
              f"({cal.frequencies[0]:.1f} - {cal.frequencies[-1]:.0f} Hz)")
        if cal.has_absolute_reference:
            print(f"  Sens Factor    : {cal.sens_factor_db:+.4f} dB")
        else:
            print("  Sens Factor    : absent")
        at = cal.response_at(np.array([20, 100, 1000, 5000, 10000, 20000.0]))
        print("  response @ 20/100/1k/5k/10k/20k Hz: "
              + " ".join(f"{v:+.2f}" for v in at))

    offset, note = resolve_spl_offset(
        cal, cfg.calibration.spl_offset_db, cfg.calibration.input_gain_db
    )
    print(f"\nSPL offset       : {offset:+.2f} dB  ({note})")
    if "UNCALIBRATED" in note:
        print("  -> levels will be dBFS, not dB SPL")
        return 0

    print(f"  0 dBFS RMS reads {offset:.1f} dB SPL")
    clip_spl = offset - 3.01
    print(f"  clipping level   : {clip_spl:.1f} dB SPL (a full-scale sine)")
    if clip_spl < 100.0:
        print(
            f"\n  NOTE: {clip_spl:.0f} dB SPL of headroom assumes the capture gain is at "
            "the\n  calibration file's reference (maximum). If you expect louder "
            "sounds than\n  that, lower the capture gain and set "
            "calibration.input_gain_db to match --\n  `noise-monitor gain` will "
            "work it out for you on Linux."
        )
    return 0


def cmd_calibrate(args) -> int:
    cfg = apply_overrides(Config.load(args.config), args)
    source = MicrophoneSource(
        cfg.audio.samplerate, cfg.audio.blocksize, cfg.audio.device, cfg.audio.channel
    )

    print(f"Fit the calibrator to the microphone and switch it on ({args.spl:g} dB SPL "
          f"@ {args.frequency:g} Hz).")
    input("Press Enter to measure... ")

    chunks: list[np.ndarray] = []
    with source:
        needed = int(args.seconds * source.samplerate)
        collected = 0
        while collected < needed:
            block = source.read(timeout=2.0)
            if block is None:
                print("timed out waiting for audio", file=sys.stderr)
                return 1
            chunks.append(block)
            collected += block.size

    signal = np.concatenate(chunks).astype(np.float64)
    # Skip the first 0.5 s so the stream has settled.
    signal = signal[int(0.5 * source.samplerate) :]

    broadband_dbfs = 20 * np.log10(max(np.sqrt(np.mean(signal**2)), 1e-12))
    tone_dbfs = _tone_level_dbfs(signal, source.samplerate, args.frequency)
    peak_dbfs = 20 * np.log10(max(np.abs(signal).max(), 1e-12))

    offset = args.spl - tone_dbfs
    print(f"\nmeasured {signal.size / source.samplerate:.1f} s")
    print(f"  peak            {peak_dbfs:7.2f} dBFS")
    print(f"  broadband RMS   {broadband_dbfs:7.2f} dBFS")
    print(f"  {args.frequency:g} Hz band     {tone_dbfs:7.2f} dBFS")
    if peak_dbfs > -1.0:
        print("  WARNING: input is clipping, reduce the capture gain and repeat")
    if broadband_dbfs - tone_dbfs > 1.0:
        print("  WARNING: significant energy outside the calibrator tone "
              "(ambient noise or a bad seal)")

    print(f"\nSPL offset = {offset:.2f} dB\n")
    print("Add this to your config:\n")
    print("[calibration]")
    print(f"spl_offset_db = {offset:.2f}")
    return 0


def _tone_level_dbfs(signal: np.ndarray, samplerate: int, frequency: float) -> float:
    """RMS level in a 1/3-octave band around `frequency`, in dBFS.

    Narrowbanding rejects room noise, so the answer reflects the calibrator
    rather than the environment.
    """
    n = 1 << int(np.ceil(np.log2(min(signal.size, samplerate * 4))))
    n = min(n, signal.size)
    from .spectrum import power_preserving_window

    window = power_preserving_window(n)
    total = 0.0
    frames = 0
    for start in range(0, signal.size - n + 1, n // 2):
        spec = np.fft.rfft(signal[start : start + n] * window)
        power = (np.abs(spec) ** 2) * 2 / (n**2)
        power[0] /= 2
        if n % 2 == 0:
            power[-1] /= 2
        freqs = np.fft.rfftfreq(n, 1 / samplerate)
        band = (freqs >= frequency / 2 ** (1 / 6)) & (freqs <= frequency * 2 ** (1 / 6))
        total += power[band].sum()
        frames += 1
    if frames == 0:
        return -120.0
    return 10 * np.log10(max(total / frames, 1e-20))


def cmd_run(args) -> int:
    cfg = apply_overrides(Config.load(args.config), args)
    cal = load_calibration(cfg)

    source = make_source(cfg, args)
    logger = (
        CsvLogger(cfg.logging.directory, cfg.logging.rotate_daily)
        if cfg.logging.enabled
        else None
    )
    engine = MonitorEngine(cfg, source, cal, logger)

    offset, note = engine.analyzer.spl_offset_db, engine.calibration_note
    print(f"calibration: {note} (offset {offset:+.2f} dB)")
    if not engine.calibrated:
        print("WARNING: no absolute reference -- levels shown are dBFS, not dB SPL")
    if logger is not None:
        print(f"logging every {cfg.logging.interval_s:g} s to {cfg.logging.directory}/")

    if args.headless is not None:
        return _run_headless(engine, args.headless)

    try:
        from .ui import run_ui
    except ImportError as exc:
        raise SystemExit(
            f"the GUI needs pyqtgraph and a Qt binding ({exc}). "
            "Install them, or use --headless SECONDS."
        )
    return run_ui(engine, cfg)


def _run_headless(engine: MonitorEngine, seconds: float) -> int:
    import time

    engine.start()
    deadline = time.monotonic() + seconds
    last_printed = None
    try:
        while time.monotonic() < deadline:
            time.sleep(0.25)
            _, state = engine.drain()
            if state.error:
                print(f"ERROR: {state.error}", file=sys.stderr)
                return 1
            iv = state.last_interval
            if iv is not None and iv.timestamp != last_printed:
                last_printed = iv.timestamp
                w = iv.weighting
                print(f"L{w}eq {iv.leq:6.1f}  L{w}max {iv.lmax:6.1f}  "
                      f"L{w}90 {iv.l90:6.1f}  clipped {iv.clipped_samples}")
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    handlers = {
        "run": cmd_run,
        "devices": cmd_devices,
        "calibrate": cmd_calibrate,
        "check": cmd_check,
        "gain": cmd_gain,
    }
    try:
        return handlers[command](args)
    except RuntimeError as exc:
        # Missing PortAudio, no matching device, and similar setup problems --
        # actionable messages, so print them without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
