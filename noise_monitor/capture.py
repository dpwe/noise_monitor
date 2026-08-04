"""Audio input: the UMIK-1 via PortAudio, plus offline sources for testing.

Every source pushes mono float32 blocks into a bounded queue. The PortAudio
callback must never block, so on overflow it drops the block and counts it --
dropped blocks end up in the CSV log rather than being silently absorbed.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

#: Blocks buffered between the capture thread and the DSP thread. At the
#: default 1024-frame block and 48 kHz this is about 2.7 s of slack.
QUEUE_DEPTH = 128


class AudioSource(ABC):
    """Common interface: start(), read() blocks, stop()."""

    def __init__(self, samplerate: int, blocksize: int):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_DEPTH)
        self.dropped_blocks = 0
        self.description = "unknown source"

    def _push(self, block: np.ndarray) -> None:
        try:
            self.queue.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next block, or None on timeout / end of stream."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_dropped(self) -> int:
        n, self.dropped_blocks = self.dropped_blocks, 0
        return n

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def _sounddevice():
    """Import sounddevice, turning both failure modes into one clear message.

    A missing PortAudio *library* raises OSError rather than ImportError, and on
    a fresh Raspberry Pi OS image that is the more likely of the two.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"cannot use the audio input: {exc}\n"
            "On Raspberry Pi OS: sudo apt install python3-sounddevice libportaudio2\n"
            "Elsewhere: pip install sounddevice\n"
            "To run without a microphone, use --synthetic or --wav FILE."
        ) from exc
    return sd


def list_input_devices() -> list[dict]:
    """All input-capable devices, as sounddevice reports them."""
    sd = _sounddevice()

    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({"index": index, **dev})
    return devices


def find_device(spec: str | None) -> int | None:
    """Resolve a device spec to a PortAudio index.

    `spec` may be an integer index, a case-insensitive substring of the device
    name, or None for the system default.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec.isdigit():
        return int(spec)
    matches = [d for d in list_input_devices() if spec.lower() in d["name"].lower()]
    if not matches:
        names = ", ".join(repr(d["name"]) for d in list_input_devices()) or "none"
        raise RuntimeError(
            f"no input device matching {spec!r}. Available input devices: {names}"
        )
    if len(matches) > 1:
        names = ", ".join(repr(d["name"]) for d in matches)
        raise RuntimeError(f"device spec {spec!r} is ambiguous, matches: {names}")
    return matches[0]["index"]


class MicrophoneSource(AudioSource):
    """Live capture from a PortAudio input device."""

    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        device: str | None = None,
        channel: int = 0,
    ):
        super().__init__(samplerate, blocksize)
        self.device_spec = device
        self.channel = channel
        self._stream = None
        self.overflows = 0

    def start(self) -> None:
        sd = _sounddevice()

        index = find_device(self.device_spec)
        info = sd.query_devices(index if index is not None else sd.default.device[0])
        channels = max(1, min(int(info["max_input_channels"]), self.channel + 1))
        if self.channel >= channels:
            raise RuntimeError(
                f"{info['name']} has {info['max_input_channels']} input channel(s), "
                f"cannot select channel {self.channel}"
            )

        def callback(indata, frames, time_info, status):
            if status:
                # input_overflow means the OS dropped samples before we saw them.
                self.overflows += 1
            self._push(np.array(indata[:, self.channel], dtype=np.float32, copy=True))

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=index,
            channels=channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self.samplerate = int(self._stream.samplerate)
        self.description = f"{info['name']} @ {self.samplerate} Hz, channel {self.channel}"

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class ArraySource(AudioSource):
    """Replays an in-memory signal. Used by tests and `--synthetic`."""

    def __init__(
        self,
        signal: np.ndarray,
        samplerate: int,
        blocksize: int,
        realtime: bool = False,
        loop: bool = False,
    ):
        super().__init__(samplerate, blocksize)
        self.signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        self.realtime = realtime
        self.loop = loop
        self.description = f"array source, {self.signal.size / samplerate:.1f} s"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import time as _time

        pos = 0
        period = self.blocksize / self.samplerate
        next_at = _time.monotonic()
        while not self._stop.is_set():
            if pos >= self.signal.size:
                if not self.loop:
                    break
                pos = 0
            block = self.signal[pos : pos + self.blocksize]
            pos += self.blocksize
            if block.size < self.blocksize and not self.loop:
                block = np.pad(block, (0, self.blocksize - block.size))
            self._push(np.array(block, copy=True))
            if self.realtime:
                next_at += period
                delay = next_at - _time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class FileSource(ArraySource):
    """Replays a WAV file, for testing the chain against a known recording."""

    def __init__(
        self,
        path: str | Path,
        blocksize: int,
        channel: int = 0,
        realtime: bool = False,
        loop: bool = False,
    ):
        import soundfile as sf  # optional dependency, only needed for this source

        data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
        super().__init__(data[:, channel], samplerate, blocksize, realtime, loop)
        self.description = f"{Path(path).name} @ {samplerate} Hz"


def synthetic_signal(
    samplerate: int,
    duration_s: float = 10.0,
    target_dbfs: float = -30.0,
    seed: int = 0,
) -> np.ndarray:
    """Pink-ish noise plus a 1 kHz tone, for exercising the UI without a mic."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * samplerate)
    white = rng.standard_normal(n)
    # Shape white noise to roughly -3 dB/octave in the frequency domain.
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / samplerate)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spec * shape, n=n)
    pink /= np.sqrt(np.mean(pink**2))

    t = np.arange(n) / samplerate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    mix = pink + tone
    mix *= 10 ** (target_dbfs / 20) / np.sqrt(np.mean(mix**2))
    return mix.astype(np.float32)
