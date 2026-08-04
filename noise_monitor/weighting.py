"""IEC 61672-1 frequency weighting and exponential time weighting.

The A- and C-weighting analogue transfer functions are defined by four pole
frequencies; both are normalised to 0 dB at 1 kHz. The digital versions here are
plain bilinear transforms of those, which is the standard approach and stays
well inside class-1 tolerance across the band that matters for dB(A) at
44.1/48 kHz. (The bilinear frequency warping pulls the response down near
Nyquist; at 48 kHz that is a fraction of a dB by 16 kHz, where A-weighting has
already applied about -7 dB.)

Time weighting is the standard exponential average of the squared signal: a
one-pole lowpass with tau = 125 ms (Fast) or 1 s (Slow).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import bilinear, lfilter, lfilter_zi, sosfilt, tf2sos, zpk2tf

# IEC 61672-1 pole frequencies (Hz).
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217
# Normalisation so that the response is exactly 0 dB at 1 kHz.
_A1000 = 1.9997  # dB of gain the A-weighting numerator needs
_C1000 = 0.0619  # dB for C-weighting

FAST_TAU = 0.125
SLOW_TAU = 1.0
#: Impulse time weighting is not implemented; Fast/Slow cover normal use.
TIME_CONSTANTS = {"fast": FAST_TAU, "slow": SLOW_TAU}


def _analog_a() -> tuple[np.ndarray, np.ndarray]:
    """Analogue A-weighting as (numerator, denominator) polynomials in s."""
    z = [0.0, 0.0, 0.0, 0.0]
    p = [
        -2 * np.pi * _F4,
        -2 * np.pi * _F4,
        -2 * np.pi * _F3,
        -2 * np.pi * _F2,
        -2 * np.pi * _F1,
        -2 * np.pi * _F1,
    ]
    k = (2 * np.pi * _F4) ** 2 * 10 ** (_A1000 / 20)
    return zpk2tf(z, p, k)


def _analog_c() -> tuple[np.ndarray, np.ndarray]:
    """Analogue C-weighting as (numerator, denominator) polynomials in s."""
    z = [0.0, 0.0]
    p = [
        -2 * np.pi * _F4,
        -2 * np.pi * _F4,
        -2 * np.pi * _F1,
        -2 * np.pi * _F1,
    ]
    k = (2 * np.pi * _F4) ** 2 * 10 ** (_C1000 / 20)
    return zpk2tf(z, p, k)


def weighting_sos(kind: str, samplerate: int) -> np.ndarray:
    """Second-order sections for the named weighting.

    ``"Z"`` (unweighted) returns a pass-through section.
    """
    kind = kind.upper()
    if kind == "Z":
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    if kind == "A":
        num, den = _analog_a()
    elif kind == "C":
        num, den = _analog_c()
    else:
        raise ValueError(f"unknown weighting {kind!r}; expected 'A', 'C' or 'Z'")
    b, a = bilinear(num, den, fs=samplerate)
    return tf2sos(b, a)


def weighting_response_db(kind: str, freqs: np.ndarray) -> np.ndarray:
    """Ideal (analogue) weighting response in dB at the given frequencies.

    Used for weighting spectrogram bands and for testing the digital filters.
    """
    kind = kind.upper()
    freqs = np.asarray(freqs, dtype=float)
    if kind == "Z":
        return np.zeros_like(freqs)
    if kind == "A":
        num, den = _analog_a()
    elif kind == "C":
        num, den = _analog_c()
    else:
        raise ValueError(f"unknown weighting {kind!r}; expected 'A', 'C' or 'Z'")
    s = 2j * np.pi * freqs
    with np.errstate(divide="ignore"):
        h = np.polyval(num, s) / np.polyval(den, s)
        return 20 * np.log10(np.abs(h))


class StreamingFilter:
    """An SOS filter that carries its state across successive blocks."""

    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=float)
        self.zi = np.zeros((self.sos.shape[0], 2))

    def __call__(self, block: np.ndarray) -> np.ndarray:
        out, self.zi = sosfilt(self.sos, block, zi=self.zi)
        return out


class StreamingFIR:
    """An FIR filter that carries its state across successive blocks."""

    def __init__(self, taps: np.ndarray):
        self.taps = np.asarray(taps, dtype=float)
        self.zi = np.zeros(len(self.taps) - 1)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        out, self.zi = lfilter(self.taps, [1.0], block, zi=self.zi)
        return out


class ExponentialLevel:
    """Exponentially time-weighted mean square, evaluated per sample.

    Feed successive blocks of the (already frequency-weighted) signal; get back
    the running mean square at each sample, from which a dB level follows. The
    state persists across blocks so block boundaries are invisible.
    """

    def __init__(self, samplerate: int, tau: float):
        self.alpha = float(np.exp(-1.0 / (samplerate * tau)))
        # One-pole: y[n] = alpha*y[n-1] + (1-alpha)*x[n]^2
        self._b = np.array([1.0 - self.alpha])
        self._a = np.array([1.0, -self.alpha])
        self._zi = np.zeros(1)
        self._primed = False

    def __call__(self, block: np.ndarray) -> np.ndarray:
        square = np.asarray(block, dtype=float) ** 2
        if not self._primed and square.size:
            # Start from the block's own level rather than from silence, so the
            # very first reading is not an artificial fade-in from -inf dB.
            self._zi = lfilter_zi(self._b, self._a) * float(square[0])
            self._primed = True
        out, self._zi = lfilter(self._b, self._a, square, zi=self._zi)
        return out
