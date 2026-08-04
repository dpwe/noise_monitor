"""Streaming STFT and mapping of FFT bins onto log-spaced display bands.

Power bookkeeping is the point of this module: the window is normalised so that
the sum of the one-sided bin powers equals the mean square of the frame
(Parseval), and the bin-to-band matrix conserves power. That makes each
spectrogram row a real band level in dB SPL rather than an arbitrary display
value, and it means the bands sum back to the broadband level.
"""

from __future__ import annotations

import numpy as np


def power_preserving_window(nfft: int) -> np.ndarray:
    """Periodic Hann scaled so that mean(w**2) == 1."""
    w = np.hanning(nfft + 1)[:-1]  # periodic, not symmetric
    return w * np.sqrt(nfft / np.sum(w**2))


class StreamingSTFT:
    """Buffers a sample stream and yields overlapping FFT power frames."""

    def __init__(self, nfft: int, hop: int, samplerate: int):
        if hop <= 0 or hop > nfft:
            raise ValueError("hop must be in (0, nfft]")
        self.nfft = nfft
        self.hop = hop
        self.samplerate = samplerate
        self.window = power_preserving_window(nfft)
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / samplerate)
        self._buf = np.zeros(0, dtype=np.float64)
        # One-sided power scaling: interior bins carry the negative-frequency
        # half too, DC and Nyquist do not.
        self._one_sided = np.full(self.freqs.size, 2.0)
        self._one_sided[0] = 1.0
        if nfft % 2 == 0:
            self._one_sided[-1] = 1.0

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Add samples; return the power spectra of any complete frames.

        Each returned array is the one-sided power per bin, in units of mean
        square amplitude, such that ``spectrum.sum()`` is the frame's mean
        square.
        """
        self._buf = np.concatenate((self._buf, np.asarray(block, dtype=np.float64)))
        frames = []
        while self._buf.size >= self.nfft:
            frame = self._buf[: self.nfft] * self.window
            spec = np.fft.rfft(frame)
            power = (np.abs(spec) ** 2) * self._one_sided / (self.nfft**2)
            frames.append(power)
            self._buf = self._buf[self.hop :]
        return frames


class BandMapper:
    """Maps FFT bin powers onto log-spaced bands, conserving total power.

    Each FFT bin occupies ``[f - df/2, f + df/2)``; each band occupies its own
    edge-to-edge range. A bin's power is split between bands in proportion to
    how much of the bin overlaps each band. Total power is preserved (up to the
    part falling outside [fmin, fmax]), so band levels are physically
    meaningful in both directions -- when bands are wider than bins they sum
    bins, and when they are narrower than bins (unavoidable at low frequency)
    they take a proportional share.
    """

    def __init__(
        self,
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
        n_bands: int,
        samplerate: int,
    ):
        freqs = np.asarray(freqs, dtype=float)
        nyquist = samplerate / 2.0
        fmax = min(fmax, nyquist)
        if fmin <= 0 or fmax <= fmin:
            raise ValueError("need 0 < fmin < fmax")

        self.band_edges = np.geomspace(fmin, fmax, n_bands + 1)
        self.band_centers = np.sqrt(self.band_edges[:-1] * self.band_edges[1:])
        self.bandwidths = np.diff(self.band_edges)
        self.n_bands = n_bands

        df = freqs[1] - freqs[0]
        bin_lo = freqs - df / 2.0
        bin_hi = freqs + df / 2.0
        bin_lo[0] = max(bin_lo[0], 0.0)

        # Overlap matrix (n_bands x n_bins), normalised by bin width so that a
        # bin's power is apportioned, never duplicated.
        lo = np.maximum(bin_lo[None, :], self.band_edges[:-1, None])
        hi = np.minimum(bin_hi[None, :], self.band_edges[1:, None])
        overlap = np.clip(hi - lo, 0.0, None) / df
        self.matrix = overlap

    def __call__(self, power: np.ndarray) -> np.ndarray:
        """Band powers (mean-square units) from one-sided bin powers."""
        return self.matrix @ power


def band_levels_db(
    band_power: np.ndarray,
    spl_offset_db: float,
    bandwidths: np.ndarray | None = None,
    scale: str = "band",
    floor: float = 1e-20,
) -> np.ndarray:
    """Band powers to dB SPL.

    ``scale="band"`` gives the total level in each band. ``scale="density"``
    divides by bandwidth first, giving a spectral density (dB SPL per Hz) which
    renders broadband noise as a flat field instead of a rising slope.
    """
    if scale == "density":
        if bandwidths is None:
            raise ValueError("density scaling needs bandwidths")
        band_power = band_power / bandwidths
    elif scale != "band":
        raise ValueError(f"unknown scale {scale!r}; expected 'band' or 'density'")
    return 10 * np.log10(np.maximum(band_power, floor)) + spl_offset_db
