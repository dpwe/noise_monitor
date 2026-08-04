import numpy as np
import pytest

# A file in exactly the shape miniDSP ships: bare-leading-dot sensitivity,
# tab-separated pairs, log-spaced from 10 Hz to 20 kHz, 0 dB at 1 kHz.
SENS_FACTOR = -0.6162
SERIAL = "7005701"


def _response(freqs: np.ndarray) -> np.ndarray:
    """A plausible UMIK-1 shape: LF roll-off, flat midband, slight HF tilt."""
    lf = -6.0 / (1.0 + (freqs / 30.0) ** 2)
    hf = -0.5 * np.log10(np.maximum(freqs, 1.0) / 1000.0) ** 2
    curve = lf + hf
    # Files are normalised to exactly 0 dB at 1 kHz.
    return curve - np.interp(np.log(1000.0), np.log(freqs), curve)


@pytest.fixture
def cal_frequencies() -> np.ndarray:
    return np.geomspace(10.054, 20016.816, 616)


@pytest.fixture
def cal_file(tmp_path, cal_frequencies):
    freqs = cal_frequencies
    resp = _response(freqs)
    path = tmp_path / f"{SERIAL}.txt"
    lines = [f'"Sens Factor ={SENS_FACTOR}dB, SERNO: {SERIAL}"']
    lines += [f"{f:.3f}\t{d:.4f}" for f, d in zip(freqs, resp)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def cal_file_90deg(tmp_path, cal_frequencies):
    """The auto-generated variant, which carries an extra quoted comment."""
    freqs = cal_frequencies
    resp = _response(freqs)
    path = tmp_path / f"{SERIAL}_90deg.txt"
    lines = [
        f"Sens Factor ={SENS_FACTOR}dB, SERNO: {SERIAL}",
        '"Auto-generated 90-degree calibration file"',
    ]
    lines += [f"{f:.3f}\t{d:.4f}" for f, d in zip(freqs, resp)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
