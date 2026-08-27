"""Parsing of `amixer contents` output.

The sample text below is the shape ALSA prints for a USB audio class device
like the UMIK-1. These tests pin the parser, not the hardware -- the values a
real Pi reports still need checking once by hand against `amixer`.
"""

import pytest

from noise_monitor import alsa
from noise_monitor.alsa import CaptureGain, parse_contents, read_capture_gain

AMIXER_CONTENTS = """\
numid=3,iface=MIXER,name='Mic Capture Switch'
  ; type=BOOLEAN,access=rw------,values=1
  : values=on
numid=2,iface=MIXER,name='Mic Capture Volume'
  ; type=INTEGER,access=rw---R--,values=1,min=0,max=24,step=0
  : values=18
  | dBscale-min=0.00dB,step=1.00dB,mute=0
numid=1,iface=CARD,name='Keep Interface'
  ; type=BOOLEAN,access=rw------,values=1
  : values=off
"""


def test_parses_control_blocks():
    controls = parse_contents(AMIXER_CONTENTS)
    # Every control gets its own block, whatever its iface. Skipping the
    # non-mixer ones used to mean not starting a new block for them, which
    # let their min/max overwrite the control above.
    assert [c["numid"] for c in controls] == [3, 2, 1]
    assert controls[2]["iface"] == "CARD"
    volume = controls[1]
    assert volume["name"] == "Mic Capture Volume"
    assert (volume["min"], volume["max"], volume["value"]) == (0, 24, 18)
    assert volume["db_min"] == 0.0
    assert volume["db_step"] == 1.0


def test_capture_gain_derives_dB_from_the_tlv_scale(monkeypatch):
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: AMIXER_CONTENTS)
    gain = read_capture_gain("1")
    assert gain is not None
    assert gain.current_db == pytest.approx(18.0)
    assert gain.max_db == pytest.approx(24.0)
    # 18 dB of 24 dB available, so the offset owed to the calibration is 6 dB.
    assert gain.input_gain_db == pytest.approx(6.0)
    assert not gain.at_reference


def test_gain_at_maximum_needs_no_offset(monkeypatch):
    text = AMIXER_CONTENTS.replace(": values=18", ": values=24")
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: text)
    gain = read_capture_gain("1")
    assert gain.input_gain_db == pytest.approx(0.0)
    assert gain.at_reference


def test_negative_db_scale_is_handled(monkeypatch):
    """Some devices express gain as attenuation from 0 dB."""
    text = AMIXER_CONTENTS.replace(
        "dBscale-min=0.00dB,step=1.00dB", "dBscale-min=-34.00dB,step=1.50dB"
    )
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: text)
    gain = read_capture_gain("1")
    assert gain.current_db == pytest.approx(-34.0 + 18 * 1.5)
    assert gain.max_db == pytest.approx(-34.0 + 24 * 1.5)
    assert gain.input_gain_db == pytest.approx(9.0)


def test_missing_tlv_returns_none_rather_than_guessing(monkeypatch):
    """Without a dB scale there is no safe conversion, so refuse to invent one."""
    text = "\n".join(
        line for line in AMIXER_CONTENTS.splitlines() if "dBscale" not in line
    )
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: text)
    assert read_capture_gain("1") is None


def test_no_capture_volume_control_returns_none(monkeypatch):
    text = AMIXER_CONTENTS.replace("Mic Capture Volume", "Mic Playback Volume")
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: text)
    assert read_capture_gain("1") is None


def test_amixer_failure_returns_none(monkeypatch):
    monkeypatch.setattr("noise_monitor.alsa._run", lambda args: None)
    assert read_capture_gain("1") is None


def test_input_gain_is_the_shortfall_from_maximum():
    gain = CaptureGain(
        card="1", control="Mic Capture Volume",
        current_db=6.0, max_db=24.0, current_raw=6, max_raw=24,
    )
    assert gain.input_gain_db == pytest.approx(18.0)


# The TLV form a real UMIK-1 on a Raspberry Pi actually prints. It reports
# endpoints only -- no per-step size -- which the dBscale-only parser could
# not read, so `noise-monitor gain` gave up on the very hardware it is for.
UMIK1_ON_A_PI = """numid=2,iface=MIXER,name='Mic Capture Switch'
  ; type=BOOLEAN,access=rw------,values=1
  : values=on
numid=3,iface=MIXER,name='Mic Capture Volume'
  ; type=INTEGER,access=rw---R--,values=1,min=0,max=127,step=0
  : values=127
  | dBminmax-min=-63.50dB,max=0.00dB
numid=1,iface=PCM,name='Capture Channel Map'
  ; type=INTEGER,access=r--v-R--,values=1,min=0,max=36,step=0
  : values=0
  | container
    | chmap-fixed=MONO
"""


def _umik_gain(monkeypatch, text=UMIK1_ON_A_PI):
    monkeypatch.setattr(alsa, "_run", lambda args: text)
    return alsa.read_capture_gain("2")


def test_a_dbminmax_control_is_read(monkeypatch):
    gain = _umik_gain(monkeypatch)
    assert gain is not None
    assert gain.control == "Mic Capture Volume"
    assert gain.current_raw == 127 and gain.max_raw == 127


def test_the_top_of_a_dbminmax_range_is_the_maximum(monkeypatch):
    gain = _umik_gain(monkeypatch)
    assert gain.current_db == pytest.approx(0.0)
    assert gain.max_db == pytest.approx(0.0)
    assert gain.input_gain_db == pytest.approx(0.0)
    assert gain.at_reference


def test_dbminmax_interpolates_between_the_endpoints(monkeypatch):
    """Half way up a -63.5..0 dB range is -31.75 dB, so 31.75 dB to add back."""
    half = UMIK1_ON_A_PI.replace("  : values=127", "  : values=64", 1)
    gain = _umik_gain(monkeypatch, half)
    assert gain.current_db == pytest.approx(-63.5 + 64 / 127 * 63.5)
    assert gain.input_gain_db == pytest.approx(63.5 - 64 / 127 * 63.5)
    assert not gain.at_reference


def test_a_non_mixer_control_does_not_leak_into_the_previous_one():
    """The UMIK-1 lists an iface=PCM control last. Its min/max is not the
    volume's, and reading it as such put the gain 160 dB out."""
    controls = {c["name"]: c for c in parse_contents(UMIK1_ON_A_PI)}
    assert controls["Mic Capture Volume"]["max"] == 127
    assert controls["Capture Channel Map"]["max"] == 36
    assert controls["Capture Channel Map"]["iface"] == "PCM"
