"""Parsing of `amixer contents` output.

The sample text below is the shape ALSA prints for a USB audio class device
like the UMIK-1. These tests pin the parser, not the hardware -- the values a
real Pi reports still need checking once by hand against `amixer`.
"""

import pytest

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
    # numid=1 is iface=CARD, not a mixer control, so it is skipped.
    assert [c["numid"] for c in controls] == [3, 2]
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
