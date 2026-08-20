# noise-monitor

Real-time calibrated spectrogram and dB(A) sound level meter for a Raspberry Pi 5
with a miniDSP UMIK-1.

The window shows a scrolling log-frequency spectrogram in dB SPL, a large
time-weighted dB(A) readout, and a level trace. In the background it appends
LAeq / LAmax / L10 / L50 / L90 to a CSV every logging interval.

Everything is calibrated from the UMIK-1's own serial-numbered file: its
frequency response corrects the spectrogram *and* the broadband meter, and its
`Sens Factor` sets the absolute dB SPL reference.

---

## Install on a Raspberry Pi 5

Raspberry Pi OS Bookworm ships Python 3.11, which is the minimum here.

```bash
sudo apt update
sudo apt install -y python3-numpy python3-scipy python3-pyqtgraph python3-pyqt5 \
                    python3-sounddevice libportaudio2 alsa-utils
```

Prefer the apt packages over pip on a Pi — numpy and scipy build from source
under pip and take a very long time on the Pi's ARM cores.

```bash
git clone <your-repo-url> ~/noise_monitor
cd ~/noise_monitor
pip install --break-system-packages --no-deps -e .
```

`--no-deps` keeps pip from replacing the apt-installed numpy/scipy. If you would
rather have an isolated environment, create the venv with access to the system
packages:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps -e .
```

### Development on a desktop

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gui,dev]"
.venv/bin/python -m pytest
```

You do not need a UMIK-1 to run or develop against it — `--synthetic` feeds a
generated signal through the whole chain, and `--wav FILE` replays a recording
(that one also needs `pip install soundfile`).

---

## Getting the calibration file

Download the serial-numbered `.txt` for **your** microphone from
<https://www.minidsp.com/products/acoustic-measurement/umik-1> — the serial is
printed on the mic's body. Files from a different unit are worse than none.

miniDSP supplies two:

| File | Use it when |
|---|---|
| `<serial>.txt` | The mic points **at** the source (0° incidence). |
| `<serial>_90deg.txt` | The mic points **up** / across the field — the usual choice for environmental noise monitoring. |

Point the config at whichever matches how you mount it:

```toml
[calibration]
cal_file = "~/noise_monitor/cal/7005701_90deg.txt"
```

Then confirm it parsed:

```bash
noise-monitor check -c config.toml
```

```
Calibration file : /home/pi/noise_monitor/cal/7005701_90deg.txt
  serial         : 7005701
  points         : 615 (10.1 - 20017 Hz)
  Sens Factor    : -0.6162 dB
SPL offset       : +94.62 dB  (Sens Factor -0.6162 dB, serial 7005701)
  0 dBFS RMS reads 94.6 dB SPL
  clipping level : 91.6 dB SPL (a full-scale sine)
```

---

## Capture gain — read this before trusting the numbers

This is the one setting that will silently make every reading wrong.

The `Sens Factor` in the calibration file is defined by miniDSP (and REW) as
*the dBFS the mic produces from a 94 dB SPL calibrator **with the input volume
at maximum***. The UMIK-1's USB gain control is a real gain, so:

```
dB SPL = dBFS_rms + (94 - sens_factor) + input_gain_db
```

where `input_gain_db` is how far the capture gain sits **below** maximum.

Two consequences:

1. **At maximum gain the mic clips at roughly 92 dB SPL.** That is fine for
   typical environmental monitoring (30–85 dB(A)) and gives the best noise
   floor, but it will overload on anything loud. `check` warns you about this,
   and the UI shows a red `CLIP` indicator when it happens.
2. **If you lower the gain for more headroom, you must tell the app.** On Linux:

   ```bash
   noise-monitor gain
   ```

   ```
   card           : 1
   control        : Mic Capture Volume
   current gain   : +18.00 dB  (step 18 of 24)
   maximum gain   : +24.00 dB

   input_gain_db  = +6.00    <- put this in your config
   ```

   Set the gain itself with `amixer -c 1 sset Mic 18`, and make it stick across
   reboots with `sudo alsactl store`.

   `noise-monitor gain` scrapes `amixer` output, which is not a stable
   interface. Cross-check it once by hand against `amixer -c 1 contents` — the
   command prints a warning if its two sources of truth disagree.

### The reliable way: an acoustic calibrator

If you have a 94 dB / 1 kHz calibrator, use it and ignore everything above —
it measures the whole chain as configured, gain setting included:

```bash
noise-monitor calibrate -c config.toml
```

Fit the calibrator, press Enter, and it prints an `spl_offset_db` to paste into
your config. That value then overrides the `Sens Factor` path entirely. Repeat
it whenever you change the capture gain.

---

## Running

```bash
noise-monitor run -c config.toml
```

Two spectrograms, half the window each:

| Panel | Span | Shows |
|---|---|---|
| Live | `ui.history_s` (30 s) | One column per FFT hop, ~21 ms. |
| Long-term average | `ui.long_span_s` (24 h) | One column per `ui.long_column_s` (3 min), with the LAeq over the same windows drawn on top in red. |

The current level, and the statistics from the last logged interval, are drawn
*over* the live panel rather than in a strip above it — the spectrograms get
the whole window.

The readout is a **rolling Leq over `analysis.display_average_s`** (1 s by
default), not the exponential Fast level — steady enough to read off a screen
across the room. Fast/Slow is still what `LAmax` and the percentiles in the CSV
are measured from, per IEC 61672.

The red trace uses `ui.level_min`..`ui.level_max` (30–60 dB), which is where
residential background noise lives; the spectrograms keep the wider
`db_min`..`db_max` colour scale, and share one colour bar. Widen the trace if
you are measuring something louder.

| Key | Action |
|---|---|
| `F` | Toggle fullscreen |
| `Q` / `Esc` | Quit |

Useful flags (all override the config file):

```bash
noise-monitor run --synthetic                 # no microphone needed
noise-monitor run --weighting C --time-weighting slow
noise-monitor run --fullscreen --history 60
noise-monitor run --long-span 12 --long-column 1    # 12 h panel, 1 min columns
noise-monitor run --average 5                       # steadier readout
noise-monitor run --headless 3600             # log for an hour, no GUI
noise-monitor devices                         # list inputs
```

---

## The CSV log

One row per `logging.interval_s`, in `logs/noise-YYYYMMDD.csv`:

```
time,timestamp,duration_s,LAeq,LAmax,LAmin,LA10,LA50,LA90,Lpeak,clipped_samples,dropped_blocks
2026-08-04T08:22:39.092,1785846159.092919,2.0,59.67,60.21,58.93,59.97,59.66,59.11,77.01,0,0
```

| Column | Meaning |
|---|---|
| `time` / `timestamp` | Interval **end**, local ISO and Unix epoch. Derived from the sample count, so rows are exactly evenly spaced. |
| `LAeq` | Equivalent continuous A-weighted level over the interval. |
| `LAmax` / `LAmin` | Extremes of the Fast (or Slow) time-weighted level. |
| `LA10` / `LA50` / `LA90` | Level exceeded 10% / 50% / 90% of the interval. `LA90` is the usual background/residual indicator, `LA10` the intrusive one. |
| `Lpeak` | Highest instantaneous sample, unweighted in time. Not `LAmax`. |
| `clipped_samples` | Samples at or above the clip threshold. **Any nonzero value means that row is suspect** — lower the capture gain. |
| `dropped_blocks` | Audio blocks lost because the DSP thread fell behind. Should be 0. |

The first few hundred milliseconds after start-up are discarded: the
exponential detector needs about five time constants to settle, and including
them would put a meaningless `LAmin` in the first row.

---

## Run it on boot

`/etc/systemd/system/noise-monitor.service`, for headless logging:

```ini
[Unit]
Description=Noise monitor
After=sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/noise_monitor
ExecStart=/usr/bin/python3 -m noise_monitor run -c config.toml --headless 86400
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now noise-monitor
```

For the GUI on the Pi's own screen, run it from the desktop session instead —
add it to `~/.config/autostart/` rather than systemd, so it inherits the display.

---

## How it works

```
UMIK-1 ──► PortAudio ──► queue ──► DSP thread ──┬──► spectrogram columns ──► Qt
                                                └──► dB(A) meter ──► CSV
```

Capture, DSP and the GUI are separate threads. The PortAudio callback only
copies into a bounded queue, so it can never block; if the DSP thread falls
behind, blocks are dropped and counted in the log rather than silently absorbed.

**Absolute level.** `dB SPL = dBFS_rms + spl_offset_db`, with `dBFS_rms` a plain
RMS against a full-scale amplitude of 1.0 — so a full-scale sine reads
−3.01 dBFS, the convention REW states it uses.

**Frequency response.** The cal file's freq/dB pairs are the mic's own
deviation, and per the REW convention they are *subtracted* from measurements.
This is applied once, as a linear-phase FIR on the time-domain signal, so the
spectrogram and the dB(A) number carry exactly the same correction. An FIR
cannot resolve fine detail below about `samplerate / numtaps` (≈23 Hz with the
default 1023 taps at 48 kHz), so the correction is approximate in the bottom
octave — where A-weighting is already 40 dB down, so it costs nothing in dB(A).
Set `correct_broadband = false` to correct only the spectrogram.

**Weighting.** A- and C-weighting are bilinear transforms of the IEC 61672-1
analogue definitions. Bilinear warping makes them under-respond near Nyquist
(−0.5 dB at 8 kHz, −6.4 dB at 16 kHz at 48 kHz sample rate), which stays inside
class 1 tolerance — the standard's high-frequency limits are asymmetric for
exactly this reason. The cost in LAeq is about 0.02 dB for a traffic-like
spectrum and 0.2 dB for pink noise. The test suite checks both the tolerance
conformance and that error budget.

**Time weighting.** Fast (125 ms) and Slow (1 s) are one-pole exponential
averages of the squared signal, evaluated per sample with state carried across
blocks, so block boundaries are invisible and `LAmax` is not quantised to the
block rate.

**Spectrogram.** A Hann window scaled so `mean(w²) = 1`, giving one-sided bin
powers that sum (Parseval) to the frame's mean square. FFT bins are mapped onto
log-spaced bands by fractional overlap, which conserves power — so each row is
a genuine band level in dB SPL, and the bands sum back to the broadband level.
There is a test asserting exactly that. Below about 250 Hz the bands are
narrower than one FFT bin, so they show a proportional share of it; raise
`nfft` for more low-frequency resolution, at the cost of time resolution.
`scale = "density"` divides by bandwidth and renders broadband noise flat.

**Long-term average.** The bottom panel averages every `ui.long_column_s` of
live columns into one, in the **power** domain. Averaging decibels instead
would badly under-report a day shaped by short loud events: the mean of 40 and
80 dB is 60 dB, but their energy mean — the thing a person actually hears the
day as — is 77 dB. So each long-term column is a genuine per-band Leq, and the
trace over it is the broadband Leq over the same window. It is stored as a
fixed array scrolled in place, so 24 hours costs the same as 24 seconds
(480 columns × 256 bands ≈ 0.5 MB), and the column still filling is shown at
the right-hand edge so the newest minutes are not blank.

## Layout

| File | |
|---|---|
| `calibration.py` | Cal file parsing, SPL offset, response correction |
| `weighting.py` | IEC 61672 A/C weighting, exponential time weighting |
| `spectrum.py` | Streaming STFT, bin→band mapping |
| `metrics.py` | Leq / Lmax / Ln accumulation per interval |
| `analysis.py` | The chain: blocks in, columns and levels out |
| `longterm.py` | Power-domain averaging behind the 24 hour panel |
| `capture.py` | PortAudio, plus synthetic and WAV sources |
| `engine.py` | DSP thread, state snapshot for the GUI |
| `ui.py` | PyQtGraph window |
| `alsa.py` | Capture gain readback (Linux) |
| `logsink.py` | CSV writer |

## Caveats

- **Not a legal-grade sound level meter.** No type approval, no periodic
  verification. The chain is built to IEC 61672 definitions and tested against
  the standard's tolerance table, but a UMIK-1 on a Pi is not a certified
  instrument.
- **`noise-monitor gain` is untested against real hardware.** The ALSA parser is
  covered by unit tests against representative `amixer` output, but nobody has
  yet run it on a Pi with a UMIK-1 attached — verify it once by hand.
- **Outdoors you need a windscreen.** Wind noise is mostly infrasonic; it will
  not move dB(A) much but it will eat your headroom and clip.
