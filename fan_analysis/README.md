# Fan activity analysis (156 W 95th St rooftop heat pump)

Built 2026-10-06 from the monitor's own `logs/noise-*.csv` and
`screenshots/*.png`.

## Running it

The inputs are a running instrument's output and are **not in this repository**
— `.gitignore` keeps `logs/` and `screenshots/` out. By default the scripts look
for them beside this directory, which is how a monitor run in place lays itself
out. Point `NOISE_MONITOR_DATA` elsewhere for a collection kept somewhere else:

    pip install -e '.[analysis]'          # matplotlib, pillow (numpy is core)
    cd fan_analysis
    NOISE_MONITOR_DATA=~/Downloads/noise_monitor python3 build_log.py
    NOISE_MONITOR_DATA=~/Downloads/noise_monitor python3 chart.py

`build_log.py` writes `fan_events.csv`; `chart.py` reads it and writes both PNGs.
Those three outputs are committed, because the inputs are not: without them the
finding could not be reproduced from this repository alone.

## Outputs
- `fan_events.csv`  - one row per detected fan-on episode
- `fan_activity.png` - heatmap of minutes of fan activity by hour and date, with a
  daily high/low temperature strip above it (full res)
- `fan_doc.png`      - same figure, 1300 px wide, for pasting into documents
- `wx_archive.json`  - cached hourly temperature (see Temperature below)

## Why it works this way
The CSV logs hold only broadband metrics (LAeq, LAmax, LA10/50/90, Lpeak) - no
spectral bands - and `../history/long_term.npz` is only a rolling 24-hour buffer.
So the 1.3-3.5 kHz fan signature survives only in the saved screenshots. The
detector therefore runs on LA90 (the noise floor), which a steady fan lifts while
traffic transients mostly do not; the screenshots are used to validate it.

## Detection (detect.py + refine.py)
1. Per-minute LA90 series across all logs.
2. Baseline = rolling 20th percentile, 91-minute centred window.
3. Candidate = LA90 >= baseline + 4.0 dB, >= 4 min, gaps <= 3 min merged, <= 60 min.
4. Shape filter: duration 5-35 min; median level >= 3.5 dB above both the 8 min
   before and the 8 min after; std within the episode <= 2.5 dB.
5. Steadiness filter (added 2026-10-06, build_log.py): median(LAeq - LA90) <= 2.0 dB
   AND median(LA10 - LA90) <= 3.5 dB over the episode.

   Why: episodes clustered at 07:00-09:00 and 18:00-21:00 turned out to be the
   kitchen radio, not the fan. Level does NOT separate them - both sit at a median
   LA90 of 49.3 dB(A). Fluctuation does: speech and music swing, so LAeq rides well
   above the LA90 floor (5-10 dB, up to 10.6 dB at 21:00), while a steady fan keeps
   them nearly equal (~0.8 dB in the quiet overnight hours). This dropped 50 of 332
   episodes, 37 of them in those two windows, and removed the spurious morning and
   early-evening peaks from the hourly profile. It also drops 13 spectrally
   confirmed episodes (probably fan + radio together), so the count is deliberately
   conservative.

## Spectral confirmation (tonal3.py)
Reads the lower (long-term) spectrogram pane of each screenshot:
- panel x 77..631 = 480 columns of 180 s, ending at the time in the filename
- frequency axis: 1 kHz at y=296, 47 px per decade
- pixels are matched to the viridis colormap; poor matches (the red LAeq overlay,
  gridlines, text) are masked out
- metric = mean(1.3-3.5 kHz) - mean(250-700 Hz); fan = metric > 25th pct + 8
Episode counts as confirmed if >= 40% of its covered 3-min slots are above that.

## Results (as of 2026-10-06, after the steadiness filter)
- 282 episodes, 4 Aug - 6 Oct 2026 (332 before the steadiness filter)
- 268 on or before Sep 17; 14 after (Sep 17 was the last time the fan was heard),
  so the false-positive floor is roughly 0.7/day
- of the 67 episodes a screenshot covers, 56 confirmed, 11 not (about 84% agreement)
- median episode 9 min; noise floor 43.1 -> 49.3 dB(A), median rise 6.2 dB, max 52.4
- 46.5 hours of fan-on time to Sep 17; activity in every hour of the day, fairly
  flat across the day (4-6 min per logged hour late morning and early afternoon),
  night 22:00-06:00 averaging 3.5 min/hour vs 4.5 daytime

## CSV columns
date, start, end, duration_min, L90_during, L90_before, rise_dB, LAeq_mean,
LAeq_max, LAeq_minus_L90, LA10_minus_L90 (the two steadiness measures), spectral
(confirmed / not_confirmed / no_screenshot).

## Temperature (added 2026-10-06)
Hourly temperature comes from the Open-Meteo reanalysis archive for 40.7963 N,
73.9681 W (nearest grid point to 10025), Aug 4 - Oct 6 2026, in Fahrenheit, local
time. chart.py downloads it once into `wx_archive.json` and reuses that file; delete
the file to refetch. Full coverage, no missing hours. The strip above the heatmap
shows each day's high (line) and high-low range (shaded), with a dashed 75 F guide.

What it shows: the response is a THRESHOLD, not a slope. On the 18 logged days whose
high stayed below 75 F the median fan activity was 0 min per logged hour; above 75 F
the median was 2.4-4.1. Daily correlation is only r=0.44 and hourly r=0.22 because a
correlation understates a step response - and the unit answers to accumulated
building heat and tenants' thermostats, not the outdoor temperature at that instant.
Hours above 85 F had the fan running 64% of the time vs 11% below 65 F. Highs drop
below 75 F from about Sep 18, which is exactly when the logs go quiet - useful
corroboration that the silence is seasonal rather than a response to the complaint.

## The screenshot geometry is tied to one window layout

`tonal3.py` reads the long-term pane by **fixed pixel coordinates** — panel
x 77..631, 1 kHz at y=296, 47 px per decade. Those were measured off the
screenshots in this collection, which are themselves not all the same size
(800x418, 800x480 and 847x418 all appear). So the spectral metric is already
approximate across the set, which is part of why it is used as corroboration —
84% agreement with the level-based detector — rather than as ground truth.

It also means **screenshots taken after the UI changed will not parse**. The
window layout moved during October 2026: the colour bar was capped at 240 px
and top-aligned, a Shot button was added beneath it, the long-term x axis gained
a stacked date line, and the level trace range went 30–60 to 30–70 dB. Any of
those shifts the pane. Before trusting a spectral label on a newer screenshot,
re-measure the four constants against it.

## Layout note
The colorbar gets its own gridspec column. Attaching it with `fig.colorbar(ax=ax)`
steals width from the heatmap, which then no longer lines up with the temperature
strip above it (the date axes drift apart by a few days at the right edge).

## Changes made when this moved into the repository
- Paths to `logs/` and `screenshots/` were absolute; they now come from
  `paths.py` and honour `NOISE_MONITOR_DATA`.
- `build_log.py` wrote `fan_events.csv` into a scratch directory while
  `chart.py` read the copy beside it, so a re-run updated a file nobody read
  and left the chart on stale events. It now writes the one that is read.
- `endtime()` was handed `Path` objects by the new path helpers and
  `re.search` raised on them, which an except-and-continue swallowed — every
  screenshot was skipped and nothing was spectrally confirmed. It takes
  `str()` now, raises a clear error when a filename carries no timestamp, and
  the caller reports what it skipped instead of passing over it in silence.
- `fan_doc.png` was documented but nothing generated it, so regenerating the
  chart left it stale. `chart.py` now writes it, by downscaling the full-size
  render (re-rendering at 96 dpi reflows the footnote off the right edge).
- A no-op line in `refine.py`, overwritten on the next line, was removed.

Verified after the move: `build_log.py` reproduces `fan_events.csv`
byte-for-byte, and `chart.py` reproduces the figures and the published summary
numbers (282 episodes, 268 to Sep 17, median 9 min, 43.1 -> 49.3 dB(A),
46.5 hours, night 3.5 vs day 4.5 min/hour, 56 of 67 spectrally confirmed).
