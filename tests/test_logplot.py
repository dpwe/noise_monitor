"""Reading the CSV logs back for plotting."""

import csv

import numpy as np
import pytest

from noise_monitor.logplot import GAP_TOLERANCE, find_logs, load_logs

FIELDS = ["time", "timestamp", "duration_s", "LAeq", "LAmax", "LA90", "Lpeak",
          "clipped_samples", "dropped_blocks"]
BASE = 1_800_000_000.0


def write_log(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def row(when, leq, **extra):
    out = {name: 0 for name in FIELDS}
    out.update({"time": "irrelevant", "timestamp": when, "duration_s": 10.0,
                "LAeq": leq, "LAmax": leq + 5, "LA90": leq - 3, "Lpeak": leq + 20})
    out.update(extra)
    return out


@pytest.fixture
def one_day(tmp_path):
    rows = [row(BASE + 10 * i, 40.0 + i) for i in range(5)]
    return write_log(tmp_path / "logs" / "noise-20260821.csv", rows)


# --- loading ----------------------------------------------------------

def test_reads_a_single_file(one_day):
    series = load_logs(one_day)
    assert len(series) == 5
    assert series.timestamps[0] == BASE
    assert series.columns["LAeq"] == pytest.approx([40, 41, 42, 43, 44])
    assert series.interval_s == pytest.approx(10.0)


def test_a_directory_loads_every_day_in_order(tmp_path):
    logs = tmp_path / "logs"
    # Written newest-first, to prove the sort is not just file order.
    write_log(logs / "noise-20260822.csv", [row(BASE + 86400 + 10 * i, 50.0) for i in range(3)])
    write_log(logs / "noise-20260821.csv", [row(BASE + 10 * i, 40.0) for i in range(3)])
    series = load_logs(logs)
    assert len(series) == 6
    assert np.all(np.diff(series.timestamps) > 0)
    assert series.columns["LAeq"][:3] == pytest.approx([40, 40, 40])


def test_level_columns_are_picked_out(one_day):
    assert set(load_logs(one_day).level_columns) == {"LAeq", "LAmax", "LA90", "Lpeak"}
    assert "clipped_samples" not in load_logs(one_day).level_columns


def test_metrics_resolve_across_the_weighting_letter(one_day):
    """A C-weighted log has LCeq where this one has LAeq."""
    series = load_logs(one_day)
    assert series.resolve("LAeq") == "LAeq"
    assert series.resolve("Leq") == "LAeq"
    assert series.resolve("L90") == "LA90"
    assert series.resolve("Lpeak") == "Lpeak"
    assert series.resolve("Lnonsense") is None


def test_a_column_missing_from_one_file_is_nan_not_a_shift(tmp_path):
    logs = tmp_path / "logs"
    write_log(logs / "noise-20260821.csv", [row(BASE, 40.0)])
    short = [f for f in FIELDS if f != "LA90"]
    write_log(logs / "noise-20260822.csv",
              [{k: v for k, v in row(BASE + 86400, 50.0).items() if k != "LA90"}], short)
    series = load_logs(logs)
    assert series.columns["LAeq"] == pytest.approx([40.0, 50.0])
    assert series.columns["LA90"][0] == pytest.approx(37.0)
    assert np.isnan(series.columns["LA90"][1])


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path):
    path = write_log(tmp_path / "noise-20260821.csv", [row(BASE, 40.0), row(BASE + 10, 41.0)])
    with open(path, "a") as fh:
        fh.write("2026-08-21T00:00:20.000,")  # power cut mid-row
    series = load_logs(path)
    assert len(series) == 2


def test_a_csv_that_is_not_ours_is_ignored(tmp_path):
    logs = tmp_path / "logs"
    write_log(logs / "noise-20260821.csv", [row(BASE, 40.0)])
    (logs / "noise-notes.csv").write_text("a,b\n1,2\n")
    series = load_logs(logs)
    assert len(series) == 1
    assert len(series.sources) == 1


def test_no_files_is_an_error_not_an_empty_plot(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_logs(tmp_path)


def test_a_file_with_no_readable_rows_is_an_error(tmp_path):
    path = write_log(tmp_path / "noise-20260821.csv", [])
    with pytest.raises(ValueError):
        load_logs(path)


# --- gaps -------------------------------------------------------------

def test_an_uninterrupted_log_is_not_broken(one_day):
    series = load_logs(one_day)
    times, values = series.broken_at_gaps("LAeq")
    assert times.size == len(series)
    assert not np.isnan(values).any()


def test_an_outage_breaks_the_line(tmp_path):
    """A straight line across an outage is a measurement nobody made."""
    rows = [row(BASE + 10 * i, 40.0) for i in range(3)]
    rows += [row(BASE + 3600 + 10 * i, 45.0) for i in range(3)]  # an hour off
    series = load_logs(write_log(tmp_path / "noise-20260821.csv", rows))
    times, values = series.broken_at_gaps("LAeq")
    assert int(np.isnan(values).sum()) == 1
    # The break sits inside the hole, moving no real sample.
    hole = int(np.flatnonzero(np.isnan(values))[0])
    assert BASE + 20 < times[hole] < BASE + 3600


def test_ordinary_jitter_does_not_count_as_an_outage(tmp_path):
    stamps = [BASE, BASE + 10, BASE + 20.5, BASE + 30, BASE + 10 * GAP_TOLERANCE + 30]
    rows = [row(t, 40.0) for t in stamps[:4]]
    series = load_logs(write_log(tmp_path / "noise-20260821.csv", rows))
    _, values = series.broken_at_gaps("LAeq")
    assert not np.isnan(values).any()


# --- file discovery ---------------------------------------------------

def test_find_logs_is_chronological(tmp_path):
    for day in ("20260823", "20260821", "20260822"):
        write_log(tmp_path / f"noise-{day}.csv", [row(BASE, 40.0)])
    assert [p.name for p in find_logs(tmp_path)] == [
        "noise-20260821.csv", "noise-20260822.csv", "noise-20260823.csv"
    ]


def test_days_takes_the_newest(tmp_path):
    for day in ("20260821", "20260822", "20260823"):
        write_log(tmp_path / f"noise-{day}.csv", [row(BASE, 40.0)])
    assert [p.name for p in find_logs(tmp_path, days=2)] == [
        "noise-20260822.csv", "noise-20260823.csv"
    ]
