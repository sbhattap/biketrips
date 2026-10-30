import pandas as pd
import pytest

from biketrips import config, data, quality

HEADER = ",_id,START_TIME,START_LAT,END_TIME,END_LNG,USER_ID,START_LNG,END_LAT"
T0, T1 = "2021-08-01 08:00:00.0", "2021-08-01 08:10:00.0"
ROWS = {  # trip id -> (start_time, start_lat, end_time, end_lng, start_lng, end_lat)
    "ok": (T0, 22.5, T1, 114.0, 114.0, 22.6),
    "dup": (T0, 22.5, T1, 114.0, 114.0, 22.6),
    "badts": ("not a time", 22.5, T1, 114.0, 114.0, 22.6),
    "back": (T1, 22.5, T0, 114.0, 114.0, 22.6),
    "zero": (T0, 22.5, T0, 114.0, 114.0, 22.6),
    "long": (T0, 22.5, "2021-08-01 20:00:00.0", 114.0, 114.0, 22.6),
    "nocoord": (T0, "", T1, 114.0, 114.0, 22.6),
    "origin": (T0, 0, T1, 114.0, 0, 22.6),
    "far": (T0, 43.9, T1, 114.0, 126.4, 22.6),
    "endfar": (T0, 22.5, T1, 120.0, 114.0, 22.6),
}
EXPECTED = {"duplicate_id": 1, "bad_timestamp": 1, "end_before_start": 1, "zero_duration": 1,
            "extreme_duration": 1, "missing_coords": 1, "zero_coords": 1, "start_out_of_bounds": 1,
            "end_out_of_bounds": 1, None: 1}


@pytest.fixture
def trips_csv(tmp_path):
    lines = [HEADER] + [f"{i},{k.replace('dup', 'ok')},{s},{sl},{e},{el},u,{sg},{ea}"
                        for i, (k, (s, sl, e, el, sg, ea)) in enumerate(ROWS.items())]
    path = tmp_path / "trips.csv"
    path.write_text("\n".join(lines))
    return path


def test_each_rule_catches_its_row(trips_csv, tmp_path):
    cfg = config.load_config()
    con = data.connect()
    log = quality.DropLog(tmp_path / "log.csv")
    counts = data.clean_trips(con, cfg, log=log, csv=trips_csv)
    assert counts == EXPECTED
    clean = con.sql("SELECT * FROM trips").df()
    assert len(clean) == 1
    # Kept trips are shifted from GCJ-02 to WGS84 and get an H3 cell.
    assert clean["end_lat"][0] == pytest.approx(22.6 - cfg["gcj_offset"]["lat"])
    assert clean["end_cell"][0].startswith("88")
    drops = log.df[log.df["action"] == "drop"]
    assert drops["rows"].sum() == len(ROWS) - 1


def test_droplog_rerun_replaces_stage(tmp_path):
    log = quality.DropLog(tmp_path / "log.csv")
    for _ in range(2):
        log.start("s")
        log.add("s", "step", "drop", 3, 10, "why")
    log.save()
    assert len(pd.read_csv(tmp_path / "log.csv")) == 1


def test_sanitize_matches_tutorial_rule():
    assert data.sanitize("Science/Culture & Education Service") == "ScienceCultureEducationService"


def test_hash_samples_are_nested():
    con = data.connect()
    ids = "SELECT 'id' || range::VARCHAR AS trip_id FROM range(20000)"
    small = {r[0] for r in con.sql(f"SELECT * FROM ({ids}) WHERE {data.sample_filter(0.01, 42, 'trip_id')}").fetchall()}
    big = {r[0] for r in con.sql(f"SELECT * FROM ({ids}) WHERE {data.sample_filter(0.1, 42, 'trip_id')}").fetchall()}
    assert small < big and 100 < len(big) < 3000
