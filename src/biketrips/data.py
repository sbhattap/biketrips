"""Loading and cleaning. Trips stay in DuckDB; spatial layers are small enough for GeoPandas."""
import re

import duckdb
import geopandas as gpd
import pandas as pd

# Rules run in this order; each trip is charged to the first rule it fails.
RULES = [
    ("missing_id", "trip_id IS NULL"),
    ("duplicate_id", "dup_rank > 1"),
    ("duplicate_trip", "trip_rank > 1 AND start_time IS NOT NULL"),
    ("bad_timestamp", "start_time IS NULL OR end_time IS NULL"),
    ("outside_window", "start_time < TIMESTAMP '{w0}' OR start_time >= TIMESTAMP '{w1}'"),
    ("end_before_start", "end_time < start_time"),
    ("zero_duration", "duration_s < {min_duration_s}"),
    ("extreme_duration", "duration_s > {max_duration_s}"),
    ("missing_coords", "start_lat IS NULL OR start_lng IS NULL OR end_lat IS NULL OR end_lng IS NULL"),
    ("zero_coords", "(start_lat = 0 AND start_lng = 0) OR (end_lat = 0 AND end_lng = 0)"),
    ("start_out_of_bounds", "NOT ({start_ok})"),
    # Start is valid by now, so these isolate GPS failures at drop-off.
    ("end_out_of_bounds", "NOT ({end_ok})"),
]
REASONS = {
    "missing_id": "no trip id", "duplicate_id": "repeated trip id, first kept",
    "duplicate_trip": "same user, start time and start point under another id, first kept",
    "outside_window": "start time outside the study window",
    "bad_timestamp": "null or unparseable timestamp", "end_before_start": "end time before start",
    "zero_duration": "duration under min_duration_s", "extreme_duration": "duration over max_duration_s",
    "missing_coords": "null coordinate", "zero_coords": "GPS (0, 0) placeholder",
    "start_out_of_bounds": "start outside Shenzhen bbox",
    "end_out_of_bounds": "start in bbox, end outside",
}


def connect():
    con = duckdb.connect()
    con.sql("INSTALL h3 FROM community; LOAD h3;")
    return con


def sample_filter(frac, seed, col="_id"):
    # Hash-based sampling is reproducible across thread counts and nested: the 1% sample is a
    # subset of the 10% sample, so noise comparisons isolate sample size.
    return "TRUE" if frac >= 1 else f"hash({col} || '{seed}') % 1000000 < {int(frac * 1e6)}"


def raw_trips_sql(csv, frac=1.0, seed=42):
    """Read every column as text so unparseable values surface as NULL instead of failing."""
    return f"""SELECT _id AS trip_id, USER_ID AS user_id,
        TRY_CAST(START_TIME AS TIMESTAMP) AS start_time, TRY_CAST(END_TIME AS TIMESTAMP) AS end_time,
        TRY_CAST(START_LAT AS DOUBLE) AS start_lat, TRY_CAST(START_LNG AS DOUBLE) AS start_lng,
        TRY_CAST(END_LAT AS DOUBLE) AS end_lat, TRY_CAST(END_LNG AS DOUBLE) AS end_lng,
        row_number() OVER (PARTITION BY _id ORDER BY START_TIME) AS dup_rank,
        -- The files overlap, and some trips appear twice under different ids.
        row_number() OVER (PARTITION BY USER_ID, START_TIME, START_LAT, START_LNG ORDER BY _id) AS trip_rank
        FROM read_csv('{csv}', all_varchar = true) WHERE {sample_filter(frac, seed)}"""


def _bounds(cfg, p):
    (lo0, lo1), (la0, la1) = cfg["cleaning"]["lng_range"], cfg["cleaning"]["lat_range"]
    return f"{p}_lng BETWEEN {lo0} AND {lo1} AND {p}_lat BETWEEN {la0} AND {la1}"


def clean_trips(con, cfg, frac=1.0, log=None, csv=None):
    """Build DuckDB tables `trips_flagged` (all rows + drop_reason) and `trips` (clean, WGS84 + H3)."""
    w0, w1 = cfg["cleaning"]["window"]
    fmt = dict(cfg["cleaning"], start_ok=_bounds(cfg, "start"), end_ok=_bounds(cfg, "end"), w0=w0, w1=w1)
    case = " ".join(f"WHEN {cond.format(**fmt)} THEN '{name}'" for name, cond in RULES)
    raw = raw_trips_sql(csv or cfg["data"]["trips_csv"], frac, cfg["seed"])
    con.sql(f"""CREATE OR REPLACE TABLE trips_flagged AS
        SELECT *, CASE {case} END AS drop_reason FROM (
        SELECT *, date_diff('second', start_time, end_time) AS duration_s FROM ({raw}))""")
    off, res = cfg["gcj_offset"], cfg["features"]["h3_res"]
    con.sql(f"""CREATE OR REPLACE TABLE trips AS SELECT trip_id, start_time, end_time, duration_s,
        start_lat - {off['lat']} AS start_lat, start_lng - {off['lng']} AS start_lng,
        end_lat - {off['lat']} AS end_lat, end_lng - {off['lng']} AS end_lng,
        h3_h3_to_string(h3_latlng_to_cell(end_lat - {off['lat']}, end_lng - {off['lng']}, {res})) AS end_cell
        FROM trips_flagged WHERE drop_reason IS NULL""")
    counts = dict(con.sql("SELECT drop_reason, count(*) FROM trips_flagged GROUP BY 1").fetchall())
    if log is not None:
        log.start("trips")
        n = sum(counts.values())
        log.add("trips", "raw rows", "info", n, n, f"{frac:.0%} of {cfg['data']['trips_csv'].name}")
        for name, _ in RULES:
            log.add("trips", name, "drop", counts.get(name, 0), n, REASONS[name])
            n -= counts.get(name, 0)
        log.add("trips", "coordinate shift", "fix", n, n, "GCJ-02 to WGS84 via gcj_offset")
    return counts


def load_boundary(cfg):
    b = gpd.read_file(cfg["data"]["boundary"])
    # The shapefile ships without a .prj; its bounds are lon/lat and it lines up with WGS84 POIs.
    return b.set_crs(4326) if b.crs is None else b.to_crs(4326)


def sanitize(name):
    return re.sub(r"[ &/]", "", name)


def poi_codes(cfg):
    codes = pd.read_excel(cfg["data"]["poi_codes"])
    return codes.assign(category=codes["Big Category"].map(sanitize))


def load_pois(cfg, log=None):
    """All POI shapefiles stacked, category taken from the file name and checked against the code map."""
    frames = []
    for path in sorted(cfg["data"]["poi_dir"].glob("*.shp")):
        g = gpd.read_file(path)
        cols = [c for c in ("id", "大类") if c in g.columns]
        frames.append(g[cols + ["geometry"]].assign(category=path.name.split("_")[1], file=path.name))
    pois = pd.concat(frames, ignore_index=True).pipe(gpd.GeoDataFrame, crs=4326)
    if log is not None:
        from biketrips.quality import audit_pois
        pois = audit_pois(pois, poi_codes(cfg), log)
    return pois


def load_transit(cfg, log=None):
    """Metro and bus stops in WGS84. Bus rows are route-stops, so stations repeat across routes."""
    off = cfg["gcj_offset"]
    metro = pd.read_csv(cfg["data"]["metro_stops"])[["stationnam", "lat", "lon"]].rename(columns={"lon": "lng"})
    bus = gpd.read_file(cfg["data"]["bus_stops"])[["id_station", "lat", "lng"]].astype({"lat": float, "lng": float})
    uniq = bus.drop_duplicates("id_station")
    if log is not None:
        log.start("transit")
        log.add("transit", "metro stations", "info", len(metro), len(metro), "metro_area.csv rows")
        log.add("transit", "bus duplicate stations", "drop", len(bus) - len(uniq), len(bus),
                "same id_station listed once per route")
        log.add("transit", "coordinate shift", "fix", len(metro) + len(uniq), len(metro) + len(uniq),
                "GCJ-02 to WGS84 via gcj_offset")
    shift = lambda d: d.assign(lat=d["lat"] - off["lat"], lng=d["lng"] - off["lng"])  # noqa: E731
    return shift(metro), shift(uniq.reset_index(drop=True))
