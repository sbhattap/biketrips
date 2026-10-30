"""Load configs/config.yaml and resolve data paths against the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASET = "Shenzhen Bike-Share Data"


def load_config(path=None):
    cfg = yaml.safe_load(open(path or ROOT / "configs" / "config.yaml"))
    d = cfg["data"]
    for key in ("raw_dir", "interim_dir", "processed_dir", "results_dir"):
        d[key] = ROOT / d[key]
    base = d["raw_dir"] / DATASET
    for key in ("trips_csv", "trips_zip", "poi_zip", "poi_dir", "poi_codes", "boundary",
                "population", "bus_stops", "metro_stops"):
        d[key] = base / d[key]
    return cfg
