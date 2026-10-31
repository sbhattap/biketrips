import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from biketrips import config, data, features

CENTER = h3.geo_to_h3(22.55, 114.05, 8)


@pytest.fixture
def grid():
    return features.full_grid(gpd.GeoDataFrame(geometry=[box(114.0, 22.5, 114.1, 22.6)], crs=4326), 8, 0.0)


def test_target_on_full_grid_keeps_zeros(grid):
    con = data.connect()
    ends = [CENTER] * 3 + [grid["h3"].iloc[0]] + ["88411c0001fffff"]  # last one is outside the grid
    con.sql("CREATE TABLE trips AS SELECT unnest($e) AS end_cell", params={"e": ends})
    t = features.trip_target(con, grid)
    assert t["h3"].is_unique and len(t) == len(grid)
    assert t["trips"].notna().all() and t["trips"].sum() == 4
    assert t["is_zero"].sum() == len(grid) - 2


def test_poi_counts_fill_true_zeros(grid):
    lat, lng = h3.h3_to_geo(CENTER)
    pois = gpd.GeoDataFrame({"category": ["Shopping", "Shopping", "MedicalService"]},
                            geometry=[Point(lng, lat)] * 3, crs=4326)
    p = features.poi_counts(pois, grid, 8).set_index("h3")
    assert p.loc[CENTER, "poi_Shopping"] == 2 and p.loc[CENTER, "poi_total"] == 3
    assert p.notna().all().all() and p["poi_total"].sum() == 3


def test_spatial_lag_sums_neighbours_only(grid):
    cells = grid.assign(x=(grid["h3"] == CENTER).astype(float))
    lag = features.spatial_lags(cells, ["x"])["lag_x"]
    ring = set(h3.k_ring(CENTER, 1)) - {CENTER}
    assert lag[cells["h3"] == CENTER].item() == 0
    assert (lag[cells["h3"].isin(ring)] == 1).all() and lag.sum() == 6


def test_morans_i_sign(grid):
    adj = features.neighbors(grid["h3"].tolist())
    smooth = grid["lng"].to_numpy()
    noise = np.random.default_rng(0).normal(size=len(grid))
    assert features.morans_i(adj, smooth) > 0.8
    assert abs(features.morans_i(adj, noise)) < 0.15


def test_built_cells_invariants():
    path = config.load_config()["data"]["processed_dir"] / "cells.parquet"
    if not path.exists():
        pytest.skip("run scripts/build_features.py first")
    c = pd.read_parquet(path)
    assert c["h3"].is_unique
    assert c["is_zero"].sum() > 0 and (c["trips"] == 0).sum() == c["is_zero"].sum()
    nan_cols = set(c.columns[c.isna().any()])
    assert nan_cols <= {"pop_density"}
    assert (c["pop_density"].isna() == c["pop_missing"].astype(bool)).all()
