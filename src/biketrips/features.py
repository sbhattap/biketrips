"""Full H3 grid over the boundary, the trip-end target, and cell-level urban features."""
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import sparse
from shapely.geometry import Polygon, mapping
from sklearn.neighbors import BallTree

from biketrips import data


def full_grid(boundary, res, buffer_deg):
    """Every res-`res` cell whose centre lies in the (slightly buffered) boundary, trips or not."""
    geom = boundary.geometry.union_all().buffer(buffer_deg)
    polys = getattr(geom, "geoms", [geom])
    cells = sorted(set().union(*(h3.polyfill(mapping(p), res, geo_json_conformant=True) for p in polys)))
    lat, lng = zip(*map(h3.h3_to_geo, cells))
    return pd.DataFrame({"h3": cells, "lat": lat, "lng": lng})


def to_gdf(cells):
    poly = [Polygon(h3.h3_to_geo_boundary(c, geo_json=True)) for c in cells["h3"]]
    return gpd.GeoDataFrame(cells, geometry=poly, crs=4326)


def neighbors(cells):
    """Sparse adjacency over k-ring 1, self excluded. Neighbours outside the grid are ignored."""
    idx = {c: i for i, c in enumerate(cells)}
    pairs = [(i, idx[n]) for c, i in idx.items() for n in h3.k_ring(c, 1) if n != c and n in idx]
    r, c = zip(*pairs)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(idx),) * 2)


def morans_i(adj, values):
    """Moran's I with row-standardised k-ring weights."""
    z = np.asarray(values, float) - np.mean(values)
    deg = np.asarray(adj.sum(1)).ravel()
    has = deg > 0
    lag = (adj @ z)[has] / deg[has]
    return len(z) / has.sum() * (z[has] @ lag) / (z @ z)


def neighbor_mean(adj, values):
    deg = np.asarray(adj.sum(1)).ravel()
    return np.where(deg > 0, adj @ np.asarray(values, float) / np.maximum(deg, 1), np.nan)


def trip_target(con, grid, log=None, week_start=None, weeks=0):
    """Trip-end counts on the full grid. Missing cells get 0 trips, which is a measured zero.

    With `weeks`, also counts per week (trips_w1, ...) and in how many weeks a cell had any trip.
    A cell empty every week looks like a gap in the operator's service, not a quiet week."""
    t0 = f"TIMESTAMP '{week_start}'"
    wk = "".join(f", count(*) FILTER (WHERE start_time >= {t0} + INTERVAL {7 * k} DAY AND start_time < "
                 f"{t0} + INTERVAL {7 * (k + 1)} DAY) AS trips_w{k + 1}" for k in range(weeks))
    counts = con.sql(f"SELECT end_cell AS h3, count(*) AS trips{wk} FROM trips GROUP BY 1").df()
    out = grid.merge(counts, on="h3", how="left")
    outside = counts.loc[~counts["h3"].isin(grid["h3"]), "trips"].sum()
    n_trips = counts["trips"].sum()
    filled = out["trips"].isna().sum()
    week_cols = [f"trips_w{k + 1}" for k in range(weeks)]
    out[["trips"] + week_cols] = out[["trips"] + week_cols].fillna(0).astype(int)
    out["is_zero"] = (out["trips"] == 0).astype(int)
    if weeks:
        out["weeks_served"] = (out[week_cols] > 0).sum(axis=1)
        out["always_served"] = (out["weeks_served"] == weeks).astype(int)
    if log is not None:
        log.start("cells")
        log.add("cells", "trip ends outside grid", "drop", outside, n_trips,
                "end cell centre outside the buffered boundary")
        log.add("cells", "cells with no trip end", "fill", filled, len(out),
                "trips = 0, is_zero = 1; zero may mean no bikes (supply), not no demand")
        if weeks:
            log.add("cells", "cells with trips in some weeks only", "flag",
                    out["weeks_served"].between(1, weeks - 1).sum(), len(out),
                    f"intermittent service; always_served = 1 needs trips in all {weeks} weeks")
    return out


def to_cells(lat, lng, res):
    return [h3.geo_to_h3(a, b, res) for a, b in zip(lat, lng)]


def poi_counts(pois, grid, res, log=None):
    """POI count per big category per cell. A cell with no POI of a category is a true zero."""
    cell = pd.Series(to_cells(pois.geometry.y, pois.geometry.x, res), name="h3")
    tab = pd.crosstab(cell, pois["category"].to_numpy()).add_prefix("poi_")
    out = grid[["h3"]].merge(tab, left_on="h3", right_index=True, how="left")
    cols = list(tab.columns)
    if log is not None:
        log.add("cells", "POIs outside grid", "drop", tab[~tab.index.isin(grid["h3"])].to_numpy().sum(),
                len(pois), "POI cell not in buffered boundary grid")
        log.add("cells", "POI counts missing", "fill", out[cols].isna().to_numpy().sum(), out[cols].size,
                "cell x category with no POI = 0 (true zero)")
    out[cols] = out[cols].fillna(0).astype(int)
    out["poi_total"] = out[cols].sum(axis=1)
    return out


def population(path, grid, res, log=None):
    """Mean population per raster pixel in each cell. No valid pixel means unknown, not zero."""
    with rasterio.open(path) as r:
        band, nodata, transform, crs = r.read(1), r.nodata, r.transform, r.crs
    rows, cols = np.nonzero(band != nodata)
    x, y = rasterio.transform.xy(transform, rows, cols)
    lng, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    px = pd.DataFrame({"h3": to_cells(lat, lng, res), "v": band[rows, cols]})
    out = grid[["h3"]].merge(px.groupby("h3")["v"].mean().rename("pop_density"), on="h3", how="left")
    out["pop_missing"] = out["pop_density"].isna().astype(int)
    if log is not None:
        log.add("cells", "population outside raster coverage", "flag", out["pop_missing"].sum(), len(out),
                "kept as NaN + pop_missing; imputed inside model pipelines on training folds only")
    return out


def transit(grid, metro, bus, res):
    tree = BallTree(np.radians(metro[["lat", "lng"]].to_numpy()), metric="haversine")
    dist, _ = tree.query(np.radians(grid[["lat", "lng"]].to_numpy()))
    stops = pd.Series(to_cells(bus["lat"], bus["lng"], res)).value_counts()
    out = pd.DataFrame({"h3": grid["h3"], "metro_dist_m": dist[:, 0] * 6_371_000,
                        "bus_stops": grid["h3"].map(stops).fillna(0).astype(int).to_numpy()})
    # Nearest stop always exists, so any NaN here is a bug, not missing data.
    assert out["metro_dist_m"].notna().all()
    return out


def spatial_lags(cells, cols):
    """Sum of each column over the k-ring 1 neighbours (self excluded)."""
    lag = neighbors(cells["h3"].tolist()) @ cells[cols].to_numpy(float)
    return pd.DataFrame(lag, columns=[f"lag_{c}" for c in cols], index=cells.index)


def feature_columns(cells):
    keep = ("poi_", "lag_", "pop_", "metro_", "bus_")
    return [c for c in cells.columns if c.startswith(keep)]


def build_cells(cfg, con, log=None):
    """One row per H3 cell on the full grid: target, flags, features, and the spatial group id."""
    res, d = cfg["features"]["h3_res"], cfg["data"]
    grid = full_grid(data.load_boundary(cfg), res, cfg["features"]["boundary_buffer_deg"])
    cells = trip_target(con, grid, log, cfg["cleaning"]["window"][0], cfg["features"]["weeks"])
    poi = poi_counts(data.load_pois(cfg, log), grid, res, log)
    metro, bus = data.load_transit(cfg, log)
    for part in (poi, population(d["population"], grid, res, log), transit(grid, metro, bus, res)):
        cells = cells.merge(part, on="h3", validate="one_to_one")
    lag_cols = [c for c in cells.columns if c.startswith("poi_")]
    cells = cells.join(spatial_lags(cells, lag_cols))
    cells["group"] = [h3.h3_to_parent(c, cfg["features"]["parent_res"]) for c in cells["h3"]]
    # Zero POIs and no known population: mostly water, mountains or the border strip.
    cells["trivial"] = ((cells["poi_total"] == 0) & (cells["pop_density"].fillna(0) == 0)).astype(int)
    if log is not None:
        log.add("cells", "trivial cells", "flag", cells["trivial"].sum(), len(cells),
                "zero POIs and zero or unknown population; metrics reported with and without them")
        log.add("cells", "trivial cells with trips", "info", (cells["trivial"] & ~cells["is_zero"]).sum(),
                cells["trivial"].sum(), "trivial cells that still have trip ends")
    return cells
