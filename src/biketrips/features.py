"""Full H3 grid over the boundary, the trip-end target, and cell-level urban features."""
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Polygon, mapping


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


def trip_target(con, grid, log=None):
    """Trip-end counts on the full grid. Missing cells get 0 trips, which is a measured zero."""
    counts = con.sql("SELECT end_cell AS h3, count(*) AS trips FROM trips GROUP BY 1").df()
    out = grid.merge(counts, on="h3", how="left")
    outside = counts.loc[~counts["h3"].isin(grid["h3"]), "trips"].sum()
    n_trips = counts["trips"].sum()
    filled = out["trips"].isna().sum()
    out["trips"] = out["trips"].fillna(0).astype(int)
    out["is_zero"] = (out["trips"] == 0).astype(int)
    if log is not None:
        log.start("cells")
        log.add("cells", "trip ends outside grid", "drop", outside, n_trips,
                "end cell centre outside the buffered boundary")
        log.add("cells", "cells with no trip end", "fill", filled, len(out),
                "trips = 0, is_zero = 1; zero may mean no bikes (supply), not no demand")
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
