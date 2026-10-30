"""Drop/fill log, POI audit, and the data-quality numbers the EDA reports."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from biketrips.data import sample_filter

COLS = ["stage", "step", "action", "rows", "of", "reason"]


class DropLog:
    """Every drop, fill, fix or flag is a row here; results/data_quality.md is rendered from it."""

    def __init__(self, path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=COLS)

    def start(self, stage):
        # Re-running a stage replaces its rows instead of appending duplicates.
        self.df = self.df[self.df["stage"] != stage]

    def add(self, stage, step, action, rows, of, reason):
        row = pd.DataFrame([[stage, step, action, int(rows), int(of), reason]], columns=COLS)
        self.df = pd.concat([self.df.astype(row.dtypes), row], ignore_index=True)

    def save(self):
        self.df.to_csv(self.path, index=False)


def write_report(log, path, intro=""):
    lines = ["# Data quality report", "", "Generated from `results/drop_log.csv`. Every drop, fill,",
             "fix and flag is listed with a row count and a reason.", ""] + ([intro, ""] if intro else [])
    for stage, g in log.df.groupby("stage", sort=False):
        lines += [f"## {stage}", "", "| step | action | rows | of | share | reason |", "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            share = f"{r.rows / r.of:.4%}" if r.of else ""
            lines.append(f"| {r.step} | {r.action} | {r.rows:,} | {r.of:,} | {share} | {r.reason} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def audit_pois(pois, codes, log):
    log.start("pois")
    n = len(pois)
    log.add("pois", "raw rows", "info", n, n, f"{pois['file'].nunique()} shapefiles")
    in_codes, in_files = set(codes["category"]), set(pois["category"])
    for cat in sorted(in_codes - in_files):
        log.add("pois", f"no shapefile: {cat}", "info", 0, n, "category in amap_poicode.xlsx only")
    for cat in sorted(in_files - in_codes):
        log.add("pois", f"not in code map: {cat}", "info", (pois["category"] == cat).sum(), n, "shapefile only")
    # Only some files carry the Chinese category; where present it must agree with the code map.
    zh = codes.drop_duplicates("category").set_index("category")["大类"]
    has_zh = pois["大类"].notna() if "大类" in pois else pd.Series(False, index=pois.index)
    bad = has_zh & (pois["大类"] != pois["category"].map(zh))
    log.add("pois", "category name check", "info", bad.sum(), has_zh.sum(),
            "rows whose Chinese 大类 disagrees with the sanitized file category")
    log.add("pois", "no Chinese category column", "info", (~has_zh).sum(), n,
            "geometry-only files, category taken from file name")
    missing = pois.geometry.isna() | pois.geometry.is_empty
    log.add("pois", "missing geometry", "drop", missing.sum(), n, "null or empty point")
    pois = pois[~missing]
    invalid = ~pois.geometry.is_valid
    log.add("pois", "invalid geometry", "drop", invalid.sum(), len(pois), "shapely is_valid is False")
    pois = pois[~invalid]
    dup_id = pois["id"].notna() & pois.duplicated("id") if "id" in pois else pd.Series(False, index=pois.index)
    log.add("pois", "duplicate id", "drop", dup_id.sum(), len(pois), "same AMap id twice, first kept")
    pois = pois[~dup_id]
    xy = pd.DataFrame({"x": pois.geometry.x.round(6), "y": pois.geometry.y.round(6), "c": pois["category"]})
    log.add("pois", "same point, same category", "flag", xy.duplicated().sum(), len(pois),
            "kept: co-located venues (malls, towers) share a geocode")
    cross = xy.drop_duplicates().duplicated(["x", "y"]).sum()
    log.add("pois", "same point, other category", "flag", cross, len(pois), "kept: multi-use addresses")
    return pois.drop(columns=["file"]).reset_index(drop=True)


def null_rates(con, table):
    return con.sql(f"SUMMARIZE {table}").df()[["column_name", "column_type", "count", "null_percentage",
                                                "min", "max", "approx_unique"]]


def drop_summary(con):
    return con.sql("""SELECT coalesce(drop_reason, 'kept') AS reason, count(*) AS n,
        round(100 * count(*) / sum(count(*)) OVER (), 4) AS pct FROM trips_flagged GROUP BY 1 ORDER BY n DESC""").df()


def invalid_points(con, decimals=1):
    """Where the bad end of each invalid trip lands, binned to a coarse lon/lat grid."""
    bad_start = "drop_reason = 'start_out_of_bounds' OR (start_lat = 0 AND start_lng = 0)"
    return con.sql(f"""SELECT drop_reason, CASE WHEN {bad_start} THEN 'start' ELSE 'end' END AS side,
        round(CASE WHEN {bad_start} THEN start_lng ELSE end_lng END, {decimals}) AS lng,
        round(CASE WHEN {bad_start} THEN start_lat ELSE end_lat END, {decimals}) AS lat, count(*) AS n
        FROM trips_flagged WHERE drop_reason IN ('zero_coords', 'start_out_of_bounds', 'end_out_of_bounds')
        GROUP BY ALL ORDER BY n DESC""").df()


def durations(con):
    q = [0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1]
    vals = con.sql(f"SELECT quantile_cont(duration_s, {q}) FROM trips_flagged").fetchall()[0][0]
    return pd.Series(vals, index=q, name="duration_s")


def duration_hist(con):
    """Trip counts in log10(seconds) bins of width 0.1, computed in DuckDB."""
    return con.sql("""SELECT round(log10(greatest(duration_s, 1)), 1) AS log10_s, count(*) AS n
        FROM trips_flagged WHERE duration_s >= 0 GROUP BY 1 ORDER BY 1""").df()


def temporal(con, table="trips"):
    return con.sql(f"""SELECT strftime(start_time, '%Y-%m-%d') AS day, hour(start_time) AS hour,
        dayname(start_time) AS weekday, count(*) AS n FROM {table} GROUP BY ALL ORDER BY day, hour""").df()


def sample_noise(con, cells, fractions, seed):
    """Per-cell trip-end counts from nested hash samples, scaled up and compared with the full count."""
    full = cells.set_index("h3")["trips"]
    rows = []
    for f in fractions:
        cond = sample_filter(f, seed, "trip_id")
        c = con.sql(f"SELECT end_cell, count(*) AS n FROM trips WHERE {cond} GROUP BY 1").df()
        est = c.set_index("end_cell")["n"].reindex(full.index, fill_value=0) / f
        pos = full > 0
        rel = (est[pos] - full[pos]).abs() / full[pos]
        rows.append({"fraction": f, "trips": int(est.sum() * f), "cells_seen": int((est > 0).sum()),
                     "false_zero_cells": int(((est == 0) & pos).sum()),
                     "median_rel_error": rel.median(), "p90_rel_error": rel.quantile(0.9),
                     "spearman_vs_full": spearmanr(est, full)[0]})
    return pd.DataFrame(rows)


def correlations(cells, cols, target="trips"):
    """Spearman correlation of each feature with the target, strongest first."""
    rho = {c: spearmanr(cells[c], cells[target], nan_policy="omit")[0] for c in cols}
    return pd.Series(rho, name="spearman").sort_values(ascending=False)


def overdispersion(y):
    y = np.asarray(y, float)
    return {"mean": y.mean(), "variance": y.var(), "var_to_mean": y.var() / y.mean(),
            "zero_share": (y == 0).mean()}
