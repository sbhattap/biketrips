"""Clean all trips, keep them in data/interim, and write the cell table to data/processed."""
from biketrips import config, data, features, quality


def main():
    cfg = config.load_config()
    d = cfg["data"]
    con = data.connect()
    log = quality.DropLog(d["results_dir"] / "drop_log.csv")
    data.clean_trips(con, cfg, frac=1.0, log=log)
    d["interim_dir"].mkdir(parents=True, exist_ok=True)
    # Timestamps are kept for later cell-by-hour work; the target here is static.
    con.sql(f"COPY trips TO '{d['interim_dir'] / 'trips_clean.parquet'}' (FORMAT parquet)")
    cells = features.build_cells(cfg, con, log)
    d["processed_dir"].mkdir(parents=True, exist_ok=True)
    cells.to_parquet(d["processed_dir"] / "cells.parquet")
    log.save()
    quality.write_report(log, d["results_dir"] / "data_quality.md")
    print(f"{len(cells)} cells, {len(features.feature_columns(cells))} features, "
          f"zero share {cells['is_zero'].mean():.1%}")


if __name__ == "__main__":
    main()
