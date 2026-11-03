# Data quality report

Generated from `results/drop_log.csv`. Every drop, fill,
fix and flag is listed with a row count and a reason.

## trips

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| raw rows | info | 44,623,001 | 44,623,001 | 100.0000% | 100% of 2*_1000w.csv |
| missing_id | drop | 0 | 44,623,001 | 0.0000% | no trip id |
| duplicate_id | drop | 0 | 44,623,001 | 0.0000% | repeated trip id, first kept |
| duplicate_trip | drop | 77,020 | 44,623,001 | 0.1726% | same user, start time and start point under another id, first kept |
| bad_timestamp | drop | 0 | 44,545,981 | 0.0000% | null or unparseable timestamp |
| outside_window | drop | 39,204 | 44,545,981 | 0.0880% | start time outside the study window |
| end_before_start | drop | 0 | 44,506,777 | 0.0000% | end time before start |
| zero_duration | drop | 7,346 | 44,506,777 | 0.0165% | duration under min_duration_s |
| extreme_duration | drop | 134 | 44,499,431 | 0.0003% | duration over max_duration_s |
| missing_coords | drop | 0 | 44,499,297 | 0.0000% | null coordinate |
| zero_coords | drop | 47,825 | 44,499,297 | 0.1075% | GPS (0, 0) placeholder |
| start_out_of_bounds | drop | 6,001 | 44,451,472 | 0.0135% | start outside Shenzhen bbox |
| end_out_of_bounds | drop | 981 | 44,445,471 | 0.0022% | start in bbox, end outside |
| coordinate shift | fix | 44,444,490 | 44,444,490 | 100.0000% | GCJ-02 to WGS84 via gcj_offset |

## cells

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| trip ends outside grid | drop | 2,583 | 44,444,490 | 0.0058% | end cell centre outside the buffered boundary |
| cells with no trip end | fill | 1,090 | 3,272 | 33.3130% | trips = 0, is_zero = 1; zero may mean no bikes (supply), not no demand |
| cells with trips in some weeks only | flag | 435 | 3,272 | 13.2946% | intermittent service; always_served = 1 needs trips in all 4 weeks |
| POIs outside grid | drop | 2 | 1,126,565 | 0.0002% | POI cell not in buffered boundary grid |
| POI counts missing | fill | 15,801 | 75,256 | 20.9963% | cell x category with no POI = 0 (true zero) |
| population outside raster coverage | flag | 998 | 3,272 | 30.5012% | kept as NaN + pop_missing; imputed inside model pipelines on training folds only |
| trivial cells | flag | 608 | 3,272 | 18.5819% | zero POIs and zero or unknown population; metrics reported with and without them |
| trivial cells with trips | info | 133 | 608 | 21.8750% | trivial cells that still have trip ends |

## pois

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| raw rows | info | 1,127,200 | 1,127,200 | 100.0000% | 23 shapefiles |
| no shapefile: CultureEducation | info | 0 | 1,127,200 | 0.0000% | category in amap_poicode.xlsx only |
| category name check | info | 0 | 40,824 | 0.0000% | rows whose Chinese 大类 disagrees with the sanitized file category |
| no Chinese category column | info | 1,086,376 | 1,127,200 | 96.3783% | geometry-only files, category taken from file name |
| missing geometry | drop | 0 | 1,127,200 | 0.0000% | null or empty point |
| invalid geometry | drop | 0 | 1,127,200 | 0.0000% | shapely is_valid is False |
| duplicate id | drop | 635 | 1,127,200 | 0.0563% | same AMap id twice, first kept |
| same point, same category | flag | 17,654 | 1,126,565 | 1.5671% | kept: co-located venues (malls, towers) share a geocode |
| same point, other category | flag | 16,439 | 1,126,565 | 1.4592% | kept: multi-use addresses |

## transit

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| metro stations | info | 234 | 234 | 100.0000% | metro_area.csv rows |
| bus duplicate stations | drop | 20,402 | 26,406 | 77.2627% | same id_station listed once per route |
| coordinate shift | fix | 6,238 | 6,238 | 100.0000% | GCJ-02 to WGS84 via gcj_offset |
