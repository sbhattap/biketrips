# Data quality report

Generated from `results/drop_log.csv`. Every drop, fill,
fix and flag is listed with a row count and a reason.

## trips

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| raw rows | info | 9,980,998 | 9,980,998 | 100.0000% | 100% of 21_1000w.csv |
| missing_id | drop | 0 | 9,980,998 | 0.0000% | no trip id |
| duplicate_id | drop | 0 | 9,980,998 | 0.0000% | repeated trip id, first kept |
| bad_timestamp | drop | 0 | 9,980,998 | 0.0000% | null or unparseable timestamp |
| end_before_start | drop | 0 | 9,980,998 | 0.0000% | end time before start |
| zero_duration | drop | 1,356 | 9,980,998 | 0.0136% | duration under min_duration_s |
| extreme_duration | drop | 50 | 9,979,642 | 0.0005% | duration over max_duration_s |
| missing_coords | drop | 0 | 9,979,592 | 0.0000% | null coordinate |
| zero_coords | drop | 11,539 | 9,979,592 | 0.1156% | GPS (0, 0) placeholder |
| start_out_of_bounds | drop | 2,025 | 9,968,053 | 0.0203% | start outside Shenzhen bbox |
| end_out_of_bounds | drop | 318 | 9,966,028 | 0.0032% | start in bbox, end outside |
| coordinate shift | fix | 9,965,710 | 9,965,710 | 100.0000% | GCJ-02 to WGS84 via gcj_offset |

## cells

| step | action | rows | of | share | reason |
|---|---|---|---|---|---|
| trip ends outside grid | drop | 534 | 9,965,710 | 0.0054% | end cell centre outside the buffered boundary |
| cells with no trip end | fill | 1,374 | 3,272 | 41.9927% | trips = 0, is_zero = 1; zero may mean no bikes (supply), not no demand |
| POIs outside grid | drop | 2 | 1,126,565 | 0.0002% | POI cell not in buffered boundary grid |
| POI counts missing | fill | 15,801 | 75,256 | 20.9963% | cell x category with no POI = 0 (true zero) |
| zero-trip cells | flag | 1,374 | 3,272 | 41.9927% | is_zero = 1 |
| zero POI and zero trips | flag | 605 | 3,272 | 18.4902% | likely water or non-urban; results reported with and without trivial zeros |

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
