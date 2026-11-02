# biketrips

Predicting shared-bike demand from urban features in Shenzhen, with honest evaluation. The original tutorial's scores look better than they are: it trained on 1% of trips, dropped every cell with zero demand, and scored models with a random split that leaks spatial information. This repo fixes all three, compares five models under a spatial split, and asks which one actually predicts demand in neighborhoods it has never seen.

## Credit

**Based on:** *Predicting Mobility Demand from Urban Features*, Climate Change AI Summer School 2023, by Konstantin Klemmer, Shafat Rahman, Felix Wagner, Florian Nachtigall, and Gabriela Yaulli Herrera.

- Original repo: https://github.com/climatechange-ai-tutorials/mobility-demand
- DOI: https://doi.org/10.5281/zenodo.11619223
- Data: Shenzhen bike-share trip data and AMap POI data, as distributed with the tutorial.
- The original notebook is kept unchanged at `notebooks/00_original_tutorial.ipynb` (MIT license, see `LICENSE`).

### What's original vs. what's new

| From the tutorial | Added in this repo |
|---|---|
| Problem framing: trip ends per H3 cell from POI counts | All ~10M trips aggregated in DuckDB instead of a 1% pandas sample |
| Shenzhen trip, POI, boundary, population and transit data | Full H3 grid over the boundary, with zero-trip cells kept and flagged |
| H3 resolution-8 grid and the Shenzhen bounding box filter | Missingness and quality audit with a logged reason for every drop or fill (`results/data_quality.md`) |
| Linear regression baseline | GCJ-02 to WGS84 fix: trips and transit stops were about 550 m off the POIs |
| Two-layer GCN on a kNN graph of cells | All 23 POI categories, population with a missingness flag, metro distance, bus stops, spatial lags |
| XGBoost exercise and the idea of tuning it | Spatial evaluation: held-out res-6 blocks and grouped inner CV, next to a random split |
| | Poisson GLM and a spatial kNN baseline |
| | Tuning for every model (grid search, Optuna for XGBoost) |
| | Count-aware metrics: Poisson deviance, Spearman, top-10% hit rate |

## Results

Held-out test scores. Spatial = 19 unseen res-6 blocks (649 cells); random = 654 random cells. Higher is better except MAE.

| Model | R² random | R² spatial | Spearman random | Spearman spatial | Top-10% hit random | Top-10% hit spatial | MAE spatial |
|---|---|---|---|---|---|---|---|
| Ridge on log1p | -0.47 | -0.10 | 0.89 | 0.88 | 0.80 | 0.62 | 3,356 |
| Spatial kNN (no features) | 0.60 | 0.36 | **0.91** | 0.78 | 0.76 | 0.49 | 3,723 |
| Poisson GLM | 0.77 | 0.49 | 0.87 | 0.88 | 0.82 | 0.69 | 2,733 |
| **XGBoost (Poisson)** | **0.80** | **0.70** | 0.90 | **0.90** | **0.85** | **0.72** | **2,182** |
| GCN (transductive) | 0.73 | 0.53 | 0.90 | 0.89 | 0.79 | 0.65 | 2,734 |

- **Every model looks better under a random split.** R² drops by 0.10 (XGBoost) to 0.37 (ridge) on unseen blocks; the top-10% hit rate drops by 0.13 to 0.27.
- **The features-free kNN shows the leak most clearly.** It has the best random-split rank correlation of any model and the worst spatial one.
- **XGBoost is the model to use for unseen neighbourhoods**, and it loses the least when moving to the spatial split.
- **Easy zeros do not inflate the headline.** Scores without trivial cells (no POIs, no known population) are in `notebooks/02_results.ipynb`. R² moves by at most 0.03 for every model except ridge (0.06).
- **The largest errors are supply gaps.** Two held-out blocks near 22.74N, 114.05E have about 340 POIs per cell but about 460 trips in total. XGBoost predicts about 130k there. A model of demand from urban form cannot know where the operator does not put bikes.

Full table (MAE, RMSE, Poisson deviance, inner-CV scores): `notebooks/02_results.ipynb` and `results/metrics.csv`. Tuning everything takes about 4 minutes on one machine (GCN on GPU).

![Random vs spatial split](results/figures/random_vs_spatial.png)
![Top-k hit rate](results/figures/topk_hit_rate.png)
![Residual map](results/figures/residual_map.png)
![Feature importance](results/figures/xgb_importance.png)

## What we found in the data

- **1% of trips is too little.** In the tutorial's 1% sample, 613 of the 1,898 cells that have trips look empty, and the median per-cell count error after scaling up is 27%.
- **42% of cells have zero trips.** The tutorial dropped them, which also removes the hardest part of the problem. Here they stay in and carry an `is_zero` flag.
- **Two coordinate systems are mixed.** Trips and transit stops use GCJ-02 (AMap's offset coordinates); POIs, the boundary and the population raster use WGS84. Uncorrected, trip ends sit about 550 m (more than one cell edge) away from the POIs they are compared with.
- **Demand is spatially smooth** (Moran's I 0.87 on log counts), so a random split puts near-copies of each test cell into training.
- **The population file is not what its name suggests.** `pop/pop.shp` holds 234 metro stations; the population layer is the raster, and 30% of cells fall outside it.
- Every drop and fill, with counts and reasons, is in [`results/data_quality.md`](results/data_quality.md). The full EDA is in `notebooks/01_eda.ipynb`.

## How it works

- **Target:** trip-end count per H3 res-8 cell (about 0.74 km²) on the full grid over the Shenzhen boundary: 3,272 cells, 42% of them zero.
- **Features (52):** POI count per big category (23) and their total, the same counts summed over the 6 neighbouring cells, population density with a `pop_missing` flag, distance to the nearest metro station, and bus stop count. Population outside the raster's coverage stays NaN and is imputed inside each model, fit on training rows only.
- **Spatial split:** cells are grouped by their H3 res-6 parent (up to 49 cells, about 36 km²). 20% of groups are the test set, touched once. Hyperparameters are tuned with `GroupKFold(5)` on the remaining groups and ranked by Poisson deviance. The random split uses the same pipeline with plain `KFold`.
- **Models:** ridge regression on `log1p(trips)`; spatial kNN (mean of the k nearest training cells, no features); Poisson GLM; XGBoost with `count:poisson` and 40 Optuna trials; and the tutorial's two-layer GCN.
- **The GCN is transductive.** Every cell is a node in one kNN graph. Test cells stay in the graph with their features visible and their labels masked, and early stopping uses held-out training blocks. This is a softer test than for the other models, which never see test cells at all. An inductive model (GraphSAGE) is listed under Next steps.
- **Metrics** are in `src/biketrips/evaluate.py`. Negative predictions are clipped to 0 before scoring (only ridge can produce them; `n_clipped` in `metrics.csv` counts them). Every run appends to `results/metrics.csv`.

## Reproduce

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/download_data.py           # about 2 GB from the tutorial's Google Drive folder
python scripts/build_features.py          # cleaning + features, about 15 s
python scripts/run_experiments.py --model all
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_results.ipynb
pytest
```

All paths, seeds (42) and search spaces are in `configs/config.yaml`. The GCN uses a GPU when one is available. DuckDB loads its `h3` community extension on first use, which needs internet access once.

## Repo layout

```
configs/config.yaml        paths, H3 settings, CV, search spaces
scripts/                   download, build features, run experiments
src/biketrips/             data, quality, features, splits, models/, tune, evaluate, plots
notebooks/                 00 original tutorial (unchanged), 01 EDA, 02 results
results/                   metrics.csv, best_params.yaml, data_quality.md, predictions, figures/
tests/                     cleaning rules, grid and features, splits and metrics
```

## Limitations

- **Zero does not always mean no demand.** A cell with no trip ends may have had no bikes. The east of the city (east of 114.25E) holds 35% of cells but 0.8% of trips, which looks like the operator's service area rather than low demand. Models are scored on these cells as true zeros.
- **One operator, eight days.** Trips run from 2021-07-31 to 2021-08-07 (a few stray trips start on Jul 28 to 30). Four more trip files ship with the data (Aug 9 to 31) and are not used here.
- **Counts are overdispersed** (variance about 20,000 times the mean). The Poisson models rank and scale cells well, but their implied uncertainty is far too narrow.
- **The GCJ-02 shift is a constant offset** estimated from POIs that carry both coordinate systems. Its spread is about 13 m, small next to a 460 m cell edge.

## Next steps

- LightGBM as a second boosted-tree baseline.
- GraphSAGE or another inductive GNN that can score areas outside the training graph.
- A cell-by-hour model on the full dataset (start and end times are already kept in `data/interim/`).
- Origin-destination flows instead of trip ends alone.
- Separating supply from demand, for example with a service-area mask or bike availability data.
- Transfer to another city.

## Citation

```bibtex
@misc{klemmer2026predicting,
  title={Predicting Mobility Demand from Urban Features},
  author={Klemmer, Konstantin and Rahman, Shafat and Wagner, Felix and Nachtigall, Florian and Herrera, Gabriela Yaulli},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.11619223},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/mobility-demand}}
}
```
