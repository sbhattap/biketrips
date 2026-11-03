# biketrips

Where will people ride shared bikes in a part of the city a model has never seen? This repo builds a city-wide benchmark for that question in Shenzhen: all ~10M trips, every H3 cell including the 42% with no trips, a logged audit of every data-quality decision, 52 urban features, and eight models tuned and scored on held-out city blocks. The best model, XGBoost with a Poisson loss, reaches R² 0.70 on unseen blocks. Most of the remaining error comes from where the operator puts bikes, not from model choice.

**Starting point:** the Climate Change AI tutorial *Predicting Mobility Demand from Urban Features* by Konstantin Klemmer, Shafat Rahman, Felix Wagner, Florian Nachtigall, and Gabriela Yaulli Herrera ([repo](https://github.com/climatechange-ai-tutorials/mobility-demand), [DOI](https://doi.org/10.5281/zenodo.11619223)). It introduced the problem, the data and the first models. See [Credit and citation](#credit-and-citation).

## What this repo does

- **Full data, not a sample.** All 9,980,998 trips are aggregated with DuckDB. A 1% sample leaves a third of active cells looking empty.
- **Every cell counts.** A full H3 grid over the city boundary (3,272 cells), with zero-trip cells kept and flagged as possible supply gaps.
- **Audited data.** Every drop and fill is logged with a row count and a reason (`results/data_quality.md`). The audit found a ~550 m datum mismatch between trips and POIs (GCJ-02 vs WGS84) and a population file that holds metro stations.
- **Richer features.** All 23 POI categories, spatial lags, population density with a missingness flag, metro distance and bus stops.
- **Honest evaluation.** Whole city blocks are held out, with grouped inner CV for tuning, and a random split is run next to it to measure the spatial leak.
- **Eight models, tuned the same way:** two linear baselines, spatial kNN, Poisson GLM, XGBoost, HistGradientBoosting, a two-part hurdle model and a GCN.
- **Count-aware metrics:** Poisson deviance, Spearman and top-10% hit rate alongside MAE, RMSE and R², with and without trivially empty cells.

## Results

Held-out test scores. Spatial = 19 unseen res-6 blocks (649 cells); random = 654 random cells. Higher is better except MAE.

| Model | R² random | R² spatial | Spearman random | Spearman spatial | Top-10% hit random | Top-10% hit spatial | MAE spatial |
|---|---|---|---|---|---|---|---|
| *Baseline LR (3 POI types, non-zero cells)* | 0.72 | 0.62 | 0.80 | 0.85 | 0.77 | 0.71 | 2,537 |
| Ridge on log1p (variant) | -0.47 | -0.10 | 0.89 | 0.88 | 0.80 | 0.62 | 3,356 |
| Spatial kNN (no features) | 0.60 | 0.36 | **0.91** | 0.78 | 0.76 | 0.49 | 3,723 |
| Poisson GLM | 0.77 | 0.49 | 0.87 | 0.88 | 0.82 | 0.69 | 2,733 |
| **XGBoost (Poisson)** | **0.80** | **0.70** | 0.90 | **0.90** | **0.85** | **0.72** | 2,182 |
| HistGradientBoosting (Poisson) | **0.80** | 0.69 | 0.90 | **0.90** | 0.82 | 0.68 | 2,182 |
| Hurdle: P(any trips) x trips if any | 0.79 | 0.69 | 0.91 | **0.90** | 0.82 | 0.69 | **2,154** |
| GCN (transductive) | 0.73 | 0.53 | 0.90 | 0.89 | 0.79 | 0.65 | 2,734 |

Baseline LR is plain least squares on raw counts of Sports & Recreation, Transportation and Medical POIs, trained only on cells with trips and not tuned. It is the simplest reasonable model and the one this project started from. It is scored on the same test cells as every other model.

- **Only boosted trees clearly beat Baseline LR on unseen blocks.** Tuned XGBoost improves R² by 0.08, MAE by 14% and Spearman by 0.05, and cuts Poisson deviance by 44%. The top-10% hit rate barely moves (0.72 vs 0.71).
- **The win comes from boosting, not from XGBoost.** sklearn's HistGradientBoosting, with the same Poisson loss and tuning budget, matches XGBoost on unseen blocks (R² 0.69 vs 0.70, same MAE). XGBoost keeps a higher top-10% hit rate (0.72 vs 0.68).
- **A two-part hurdle model helps a little, but not where it matters most.** It predicts "any trips?" and "how many?" with the same trees. Against the one-part version it lowers spatial MAE by 1% and Poisson deviance by 3%, and drops the median prediction on zero cells from 4 trips to 1. It does not fix the supply gaps below: those cells look busy, so the classifier is confident they have trips.
- **More features and fancier models are not enough on their own.** The Poisson GLM and the GCN use all 52 features but have lower spatial R² than the 3-feature Baseline LR. The ridge-on-log variant, an early design choice here, is the worst model on R² (its back-transform distorts large cells), though it ranks cells well. Duan's smearing correction made it worse (random R² -5.6): log errors are far larger on some cells than others, so one constant factor inflates the biggest predictions.
- **Baseline LR has its own flaw: negative counts.** It predicts below zero for 307 of 649 spatial test cells (clipped to 0 before scoring).
- **A random split flatters every model that scores positive R².** Moving to unseen blocks costs 0.10 (XGBoost and Baseline LR) to 0.28 (Poisson GLM). Ridge is negative under both splits. Every model loses top-10% hit rate (0.07 to 0.27).
- **The features-free kNN shows the leak most clearly.** It has the best random-split rank correlation of any model and the worst spatial one.
- **Easy zeros do not inflate the headline.** Without trivial cells (no POIs, no known population), R² moves by at most 0.03 for every model except ridge (0.06). See `notebooks/02_results.ipynb`.
- **The largest errors are supply gaps.** Two held-out blocks near 22.74N, 114.05E have about 340 POIs per cell but about 460 trips in total. All three tree models predict 126k to 134k there. A model of demand from urban form cannot know where the operator does not put bikes.

Full table (MAE, RMSE, Poisson deviance, clipping counts, inner-CV scores): `notebooks/02_results.ipynb` and `results/metrics.csv`. Tuning everything takes about 10 minutes on one machine (GCN on GPU).

![Random vs spatial split](results/figures/random_vs_spatial.png)
![Top-k hit rate](results/figures/topk_hit_rate.png)
![Residual map](results/figures/residual_map.png)
![Feature importance](results/figures/xgb_importance.png)

## What we found in the data

- **1% of trips is too little.** In a 1% sample, 613 of the 1,898 cells that have trips look empty, and the median per-cell count error after scaling up is 27%.
- **42% of cells have zero trips.** Dropping them is tempting, but it removes the hardest part of the problem. Here they stay in and carry an `is_zero` flag.
- **Two coordinate systems are mixed.** Trips and transit stops use GCJ-02 (AMap's offset coordinates); POIs, the boundary and the population raster use WGS84. Uncorrected, trip ends sit about 550 m (more than one cell edge) away from the POIs they are compared with.
- **Demand is spatially smooth** (Moran's I 0.87 on log counts), so a random split puts near-copies of each test cell into training.
- **The population file is not what its name suggests.** `pop/pop.shp` holds 234 metro stations; the population layer is the raster, and 30% of cells fall outside it.
- Every drop and fill, with counts and reasons, is in [`results/data_quality.md`](results/data_quality.md). The full EDA is in `notebooks/01_eda.ipynb`.

## How it works

- **Target:** trip-end count per H3 res-8 cell (about 0.74 km²) on the full grid over the Shenzhen boundary: 3,272 cells, 42% of them zero.
- **Features (52):** POI count per big category (23) and their total, the same counts summed over the 6 neighbouring cells, population density with a `pop_missing` flag, distance to the nearest metro station, and bus stop count. Population outside the raster's coverage stays NaN and is imputed inside each model, fit on training rows only.
- **Spatial split:** cells are grouped by their H3 res-6 parent (up to 49 cells, about 36 km²). 20% of groups are the test set, touched once. Hyperparameters are tuned with `GroupKFold(5)` on the remaining groups and ranked by Poisson deviance. The random split uses the same pipeline with plain `KFold`.
- **Models:** Baseline LR (untuned); ridge regression on `log1p(trips)`; spatial kNN (mean of the k nearest training cells, no features); Poisson GLM; XGBoost with `count:poisson` and 40 Optuna trials; sklearn's HistGradientBoosting with a Poisson loss and the same budget; a hurdle model (HistGradientBoosting classifier for P(trips > 0) times a Poisson regressor fit on cells with trips, shared hyperparameters); and a two-layer GCN.
- **The GCN is transductive.** Every cell is a node in one kNN graph. Test cells stay in the graph with their features visible and their labels masked, and early stopping uses held-out training blocks. This is a softer test than for the other models, which never see test cells at all. An inductive model (GraphSAGE) is listed under Next steps.
- **Metrics** are in `src/biketrips/evaluate.py`. Negative predictions are clipped to 0 before scoring (only the two linear models produce them; `n_clipped` in `metrics.csv` counts them). Every run appends to `results/metrics.csv`.

## Reproduce

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/download_data.py           # about 2 GB from the public Google Drive folder
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
notebooks/                 00 starting-point tutorial (unchanged), 01 EDA, 02 results
results/                   metrics.csv, best_params.yaml, data_quality.md, predictions, figures/
tests/                     cleaning rules, grid and features, splits and metrics
```

## Limitations

- **Zero does not always mean no demand.** A cell with no trip ends may have had no bikes. The east of the city (east of 114.25E) holds 35% of cells but 0.8% of trips, which looks like the operator's service area rather than low demand. Models are scored on these cells as true zeros.
- **One operator, eight days.** Trips run from 2021-07-31 to 2021-08-07 (a few stray trips start on Jul 28 to 30). Four more trip files ship with the data (Aug 9 to 31) and are not used here.
- **Counts are overdispersed** (variance about 20,000 times the mean). The Poisson models rank and scale cells well, but their implied uncertainty is far too narrow.
- **The GCJ-02 shift is a constant offset** estimated from POIs that carry both coordinate systems. Its spread is about 13 m, small next to a 460 m cell edge.

## Next steps

- A zero-truncated count part for the hurdle model, or a negative binomial GLM for the overdispersion.
- GraphSAGE or another inductive GNN that can score areas outside the training graph.
- A cell-by-hour model on the full dataset (start and end times are already kept in `data/interim/`).
- Origin-destination flows instead of trip ends alone.
- Separating supply from demand, for example with a service-area mask or bike availability data.
- Transfer to another city.

## Credit and citation

This project started from *Predicting Mobility Demand from Urban Features*, Climate Change AI Summer School 2023, by Konstantin Klemmer, Shafat Rahman, Felix Wagner, Florian Nachtigall, and Gabriela Yaulli Herrera ([repo](https://github.com/climatechange-ai-tutorials/mobility-demand), [DOI](https://doi.org/10.5281/zenodo.11619223)).

- **From the tutorial:** the problem framing (trip ends per H3 cell from POI counts), the Shenzhen trip, POI, boundary, population and transit data, the H3 resolution-8 grid, the linear regression used as Baseline LR, the two-layer GCN architecture, and the suggestion to try gradient boosting.
- **Everything else** listed under [What this repo does](#what-this-repo-does) was added here.
- **Data:** Shenzhen bike-share trip data and AMap POI data, as distributed with the tutorial.
- The tutorial notebook is kept unchanged at `notebooks/00_original_tutorial.ipynb` (MIT license, see `LICENSE`).

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
