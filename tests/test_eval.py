import numpy as np
import pandas as pd
import pytest

from biketrips import evaluate, splits
from biketrips.models import make


@pytest.fixture
def cells():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({"h3": [f"c{i}" for i in range(n)], "group": rng.integers(0, 40, n),
                         "lat": rng.uniform(22.4, 22.8, n), "lng": rng.uniform(113.8, 114.4, n),
                         "f1": rng.poisson(5, n), "f2": rng.poisson(2, n),
                         "trips": rng.poisson(50, n), "trivial": (rng.random(n) < 0.1).astype(int)})


def test_spatial_holdout_and_folds_keep_groups_apart(cells):
    test = splits.holdout(cells, "spatial", 0.2, 42)
    assert not set(cells.group[test]) & set(cells.group[~test])
    for tr, va in splits.inner_folds(cells, "spatial", 5, 42):
        assert not set(cells.group.iloc[tr]) & set(cells.group.iloc[va])
    assert splits.holdout(cells, "random", 0.2, 42).sum() == 80


def test_metrics_on_perfect_and_negative_predictions():
    y = np.array([0, 1, 5, 10, 100.0])
    m = evaluate.metrics(y, y, top_frac=0.2)
    assert m["mae"] == 0 and m["poisson_deviance"] == pytest.approx(0, abs=1e-5) and m["top10_hit"] == 1
    m = evaluate.metrics(y, y - 2, top_frac=0.2)
    assert m["n_clipped"] == 2 and np.isfinite(m["poisson_deviance"])


def test_top_hit_rate_counts_overlap():
    y = np.arange(10.0)
    assert evaluate.top_hit_rate(y, -y, 0.2) == 0
    assert evaluate.top_hit_rate(y, y, 0.2) == 1


def test_baseline_lr_uses_its_three_pois_and_nonzero_cells(cells):
    from biketrips.models.baseline_lr import BASELINE_FEATURES
    df = cells.assign(**{f: cells["f1"] for f in BASELINE_FEATURES}, trips=cells["trips"] * (cells.index % 2))
    m = make("baseline_lr", [], {"nonzero_only": True}).fit(df, df["trips"])
    full = make("baseline_lr", [], {"nonzero_only": False}).fit(df, df["trips"])
    assert m.model.intercept_ > full.model.intercept_  # zeros pull the all-cells fit down


@pytest.mark.parametrize("name,params", [("linear", {"alpha": 1.0}), ("knn", {"k": 5}),
                                         ("poisson", {"alpha": 0.01}), ("xgb", {"n_estimators": 20}),
                                         ("hgb", {"max_iter": 20}), ("hurdle", {"max_iter": 20}),
                                         ("gcn", {"hidden": 8, "k": 4, "lr": 0.01, "dropout": 0.0,
                                                  "max_epochs": 20, "patience": 5, "val_frac": 0.2})])
def test_models_share_interface(cells, name, params):
    pred = make(name, ["f1", "f2"], params, cells=cells).fit(cells.iloc[:300], cells.trips[:300]).predict(cells.iloc[300:])
    assert pred.shape == (100,) and np.isfinite(pred).all()


def test_metrics_csv_roundtrip(tmp_path):
    path = tmp_path / "m.csv"
    evaluate.append_metrics(path, "knn", "spatial", "test", {"mae": 1.0})
    m = pd.read_csv(path)
    assert list(m.columns) == ["model", "split_type", "fold", "metric", "value", "timestamp", "git_sha"]
