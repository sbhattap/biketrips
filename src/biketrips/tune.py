"""Hyperparameter search on inner folds. Candidates are ranked by mean Poisson deviance."""
import numpy as np
from sklearn.model_selection import ParameterGrid

from biketrips.evaluate import metrics
from biketrips.models import make

SELECT = "poisson_deviance"


def cross_validate(name, params, train, folds, ctx):
    out = []
    for tr, va in folds:
        model = make(name, ctx["features"], params, ctx["seed"], ctx["cells"])
        model.fit(train.iloc[tr], train["trips"].iloc[tr])
        out.append(metrics(train["trips"].iloc[va], model.predict(train.iloc[va]), ctx["top_frac"]))
    return out


def grid_search(name, grid, train, folds, ctx):
    grid = {k: v if isinstance(v, list) else [v] for k, v in grid.items()}
    best = None
    for params in ParameterGrid(grid):
        cv = cross_validate(name, params, train, folds, ctx)
        loss = np.mean([m[SELECT] for m in cv])
        if best is None or loss < best[0]:
            best = (loss, params, cv)
    return best[1], best[2]


def search(name, cfg, train, folds, ctx):
    """Returns (best params, per-fold metrics of the best params)."""
    return grid_search(name, cfg["models"][name], train, folds, ctx)
