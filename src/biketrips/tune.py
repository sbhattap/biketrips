"""Hyperparameter search on inner folds. Candidates are ranked by mean Poisson deviance."""
import numpy as np
import optuna
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


def optuna_search(name, space, n_trials, train, folds, ctx):
    """TPE over `space` ({param: [low, high]}); int bounds give int params, rates are log-scaled."""
    cvs = {}

    def objective(trial):
        params = {k: trial.suggest_int(k, lo, hi) if isinstance(lo, int) and isinstance(hi, int)
                  else trial.suggest_float(k, lo, hi, log=k in ("learning_rate", "reg_lambda", "l2_regularization"))
                  for k, (lo, hi) in space.items()}
        cvs[trial.number] = cross_validate(name, params, train, folds, ctx)
        return np.mean([m[SELECT] for m in cvs[trial.number]])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=ctx["seed"]))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, cvs[study.best_trial.number]


def out_of_fold(name, params, cells, folds, ctx):
    """A prediction for every cell from a model that never saw that cell's fold, for city-wide maps."""
    pred = np.zeros(len(cells))
    for tr, va in folds:
        model = make(name, ctx["features"], params, ctx["seed"], ctx["cells"])
        pred[va] = model.fit(cells.iloc[tr], cells["trips"].iloc[tr]).predict(cells.iloc[va])
    return pred


def search(name, cfg, train, folds, ctx):
    """Returns (best params, per-fold metrics of the best params)."""
    mcfg = cfg["models"][name]
    if "space" in mcfg:
        return optuna_search(name, mcfg["space"], mcfg["n_trials"], train, folds, ctx)
    return grid_search(name, mcfg, train, folds, ctx)
