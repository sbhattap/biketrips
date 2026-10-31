"""Metrics and the metrics.csv writer."""
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from biketrips.config import ROOT


def top_hit_rate(y, pred, frac):
    """Share of the truly top-`frac` cells that the model also ranks in its top `frac`."""
    k = max(1, int(np.ceil(frac * len(y))))
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-pred)[:k])) / k


def poisson_deviance(y, pred):
    # Floor at 1e-6 so a zero prediction for a cell with trips is heavily penalised, not infinite.
    p = np.maximum(pred, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        ylogy = np.where(y > 0, y * np.log(y / p), 0.0)
    return float(np.mean(2 * (ylogy - (y - p))))


def metrics(y, pred, top_frac=0.1):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n_clipped = int((pred < 0).sum())
    pred = np.clip(pred, 0, None)  # counts cannot be negative; n_clipped records how often this bites
    return {"mae": mean_absolute_error(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5,
            "poisson_deviance": poisson_deviance(y, pred), "spearman": spearmanr(y, pred)[0],
            "top10_hit": top_hit_rate(y, pred, top_frac), "r2": r2_score(y, pred), "n_clipped": n_clipped}


def score(cells, pred, top_frac=0.1):
    """Metrics on all test cells and on non-trivial cells only (suffix `_nontrivial`)."""
    out = metrics(cells["trips"], pred, top_frac)
    keep = cells["trivial"].to_numpy() == 0
    out.update({f"{k}_nontrivial": v for k, v in metrics(cells["trips"][keep], pred[keep], top_frac).items()})
    return out


def git_sha():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def append_metrics(path, model, split_type, fold, values):
    rows = pd.DataFrame({"model": model, "split_type": split_type, "fold": fold, "metric": list(values),
                         "value": list(values.values()),
                         "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "git_sha": git_sha()})
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def latest(path):
    """Most recent run per (model, split_type), since metrics.csv is append-only."""
    m = pd.read_csv(path)
    last = m.groupby(["model", "split_type"])["timestamp"].transform("max")
    return m[m["timestamp"] == last]
