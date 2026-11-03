"""Tune each model on inner folds, score once on the held-out test set, for both split types."""
import argparse

import matplotlib
import pandas as pd
import yaml

from biketrips import config, evaluate, features, plots, splits, tune
from biketrips.models import make

MODELS = ["baseline_lr", "linear", "knn", "poisson", "xgb", "hgb", "hurdle", "gcn"]


def run(name, cfg, cells):
    res, cv = cfg["data"]["results_dir"], cfg["cv"]
    ctx = {"features": features.feature_columns(cells), "seed": cfg["seed"], "cells": cells,
           "top_frac": cv["top_frac"]}
    best, preds, ts = {}, [], evaluate.now()
    for split in ("random", "spatial"):
        test = splits.holdout(cells, split, cv["test_frac"], cfg["seed"])
        train, held = cells[~test].reset_index(drop=True), cells[test].reset_index(drop=True)
        folds = splits.inner_folds(train, split, cv["n_splits"], cfg["seed"])
        params, cv_metrics = tune.search(name, cfg, train, folds, ctx)
        for i, m in enumerate(cv_metrics, 1):
            evaluate.append_metrics(res / "metrics.csv", name, split, f"cv{i}", m, ts)
        model = make(name, ctx["features"], params, cfg["seed"], cells).fit(train, train["trips"])
        pred = model.predict(held)
        test_metrics = evaluate.score(held, pred, cv["top_frac"])
        evaluate.append_metrics(res / "metrics.csv", name, split, "test", test_metrics, ts)
        preds.append(held[["h3", "trips"]].assign(model=name, split_type=split, pred=pred))
        best[split] = {k: v for k, v in params.items() if k != "cells"}
        print(f"{name:8s} {split:8s} test cells {len(held)} best {best[split]}")
    return best, pd.concat(preds), model


def update_outputs(res, name, best, preds):
    path = res / "best_params.yaml"
    allp = yaml.safe_load(path.read_text()) if path.exists() else {}
    allp[name] = best
    path.write_text(yaml.safe_dump(allp, sort_keys=False))
    ppath = res / "predictions.csv"
    old = pd.read_csv(ppath) if ppath.exists() else pd.DataFrame()
    pd.concat([old[old.get("model") != name] if len(old) else old, preds]).to_csv(ppath, index=False)


def save_importance(res, imp):
    matplotlib.use("Agg")
    imp.rename("total_gain").to_csv(res / "xgb_importance.csv", index_label="feature")
    ax = plots.importance(imp, title="XGBoost feature importance, spatial split (total gain)")
    ax.figure.tight_layout()
    ax.figure.savefig(res / "figures" / "xgb_importance.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS + ["all"], default="all")
    names = MODELS if (m := ap.parse_args().model) == "all" else [m]
    cfg = config.load_config()
    cells = pd.read_parquet(cfg["data"]["processed_dir"] / "cells.parquet")
    for name in names:
        best, preds, model = run(name, cfg, cells)
        update_outputs(cfg["data"]["results_dir"], name, best, preds)
        if name == "xgb":  # importance from the spatial-split model
            save_importance(cfg["data"]["results_dir"], model.importance())
    m = evaluate.latest(cfg["data"]["results_dir"] / "metrics.csv")
    m = m[(m["fold"] == "test") & m["metric"].isin(["mae", "spearman", "top10_hit", "r2", "r2_nontrivial"])]
    print(m.pivot_table(index=["model", "split_type"], columns="metric", values="value").round(3))


if __name__ == "__main__":
    main()
