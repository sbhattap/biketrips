"""Held-out test masks and inner CV folds, random or blocked by H3 parent cell."""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def holdout(cells, kind, test_frac, seed):
    """Boolean test mask. Spatial holds out whole res-6 parent groups, so test cells have no
    training neighbours inside their block."""
    rng = np.random.default_rng(seed)
    if kind == "spatial":
        groups = np.sort(cells["group"].unique())
        test = rng.choice(groups, round(test_frac * len(groups)), replace=False)
        return cells["group"].isin(test).to_numpy()
    mask = np.zeros(len(cells), bool)
    mask[rng.choice(len(cells), round(test_frac * len(cells)), replace=False)] = True
    return mask


def inner_folds(cells, kind, n_splits, seed):
    if kind == "spatial":
        cv = GroupKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(cells, groups=cells["group"]))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(cells))
