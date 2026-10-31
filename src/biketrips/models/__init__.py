"""Five models behind one interface: Model(features, seed, **params).fit(X, y).predict(X).

X is always a slice of the cell table, so models can pick feature columns, coordinates,
or (for the GCN) graph positions by H3 index.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class Model:
    def __init__(self, features, seed=42, **params):
        self.features, self.seed, self.params = features, seed, params


def tabular(estimator):
    """log1p tames heavy-tailed counts; the median imputer for population is fit on training rows only."""
    return make_pipeline(FunctionTransformer(np.log1p), SimpleImputer(strategy="median"), StandardScaler(),
                         estimator)


def make(name, features, params, seed=42, cells=None):
    from biketrips.models import knn, linear
    registry = {"linear": linear.Linear, "knn": knn.SpatialKNN}
    if name == "gcn":
        params = dict(params, cells=cells)
    return registry[name](features, seed, **params)
