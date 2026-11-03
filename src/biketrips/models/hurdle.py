import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from biketrips.models import Model


class Hurdle(Model):
    """Prediction = P(any trips) x expected trips given some. Zero cells inside busy areas (likely
    supply gaps) are handled by the classifier instead of dragging the count model down.

    Both parts are the same boosted trees as `hgb` with shared hyperparameters, so hurdle vs hgb
    isolates the two-part structure. The count part is a plain Poisson fit on positive cells, not
    zero-truncated: the median positive cell has about 600 trips, so truncation only matters for
    the smallest cells. Single-threaded for the same reason as `hgb`."""

    @threadpool_limits.wrap(limits=1)
    def fit(self, X, y):
        X, y = X[self.features], np.asarray(y)
        kw = dict(early_stopping=False, random_state=self.seed, **self.params)
        self.clf = HistGradientBoostingClassifier(**kw).fit(X, y > 0)
        self.reg = HistGradientBoostingRegressor(loss="poisson", **kw).fit(X[y > 0], y[y > 0])
        return self

    @threadpool_limits.wrap(limits=1)
    def predict(self, X):
        X = X[self.features]
        return self.clf.predict_proba(X)[:, 1] * self.reg.predict(X)
