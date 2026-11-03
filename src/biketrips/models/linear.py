import numpy as np
from sklearn.linear_model import Ridge

from biketrips.models import Model, tabular


class Linear(Model):
    """Ridge on log1p(trips) with log1p features, back-transformed for scoring. A log-scale,
    all-features variant of Baseline LR (models/baseline_lr.py)."""

    def fit(self, X, y):
        self.pipe = tabular(Ridge(alpha=self.params["alpha"])).fit(X[self.features], np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.pipe.predict(X[self.features]))
