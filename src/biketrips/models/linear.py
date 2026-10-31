import numpy as np
from sklearn.linear_model import Ridge

from biketrips.models import Model, tabular


class Linear(Model):
    """Tutorial baseline, fit on log1p(trips) and back-transformed for scoring."""

    def fit(self, X, y):
        self.pipe = tabular(Ridge(alpha=self.params["alpha"])).fit(X[self.features], np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.pipe.predict(X[self.features]))
