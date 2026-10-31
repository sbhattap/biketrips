import numpy as np
from sklearn.neighbors import BallTree

from biketrips.models import Model


class SpatialKNN(Model):
    """Mean trips of the k nearest training cells. Uses no urban features at all, so it measures how
    much of the score is just spatial smoothness."""

    def fit(self, X, y):
        self.tree = BallTree(np.radians(X[["lat", "lng"]].to_numpy()), metric="haversine")
        self.y = np.asarray(y, float)
        return self

    def predict(self, X):
        _, idx = self.tree.query(np.radians(X[["lat", "lng"]].to_numpy()), k=self.params["k"])
        return self.y[idx].mean(axis=1)
