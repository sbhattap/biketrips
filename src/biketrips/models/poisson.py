from sklearn.linear_model import PoissonRegressor

from biketrips.models import Model, tabular


class PoissonGLM(Model):
    """Log-link GLM on standardised log1p features. Fits the mean well enough to rank cells, but
    with variance/mean near 20,000 its implied uncertainty is far too narrow."""

    def fit(self, X, y):
        glm = PoissonRegressor(alpha=self.params["alpha"], solver="newton-cholesky", max_iter=1000)
        self.pipe = tabular(glm).fit(X[self.features], y)
        return self

    def predict(self, X):
        return self.pipe.predict(X[self.features])
