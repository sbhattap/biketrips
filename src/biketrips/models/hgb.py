from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from biketrips.models import Model


class HGB(Model):
    """sklearn's LightGBM-style boosted trees with a Poisson loss. A second tree library checks that
    XGBoost's lead comes from boosting itself, not from one library. NaN population gets a learned
    split direction, as in XGBoost.

    One thread: on 3k rows OpenMP start-up and spinning cost more than the work (8 threads ran 10x
    slower than 1 on a shared 24-core machine)."""

    @threadpool_limits.wrap(limits=1)
    def fit(self, X, y):
        self.model = HistGradientBoostingRegressor(loss="poisson", early_stopping=False, random_state=self.seed,
                                                   **self.params).fit(X[self.features], y)
        return self

    @threadpool_limits.wrap(limits=1)
    def predict(self, X):
        return self.model.predict(X[self.features])
