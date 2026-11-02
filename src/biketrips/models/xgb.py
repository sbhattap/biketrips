import pandas as pd
from xgboost import XGBRegressor

from biketrips.models import Model


class XGB(Model):
    """Poisson-objective boosted trees. Missing population is left as NaN: XGBoost learns a default
    split direction for it on the training rows, so no imputation is needed."""

    def fit(self, X, y):
        self.model = XGBRegressor(objective="count:poisson", tree_method="hist", random_state=self.seed,
                                  n_jobs=8, **self.params).fit(X[self.features], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])

    def importance(self):
        gain = self.model.get_booster().get_score(importance_type="total_gain")
        return pd.Series(gain).reindex(self.features, fill_value=0).sort_values(ascending=False)
