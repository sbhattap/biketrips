from sklearn.linear_model import LinearRegression

from biketrips.models import Model

# The three POI categories of the original baseline setup.
BASELINE_FEATURES = ["poi_SportsRecreation", "poi_TransportationService", "poi_MedicalService"]


class BaselineLR(Model):
    """Baseline LR: plain least squares on raw counts of three POI types,
    trained only on cells with trips. Not tuned; it is scored on the same test cells as every model."""

    def fit(self, X, y):
        keep = (y > 0).to_numpy() if self.params["nonzero_only"] else slice(None)
        self.model = LinearRegression().fit(X[BASELINE_FEATURES][keep], y[keep])
        return self

    def predict(self, X):
        return self.model.predict(X[BASELINE_FEATURES])
