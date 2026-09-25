"""Small research forecast-model abstraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ForecastModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...
    def predict_distribution(self, X, residuals):
        point = np.asarray(self.predict(X), dtype=float)
        residuals = np.asarray(residuals, dtype=float)
        return {"p10": point + np.nanpercentile(residuals, 10), "p50": point, "p90": point + np.nanpercentile(residuals, 90)}
    @abstractmethod
    def metadata(self): ...

class PersistenceModel(ForecastModel):
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=float)
    def metadata(self):
        return {"model_id": "PERSISTENCE_DELTA", "model_version": "1.0", "research_status": "BASELINE"}

class RFResearchModel(ForecastModel):
    def __init__(self, n_estimators=100, max_depth=5, random_state=42, n_jobs=1):
        self.config = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state, "n_jobs": n_jobs}
        self.model = None
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**self.config).fit(X, y)
        return self
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("RFResearchModel must be fitted before predict")
        return self.model.predict(X)
    def metadata(self):
        return {"model_id": "RF_STANDARD", "model_version": "frozen_research_v1", "research_status": "RESEARCH_ONLY", **self.config}
