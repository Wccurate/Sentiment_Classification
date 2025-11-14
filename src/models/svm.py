from __future__ import annotations

import joblib
from dataclasses import dataclass
from typing import Any, List

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.datasets import Example


@dataclass
class SVMConfig:
    C: float = 1.0
    class_weight: Any = None


class SVMClassifier:
    def __init__(self, cfg: SVMConfig):
        self.cfg = cfg
        # Standardize if dense; if sparse input stays sparse, scaler is skipped by SVC.
        self.model = Pipeline([
            ("clf", LinearSVC(C=cfg.C, class_weight=cfg.class_weight))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

