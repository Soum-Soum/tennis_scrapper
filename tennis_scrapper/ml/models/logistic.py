from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from ml.preprocess_data import ColsData

from .base import ModelWrapper, drop_cols

def build_logreg(
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    defaults: dict[str, Any] = dict(
        penalty="l2",
        C=1.0,                   # inverse of regularization strength
        fit_intercept=True,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)

class LogisticRegressionWrapper(ModelWrapper):

    def __init__(self, model: LogisticRegression):
        super().__init__(model)
        self._features_names = None

    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: pd.DataFrame,
              y_val: pd.Series,
              cols_data: ColsData) -> None:
        
        X_train = drop_cols(X_train, cols_data)
        X_val = drop_cols(X_val, cols_data)
        self._features_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)

    def feature_importance(self) -> pd.DataFrame:
        # For logistic regression, use absolute value of coefficients as importance
        if not self.feature_names_:
            raise RuntimeError("Model not trained: feature_names_ is missing.")
        coef = self.model.coef_.ravel()  # binary classification -> shape (1, n_features)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "coefficient": coef,
                    "importance": np.abs(coef),
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    def save_model(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()[self._features_names].reindex(columns=self._features_names)
        probas = self.model.predict_proba(X)[:, 1]
        predicted_class = self.model.predict(X)
        return pd.DataFrame({
            "predicted_class": predicted_class,
            "predicted_proba": probas
        })


    @classmethod
    def from_params(
        cls, logreg_kwargs: dict
    ) -> "LogisticRegressionWrapper":
        model = build_logreg(**logreg_kwargs)
        return cls(model)

    @classmethod
    def from_model(cls, model_path: Path) -> "LogisticRegressionWrapper":
        model = joblib.load(model_path)
        return cls(model)
