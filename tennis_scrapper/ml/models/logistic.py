from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from .base import ModelWrapper

class LogisticRegressionWrapper(ModelWrapper):

    def __init__(self, model: LogisticRegression):
        super().__init__(model)

    def feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        # For logistic regression, use absolute value of coefficients as importance
        coefs = self.model.coef_.ravel() if hasattr(self.model, "coef_") else np.zeros(len(X_train.columns))
        feat_importance_df = (
            pd.DataFrame({"feature": X_train.columns, "importance": np.abs(coefs)})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
        return feat_importance_df

    def save_model(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        probas = self.model.predict_proba(X)[:, 1]
        predicted_class = self.model.predict(X)
        return pd.DataFrame({
            "predicted_class": predicted_class,
            "predicted_proba": probas,
        })

    @classmethod
    def from_params(cls, params: dict, y_train=None) -> "LogisticRegressionWrapper":
        # compute class_weight using scale_pos_weight if y_train provided
        if "class_weight" not in params and y_train is not None:
            pos = np.sum(y_train == 1)
            neg = np.sum(y_train == 0)
            if pos > 0:
                # sklearn expects dict or 'balanced'; we convert to dict
                weight = neg / pos
                params["class_weight"] = {0: 1.0, 1: weight}
        model = LogisticRegression(**params)
        return cls(model)

    @classmethod
    def from_model(cls, model_path: Path) -> "LogisticRegressionWrapper":
        model = joblib.load(model_path)
        return cls(model)
