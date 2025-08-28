from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Type, Any

from tennis_scrapper.ml.preprocess_data import ColsData


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = neg/pos for class imbalance handling in XGBoost."""
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    return (neg / pos) if pos > 0 else 1.0


def drop_cols(X: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    X = X.copy()
    X = X.drop(cols_data.categorical, axis=1, errors="ignore")
    X = X.drop(cols_data.other, axis=1, errors="ignore")
    X = X.drop(cols_data.target, axis=1, errors="ignore")
    return X


class ModelWrapper(ABC):
    """Abstract interface for model wrapper classes.

    Concrete wrappers (XGB, LogisticRegression, ...) should implement this API so
    the rest of the codebase can use them interchangeably.
    """

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, save_path: Path) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with at least 'predicted_class' and 'predicted_proba'."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_params(
        cls: Type["ModelWrapper"], params: dict, y_train=None
    ) -> "ModelWrapper":
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_model(cls: Type["ModelWrapper"], model_path: Path) -> "ModelWrapper":
        raise NotImplementedError()
