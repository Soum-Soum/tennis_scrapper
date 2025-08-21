from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

from ml.models.base import compute_scale_pos_weight

def build_xgb(
    scale_pos_weight: float,
    random_state: int = 42,
    **kwargs,
) -> XGBClassifier:
    defaults = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        tree_method="hist",
        gpu_id=0,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)

class XgbClassifierWrapper:

    def __init__(self, model: XGBClassifier):
        self.model = model
   
    def feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        importances = self.model.feature_importances_
        feat_importance_df = (
            pd.DataFrame({"feature": X_train.columns, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
        return feat_importance_df

    def save_model(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(save_path)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        probas = self.model.predict_proba(X)[:, 1]
        predicted_class = self.model.predict(X)
        return pd.DataFrame({
            "predicted_class": predicted_class,
            "predicted_proba": probas
        })

    @classmethod
    def from_params(cls, xgb_kwargs: dict, y_train: pd.Series) -> "XgbClassifierWrapper":
        scale_pos_weight = compute_scale_pos_weight(y_train)
        model = build_xgb(scale_pos_weight=scale_pos_weight, **xgb_kwargs)
        return cls(model)
        
    @classmethod
    def from_model(cls, model_path: Path) -> "XgbClassifierWrapper":
        model = XGBClassifier()
        model.load_model(model_path)
        return cls(model)
