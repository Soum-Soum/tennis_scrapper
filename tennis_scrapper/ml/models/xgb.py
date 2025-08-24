from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

from ml.models.base import ModelWrapper, compute_scale_pos_weight, drop_cols
from ml.preprocess_data import ColsData

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

class XgbClassifierWrapper(ModelWrapper):

    def __init__(self, model: XGBClassifier):
        super().__init__(model)

    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: pd.DataFrame,
              y_val: pd.Series,
              cols_data: ColsData) -> None:
        
        X_train = drop_cols(X_train, cols_data)
        X_val = drop_cols(X_val, cols_data)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_names = self.model.get_booster().feature_names
        X = X.copy()[feature_names].reindex(columns=feature_names)
        probas = self.model.predict_proba(X)[:, 1]
        predicted_class = self.model.predict(X)
        return pd.DataFrame({
            "predicted_class": predicted_class,
            "predicted_proba": probas
        })
        
    def feature_importance(self) -> pd.DataFrame:
        booster = self.model.get_booster()
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": booster.feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    def save_model(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(save_path)

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
