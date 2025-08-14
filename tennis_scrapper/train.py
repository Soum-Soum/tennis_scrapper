import json
from pathlib import Path
from catboost import CatBoostClassifier
import joblib
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from xgboost import XGBClassifier

from plot import save_all_plots


class ColsData(BaseModel):
    numerical: list[str]
    categorical: list[str]
    other: list[str]
    target: str
    date_column: str

    @property
    def all_columns(self) -> list[str]:
        return (
            self.numerical
            + self.categorical
            + self.other
            + [self.target, self.date_column]
        )


def randomly_flip_players(
    df: pd.DataFrame, target_col: str, flip_prob: float = 0.5
) -> pd.DataFrame:
    logger.info("Randomly flipping match data...")
    df = df.copy()

    # Create a mask for rows to flip
    flip_mask = np.random.rand(len(df)) < flip_prob

    # Identify player_1 and player_2 columns
    p1_cols = sorted([col for col in df.columns if col.startswith("player_1_")])
    p2_cols = sorted([col for col in df.columns if col.startswith("player_2_")])

    if len(p1_cols) != len(p2_cols):
        raise ValueError("player_1_ and player_2_ column counts do not match.")

    # Swap player_1_* and player_2_* for selected rows
    df.loc[flip_mask, p1_cols + p2_cols] = df.loc[flip_mask, p2_cols + p1_cols].values

    # Flip the target: if player_1 won, after flipping it's player_2 who wins
    df.loc[flip_mask, target_col] = 1 - df.loc[flip_mask, target_col]

    return df


def validate_cols(df: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    logger.info("Validating DataFrame columns against cols_data...")
    for col in cols_data.numerical + cols_data.categorical:
        assert col in df.columns, f"Column {col} not found in DataFrame"

    known_cols = set(cols_data.all_columns)
    cols_in_df = set(df.columns)
    for col in cols_in_df - known_cols:
        logger.warning(f"Column {col} in DataFrame not found in cols_data")
        df = df.drop(columns=col, axis=1)

    return df


def time_split_shuffle(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    target_col: str,
    date_col: str = "date",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Splitting DataFrame into train and validation sets...")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values(date_col)

    train_df = df_sorted[df_sorted[date_col] < split_date].copy().drop(columns=date_col)
    val_df = df_sorted[df_sorted[date_col] >= split_date].copy().drop(columns=date_col)

    def post_process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

    X_train, y_train = post_process_data(train_df)
    X_val, y_val = post_process_data(val_df)

    logger.info(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")

    return X_train, X_val, y_train, y_val


def normalize_numerical(
    X_train: pd.DataFrame, X_val: pd.DataFrame, numerical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    logger.info("Normalizing numerical columns...")

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    cols_to_scale = [col for col in numerical_cols if col in X_train.columns]

    scaler = StandardScaler()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train_scaled, X_val_scaled, scaler


def compute_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing difference columns")
    for col in df.columns:
        if "player_1" not in col:
            continue

        assert (
            col.replace("player_1", "player_2") in df.columns
        ), f"Missing corresponding player_2 column for {col}"

        if df[col].dtype not in [float, int, np.float64, np.int64]:
            continue

        col2 = col.replace("player_1", "player_2")

        df[col.replace("player_1", "diff")] = df[col] - df[col2]

    return df


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = neg/pos for class imbalance handling in XGBoost."""
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    return (neg / pos) if pos > 0 else 1.0


def get_feature_importance(model: XGBClassifier, X_train: pd.DataFrame) -> pd.DataFrame:

    importances = model.feature_importances_  # numpy array aligné sur X_train.columns

    feat_importance_df = (
        pd.DataFrame({"feature": X_train.columns, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    return feat_importance_df


def build_xgb(
    scale_pos_weight: float,
    random_state: int = 42,
    **kwargs,
) -> XGBClassifier:
    """Instantiate an XGBClassifier with sensible defaults; kwargs override them."""
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


def build_catboost(
    random_state: int = 42,
    **kwargs,
) -> CatBoostClassifier:
    """Instantiate a CatBoostClassifier with sensible defaults; kwargs override them."""
    defaults = dict(
        iterations=2000,
        depth=6,
        learning_rate=0.05,
        random_state=random_state,
        verbose=100,
    )
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


if __name__ == "__main__":
    RANDOM_STATE = 42

    data_save_path = Path("output/data")
    if (data_save_path / "X_train.parquet").exists():
        logger.info("Data already processed, loading from disk.")
        X_train_scaled = pd.read_parquet(data_save_path / "X_train_scaled.parquet")
        X_val_scaled = pd.read_parquet(data_save_path / "X_val_scaled.parquet")
        y_train = pd.read_parquet(data_save_path / "y_train.parquet")["result"]
        y_val = pd.read_parquet(data_save_path / "y_val.parquet")["result"]
    else:

        with open("cols_data.json") as f:
            cols_data = ColsData.model_validate(json.load(f))

        chunks = list(Path("output/chunks/").glob("*.parquet"))
        df = pd.concat(list(map(pd.read_parquet, chunks))).copy()
        df["result"] = 1

        df_flipped = randomly_flip_players(df, target_col=cols_data.target)

        df_flipped = compute_diff_columns(df_flipped)

        df_flipped = validate_cols(df_flipped, cols_data)

        df_flipped = df_flipped.drop(columns=cols_data.categorical)
        df_flipped = df_flipped.drop(columns=cols_data.other)

        split_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
        X_train, X_val, y_train, y_val = time_split_shuffle(
            df_flipped,
            split_date=split_date,
            target_col=cols_data.target,
            date_col=cols_data.date_column,
            random_state=RANDOM_STATE,
        )

        X_train.to_parquet(data_save_path / "X_train.parquet")
        X_val.to_parquet(data_save_path / "X_val.parquet")

        X_train_scaled, X_val_scaled, scaler = normalize_numerical(
            X_train, X_val, cols_data.numerical
        )

        joblib.dump(scaler, data_save_path / "scaler.pkl")
        X_train_scaled.to_parquet(data_save_path / "X_train_scaled.parquet")
        X_val_scaled.to_parquet(data_save_path / "X_val_scaled.parquet")
        pd.DataFrame(y_train).to_parquet(data_save_path / "y_train.parquet")
        pd.DataFrame(y_val).to_parquet(data_save_path / "y_val.parquet")

    spw = compute_scale_pos_weight(y_train)
    xgb_kwargs = {"n_estimators": 1200, "max_depth": 8, "min_child_weight": 15}
    # xgb_kwargs = {}
    model = build_xgb(scale_pos_weight=spw, random_state=RANDOM_STATE, **xgb_kwargs)
    # model = build_catboost(random_state=RANDOM_STATE, **xgb_kwargs)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[((X_train_scaled, y_train)), (X_val_scaled, y_val)],
        verbose=100,
    )
    model.save_model("classifier.json")

    model_loaded = XGBClassifier()
    model_loaded.load_model("classifier.json")

    save_all_plots(
        y_pred_list=[model_loaded.predict_proba(X_val_scaled)[:, 1]],
        y_true_list=[y_val],
        save_dir=Path("output/plots"),
    )
