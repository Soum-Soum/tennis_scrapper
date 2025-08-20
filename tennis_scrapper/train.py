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

import typer
from xgboost import XGBClassifier

from plot import plot_feature_importances, save_all_plots


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
    split_date: datetime,
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


# def build_catboost(
#     random_state: int = 42,
#     **kwargs,
# ) -> CatBoostClassifier:
#     """Instantiate a CatBoostClassifier with sensible defaults; kwargs override them."""
#     defaults = dict(
#         iterations=2000,
#         depth=6,
#         learning_rate=0.05,
#         random_state=random_state,
#         verbose=100,
#     )
#     defaults.update(kwargs)
#     return CatBoostClassifier(**defaults)


def save_dfs_for_cache(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scaler: StandardScaler,
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    save_dir: Path,
):
    X_train.to_parquet(save_dir / "X_train.parquet")
    X_val.to_parquet(save_dir / "X_val.parquet")
    pd.DataFrame(y_train).to_parquet(save_dir / "y_train.parquet")
    pd.DataFrame(y_val).to_parquet(save_dir / "y_val.parquet")

    joblib.dump(scaler, save_dir / "scaler.pkl")
    X_train_scaled.to_parquet(save_dir / "X_train_scaled.parquet")
    X_val_scaled.to_parquet(save_dir / "X_val_scaled.parquet")


app = typer.Typer()


@app.command()
def train_model(
    base_dir: Path = typer.Option(
        default="output", help="Path to save the trained model"
    ),
    split_date: datetime = typer.Option(
        default=datetime.strptime("2025-01-01", "%Y-%m-%d"),
        help="Date to split the training and validation sets",
    ),
    use_cache: bool = typer.Option(default=False, help="Whether to use cached data"),
    random_state: int = typer.Option(
        default=42, help="Random state for reproducibility"
    ),
):

    data_save_path = base_dir / "data"
    if use_cache:
        X_train_scaled = pd.read_parquet(data_save_path / "X_train_scaled.parquet")
        X_val_scaled = pd.read_parquet(data_save_path / "X_val_scaled.parquet")
        y_train = pd.read_parquet(data_save_path / "y_train.parquet")["result"]
        y_val = pd.read_parquet(data_save_path / "y_val.parquet")["result"]

    else:

        with open("resources/cols_data.json") as f:
            logger.info("Loading columns data from JSON")
            cols_data = ColsData.model_validate(json.load(f))

        chunks = list((base_dir / "chunks/").glob("*.parquet"))
        df = pd.concat(list(map(pd.read_parquet, chunks))).copy()
        logger.info(f"Loaded {len(df)} records from {len(chunks)} chunks")
        df["result"] = 0

        df_flipped = randomly_flip_players(df, target_col=cols_data.target)

        df_flipped = compute_diff_columns(df_flipped)

        df_flipped = validate_cols(df_flipped, cols_data)

        df_flipped = df_flipped.drop(columns=cols_data.categorical)
        df_flipped = df_flipped.drop(columns=cols_data.other)

        X_train, X_val, y_train, y_val = time_split_shuffle(
            df_flipped,
            split_date=split_date,
            target_col=cols_data.target,
            date_col=cols_data.date_column,
            random_state=random_state,
        )

        X_train_scaled, X_val_scaled, scaler = normalize_numerical(
            X_train, X_val, cols_data.numerical
        )

        save_dfs_for_cache(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            scaler=scaler,
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            save_dir=data_save_path,
        )

    spw = compute_scale_pos_weight(y_train)
    xgb_kwargs = {"n_estimators": 1200, "max_depth": 8, "min_child_weight": 15}
    model = build_xgb(scale_pos_weight=spw, random_state=random_state, **xgb_kwargs)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[((X_train_scaled, y_train)), (X_val_scaled, y_val)],
        verbose=100,
    )

    model_save_dir = base_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_dir / "classifier.json")
    model_loaded = XGBClassifier()
    model_loaded.load_model(model_save_dir / "classifier.json")

    plot_save_dir = base_dir / "plots"
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    save_all_plots(
        y_pred_list=[model_loaded.predict_proba(X_val_scaled)[:, 1]],
        y_true_list=[y_val],
        save_dir=plot_save_dir,
    )

    fi = model_loaded.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": X_train_scaled.columns, "importance": fi}
    ).sort_values(by="importance", ascending=False)
    plot_feature_importances(
        fi_df["importance"],
        feature_names=fi_df["feature"],
        top_k=20,
        save_path=plot_save_dir / "feature_importances.png",
    )

    # mean_imp = fi_df["importance"].mean()
    # logger.info(f"Mean feature importance: {mean_imp}")
    # features_to_keep = fi_df[fi_df["importance"] > mean_imp]["feature"]
    # logger.info(
    #     f"Removing {len(X_train_scaled.columns) - len(features_to_keep)} features"
    # )
    # X_train_filtered = X_train_scaled[features_to_keep]
    # X_val_filtered = X_val_scaled[features_to_keep]

    # xgb_kwargs = {"n_estimators": 1200, "max_depth": 12, "min_child_weight": 15}
    # model_2 = build_xgb(scale_pos_weight=spw, random_state=random_state, **xgb_kwargs)
    # model_2.fit(
    #     X_train_filtered,
    #     y_train,
    #     eval_set=[((X_train_filtered, y_train)), (X_val_filtered, y_val)],
    #     verbose=100,
    # )

    # plot_save_dir_2 = base_dir / "plots_filtered"
    # plot_save_dir_2.mkdir(parents=True, exist_ok=True)
    # save_all_plots(
    #     y_pred_list=[model_2.predict_proba(X_val_filtered)[:, 1]],
    #     y_true_list=[y_val],
    #     save_dir=plot_save_dir_2,
    # )


if __name__ == "__main__":
    app()
