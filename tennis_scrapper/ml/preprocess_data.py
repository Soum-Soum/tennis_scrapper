from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

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

    def validate_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating DataFrame columns against cols_data...")
        for col in self.numerical + self.categorical:
            assert col in df.columns, f"Column {col} not found in DataFrame"

        known_cols = set(self.all_columns)
        cols_in_df = set(df.columns)
        for col in cols_in_df - known_cols:
            logger.warning(f"Column {col} in DataFrame not found in cols_data")
            df = df.drop(columns=col, axis=1)

        return df


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


def time_split_shuffle(
    df: pd.DataFrame,
    split_date: datetime,
    target_col: str,
    date_col: str,
    random_state: int,
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

def preprocess_commons(X_df: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    X_df_flipped = randomly_flip_players(X_df, target_col=cols_data.target)
    X_df_flipped = compute_diff_columns(X_df_flipped)
    X_df_flipped = cols_data.validate_cols(X_df_flipped)
    X_df_flipped = X_df_flipped.drop(columns=cols_data.categorical)
    X_df_flipped = X_df_flipped.drop(columns=cols_data.other)
    return X_df_flipped
    

def preprocess_dataframe_train(X_df: pd.DataFrame, cols_data: ColsData, split_date:date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, StandardScaler]:
    X_df["result"] = 0
    X_df_flipped = preprocess_commons(X_df, cols_data)
    X_train, X_val, y_train, y_val = time_split_shuffle(
        X_df_flipped,
        split_date=split_date,
        target_col=cols_data.target,
        date_col=cols_data.date_column,
        random_state=RANDOM_STATE,
    )
    X_train_scaled, X_val_scaled, scaler = normalize_numerical(
        X_train, X_val, cols_data.numerical
    )
    return X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, scaler

def preprocess_dataframe_predict(X_df: pd.DataFrame, cols_data: ColsData, scaler: StandardScaler) -> pd.DataFrame:
    X_df_flipped = preprocess_commons(X_df, cols_data).copy()
    X_df_scaled = scaler.transform(X_df_flipped[cols_data.numerical])
    X_df_flipped[cols_data.numerical] = X_df_scaled
    return X_df_flipped

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
    save_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(save_dir / "X_train.parquet")
    X_val.to_parquet(save_dir / "X_val.parquet")
    pd.DataFrame(y_train).to_parquet(save_dir / "y_train.parquet")
    pd.DataFrame(y_val).to_parquet(save_dir / "y_val.parquet")

    joblib.dump(scaler, save_dir / "scaler.pkl")
    X_train_scaled.to_parquet(save_dir / "X_train_scaled.parquet")
    X_val_scaled.to_parquet(save_dir / "X_val_scaled.parquet")


