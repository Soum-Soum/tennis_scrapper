from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PLAYER_1_PREFIX = "player_1"
PLAYER_2_PREFIX = "player_2"

class ColsData(BaseModel):
    numerical: list[str]
    categorical: list[str]
    other: list[str]
    target: str
    date_column: str
    fill_na_with_zero: list[str] 

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

def get_players_cols(df: pd.DataFrame):
    p1_cols = sorted([col for col in df.columns if PLAYER_1_PREFIX in col])
    p2_cols = sorted([col for col in df.columns if PLAYER_2_PREFIX in col])
    assert len(p1_cols) == len(p2_cols), f"{PLAYER_1_PREFIX} and {PLAYER_2_PREFIX} column counts do not match."
    return p1_cols, p2_cols

def randomly_flip_players(
    df: pd.DataFrame, target_col: Optional[str] = None, flip_prob: float = 0.5
) -> pd.DataFrame:
    logger.info("Randomly flipping match data...")
    df = df.copy()

    # Create a mask for rows to flip
    flip_mask = np.random.rand(len(df)) < flip_prob

    # Identify player_1 and player_2 columns
    p1_cols, p2_cols = get_players_cols(df)

    if len(p1_cols) != len(p2_cols):
        raise ValueError(f"{PLAYER_1_PREFIX} and {PLAYER_2_PREFIX} column counts do not match.")

    # Swap player_1_* and player_2_* for selected rows
    df.loc[flip_mask, p1_cols + p2_cols] = df.loc[flip_mask, p2_cols + p1_cols].values

    # Flip the target: if player_1 won, after flipping it's player_2 who wins
    if target_col is not None:
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


    scaler = StandardScaler()

    cols_to_scale = numerical_cols
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train_scaled, X_val_scaled, scaler


def compute_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing difference columns")
    player_1_cols, player_2_cols = get_players_cols(df)
    player_1_cols = list(filter(lambda c: df[c].dtype in [float, int, np.float64, np.int64], player_1_cols))
    player_2_cols = list(filter(lambda c: df[c].dtype in [float, int, np.float64, np.int64], player_2_cols))
    assert len(player_1_cols) == len(player_2_cols), "Player 1 and Player 2 numerical columns do not match."
        
    left = df[player_1_cols].copy()
    right = df[player_2_cols].copy()
    right.columns = left.columns

    diff_df = (left - right)
    diff_df.columns = [c.replace(PLAYER_1_PREFIX, "diff") for c in left.columns]
    df = pd.concat([df, diff_df], axis=1)

    return df

def fill_nas_odds(df: pd.DataFrame) -> pd.DataFrame:
    elo_based_proba_player_1 = 1 / (1 + 10 ** ((df["player_2_elo"] - df["player_1_elo"]) / 400))
    elo_based_proba_player_2 = 1 - elo_based_proba_player_1
    elo_based_odd_player_1 = 1 / elo_based_proba_player_1
    elo_based_odd_player_2 = 1 / elo_based_proba_player_2

    df["player_1_odds"].fillna(elo_based_odd_player_1, inplace=True)
    df["player_2_odds"].fillna(elo_based_odd_player_2, inplace=True)

    return df

def fill_nas(df: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    logger.info("Filling NaN values...")
    for col in cols_data.fill_na_with_zero:
        df[col] = df[col].fillna(0)

    df = fill_nas_odds(df)


    subset = set(df.columns).intersection(set(cols_data.numerical))
    df_wo_nas = df.dropna(subset=subset)
    assert df_wo_nas.shape[0] > 0.95 * df.shape[0], "More than 5% of the data is missing."

    return df_wo_nas

def compute_pca_features(X_train: pd.DataFrame, X_val: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    logger.info("Computing PCA features...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train[cols_data.numerical])
    X_val_pca = pca.transform(X_val[cols_data.numerical])
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f"pca_{i+1}" for i in range(X_train_pca.shape[1])])
    X_val_pca_df = pd.DataFrame(X_val_pca, columns=[f"pca_{i+1}" for i in range(X_val_pca.shape[1])])
    return X_train_pca_df, X_val_pca_df

def preprocess_commons(X_df: pd.DataFrame, cols_data: ColsData) -> pd.DataFrame:
    X_df = fill_nas(X_df, cols_data)
    X_df = compute_diff_columns(X_df)
    X_df = cols_data.validate_cols(X_df)
    # X_df = X_df.drop(columns=cols_data.categorical)
    # X_df = X_df.drop(columns=cols_data.other)
    return X_df
    

def preprocess_dataframe_train(X_df: pd.DataFrame, cols_data: ColsData, split_date:date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, StandardScaler]:
    X_df["result"] = 0
    X_df_flipped = randomly_flip_players(X_df, target_col=cols_data.target)
    X_df_preprocessed = preprocess_commons(X_df_flipped, cols_data)
    X_train, X_val, y_train, y_val = time_split_shuffle(
        X_df_preprocessed,
        split_date=split_date,
        target_col=cols_data.target,
        date_col=cols_data.date_column,
        random_state=RANDOM_STATE,
    )
    X_train_scaled, X_val_scaled, scaler = normalize_numerical(
        X_train, X_val, cols_data.numerical
    )
    
    # X_train_pca, X_val_pca = compute_pca_features(X_train_scaled, X_val_scaled, cols_data)
    # X_train_scaled = pd.concat([X_train_scaled, X_train_pca], axis=1)
    # X_val_scaled = pd.concat([X_val_scaled, X_val_pca], axis=1)

    return X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, scaler

def preprocess_dataframe_predict(X_df: pd.DataFrame, cols_data: ColsData, scaler: StandardScaler) -> pd.DataFrame:
    # X_df_flipped = randomly_flip_players(X_df)
    X_df_preprocessed = preprocess_commons(X_df, cols_data)
    X_df_preprocessed[cols_data.numerical] = scaler.transform(X_df_preprocessed[cols_data.numerical])
    return X_df_preprocessed

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


