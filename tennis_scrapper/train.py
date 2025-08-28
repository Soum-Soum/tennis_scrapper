import json
from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime

import typer

from plot import plot_feature_importances, save_all_plots
from ml.models.xgb import XgbClassifierWrapper
from ml.preprocess_data import ColsData, preprocess_dataframe_train, save_dfs_for_cache


app = typer.Typer()


@app.command()
def train_model(
    base_dir: Path = typer.Option(
        default="output", help="Path to save the base dir"
    ),
    split_date: datetime = typer.Option(
        default=datetime.strptime("2025-01-01", "%Y-%m-%d"),
        help="Date to split the training and validation sets",
    ),
    use_cache: bool = typer.Option(default=False, help="Whether to use cached data"),
    add_pca: bool = typer.Option(default=False, help="Whether to add PCA features"),
    min_history_size: int = typer.Option(default=10, help="Minimum number of matches to store in history"),
):

    with open("resources/cols_data.json") as f:
        logger.info("Loading columns data from JSON")
        cols_data = ColsData.model_validate(json.load(f))
        
    data_save_path = base_dir / "data"
    if use_cache:
        X_train_scaled = pd.read_parquet(data_save_path / "X_train_scaled.parquet")
        X_val_scaled = pd.read_parquet(data_save_path / "X_val_scaled.parquet")
        y_train = pd.read_parquet(data_save_path / "y_train.parquet")["result"]
        y_val = pd.read_parquet(data_save_path / "y_val.parquet")["result"]

    else:

        chunks = list((base_dir / "chunks/").glob("*.parquet"))
        X_df = pd.concat(list(map(pd.read_parquet, chunks))).copy()
        logger.info(f"Loaded {len(X_df)} records from {len(chunks)} chunks")
        X_df["result"] = 0

        X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, scaler = preprocess_dataframe_train(X_df, cols_data, split_date, min_history_size)

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

    xgb_kwargs = {"n_estimators": 600, "max_depth": 6, "min_child_weight": 15}
    model_wrapper = XgbClassifierWrapper.from_params(xgb_kwargs, y_train)
    # model_wrapper = LogisticRegressionWrapper.from_params({})

    model_wrapper.train(X_train_scaled, y_train, X_val_scaled, y_val, cols_data)

    model_wrapper.save_model(save_path=base_dir / "model" / "classifier.json")
    predictions_df = model_wrapper.predict(X_val_scaled)


    plot_save_dir = base_dir / "plots"
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    save_all_plots(
        y_pred_list=[predictions_df["predicted_proba"]],
        y_true_list=[y_val],
        save_dir=plot_save_dir,
    )

    fi_df = model_wrapper.feature_importance()
    plot_feature_importances(
        fi_df["importance"],
        feature_names=fi_df["feature"],
        top_k=20,
        save_path=plot_save_dir / "feature_importances.png",
    )


if __name__ == "__main__":
    app()
