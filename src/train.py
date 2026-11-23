"""Train the RandomForest model, log it to MLflow, and save it to artifacts/.

Usage (from the project root):
    python -m src.train
    python -m src.train --max-depth 15 --run-name rf-depth15
"""
import time

import click
import joblib
import mlflow
from mlflow import MlflowClient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.config import load_config
from src.data import clean, load_raw, split_by_date
from src.pipeline import build_pipeline , TRUSTED_TYPES                     
from src.features import TARGET, FEATURES  

#from src.model_io import save_model, TRUSTED_TYPES 


def train(cfg: dict, run_name: str = "rf-baseline") -> dict:
    # ---- Data ----
    df = clean(load_raw())
    train_df, val_df = split_by_date(df, cutoff=cfg["split"]["cutoff"])
    n = min(cfg["split"]["train_sample"], len(train_df))
    train_df = train_df.sample(n, random_state=cfg["model"]["random_state"])

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_val, y_val = val_df.drop(columns=[TARGET]), val_df[TARGET]

    # ---- MLflow ----
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    params = {**cfg["model"], "n_jobs": -1}

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"stage": "baseline", "model_type": "random_forest"})
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(X_train), "val_rows": len(X_val),
                           "cutoff": str(cfg["split"]["cutoff"]), "features": ",".join(FEATURES)})

        t0 = time.perf_counter()
        pipeline = build_pipeline(params).fit(X_train, y_train)       
        train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_val) 
        predict_ms_per_row = (time.perf_counter() - t0) * 1000 / len(X_val)
       
          

        # Save to artifacts/ (the file the API will load)
        artifacts_dir = cfg["paths"]["artifacts"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        #model_path = artifacts_dir / "model.skops" 
        #save_model(pipeline, model_path)

        metrics = {
            "val_rmse": root_mean_squared_error(y_val, y_pred),
            "val_mae": mean_absolute_error(y_val, y_pred),
            "val_r2": r2_score(y_val, y_pred),
            "train_rmse": root_mean_squared_error(y_train, pipeline.predict(X_train)),
            "baseline_rmse": root_mean_squared_error(y_val, np.full(len(y_val), y_train.mean())),
            "train_seconds": train_seconds,
            "predict_ms_per_row": predict_ms_per_row,
            #"model_size_mb": model_path.stat().st_size / 1e6,
        }
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(pipeline, name="model", input_example=X_val.head(3),
                                 skops_trusted_types=TRUSTED_TYPES,
                                 registered_model_name="taxi-duration")

         # Size of the logged model: sum of its files, read from MLflow (no download)
        files = MlflowClient().list_logged_model_artifacts(model_info.model_id)
        model_size_mb = sum(f.file_size or 0 for f in files) / 1e6

        mlflow.log_metrics({**metrics, "model_size_mb": model_size_mb})
        print(f"Registered taxi-duration v{model_info.registered_model_version} "
              f"({model_size_mb:.1f} MB)")

   
    for k, v in metrics.items():
        print(f"  {k:20s} {v:.3f}")
    return metrics


@click.command()
@click.option("--max-depth", type=int, default=None, help="Override model.max_depth from config")
@click.option("--n-estimators", type=int, default=None, help="Override model.n_estimators from config")
@click.option("--run-name", default="rf-baseline", show_default=True)
def main(max_depth, n_estimators, run_name):
    cfg = load_config()                       # read fresh every run
    if max_depth is not None:
        cfg["model"]["max_depth"] = max_depth
    if n_estimators is not None:
        cfg["model"]["n_estimators"] = n_estimators
    train(cfg, run_name)


if __name__ == "__main__":
    main()