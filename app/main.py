"""Run from the project root (MLflow server must be running):
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from app.schema import TripRequest, TripResponse
from src.config import load_config

state: dict = {}


def load_registry_model(cfg: dict):
    """Load the model that the alias points to, plus a readable version string."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model"]
    alias = cfg["serving"]["model_alias"]

    mv = MlflowClient().get_model_version_by_alias(name, alias)       # which version is live
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    version = f"{name} v{mv.version} ({alias}, run {mv.run_id[:8]})"
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    try:
        state["model"], state["version"] = load_registry_model(cfg)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model from MLflow at {cfg['mlflow']['tracking_uri']}. "
            f"Is the MLflow server running and does the alias exist? ({e})"
        ) from e
    yield
    state.clear()


app = FastAPI(title="NYC Taxi Trip Duration", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "model_version": state.get("version")}


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([trip.model_dump()])
    duration = float(state["model"].predict(df)[0])
    return TripResponse(duration_minutes=round(duration, 2), model_version=state["version"])