"""API tests. They never contact MLflow: the model loader is replaced with a fake
(or a small real pipeline), so the tests run without a server or a trained model."""
from types import SimpleNamespace

import mlflow.sklearn  # real module, so it can be patched (mlflow loads it lazily)
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src.pipeline import build_pipeline

VERSION = "taxi-duration v1 (champion, run test1234)"

PAYLOAD = {
    "tpep_pickup_datetime": "2025-01-15T08:30:00",
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 2.4,
}
REQUEST_COLUMNS = ["tpep_pickup_datetime", "PULocationID", "DOLocationID", "trip_distance"]


class FakeModel:
    """Records what the API passes to predict() and returns a fixed value."""

    def __init__(self, value=12.345):
        self.value = value
        self.last_df = None

    def predict(self, df):
        self.last_df = df
        return np.array([self.value])


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def client(monkeypatch, fake_model):
    monkeypatch.setattr(main, "load_registry_model", lambda cfg: (fake_model, VERSION))
    with TestClient(main.app) as c:          # 'with' runs the lifespan (startup/shutdown)
        yield c


# ---------- /health ----------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": VERSION}


# ---------- /predict: valid requests ----------
def test_predict_ok(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    assert set(r.json()) == {"duration_minutes", "model_version"}


def test_predict_rounds_to_two_decimals(client):
    assert client.post("/predict", json=PAYLOAD).json()["duration_minutes"] == 12.35


def test_predict_returns_model_version(client):
    assert client.post("/predict", json=PAYLOAD).json()["model_version"] == VERSION


def test_predict_passes_one_row_with_request_columns(client, fake_model):
    client.post("/predict", json=PAYLOAD)
    df = fake_model.last_df
    assert isinstance(df, pd.DataFrame) and len(df) == 1
    assert list(df.columns) == REQUEST_COLUMNS


def test_predict_parses_datetime(client, fake_model):
    client.post("/predict", json=PAYLOAD)
    ts = pd.Timestamp(fake_model.last_df.loc[0, "tpep_pickup_datetime"])
    assert (ts.hour, ts.dayofweek) == (8, 2)      # Wednesday 08:30


def test_predict_accepts_numbers_as_strings(client):
    r = client.post("/predict", json={**PAYLOAD, "trip_distance": "2.4", "PULocationID": "161"})
    assert r.status_code == 200


@pytest.mark.parametrize("field, value", [
    ("PULocationID", 1), ("PULocationID", 265),
    ("DOLocationID", 1), ("DOLocationID", 265),
    ("trip_distance", 0.01), ("trip_distance", 150),   # no upper distance limit
])
def test_predict_accepts_boundary_values(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 200


# ---------- /predict: invalid requests -> 422 ----------
@pytest.mark.parametrize("field, value", [
    ("trip_distance", 0),
    ("trip_distance", -1),
    ("trip_distance", "far"),
    ("PULocationID", 0),
    ("PULocationID", 266),
    ("DOLocationID", 0),
    ("DOLocationID", 266),
    ("PULocationID", 1.5),
    ("tpep_pickup_datetime", "not-a-date"),
    ("tpep_pickup_datetime", None),
])
def test_predict_rejects_invalid_values(client, field, value):
    r = client.post("/predict", json={**PAYLOAD, field: value})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == field     # error points at the bad field


@pytest.mark.parametrize("missing", REQUEST_COLUMNS)
def test_predict_rejects_missing_field(client, missing):
    body = {k: v for k, v in PAYLOAD.items() if k != missing}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_rejects_empty_body(client):
    assert client.post("/predict").status_code == 422


def test_predict_get_not_allowed(client):
    assert client.get("/predict").status_code == 405


# ---------- model state / startup ----------
def test_predict_503_when_model_not_loaded(client):
    main.state.pop("model")
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 503
    assert client.get("/health").json()["model_loaded"] is False


def test_startup_fails_with_clear_message_when_mlflow_unavailable(monkeypatch):
    def broken(cfg):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(main, "load_registry_model", broken)
    with pytest.raises(RuntimeError, match="Could not load model from MLflow"):
        with TestClient(main.app):
            pass


def test_state_cleared_on_shutdown(monkeypatch, fake_model):
    monkeypatch.setattr(main, "load_registry_model", lambda cfg: (fake_model, VERSION))
    with TestClient(main.app):
        assert "model" in main.state
    assert main.state == {}


# ---------- load_registry_model() ----------
def test_load_registry_model_uses_config(monkeypatch):
    calls = {}
    cfg = {
        "mlflow": {"tracking_uri": "http://example:5000", "registered_model": "taxi-duration"},
        "serving": {"model_alias": "champion"},
    }

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            calls["alias"] = (name, alias)
            return SimpleNamespace(version="7", run_id="abcdef1234567890")

    def fake_load_model(uri):
        calls["model_uri"] = uri
        return "MODEL"

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.setdefault("uri", uri))
    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load_model)
    monkeypatch.setattr(main, "MlflowClient", FakeClient)

    model, version = main.load_registry_model(cfg)

    assert calls["uri"] == "http://example:5000"
    assert calls["alias"] == ("taxi-duration", "champion")
    assert calls["model_uri"] == "models:/taxi-duration@champion"
    assert model == "MODEL"
    assert version == "taxi-duration v7 (champion, run abcdef12)"


# ---------- end to end with a real (small) pipeline ----------
def test_predict_with_real_pipeline(monkeypatch):
    train = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-10 08:00", "2025-01-11 18:00"] * 20),
        "PULocationID": [161, 132] * 20,
        "DOLocationID": [236, 48] * 20,
        "trip_distance": [2.5, 17.0] * 20,
    })
    pipe = build_pipeline({"n_estimators": 5, "random_state": 42}).fit(train, [15.0, 40.0] * 20)
    monkeypatch.setattr(main, "load_registry_model", lambda cfg: (pipe, VERSION))

    with TestClient(main.app) as c:
        r = c.post("/predict", json=PAYLOAD)

    assert r.status_code == 200
    assert 15.0 <= r.json()["duration_minutes"] <= 40.0
