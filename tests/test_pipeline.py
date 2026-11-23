import pandas as pd
from src.pipeline import build_pipeline, TRUSTED_TYPES
import skops.io as sio



def test_trusted_types_match_pipeline(tmp_path):
    df = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-10 08:00", "2025-01-11 18:00"] * 10),
            "PULocationID": [161, 132] * 10, "DOLocationID": [236, 48] * 10,
            "trip_distance": [2.5, 17.0] * 10,
        })
    y = [15.0, 40.0] * 10
    pipe = build_pipeline({"n_estimators": 2, "random_state": 0}).fit(df, y)
    path = tmp_path / "m.skops"
    sio.dump(pipe, path)
    assert sorted(sio.get_untrusted_types(file=path)) == sorted(TRUSTED_TYPES)


def test_pipeline_fits_and_predicts_raw_rows():
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-10 08:00", "2025-01-11 18:00"] * 10),
        "PULocationID": [161, 132] * 10, "DOLocationID": [236, 48] * 10,
        "trip_distance": [2.5, 17.0] * 10,
    })
    y = [15.0, 40.0] * 10
    pipe = build_pipeline({"n_estimators": 5, "random_state": 42}).fit(df, y)

    request = pd.DataFrame([{"tpep_pickup_datetime": "2025-01-15T08:30:00",
                             "PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.4}])
    pred = pipe.predict(request)
    assert pred.shape == (1,) and pred[0] > 0