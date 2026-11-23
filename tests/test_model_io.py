# import numpy as np
# import pandas as pd
# import pytest

# from src.model_io import load_model, save_model
# from src.pipeline import build_pipeline


# @pytest.fixture
# def fitted_pipeline():
#     df = pd.DataFrame({
#         "tpep_pickup_datetime": pd.to_datetime(["2025-01-10 08:00", "2025-01-11 18:00"] * 10),
#         "PULocationID": [161, 132] * 10, "DOLocationID": [236, 48] * 10,
#         "trip_distance": [2.5, 17.0] * 10,
#     })
#     return build_pipeline({"n_estimators": 5, "random_state": 42}).fit(df, [15.0, 40.0] * 10), df


# def test_save_load_roundtrip(tmp_path, fitted_pipeline):
#     pipe, df = fitted_pipeline
#     path = tmp_path / "model.skops"
#     save_model(pipe, path)
#     loaded = load_model(path)
#     assert np.allclose(loaded.predict(df), pipe.predict(df))   # identical predictions


# def test_load_missing_file_raises(tmp_path):
#     with pytest.raises(FileNotFoundError):
#         load_model(tmp_path / "nope.skops")