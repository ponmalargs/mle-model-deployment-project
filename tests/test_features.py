import pandas as pd
import pytest

from src.features import FEATURES, prepare_features


@pytest.fixture
def trips():
    return pd.DataFrame({
        # 2025-01-15 is a Wednesday (dayofweek = 2); 2025-01-19 is a Sunday (6)
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-15 08:30:00", "2025-01-19 23:05:00"]),
        "tpep_dropoff_datetime": pd.to_datetime(["2025-01-15 08:45:00", "2025-01-19 23:20:00"]),
        "PULocationID": [161, 132],
        "DOLocationID": [236, 48],
        "trip_distance": [2.4, 17.8],
        "total_amount": [20.0, 70.0],      # extra column -> must not appear in output
        "duration": [15.0, 15.0],          # target -> must not appear in output
    })


def test_columns_equal_features(trips):
    X = prepare_features(trips)
    assert list(X.columns) == FEATURES


def test_target_and_extra_columns_excluded(trips):
    X = prepare_features(trips)
    assert "duration" not in X.columns
    assert "total_amount" not in X.columns


def test_hour_and_weekday_correct(trips):
    X = prepare_features(trips)
    assert X["pickup_hour"].tolist() == [8, 23]
    assert X["pickup_dow"].tolist() == [2, 6]    # Monday=0 ... Sunday=6


def test_no_nulls(trips):
    X = prepare_features(trips)
    assert not X.isna().any().any()


def test_same_number_of_rows(trips):
    assert len(prepare_features(trips)) == len(trips)


def test_works_with_string_timestamp():
    """The API sends the pickup time as a string."""
    df = pd.DataFrame([{"tpep_pickup_datetime": "2025-01-15T08:30:00",
                        "PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.4}])
    X = prepare_features(df)
    assert X.loc[0, "pickup_hour"] == 8 and X.loc[0, "pickup_dow"] == 2


def test_does_not_modify_input(trips):
    before = trips.copy()
    prepare_features(trips)
    pd.testing.assert_frame_equal(trips, before)
