import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Each row tests one cleaning rule. 'expect' says whether clean() should keep it."""
    rows = [
        # pickup,              dropoff,               PU,  DO,  dist, total,  expect
        ("2025-01-10 08:00:00", "2025-01-10 08:15:00", 161, 236, 2.5, 20.0, True),   # valid
        ("2024-12-31 23:50:00", "2025-01-01 00:05:00", 161, 236, 2.5, 20.0, False),  # before Jan
        ("2025-02-01 00:10:00", "2025-02-01 00:20:00", 161, 236, 2.5, 20.0, False),  # after Jan
        ("2025-01-10 09:00:00", "2025-01-10 08:50:00", 161, 236, 2.5, 20.0, False),  # negative duration
        ("2025-01-10 09:00:00", "2025-01-10 09:00:00", 161, 236, 2.5, 20.0, False),  # zero duration
        ("2025-01-10 10:00:00", "2025-01-10 12:00:00", 161, 236, 2.5, 20.0, False),  # > 60 min
        ("2025-01-10 11:00:00", "2025-01-10 11:20:00", 161, 236, 2.5, -20.0, False),  # negative amount
        ("2025-01-10 12:00:00", "2025-01-10 12:20:00", 161, 236, 0.0, 20.0, False),  # zero distance
    ]
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
            "trip_distance", "total_amount", "expect"]
    df = pd.DataFrame(rows, columns=cols)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    return df