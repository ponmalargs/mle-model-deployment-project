import pandas as pd

TARGET = "duration"
FEATURES = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour", "pickup_dow"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = ts.dt.hour
    df["pickup_dow"] = ts.dt.dayofweek
    return df[FEATURES]