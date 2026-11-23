import pandas as pd
from pathlib import Path
from src.config import CFG, file_name


RAW_DIR = CFG["paths"]["raw"]
YEAR, MONTHS = CFG["run"]["year"], CFG["run"]["months"]
# YAML reads 2025-01-01 as a datetime.date; pandas needs a Timestamp to compare
START_DATE = pd.Timestamp(CFG["run"]["start_date"])
END_DATE = pd.Timestamp(CFG["run"]["end_date"])


COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID",
    "trip_distance", 
    "total_amount"            # used for cleaning only, not as a feature
]


def load_raw(year: int = YEAR, month: int = MONTHS[0],
             raw_dir: Path = RAW_DIR,
             columns: list[str] | None = COLUMNS) -> pd.DataFrame:
    """Read the raw taxi parquet file for one year/month."""
    file_path = Path(raw_dir) / file_name(year, month)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Please run `python -m src.download_files` first.")
    return pd.read_parquet(file_path, columns=columns)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. January 2025 pickups only
    df = df[(df.tpep_pickup_datetime >= START_DATE) & (df.tpep_pickup_datetime < END_DATE)]
    # 2. Remove refunds / reversals
    df = df[df.total_amount >= 0]
    # 3. Target + duration scope
    df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[df.duration.between(1, 60)]
    # 4. Distance
    df = df[(df.trip_distance > 0)] #& (df.trip_distance <= 100)]
    return df.drop_duplicates()


def split_by_date(df: pd.DataFrame, cutoff=CFG["split"]["cutoff"]):
    cutoff = pd.Timestamp(cutoff)
    train = df[df.tpep_pickup_datetime < cutoff]
    val = df[df.tpep_pickup_datetime >= cutoff]
    return train, val