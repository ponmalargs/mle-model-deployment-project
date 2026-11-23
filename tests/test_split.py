import pandas as pd
import pytest

from src.data import split_by_date

CUTOFF = "2025-01-25"


@pytest.fixture
def trips():
    """Pickups spread across January, including one exactly at the cutoff."""
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2025-01-01 08:00:00",
            "2025-01-24 23:59:59",   # last second before cutoff -> train
            "2025-01-25 00:00:00",   # exactly at cutoff -> validation
            "2025-01-31 18:30:00",
        ]),
        "duration": [10.0, 12.0, 15.0, 20.0],
    })


def test_split_no_rows_lost(trips):
    train, val = split_by_date(trips, cutoff=CUTOFF)
    assert len(train) + len(val) == len(trips)


def test_split_no_overlap(trips):
    train, val = split_by_date(trips, cutoff=CUTOFF)
    assert set(train.index).isdisjoint(val.index)


def test_split_train_before_cutoff(trips):
    train, _ = split_by_date(trips, cutoff=CUTOFF)
    assert (train["tpep_pickup_datetime"] < pd.Timestamp(CUTOFF)).all()


def test_split_val_on_or_after_cutoff(trips):
    _, val = split_by_date(trips, cutoff=CUTOFF)
    assert (val["tpep_pickup_datetime"] >= pd.Timestamp(CUTOFF)).all()


def test_split_cutoff_row_goes_to_validation(trips):
    train, val = split_by_date(trips, cutoff=CUTOFF)
    assert len(train) == 2 and len(val) == 2


def test_split_uses_config_cutoff_by_default(trips):
    train, val = split_by_date(trips)          # cutoff from config.yaml
    assert len(train) > 0 and len(val) > 0
    assert (len(train) + len(val)) == len(trips)
