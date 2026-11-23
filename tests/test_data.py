import pandas as pd
import pytest
from src.config import file_name
from src.data import load_raw, clean


# ---------- clean() ----------
def test_clean_keeps_only_valid_rows(raw_df):
    out = clean(raw_df.drop(columns="expect"))
    assert len(out) == raw_df["expect"].sum()


def test_clean_creates_duration_in_minutes(raw_df):
    out = clean(raw_df.drop(columns="expect"))
    assert "duration" in out.columns
    assert out["duration"].iloc[0] == 15


def test_clean_duration_within_limits(raw_df):
    out = clean(raw_df.drop(columns="expect"))
    assert out["duration"].between(1, 60).all()


def test_clean_only_january(raw_df):
    out = clean(raw_df.drop(columns="expect"))
    assert (out["tpep_pickup_datetime"].dt.month == 1).all()
    assert (out["tpep_pickup_datetime"].dt.year == 2025).all()


def test_clean_no_negative_amounts(raw_df):
    out = clean(raw_df.drop(columns="expect"))
    assert (out["total_amount"] >= 0).all()


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean(raw_df.drop(columns="expect"))
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_empty_dataframe(raw_df):
    out = clean(raw_df.drop(columns="expect").iloc[0:0])
    assert out.empty


# ---------- load_raw() ----------
def test_load_raw_reads_requested_month(tmp_path, raw_df):
    df = raw_df.drop(columns="expect")
    df.to_parquet(tmp_path / file_name(2025, 1))
    df.iloc[:2].to_parquet(tmp_path / file_name(2025, 2))   # other month, must be ignored

    out = load_raw(year=2025, month=1, raw_dir=tmp_path)
    assert len(out) == len(df)


def test_load_raw_selects_columns(tmp_path, raw_df):
    raw_df.drop(columns="expect").to_parquet(tmp_path / file_name(2025, 1))
    out = load_raw(year=2025, month=1, raw_dir=tmp_path,
                   columns=["trip_distance", "PULocationID"])
    assert list(out.columns) == ["trip_distance", "PULocationID"]


def test_load_raw_missing_month_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(year=2025, month=3, raw_dir=tmp_path)
