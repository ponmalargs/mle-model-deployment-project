import pytest
from src.config import file_name, file_url


@pytest.mark.parametrize("year, month, expected", [
    (2025, 1,  "yellow_tripdata_2025-01.parquet"),
    (2025, 12, "yellow_tripdata_2025-12.parquet"),
    (2024, 9,  "yellow_tripdata_2024-09.parquet"),   # month padded to 2 digits
])
def test_file_name(year, month, expected):
    assert file_name(year, month) == expected


def test_file_url():
    url = file_url(2025, 1)
    assert url.startswith("https://")
    assert url.endswith("yellow_tripdata_2025-01.parquet")


@pytest.mark.parametrize("month", [0, 13])
def test_invalid_month_raises(month):
    with pytest.raises(ValueError):
        file_name(2025, month)