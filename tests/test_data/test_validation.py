import pandas as pd
import pytest

from src.data.validation import (
    check_duplicates,
    check_nulls,
    run_validation,
    validate_ranges,
    validate_schema,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "month":    [1, 2, 3],
        "distance": [500.0, 1000.0, None],
    })


def test_validate_schema_passes_with_correct_columns(df):
    validate_schema(df, ["month", "distance"])  # should not raise


def test_validate_schema_raises_on_missing_column(df):
    with pytest.raises(ValueError, match="missing"):
        validate_schema(df, ["month", "distance", "nonexistent"])


def test_check_nulls_raises_when_threshold_exceeded(df):
    # distance has 1/3 nulls = 33% — above the 1% threshold
    with pytest.raises(ValueError):
        check_nulls(df, {"distance": 0.01})


def test_check_nulls_passes_within_threshold(df):
    check_nulls(df, {"distance": 0.5})  # 33% < 50%, should pass without raising


def test_check_nulls_unknown_column_raises(df):
    with pytest.raises(ValueError):
        check_nulls(df, {"nonexistent": 0.1})


def test_check_duplicates_does_not_raise_on_clean_data(df):
    check_duplicates(df)  # should complete without raising


def test_check_duplicates_does_not_raise_on_duplicates():
    # check_duplicates warns but never raises — duplicates are reported, not blocked
    df_dup = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    check_duplicates(df_dup)  # should complete without raising


def test_validate_ranges_passes_within_bounds():
    df = pd.DataFrame({"distance": [100.0, 500.0, 2000.0]})
    validate_ranges(df, "distance", 0, 5000)  # should not raise


def test_validate_ranges_raises_on_out_of_range():
    df = pd.DataFrame({"distance": [100.0, -50.0, 2000.0]})
    with pytest.raises(ValueError):
        validate_ranges(df, "distance", 0, 5000)


def test_run_validation_passes_clean_data():
    clean = pd.DataFrame({"month": [1, 2], "distance": [500.0, 1000.0]})
    run_validation(clean, ["month", "distance"])  # should not raise
