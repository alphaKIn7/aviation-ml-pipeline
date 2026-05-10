import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    encode_categoricals,
    handle_missing_values,
    remove_outliers,
    scale_features,
    split_data,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "month":      [1, 2, 3, 4, 5],
        "distance":   [500.0, None, 1500.0, 800.0, 300.0],
        "carrier":    ["AA", "UA", "DL", "AA", "UA"],
        "is_delayed": [0, 1, 0, 1, 0],
    })


# ── handle_missing_values ────────────────────────────────────────────────────

def test_handle_missing_values_median_fills_nulls(df):
    result = handle_missing_values(df, {"distance": "median"})
    assert result["distance"].isnull().sum() == 0


def test_handle_missing_values_mean_fills_nulls(df):
    result = handle_missing_values(df, {"distance": "mean"})
    assert result["distance"].isnull().sum() == 0


def test_handle_missing_values_invalid_strategy_raises(df):
    with pytest.raises(ValueError):
        handle_missing_values(df, {"distance": "invalid_strategy"})


def test_handle_missing_values_unknown_column_raises(df):
    with pytest.raises(ValueError):
        handle_missing_values(df, {"nonexistent": "mean"})


# ── encode_categoricals ───────────────────────────────────────────────────────

def test_encode_categoricals_removes_original_column(df):
    result = encode_categoricals(df, ["carrier"])
    assert "carrier" not in result.columns


def test_encode_categoricals_creates_dummy_columns(df):
    result = encode_categoricals(df, ["carrier"])
    assert any(col.startswith("carrier_") for col in result.columns)


def test_encode_categoricals_output_is_numeric(df):
    result = encode_categoricals(df, ["carrier"])
    dummy_cols = [c for c in result.columns if c.startswith("carrier_")]
    for col in dummy_cols:
        assert result[col].dtype in [bool, np.uint8, int]


def test_encode_categoricals_missing_column_raises(df):
    with pytest.raises(ValueError):
        encode_categoricals(df, ["nonexistent"])


# ── scale_features ────────────────────────────────────────────────────────────

def test_scale_features_returns_scalers_dict():
    clean = pd.DataFrame({"distance": [100.0, 200.0, 300.0, 400.0, 500.0]})
    _, scalers = scale_features(clean, {"distance": "standard"})
    assert "distance" in scalers


def test_scale_features_standard_mean_near_zero():
    clean = pd.DataFrame({"distance": [100.0, 200.0, 300.0, 400.0, 500.0]})
    result, _ = scale_features(clean, {"distance": "standard"})
    assert abs(result["distance"].mean()) < 1e-9


def test_scale_features_minmax_range():
    clean = pd.DataFrame({"distance": [100.0, 200.0, 300.0, 400.0, 500.0]})
    result, _ = scale_features(clean, {"distance": "minmax"})
    assert result["distance"].min() >= 0.0
    assert result["distance"].max() <= 1.0


# ── split_data ────────────────────────────────────────────────────────────────

def test_split_data_correct_total_rows():
    df = pd.DataFrame({"f": range(100), "target": [0, 1] * 50})
    X_train, X_test, y_train, y_test = split_data(df, "target", test_size=0.2)
    assert len(X_train) + len(X_test) == 100


def test_split_data_target_not_in_X():
    df = pd.DataFrame({"f": range(20), "target": [0, 1] * 10})
    X_train, X_test, _, _ = split_data(df, "target")
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_split_data_missing_target_raises(df):
    with pytest.raises(ValueError):
        split_data(df, "nonexistent_target")


# ── remove_outliers ───────────────────────────────────────────────────────────

def test_remove_outliers_reduces_row_count():
    df = pd.DataFrame({"distance": [100, 200, 300, 200, 150, 99999]})
    result = remove_outliers(df, ["distance"])
    assert len(result) < len(df)


def test_remove_outliers_missing_column_raises():
    df = pd.DataFrame({"distance": [100, 200, 300]})
    with pytest.raises(ValueError):
        remove_outliers(df, ["nonexistent"])
