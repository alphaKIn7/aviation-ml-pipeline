import pandas as pd
import pytest

from src.features.build_features import (
    build_all_features,
    create_aggregation_features,
    create_route_feature,
    create_target,
    create_temporal_features,
    drop_columns,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "arr_delay":         [10.0, -5.0, 20.0, 5.0, 30.0],
        "dep_delay":         [5.0, -3.0, 15.0, 2.0, 25.0],
        "crs_dep_time":      [800, 1200, 1800, 600, 2100],
        "crs_arr_time":      [1000, 1400, 2000, 800, 2300],
        "day_of_week":       [1, 5, 7, 3, 6],
        "month":             [1, 6, 12, 4, 8],
        "origin":            ["JFK", "LAX", "ORD", "ATL", "JFK"],
        "dest":              ["LAX", "ORD", "JFK", "JFK", "ORD"],
        "op_unique_carrier": ["AA", "UA", "DL", "AA", "UA"],
        "cancelled":         [0, 0, 0, 0, 0],
    })


# ── create_target ─────────────────────────────────────────────────────────────

def test_create_target_adds_column(df):
    result = create_target(df)
    assert "is_delayed" in result.columns


def test_create_target_correct_threshold(df):
    result = create_target(df, delay_col="arr_delay", threshold=15)
    # arr_delay values: 10, -5, 20, 5, 30 → delayed if > 15: F, F, T, F, T
    assert result["is_delayed"].tolist() == [False, False, True, False, True]


def test_create_target_does_not_mutate_input(df):
    original_cols = set(df.columns)
    create_target(df)
    assert set(df.columns) == original_cols  # original df unchanged


# ── create_temporal_features ──────────────────────────────────────────────────

def test_create_temporal_creates_dep_hour(df):
    result = create_temporal_features(df)
    assert "dep_hour" in result.columns


def test_create_temporal_dep_hour_correct_value(df):
    result = create_temporal_features(df)
    # crs_dep_time = 800 → dep_hour = 8
    assert result["dep_hour"].iloc[0] == 8


def test_create_temporal_creates_dep_hour_bin(df):
    result = create_temporal_features(df)
    assert "dep_hour_bin" in result.columns
    assert set(result["dep_hour_bin"].unique()).issubset(
        {"morning", "afternoon", "evening", "night"}
    )


def test_create_temporal_creates_weekend_flag(df):
    result = create_temporal_features(df)
    assert "is_weekend" in result.columns


def test_create_temporal_creates_holiday_month_flag(df):
    result = create_temporal_features(df)
    assert "is_holiday_month" in result.columns
    # month=1 is a holiday month (Jan), month=4 is not
    jan_row = result[result["month"] == 1]["is_holiday_month"].iloc[0]
    apr_row = result[result["month"] == 4]["is_holiday_month"].iloc[0]
    assert jan_row is True or jan_row == 1
    assert apr_row is False or apr_row == 0


# ── create_route_feature ──────────────────────────────────────────────────────

def test_create_route_feature_combines_origin_dest(df):
    result = create_route_feature(df)
    assert "route" in result.columns
    assert result["route"].iloc[0] == "JFK_LAX"


# ── create_aggregation_features ───────────────────────────────────────────────

def test_create_aggregation_features_creates_carrier_avg(df):
    df = create_route_feature(df)  # route is needed for route_avg_delay
    result = create_aggregation_features(df)
    assert "carrier_avg_delay" in result.columns
    assert "origin_avg_delay" in result.columns
    assert "route_avg_delay" in result.columns


# ── drop_columns ──────────────────────────────────────────────────────────────

def test_drop_columns_removes_listed_columns(df):
    result = drop_columns(df, ["month", "cancelled"])
    assert "month" not in result.columns
    assert "cancelled" not in result.columns


def test_drop_columns_ignores_nonexistent(df):
    result = drop_columns(df, ["nonexistent_column"])
    assert set(result.columns) == set(df.columns)


# ── build_all_features ────────────────────────────────────────────────────────

def test_build_all_features_creates_target(df):
    result = build_all_features(df, cols_to_drop=[])
    assert "is_delayed" in result.columns


def test_build_all_features_drops_requested_columns(df):
    result = build_all_features(df, cols_to_drop=["cancelled"])
    assert "cancelled" not in result.columns
