"""
Shared pytest fixtures available to every test file in the suite.
pytest discovers this file automatically — no imports needed in test files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_flight_df():
    """Minimal flight-like DataFrame that mirrors the raw CSV structure."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "month":             np.random.randint(1, 13, n),
        "day_of_week":       np.random.randint(1, 8, n),
        "crs_dep_time":      np.random.choice([600, 800, 1200, 1600, 2000], n),
        "crs_arr_time":      np.random.choice([800, 1000, 1400, 1800, 2200], n),
        "distance":          np.random.uniform(200, 2500, n),
        "crs_elapsed_time":  np.random.uniform(60, 300, n),
        "op_unique_carrier": np.random.choice(["AA", "UA", "DL"], n),
        "origin":            np.random.choice(["JFK", "LAX", "ORD", "ATL"], n),
        "dest":              np.random.choice(["JFK", "LAX", "ORD", "ATL"], n),
        "arr_delay":         np.random.uniform(-20, 90, n),
        "dep_delay":         np.random.uniform(-15, 75, n),
        "cancelled":         [0] * n,
    })


@pytest.fixture
def sample_X():
    """Simple numerical feature matrix for model tests."""
    np.random.seed(0)
    return pd.DataFrame({
        "f1": np.random.uniform(0, 10, 40),
        "f2": np.random.uniform(0, 100, 40),
        "f3": np.random.uniform(0, 1, 40),
    })


@pytest.fixture
def sample_y():
    """Binary target series aligned with sample_X."""
    np.random.seed(0)
    return pd.Series(np.random.randint(0, 2, 40), name="is_delayed")
