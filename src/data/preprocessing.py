from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    allowed = {"mean", "median", "mode", "drop"}
    df = df.copy()

    unknown_cols = [col for col in strategies if col not in df.columns]
    if unknown_cols:
        raise ValueError(f"Columns not found in DataFrame: {unknown_cols}")

    invalid = {col: s for col, s in strategies.items() if s not in allowed}
    if invalid:
        raise ValueError(f"Invalid strategies {invalid}. Allowed: {allowed}")

    drop_cols = [col for col, s in strategies.items() if s == "drop"]
    if drop_cols:
        before = len(df)
        df = df.dropna(subset=drop_cols)
        logger.info(f"Dropped {before - len(df):,} rows due to nulls in: {drop_cols}")

    for col, strategy in strategies.items():
        if strategy == "drop" or df[col].isnull().sum() == 0:
            continue
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Cannot use 'mean' on non-numeric column '{col}'")
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Cannot use 'median' on non-numeric column '{col}'")
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])

    logger.info(f"Missing values handled for {len(strategies)} columns")
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str], drop_first: bool = False) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    logger.info(f"One-hot encoded {len(columns)} columns (drop_first={drop_first})")
    return df


def scale_features(
    df: pd.DataFrame, strategies: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, StandardScaler | MinMaxScaler]]:
    allowed = {"standard", "minmax"}
    df = df.copy()

    unknown_cols = [col for col in strategies if col not in df.columns]
    if unknown_cols:
        raise ValueError(f"Columns not found in DataFrame: {unknown_cols}")

    invalid = {col: s for col, s in strategies.items() if s not in allowed}
    if invalid:
        raise ValueError(f"Invalid scaling methods {invalid}. Allowed: {allowed}")

    scalers = {}
    for col, method in strategies.items():
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    logger.info(f"Scaled {len(strategies)} columns")
    return df, scalers


def remove_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr") -> pd.DataFrame:
    if method != "iqr":
        raise ValueError(f"Supported methods: 'iqr', got '{method}'")

    df = df.copy()
    before = len(df)

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    logger.info(f"Removed {before - len(df):,} outlier rows using IQR method")
    return df


def split_data(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Data split — Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
        f"Target: '{target}'"
    )
    return X_train, X_test, y_train, y_test
