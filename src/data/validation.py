import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = set(expected_columns) - set(df.columns)
    extra   = set(df.columns) - set(expected_columns)

    if missing:
        raise ValueError(f"Data is missing expected columns: {missing}")
    if extra:
        logger.warning(f"Data has unexpected extra columns (will be ignored): {extra}")

    logger.info("Schema validation passed")


def check_nulls(df: pd.DataFrame, threshold: float = 0.2) -> None:
    null_ratios = df.isnull().mean()
    failed = null_ratios[null_ratios > threshold]

    if not failed.empty:
        details = "\n".join(f"  {col}: {ratio:.1%} null" for col, ratio in failed.items())
        raise ValueError(f"Columns exceed null threshold ({threshold:.0%}):\n{details}")

    logger.info(f"Null check passed — all columns below {threshold:.0%} null threshold")


def check_duplicates(df: pd.DataFrame) -> None:
    n_duplicates = df.duplicated().sum()

    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates:,} duplicate rows — consider deduplication in preprocessing")
    else:
        logger.info("Duplicate check passed — no duplicate rows found")


def validate_ranges(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]

    if not out_of_range.empty:
        raise ValueError(
            f"Column '{column}' has {len(out_of_range):,} rows outside "
            f"expected range [{min_val}, {max_val}]. "
            f"Min found: {df[column].min()}, Max found: {df[column].max()}"
        )

    logger.info(f"Range check passed for '{column}' — all values within [{min_val}, {max_val}]")


def run_validation(df: pd.DataFrame, expected_columns: list[str]) -> None:
    logger.info(f"Starting validation on DataFrame with {len(df):,} rows")
    validate_schema(df, expected_columns)
    check_nulls(df)
    check_duplicates(df)
    logger.info("All validation checks passed")
