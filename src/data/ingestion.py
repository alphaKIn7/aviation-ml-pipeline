import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.io_handler import load_dataframe

logger = get_logger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")
    df = load_dataframe(path)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def load_from_directory(dir_path: str) -> pd.DataFrame:
    directory = Path(dir_path)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    csv_files = list(directory.glob("*.csv"))
    parquet_files = list(directory.glob("*.parquet"))
    all_files = csv_files + parquet_files

    if not all_files:
        raise FileNotFoundError(f"No CSV or Parquet files found in: {dir_path}")

    logger.info(f"Found {len(all_files)} file(s) in {dir_path} — combining into one DataFrame")

    frames = [load_dataframe(str(f)) for f in all_files]

    reference_columns = set(frames[0].columns)
    for i, (frame, file) in enumerate(zip(frames[1:], all_files[1:]), start=1):
        if set(frame.columns) != reference_columns:
            missing = reference_columns - set(frame.columns)
            extra = set(frame.columns) - reference_columns
            raise ValueError(
                f"Column mismatch in file '{file.name}'.\n"
                f"  Missing columns: {missing or 'none'}\n"
                f"  Extra columns:   {extra or 'none'}"
            )

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined DataFrame: {len(df):,} rows and {len(df.columns)} columns")
    return df
