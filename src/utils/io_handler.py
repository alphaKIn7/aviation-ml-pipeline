# ──────────────────────────────────────────────
# io_handler.py — File I/O Utilities
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Centralized functions for reading/writing files of different
#   formats. Every module that needs to save or load data calls
#   functions here instead of writing its own I/O code.
#
# FUNCTIONS TO BUILD HERE:
#   - load_yaml(path)                 → Read YAML config files
#   - save_yaml(data, path)           → Write YAML files
#   - save_pickle(obj, path)          → Serialize Python objects
#   - load_pickle(path)               → Deserialize Python objects
#   - save_dataframe(df, path, fmt)   → Save DataFrame (csv/parquet)
#   - load_dataframe(path)            → Load DataFrame
#   - ensure_dir(path)                → Create directory if not exists
#
# INDUSTRY PATTERN:
#   Centralizing I/O prevents scattered file handling bugs and makes
#   it easy to switch formats (e.g., CSV → Parquet) project-wide.
# ──────────────────────────────────────────────


import pandas as pd
import yaml
import pickle
from pathlib import Path

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def save_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)
    
def save_dataframe(df: pd.DataFrame, path: str, fmt: str = "csv") -> None:
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Invalid format. Choose 'csv' or 'parquet'.")

def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Invalid format. Choose 'csv' or 'parquet'.")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(obj: object, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)