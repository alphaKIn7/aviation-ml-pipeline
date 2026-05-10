import pandas as pd
import pytest

from src.data.ingestion import load_csv, load_from_directory


def test_load_csv_returns_dataframe(tmp_path):
    csv_file = tmp_path / "flights.csv"
    pd.DataFrame({"month": [1, 2], "distance": [500.0, 1000.0]}).to_csv(csv_file, index=False)
    df = load_csv(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_csv_correct_columns(tmp_path):
    csv_file = tmp_path / "flights.csv"
    pd.DataFrame({"month": [1], "distance": [500.0]}).to_csv(csv_file, index=False)
    df = load_csv(str(csv_file))
    assert "month" in df.columns
    assert "distance" in df.columns


def test_load_csv_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_csv("does/not/exist.csv")


def test_load_from_directory_combines_files(tmp_path):
    for i in range(3):
        pd.DataFrame({"a": [i], "b": [i * 10]}).to_csv(
            tmp_path / f"file_{i}.csv", index=False
        )
    df = load_from_directory(str(tmp_path))
    assert len(df) == 3
    assert set(df.columns) == {"a", "b"}


def test_load_from_directory_empty_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_directory(str(tmp_path))


def test_load_from_directory_column_mismatch_raises_error(tmp_path):
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(tmp_path / "f1.csv", index=False)
    pd.DataFrame({"a": [1], "c": [3]}).to_csv(tmp_path / "f2.csv", index=False)
    with pytest.raises(ValueError, match="Column mismatch"):
        load_from_directory(str(tmp_path))
