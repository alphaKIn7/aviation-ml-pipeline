# ──────────────────────────────────────────────
# test_data_ingestion.py — Unit Tests for Data Ingestion
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Verify that data loading functions work correctly.
#   Test edge cases: missing files, wrong formats, empty data.
#
# TESTS TO WRITE:
#   - test_load_csv_returns_dataframe
#   - test_load_csv_missing_file_raises_error
#   - test_load_csv_empty_file_raises_error
#   - test_loaded_data_has_expected_columns
#
# INDUSTRY PATTERN:
#   Use pytest fixtures to create temporary test data files.
#   Never test against real production data — use small, synthetic datasets.
# ──────────────────────────────────────────────
