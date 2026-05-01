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
