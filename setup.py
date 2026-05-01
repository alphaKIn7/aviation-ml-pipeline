# ──────────────────────────────────────────────
# setup.py — Package Installation
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Makes the src/ directory an installable Python package.
#   After running `pip install -e .`, you can import from anywhere:
#     from src.data.ingestion import load_csv
#
# WHY `pip install -e .` (EDITABLE MODE)?
#   Without this, Python can't find your src/ modules when running
#   scripts or tests. The `-e` flag means changes to src/ are
#   reflected immediately without reinstalling.
#
# USAGE:
#   pip install -e .
# ──────────────────────────────────────────────
