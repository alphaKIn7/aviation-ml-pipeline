# ──────────────────────────────────────────────
# ingestion.py — Data Ingestion
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Load raw data from its source (CSV, database, API, S3, etc.)
#   and return it as a pandas DataFrame.
#
# WHY A SEPARATE FILE?
#   Data sources change. Today it's a CSV, tomorrow it's a database,
#   next month it's a streaming API. Isolating ingestion means you
#   only change THIS file when the source changes — nothing else breaks.
#
# FUNCTIONS TO BUILD HERE:
#   - load_csv(path)            → Load from local CSV
#   - load_from_database(conn)  → Load from SQL database
#   - load_from_api(url)        → Load from REST API
#
# INDUSTRY PATTERN:
#   Always return a raw, unmodified DataFrame. Never clean or transform
#   data here — that's preprocessing.py's job.
# ──────────────────────────────────────────────
