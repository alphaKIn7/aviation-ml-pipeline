# ──────────────────────────────────────────────
# validation.py — Data Validation / Quality Gates
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Validate that incoming data meets expected schemas and quality
#   standards BEFORE any processing happens. Fail fast, fail loud.
#
# WHY THIS MATTERS:
#   "Garbage in, garbage out." In production ML, data drift and schema
#   changes are the #1 cause of silent model degradation. This file
#   acts as a checkpoint — if data doesn't pass, the pipeline stops.
#
# FUNCTIONS TO BUILD HERE:
#   - validate_schema(df, expected_columns)    → Column names & types
#   - check_nulls(df, threshold)               → Null percentage gates
#   - check_duplicates(df)                     → Duplicate row detection
#   - validate_ranges(df, column, min, max)    → Value range checks
#
# INDUSTRY PATTERN:
#   In production, teams use libraries like Great Expectations or
#   Pandera for this. We'll start with manual checks, then graduate.
# ──────────────────────────────────────────────
