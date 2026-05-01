# ──────────────────────────────────────────────
# preprocessing.py — Data Preprocessing
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Transform raw data into a clean, model-ready format.
#   This is where you handle missing values, encode categoricals,
#   scale numericals, and remove outliers.
#
# WHY SEPARATE FROM INGESTION?
#   Ingestion = "get the data."  Preprocessing = "clean the data."
#   Keeping them separate means you can swap data sources without
#   touching your cleaning logic, and vice versa.
#
# FUNCTIONS TO BUILD HERE:
#   - handle_missing_values(df, strategy)      → Impute or drop nulls
#   - encode_categoricals(df, columns)         → Label/One-hot encoding
#   - scale_features(df, columns, method)      → StandardScaler, MinMax
#   - remove_outliers(df, columns, method)     → IQR, Z-score
#   - split_data(df, target, test_size)        → Train/test split
#
# INDUSTRY PATTERN:
#   Use sklearn Pipelines here so the same transformations are applied
#   identically during training AND inference. Never fit on test data.
# ──────────────────────────────────────────────
