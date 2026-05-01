# ──────────────────────────────────────────────
# src/features/ — Feature Engineering Layer
# ──────────────────────────────────────────────
# This layer sits BETWEEN preprocessing and modeling.
# Raw cleaned data goes in → rich, informative features come out.
#
# Files in this package:
#   build_features.py   → Create new features from existing columns
#   select_features.py  → Pick the best features (filter, wrapper, embedded)
