# ──────────────────────────────────────────────
# select_features.py — Feature Selection
# ──────────────────────────────────────────────
#
# PURPOSE:
#   After engineering many features, select only the ones that
#   actually help the model. Too many features = overfitting,
#   slow training, and hard-to-explain models.
#
# METHODS TO IMPLEMENT:
#   1. Filter methods     → Correlation matrix, variance threshold
#   2. Wrapper methods    → Recursive Feature Elimination (RFE)
#   3. Embedded methods   → Feature importances from tree models
#
# FUNCTIONS TO BUILD HERE:
#   - remove_low_variance(df, threshold)
#   - remove_high_correlation(df, threshold)
#   - select_k_best(df, target, k)
#   - recursive_feature_elimination(df, target, model)
#   - get_feature_importances(model, feature_names)
#
# INDUSTRY PATTERN:
#   Always do feature selection on training data only, then apply
#   the same selected features to test data.
# ──────────────────────────────────────────────
