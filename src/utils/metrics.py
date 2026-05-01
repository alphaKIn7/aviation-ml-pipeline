# ──────────────────────────────────────────────
# metrics.py — Custom Metrics
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Define custom evaluation metrics that are specific to your
#   business problem but not available in sklearn out of the box.
#
# EXAMPLES FOR AVIATION:
#   - weighted_delay_score   → Penalize long delays more than short ones
#   - on_time_performance    → % of flights within 15 min of schedule
#   - cost_weighted_accuracy → Misclassifying a cancellation costs more
#
# FUNCTIONS TO BUILD HERE:
#   - custom_scorer(y_true, y_pred)     → Your domain-specific metric
#   - make_sklearn_scorer(func)         → Wrap for use in GridSearchCV
#
# INDUSTRY PATTERN:
#   Always align your ML metrics with BUSINESS metrics. A model with
#   95% accuracy is useless if it misses the cases that cost money.
# ──────────────────────────────────────────────
