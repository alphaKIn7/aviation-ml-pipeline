# ──────────────────────────────────────────────
# build_features.py — Feature Engineering
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Create new, more informative features from the cleaned data.
#   This is often where the biggest model performance gains come from —
#   not from tuning hyperparameters, but from better features.
#
# EXAMPLES FOR AVIATION DATA:
#   - flight_duration_hours   → Arrival - Departure time
#   - is_peak_hour            → Boolean: departure between 7-9 AM or 5-7 PM
#   - day_of_week             → Extract from date column
#   - route_frequency         → How often this route is flown
#   - rolling_avg_delay       → Average delay for this airline over last 7 days
#
# FUNCTIONS TO BUILD HERE:
#   - create_temporal_features(df)       → Day, month, hour, is_weekend
#   - create_interaction_features(df)    → Combinations of existing features
#   - create_aggregation_features(df)    → Group-level statistics
#   - create_domain_features(df)         → Aviation-specific features
#
# INDUSTRY PATTERN:
#   Feature engineering code should be idempotent — running it twice
#   on the same data should produce the same result. Never mutate in place.
# ──────────────────────────────────────────────
