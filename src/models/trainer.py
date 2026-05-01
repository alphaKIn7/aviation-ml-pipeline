# ──────────────────────────────────────────────
# trainer.py — Model Training
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Train machine learning models with cross-validation and
#   hyperparameter tuning. Save the best model to disk.
#
# WHY SEPARATE FROM EVALUATION?
#   Training = "fit the model."  Evaluation = "judge the model."
#   A trainer should not decide if a model is good — that's the
#   evaluator's job. This separation enables A/B testing of models.
#
# FUNCTIONS TO BUILD HERE:
#   - get_model(model_name, params)              → Model factory
#   - train_model(model, X_train, y_train)       → Fit and return model
#   - cross_validate_model(model, X, y, folds)   → K-fold CV scores
#   - tune_hyperparameters(model, X, y, grid)    → GridSearch / Optuna
#   - save_model(model, path)                    → Serialize to disk
#   - load_model(path)                           → Deserialize from disk
#
# INDUSTRY PATTERN:
#   Use a "model factory" pattern — a function that takes a string name
#   (e.g., "xgboost") and returns the right model object. This lets you
#   swap models via config without changing code.
# ──────────────────────────────────────────────
