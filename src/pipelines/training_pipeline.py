from __future__ import annotations

from src.data.ingestion import load_csv
from src.data.preprocessing import (
    encode_categoricals,
    handle_missing_values,
    remove_outliers,
    scale_features,
    split_data,
)
from src.data.validation import run_validation
from src.features.build_features import build_all_features
from src.features.select_features import select_k_best
from src.models.evaluator import (
    compute_metrics,
    generate_evaluation_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.models.trainer import (
    cross_validate_model,
    get_model,
    save_model,
    train_model,
)
from src.utils.config import load_config
from src.utils.io_handler import ensure_dir, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RAW_REQUIRED_COLUMNS = [
    "month", "day_of_week", "crs_dep_time", "crs_arr_time",
    "distance", "crs_elapsed_time", "op_unique_carrier",
    "origin", "dest", "arr_delay", "dep_delay", "cancelled",
]

# Models that support category dtype natively — no dummy explosion needed.
# XGBoost 2.0+ and LightGBM can split directly on category columns,
# finding the optimal grouping of categories at each node.
_NATIVE_CAT_MODELS = {"xgboost", "lightgbm"}

# Linear models require purely numerical input. A value like "JFK"=3 would
# imply JFK is 3x something, which is nonsense. One-hot encoding removes
# that false ordinal relationship by turning each category into a 0/1 flag.
_LINEAR_MODELS = {"logistic"}


def run_training_pipeline(config_path: str) -> dict:
    """
    Execute the full training workflow from raw CSV to saved, evaluated model.

    Encoding strategy (chosen automatically based on model type):
    - Linear models  (logistic)         → one-hot encoding via pd.get_dummies
    - Tree models    (xgboost, lightgbm) → native category dtype; no dummy explosion
    - sklearn trees  (random_forest)    → one-hot encoding (sklearn has no native cat support)

    Returns
    -------
    dict
        Evaluation metrics from the held-out test set.
    """
    # ── 1. Config ─────────────────────────────────────────────────────────────
    config = load_config(config_path)

    data_cfg      = config["data"]
    feat_cfg      = config["features"]
    model_cfg     = config["model"]
    train_cfg     = config["training"]
    artifacts_cfg = config["artifacts"]

    target_col   = data_cfg["target_column"]
    random_state = data_cfg.get("random_state", 42)
    test_size    = data_cfg.get("test_size", 0.2)
    threshold    = train_cfg.get("prediction_threshold", 0.5)
    model_name   = model_cfg["name"]

    # ── 2. Ingest ──────────────────────────────────────────────────────────────
    logger.info("── Step 1: Ingestion ──")
    df = load_csv(data_cfg["raw_path"])
    df = df[df["cancelled"] == 0].copy()
    df = df.dropna(subset=["arr_delay"])
    logger.info(f"After filtering: {len(df):,} rows")

    # ── 3. Validate raw schema ─────────────────────────────────────────────────
    logger.info("── Step 2: Validation ──")
    run_validation(df, _RAW_REQUIRED_COLUMNS, feat_cfg.get("null_thresholds", {}))

    # ── 4. Feature engineering ─────────────────────────────────────────────────
    logger.info("── Step 3: Feature Engineering ──")
    df = build_all_features(df, feat_cfg["drop_columns"])

    # ── 5. Trim to config-defined feature set ──────────────────────────────────
    all_feature_cols = (
        feat_cfg["numerical"]
        + feat_cfg["categorical"]
        + feat_cfg.get("binary", [])
    )
    cols_to_keep = [c for c in all_feature_cols + [target_col] if c in df.columns]
    df = df[cols_to_keep]
    logger.info(f"Feature set trimmed to {len(cols_to_keep)} columns (+ target)")

    # ── 6. Missing value handling ──────────────────────────────────────────────
    logger.info("── Step 4: Preprocessing ──")
    null_strategies = {
        k: v for k, v in feat_cfg.get("null_strategies", {}).items()
        if k in df.columns
    }
    if null_strategies:
        df = handle_missing_values(df, null_strategies)
    df = df.dropna()
    logger.info(f"After null handling: {len(df):,} rows")

    # ── 7. Outlier removal ─────────────────────────────────────────────────────
    outlier_cols = [c for c in feat_cfg.get("outlier_columns", []) if c in df.columns]
    if outlier_cols:
        df = remove_outliers(df, outlier_cols)
    logger.info(f"After outlier removal: {len(df):,} rows")

    # ── 8. Binary features → int ───────────────────────────────────────────────
    binary_cols = [c for c in feat_cfg.get("binary", []) if c in df.columns]
    if binary_cols:
        df[binary_cols] = df[binary_cols].astype(int)
        logger.info(f"Converted {len(binary_cols)} binary columns to int")

    # ── 9. Encode categorical columns (strategy depends on model type) ─────────
    cat_cols = [c for c in feat_cfg["categorical"] if c in df.columns]

    if model_name in _NATIVE_CAT_MODELS:
        # Tree models (XGBoost, LightGBM): mark columns as category dtype.
        # The model splits directly on categories — no 715-column dummy explosion.
        # Result: 18 clean features instead of 715 sparse ones.
        for col in cat_cols:
            df[col] = df[col].astype("category")
        encoding_type = "native"
        logger.info(
            f"Tree model '{model_name}' — marked {len(cat_cols)} columns "
            f"as category dtype (no dummies created)"
        )

    else:
        # Linear models and sklearn trees need numerical input — use one-hot.
        df = encode_categoricals(df, cat_cols, drop_first=feat_cfg.get("drop_first", False))
        remaining_bool = df.select_dtypes(include="bool").columns.tolist()
        if remaining_bool:
            df[remaining_bool] = df[remaining_bool].astype(int)
        encoding_type = "onehot"
        logger.info(
            f"Model '{model_name}' — one-hot encoded {len(cat_cols)} columns"
        )

    # ── 10. Scale numerical features ───────────────────────────────────────────
    scaling_cfg = {
        k: v for k, v in feat_cfg.get("scaling", {}).items() if k in df.columns
    }
    df, scalers = scale_features(df, scaling_cfg)

    # ── 11. Train / test split ─────────────────────────────────────────────────
    logger.info("── Step 5: Split ──")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col, test_size=test_size, random_state=random_state
    )

    # ── 12. Feature selection ─────────────────────────────────────────────────
    logger.info("── Step 6: Feature Selection ──")

    if encoding_type == "native":
        # f_classif inside select_k_best needs numeric values to compute F-scores.
        # Category dtype is not numeric, so we convert to integer codes ONLY for
        # the scoring step — the actual training data keeps category dtype intact.
        X_score = X_train.copy()
        for col in X_score.select_dtypes(include="category").columns:
            X_score[col] = X_score[col].cat.codes
        _, selected_features = select_k_best(X_score, y_train, k=20)
        # Apply the selected column names to the ORIGINAL X (with category dtype)
        X_train = X_train[selected_features]
        X_test  = X_test[selected_features]
    else:
        X_train, selected_features = select_k_best(X_train, y_train, k=20)
        X_test = X_test[selected_features]

    logger.info(f"Selected {len(selected_features)} features: {selected_features}")

    # ── 13. Build model params ─────────────────────────────────────────────────
    # Copy hyperparams so we don't mutate the loaded config dict.
    model_params = dict(model_cfg["hyperparameters"])
    if model_name == "xgboost" and encoding_type == "native":
        # Required flag for XGBoost to accept and correctly split category columns.
        model_params["enable_categorical"] = True

    # ── 14. Cross-validation ───────────────────────────────────────────────────
    logger.info("── Step 7: Cross-Validation ──")
    model = get_model(model_name, model_params)
    cv_results = cross_validate_model(
        model, X_train, y_train,
        folds=train_cfg.get("cross_validation_folds", 5),
        metric=train_cfg.get("metric", "roc_auc"),
    )

    # ── 15. Train final model ──────────────────────────────────────────────────
    logger.info("── Step 8: Training ──")
    model = get_model(model_name, model_params)
    model = train_model(
        model, X_train, y_train,
        early_stopping_rounds=train_cfg.get("early_stopping_rounds"),
    )

    # ── 16. Evaluate ──────────────────────────────────────────────────────────
    logger.info("── Step 9: Evaluation ──")
    logger.info(f"Using prediction threshold: {threshold}")

    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )
    y_pred = (
        (y_proba >= threshold).astype(int)
        if y_proba is not None
        else model.predict(X_test)
    )

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["cv_mean"]   = cv_results["mean"]
    metrics["cv_std"]    = cv_results["std"]
    metrics["threshold"] = threshold

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_path = artifacts_cfg["plots_path"]
    ensure_dir(plots_path)
    plot_confusion_matrix(y_test, y_pred, f"{plots_path}confusion_matrix.png")
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, f"{plots_path}roc_curve.png")
    plot_feature_importance(model, selected_features,
                            f"{plots_path}feature_importance.png")

    # ── 17. Save artifacts ─────────────────────────────────────────────────────
    logger.info("── Step 10: Saving Artifacts ──")
    model_dir = artifacts_cfg["model_path"]
    ensure_dir(model_dir)

    save_model(model,              f"{model_dir}best_model.pkl")
    save_pickle(scalers,           f"{model_dir}scalers.pkl")
    save_pickle(selected_features, f"{model_dir}feature_names.pkl")

    # model_info.pkl is read by the inference pipeline to apply the same
    # encoding strategy that was used during training.
    save_pickle(
        {
            "model_name":         model_name,
            "encoding_type":      encoding_type,
            "categorical_columns": cat_cols,
        },
        f"{model_dir}model_info.pkl",
    )

    generate_evaluation_report(
        {
            "metrics":        metrics,
            "cv":             cv_results,
            "model":          model_name,
            "encoding":       encoding_type,
            "features":       selected_features,
            "threshold":      threshold,
        },
        f"{artifacts_cfg['reports_path']}evaluation_report.json",
    )

    logger.info("Training pipeline complete.")
    logger.info(f"Test ROC-AUC : {metrics.get('roc_auc', 'N/A'):.4f}")
    logger.info(f"Delay capture: {metrics.get('delay_capture_rate', 'N/A'):.4f}")
    return metrics
