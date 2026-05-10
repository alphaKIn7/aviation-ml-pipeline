from __future__ import annotations

from typing import Union

import pandas as pd

from src.data.preprocessing import encode_categoricals
from src.features.build_features import (
    create_aggregation_features,
    create_route_feature,
    create_temporal_features,
)
from src.models.predictor import (
    load_model_artifacts,
    predict,
    predict_proba,
    preprocess_input,
)
from src.utils.config import load_config
from src.utils.io_handler import load_dataframe, load_pickle, save_dataframe, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_inference_pipeline(
    config_path: str,
    input_data: Union[str, pd.DataFrame],
    output_path: str = None,
) -> pd.DataFrame:
    """
    Run predictions on new flight data using the saved model artifacts.

    Parameters
    ----------
    config_path : str
        Path to inference_config.yaml.
    input_data : str or pd.DataFrame
        Either a file path (CSV / Parquet) or an already-loaded DataFrame.
    output_path : str, optional
        If provided, the predictions DataFrame is saved here as a CSV.

    Returns
    -------
    pd.DataFrame
        Original input rows with two new columns appended:
        - predicted_delay    (0 = on time, 1 = delayed)
        - delay_probability  (float 0.0 – 1.0)

    Why use the SAVED pipeline and not re-fit?
    ------------------------------------------
    The model's internal weights were calibrated to training-scale data.
    If you scaled inference data differently (different mean, different std),
    a feature value that meant "above average" during training now means
    something else — the model's predictions become meaningless. Loading the
    saved scaler objects and calling .transform() guarantees identical scaling.
    """
    # ── 1. Config ─────────────────────────────────────────────────────────────
    config = load_config(config_path)
    model_cfg  = config["model"]
    pre_cfg    = config["preprocessing"]

    # ── 2. Load model artifacts ───────────────────────────────────────────────
    logger.info("── Step 1: Loading model artifacts ──")
    model, scalers, feature_names = load_model_artifacts(
        model_path=model_cfg["path"],
        scalers_path=pre_cfg["pipeline_path"].replace(
            "preprocessing_pipeline.pkl", "scalers.pkl"
        ),
        feature_names_path=pre_cfg["pipeline_path"].replace(
            "preprocessing_pipeline.pkl", "feature_names.pkl"
        ),
    )

    # ── 3. Load input data ────────────────────────────────────────────────────
    logger.info("── Step 2: Loading input data ──")
    if isinstance(input_data, str):
        df = load_dataframe(input_data)
    else:
        df = input_data.copy()
    logger.info(f"Input: {len(df):,} rows")
    original = df.copy()

    # ── 4. Apply the same feature engineering as training ─────────────────────
    # Only apply transformations that use PRE-FLIGHT information.
    # We do NOT call create_log_features (uses actual dep_delay — leakage)
    # and do NOT call create_target (no arr_delay at inference time).
    logger.info("── Step 3: Feature engineering ──")
    df = create_temporal_features(df)
    df = create_route_feature(df)
    df = create_aggregation_features(df)

    # ── 5. Encode categoricals (using same strategy as training) ─────────────
    logger.info("── Step 4: Encoding ──")

    # Load the encoding info saved during training so inference uses the exact
    # same strategy. If training used native categorical (XGBoost/LightGBM),
    # inference must also mark those columns as category dtype — not one-hot.
    model_info = load_pickle(
        pre_cfg["pipeline_path"].replace("preprocessing_pipeline.pkl", "model_info.pkl")
    )
    cat_cols      = [c for c in model_info["categorical_columns"] if c in df.columns]
    encoding_type = model_info["encoding_type"]

    if encoding_type == "native":
        for col in cat_cols:
            df[col] = df[col].astype("category")
        logger.info(f"Applied native category dtype to {len(cat_cols)} columns")
    else:
        df = encode_categoricals(df, cat_cols, drop_first=False)
        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
        logger.info(f"One-hot encoded {len(cat_cols)} columns")

    # ── 6. Apply saved scalers + align to training column set ─────────────────
    logger.info("── Step 5: Preprocessing ──")
    X = preprocess_input(df, scalers, feature_names)

    # ── 7. Predict ────────────────────────────────────────────────────────────
    logger.info("── Step 6: Predicting ──")
    threshold = config.get("model", {}).get("prediction_threshold", 0.5)
    original["delay_probability"] = predict_proba(model, X).values
    original["predicted_delay"]   = predict(model, X, threshold=threshold).values

    logger.info(
        f"Predicted delay rate: {original['predicted_delay'].mean():.1%}"
    )

    # ── 8. Optionally save ────────────────────────────────────────────────────
    if output_path:
        ensure_dir(output_path.rsplit("/", 1)[0])
        save_dataframe(original, output_path)
        logger.info(f"Predictions saved → {output_path}")

    logger.info("Inference pipeline complete.")
    return original
