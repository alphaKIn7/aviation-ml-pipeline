from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io_handler import ensure_dir, load_pickle, load_dataframe, save_dataframe
from src.utils.logger import get_logger
from src.models.trainer import load_model

logger = get_logger(__name__)


def load_model_artifacts(
    model_path: str,
    scalers_path: str,
    feature_names_path: str,
) -> tuple[Any, dict, list[str]]:
    """
    Load all three artifacts that were saved together during training.

    These three always travel as a group:
    - model:         the fitted XGBoost / RF / etc. object
    - scalers:       the fitted scaler objects (StandardScaler, MinMaxScaler)
    - feature_names: the exact ordered list of columns the model was trained on

    Why save scalers separately from the model?
    The model's learned weights (split thresholds, leaf values) are calibrated
    to the SCALED version of the data. If you scale inference data differently
    — even slightly — the predictions become meaningless. Saving and reloading
    the exact same scaler objects guarantees the same transformation every time.
    """
    model = load_model(model_path)
    scalers = load_pickle(scalers_path)
    feature_names = load_pickle(feature_names_path)

    logger.info(
        f"Artifacts loaded — model: {type(model).__name__}, "
        f"scalers: {len(scalers)}, features: {len(feature_names)}"
    )
    return model, scalers, feature_names


def preprocess_input(
    df: pd.DataFrame,
    scalers: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Apply the saved preprocessing transformations to new inference data.

    Two critical rules:
    1.  Call .transform() on saved scalers — NEVER .fit_transform().
        fit_transform would compute new mean/std from the inference data,
        which is different from training data statistics, breaking the model.

    2.  After encoding, reindex to match training column names exactly.
        At inference you may see a carrier or airport that wasn't in training —
        those get no column (dropped). You may be missing carriers that were
        in training — those get a column filled with 0. Either way, the
        DataFrame ends up with exactly the same shape the model expects.
    """
    df = df.copy()

    # Apply each saved scaler (transform only, not fit)
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
        else:
            logger.warning(f"Scaler column '{col}' not in input — filling with 0")
            df[col] = 0.0

    # Align columns to the exact set seen during training
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Return binary class predictions: 0 = on time, 1 = delayed.

    Uses threshold on predict_proba when available, so the same cutoff
    configured during training is applied consistently at inference.
    Default of 0.5 matches sklearn convention; pass 0.3 to catch more delays.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
    else:
        preds = model.predict(X)

    delay_rate = preds.mean()
    logger.info(
        f"Predicted {len(preds):,} flights "
        f"(threshold={threshold}) — delay rate: {delay_rate:.1%}"
    )
    return pd.Series(preds, index=X.index, name="predicted_delay")


def predict_proba(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Return the probability of delay (0.0 to 1.0) for each flight.

    This is more actionable than a binary label. A probability of 0.51 and
    0.97 both classify as "delayed" but you should react very differently:
    - 0.51 → borderline, maybe monitor
    - 0.97 → high confidence, alert the passenger, pre-position crew

    Airlines use probability thresholds (not just 0.5) tuned to their own
    cost models — missing a delay costs X, a false alarm costs Y.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{type(model).__name__} does not support predict_proba")

    proba = model.predict_proba(X)[:, 1]  # column 1 = P(delayed)
    logger.info(
        f"Delay probability — mean: {proba.mean():.3f}, "
        f"max: {proba.max():.3f}, min: {proba.min():.3f}"
    )
    return pd.Series(proba, index=X.index, name="delay_probability")


def predict_batch(
    model_path: str,
    scalers_path: str,
    feature_names_path: str,
    data_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    End-to-end batch inference: load artifacts, load data, predict, save results.

    This is the function a scheduled job or pipeline step calls.
    In production you'd replace data_path with a database query or a
    message queue consumer — but the rest of the logic stays the same.
    """
    model, scalers, feature_names = load_model_artifacts(
        model_path, scalers_path, feature_names_path
    )

    raw = load_dataframe(data_path)
    logger.info(f"Batch input: {len(raw):,} rows from {data_path}")

    X = preprocess_input(raw, scalers, feature_names)

    results = raw.copy()
    results["delay_probability"] = predict_proba(model, X).values
    results["predicted_delay"]   = predict(model, X, threshold=0.5).values

    ensure_dir(str(Path(output_path).parent))
    save_dataframe(results, output_path)
    logger.info(f"Batch predictions saved → {output_path}")
    return results
