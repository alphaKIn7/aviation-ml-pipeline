from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from src.utils.logger import get_logger

logger = get_logger(__name__)


def delay_capture_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    What fraction of actual delays did the model catch?

    This is recall for the delayed class. In aviation, missing a delay
    (false negative) is typically worse than a false alarm (false positive)
    — passengers prefer to be warned of a possible delay than to be
    surprised by one. Use this metric to track that specific risk.

    Formula: true_positives / (true_positives + false_negatives)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    actual_delays = (y_true == 1).sum()

    if actual_delays == 0:
        return 0.0

    score = float(true_positives / actual_delays)
    logger.info(f"Delay capture rate: {score:.4f}")
    return score


def on_time_performance(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    What fraction of flights were correctly classified overall?

    Aviation industry KPI language for accuracy. Same formula, but the name
    makes the metric immediately legible to non-technical stakeholders.

    Formula: correct predictions / total predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    score = float((y_true == y_pred).mean())
    logger.info(f"On-time performance (accuracy): {score:.4f}")
    return score


def false_alarm_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    What fraction of on-time flights were incorrectly flagged as delayed?

    False alarms erode passenger trust and cause unnecessary disruption.
    There is always a tradeoff: tuning the model to catch more delays
    (higher delay_capture_rate) will also raise this rate.

    Formula: false_positives / (false_positives + true_negatives)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    actual_on_time = (y_true == 0).sum()

    if actual_on_time == 0:
        return 0.0

    score = float(false_positives / actual_on_time)
    logger.info(f"False alarm rate: {score:.4f}")
    return score


# Sklearn-compatible scorer — pass this to cross_val_score or GridSearchCV
# when you want to optimise for catching delays rather than raw accuracy.
delay_capture_scorer = make_scorer(delay_capture_rate)
