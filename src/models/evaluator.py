from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.io_handler import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import delay_capture_rate, false_alarm_rate, on_time_performance

logger = get_logger(__name__)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[pd.Series] = None,
) -> dict:
    """
    Compute all evaluation metrics and return them as a plain dictionary.

    Returning a dict (not printing) keeps this function decoupled from
    where the results go. The pipeline layer decides whether to log to
    MLflow, save to JSON, print to console, or all three — evaluator
    does not need to know or care.
    """
    metrics = {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "precision":           float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":              float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                  float(f1_score(y_true, y_pred, zero_division=0)),
        "delay_capture_rate":  delay_capture_rate(y_true, y_pred),
        "false_alarm_rate":    false_alarm_rate(y_true, y_pred),
        "on_time_performance": on_time_performance(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    save_path: str,
) -> None:
    """
    Heatmap of true vs predicted classes.

    The four cells tell the whole story:
    - Top-left (TN): correctly predicted on-time
    - Top-right (FP): false alarms — predicted delayed, actually on-time
    - Bottom-left (FN): missed delays — predicted on-time, actually delayed
    - Bottom-right (TP): correctly caught delays
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["On Time", "Delayed"],
        yticklabels=["On Time", "Delayed"],
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14},
    )
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_title("Confusion Matrix — Flight Delay Prediction", fontweight="bold", pad=12)
    ax.text(
        0.5, -0.14,
        f"True Negatives (correct on-time): {tn:,}   "
        f"False Positives (false alarms): {fp:,}\n"
        f"False Negatives (missed delays): {fn:,}   "
        f"True Positives (caught delays): {tp:,}",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444",
    )

    ensure_dir(str(Path(save_path).parent))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(
    y_true: pd.Series,
    y_proba: pd.Series,
    save_path: str,
) -> None:
    """
    ROC curve with AUC score.

    The X-axis is the false alarm rate; the Y-axis is the delay capture rate.
    A perfect model hugs the top-left corner (catches all delays, zero false
    alarms). The diagonal dashed line is a random classifier (AUC = 0.5).
    Your model should be well above it.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1a4691", lw=2, label=f"Model  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1.5, linestyle="--",
            label="Random baseline  (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1a4691")

    ax.set_xlabel("False Positive Rate  (fraction of on-time flights flagged)", fontsize=10)
    ax.set_ylabel("True Positive Rate  (fraction of delays caught)", fontsize=10)
    ax.set_title("ROC Curve — Flight Delay Prediction", fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)

    ensure_dir(str(Path(save_path).parent))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    save_path: str,
    top_n: int = 20,
) -> None:
    """Bar chart of the top N most important features from a tree-based model."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_ — skipping plot")
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names,
                      "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")  # flip so the highest bar is at the top
    )

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    colors = [
        "#1a4691" if i == len(importance_df) - 1 else "#5b87cc"
        for i in range(len(importance_df))
    ]
    importance_df.plot(kind="barh", x="feature", y="importance",
                       ax=ax, color=colors, legend=False, edgecolor="white")

    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(f"Top {min(top_n, len(feature_names))} Feature Importances",
                 fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    ensure_dir(str(Path(save_path).parent))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved → {save_path}")


def generate_evaluation_report(results: dict, save_path: str) -> None:
    """Persist evaluation results as a JSON file for tracking and comparison."""
    ensure_dir(str(Path(save_path).parent))
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation report saved → {save_path}")
