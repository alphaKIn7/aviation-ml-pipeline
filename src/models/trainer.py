from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

from src.utils.io_handler import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Registry of supported models — add new ones here without touching any
# other code. The key matches the "model.name" value in train_config.yaml.
SUPPORTED_MODELS: dict[str, type] = {
    "xgboost":       XGBClassifier,
    "lightgbm":      LGBMClassifier,
    "random_forest": RandomForestClassifier,
    "logistic":      LogisticRegression,
}

# Maps human-readable metric names (from config) to sklearn scoring strings.
METRIC_MAP: dict[str, str] = {
    "accuracy": "accuracy",
    "f1":       "f1",
    "roc_auc":  "roc_auc",
    "rmse":     "neg_root_mean_squared_error",
    "mae":      "neg_mean_absolute_error",
}


def get_model(model_name: str, params: dict) -> Any:
    """
    Model factory — returns a configured model object from a string name.

    This is the key production pattern: the pipeline never hardcodes a model
    class. It reads "model.name" from the config and calls this function.
    To swap XGBoost for LightGBM, you change one line in train_config.yaml —
    no Python code changes needed.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported options: {list(SUPPORTED_MODELS.keys())}"
        )
    model = SUPPORTED_MODELS[model_name](**params)
    logger.info(f"Initialized '{model_name}' with params: {params}")
    return model


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Any:
    """
    Fit the model on training data and return the fitted model.

    XGBoost supports early stopping: if a validation set is provided, training
    halts automatically when the validation score stops improving for
    early_stopping_rounds consecutive rounds. This prevents overfitting without
    having to guess the right number of trees upfront.

    All other models (LightGBM, RandomForest, Logistic) ignore the validation
    set and train for the full number of estimators defined in their params.
    """
    start = time.time()

    if isinstance(model, XGBClassifier) and X_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.1f}s")
    return model


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = 5,
    metric: str = "roc_auc",
) -> dict:
    """
    Evaluate model quality using K-fold cross-validation.

    Why StratifiedKFold instead of plain KFold:
    The dataset is imbalanced — only ~16% of flights are delayed. A regular
    KFold split might accidentally put most delayed flights into one fold,
    making some folds trivially easy and others unfairly hard. StratifiedKFold
    guarantees every fold has the same delayed/on-time ratio as the full dataset.

    Why cross-validation at all:
    A single train/test split gives you one score that depends heavily on which
    specific rows ended up in the test set. K-fold gives you K scores and you
    report the mean ± std — a much more honest picture of real performance.

    Returns a dict so the pipeline layer can log or store the results however it
    needs to (MLflow, a CSV, a JSON report, etc.) without this function caring.
    """
    scoring = METRIC_MAP.get(metric, metric)
    # StratifiedKFold shuffles before splitting so row order doesn't bias folds.
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    start = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - start

    results = {
        "metric": metric,
        "scores": scores.tolist(),
        "mean":   float(scores.mean()),
        "std":    float(scores.std()),
    }
    logger.info(
        f"CV ({folds} folds, {metric}): "
        f"mean={results['mean']:.4f} ± {results['std']:.4f}  [{elapsed:.1f}s]"
    )
    return results


def tune_hyperparameters(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    folds: int = 5,
    metric: str = "roc_auc",
    n_iter: int = 20,
) -> tuple[Any, dict]:
    """
    Search for the best hyperparameters using RandomizedSearchCV.

    Why Randomized and not Grid search:
    GridSearchCV tries every single combination in param_grid. If you have
    3 parameters with 5 values each, that's 125 combinations × 5 folds =
    625 model fits. RandomizedSearchCV samples n_iter random combinations
    instead, so 20 iterations × 5 folds = 100 fits — far faster with nearly
    the same quality of result on large grids.

    Returns both the best fitted model and its best params, so the caller
    can log the params to an experiment tracker or save them to the config.
    """
    scoring = METRIC_MAP.get(metric, metric)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True,
    )

    logger.info(
        f"Hyperparameter search starting: "
        f"{n_iter} iterations, {folds} folds, metric={metric}"
    )
    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    logger.info(
        f"Best {metric}: {search.best_score_:.4f} "
        f"| Params: {search.best_params_} [{elapsed:.1f}s]"
    )
    return search.best_estimator_, search.best_params_


def save_model(model: Any, path: str) -> None:
    """
    Serialize the trained model to disk using joblib.

    Why joblib over pickle:
    Pickle is Python's built-in serializer and works for anything. joblib is a
    thin wrapper that adds one important optimization: it memory-maps large numpy
    arrays instead of copying them into the file byte-by-byte. A Random Forest
    with 200 trees can have millions of float values — joblib saves and loads
    these 3-10x faster than pickle and produces smaller files.
    """
    ensure_dir(str(Path(path).parent))
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str) -> Any:
    """Load a previously saved model from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No model file found at: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model
