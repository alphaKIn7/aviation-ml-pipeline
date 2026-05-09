import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> tuple[pd.DataFrame, list[str]]:
    num_cols = df.select_dtypes(include="number").columns
    variances = df[num_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()

    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        logger.warning(f"Removed {len(low_var_cols)} low-variance columns: {low_var_cols}")
    else:
        logger.info("No low-variance columns found")

    return df, low_var_cols


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    num_cols = df.select_dtypes(include="number").columns
    corr_matrix = df[num_cols].corr().abs()

    # Look at only the upper triangle of the matrix to avoid counting pairs twice
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # For each pair above threshold, drop the second column
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)
        logger.warning(f"Removed {len(to_drop)} highly correlated columns: {to_drop}")
    else:
        logger.info(f"No columns with correlation above {threshold} found")

    return df, to_drop


def select_k_best(
    X: pd.DataFrame, y: pd.Series, k: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError("No numerical columns found for SelectKBest")

    k = min(k, len(num_cols))
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X[num_cols], y)

    selected = [num_cols[i] for i in selector.get_support(indices=True)]
    logger.info(f"SelectKBest selected {len(selected)} features: {selected}")
    return X[selected], selected


def recursive_feature_elimination(
    X: pd.DataFrame, y: pd.Series, n_features: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError("No numerical columns found for RFE")

    n_features = min(n_features, len(num_cols))

    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X[num_cols], y)

    selected = [num_cols[i] for i, s in enumerate(rfe.support_) if s]
    logger.info(f"RFE selected {len(selected)} features: {selected}")
    return X[selected], selected


def get_feature_importances(model, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute. Use a tree-based model.")

    importance_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 features by importance:\n{importance_df.head().to_string()}")
    return importance_df
