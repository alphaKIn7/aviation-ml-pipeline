import pytest

from src.models.trainer import (
    cross_validate_model,
    get_model,
    load_model,
    save_model,
    train_model,
)


# ── get_model (factory) ───────────────────────────────────────────────────────

def test_get_model_returns_xgboost():
    model = get_model("xgboost", {"n_estimators": 5, "random_state": 42})
    assert type(model).__name__ == "XGBClassifier"


def test_get_model_returns_random_forest():
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    assert type(model).__name__ == "RandomForestClassifier"


def test_get_model_returns_lightgbm():
    model = get_model("lightgbm", {"n_estimators": 5, "random_state": 42})
    assert type(model).__name__ == "LGBMClassifier"


def test_get_model_returns_logistic():
    model = get_model("logistic", {"random_state": 42})
    assert type(model).__name__ == "LogisticRegression"


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model", {})


# ── train_model ───────────────────────────────────────────────────────────────

def test_train_model_returns_fitted_model(sample_X, sample_y):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    fitted = train_model(model, sample_X, sample_y)
    preds = fitted.predict(sample_X)
    assert len(preds) == len(sample_y)


def test_train_model_predictions_are_binary(sample_X, sample_y):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    fitted = train_model(model, sample_X, sample_y)
    preds = fitted.predict(sample_X)
    assert set(preds).issubset({0, 1})


# ── cross_validate_model ──────────────────────────────────────────────────────

def test_cross_validate_returns_expected_keys(sample_X, sample_y):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    results = cross_validate_model(model, sample_X, sample_y, folds=3, metric="accuracy")
    assert "mean" in results
    assert "std" in results
    assert "scores" in results
    assert "metric" in results


def test_cross_validate_correct_number_of_folds(sample_X, sample_y):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    results = cross_validate_model(model, sample_X, sample_y, folds=4, metric="accuracy")
    assert len(results["scores"]) == 4


def test_cross_validate_mean_is_between_0_and_1(sample_X, sample_y):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    results = cross_validate_model(model, sample_X, sample_y, folds=3, metric="accuracy")
    assert 0.0 <= results["mean"] <= 1.0


# ── save_model / load_model ───────────────────────────────────────────────────

def test_save_and_load_roundtrip_produces_same_predictions(sample_X, sample_y, tmp_path):
    model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})
    fitted = train_model(model, sample_X, sample_y)

    model_path = str(tmp_path / "model.pkl")
    save_model(fitted, model_path)
    loaded = load_model(model_path)

    assert list(fitted.predict(sample_X)) == list(loaded.predict(sample_X))


def test_load_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model("does/not/exist.pkl")
