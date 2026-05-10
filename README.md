# Aviation ML Pipeline — Flight Delay Prediction

An end-to-end machine learning pipeline that predicts US domestic flight delays
using 2024 BTS (Bureau of Transportation Statistics) flight data.
Built with production-grade practices: modular architecture, config-driven
everything, unit tested, and trained on 6.5 million flights.

---

## Results

| Metric | Value |
|---|---|
| Dataset | 7M US domestic flights (2024) |
| Model | XGBoost with native categorical encoding |
| ROC-AUC | 0.70 |
| Delay capture rate | 89.7% (catches 9 in 10 actual delays) |
| Prediction threshold | 0.35 |
| Training time | ~2.5 minutes on full dataset |
| Test suite | 58 unit tests, all passing |

---

## Project Structure

```
aviation-ml-pipeline/
│
├── configs/
│   ├── train_config.yaml        # Single source of truth for all training params
│   └── inference_config.yaml   # Model paths and serving configuration
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Load CSV / Parquet, combine directories
│   │   ├── validation.py        # Schema checks, null thresholds, duplicate detection
│   │   └── preprocessing.py     # Missing values, encoding, scaling, train/test split
│   │
│   ├── features/
│   │   ├── build_features.py    # Feature engineering (temporal, aggregation, route)
│   │   └── select_features.py   # SelectKBest, RFE, feature importance
│   │
│   ├── models/
│   │   ├── trainer.py           # Model factory, cross-validation, save/load
│   │   ├── evaluator.py         # Metrics, confusion matrix, ROC curve, reports
│   │   └── predictor.py         # Load artifacts, preprocess input, batch inference
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py  # Orchestrates full training workflow (10 steps)
│   │   └── inference_pipeline.py # Orchestrates batch prediction workflow
│   │
│   └── utils/
│       ├── logger.py            # Centralised logging
│       ├── config.py            # Config loading and validation
│       ├── io_handler.py        # File I/O (YAML, CSV, Parquet, Pickle)
│       └── metrics.py           # Aviation business metrics (delay capture rate, etc.)
│
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_data/               # Tests for ingestion, validation, preprocessing
│   ├── test_features/           # Tests for feature engineering
│   └── test_models/             # Tests for training, save/load, cross-validation
│
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis on flight data
│
├── scripts/
│   ├── train.py                 # CLI entry point for training
│   └── predict.py               # CLI entry point for batch inference
│
├── artifacts/                   # Saved models, plots, reports (git-ignored)
├── data/                        # Raw and processed data (git-ignored)
├── docker/                      # Dockerfile for reproducible environment
├── requirements.txt             # Production dependencies
└── requirements-dev.txt         # Development dependencies (pytest, etc.)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/alphaKIn7/aviation-ml-pipeline.git
cd aviation-ml-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> **macOS note:** XGBoost requires OpenMP. If you see a `libomp.dylib` error:
> ```bash
> brew install libomp
> ```

### 2. Add your data

Download the 2024 BTS On-Time Performance dataset and place it at:
```
data/raw/flight_data_2024.csv
```

### 3. Train the model

```bash
python scripts/train.py --config configs/train_config.yaml
```

Or directly in Python:

```python
from src.pipelines.training_pipeline import run_training_pipeline

metrics = run_training_pipeline("configs/train_config.yaml")
print(metrics)
```

### 4. Run batch inference

```python
from src.pipelines.inference_pipeline import run_inference_pipeline

results = run_inference_pipeline(
    config_path="configs/inference_config.yaml",
    input_data="path/to/new_flights.csv",
    output_path="predictions.csv",
)
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Pipeline Steps

The training pipeline runs 10 steps automatically:

| Step | What happens |
|---|---|
| 1. Ingest | Load CSV, filter cancelled flights, drop rows with unknown arrival delay |
| 2. Validate | Check schema, null rates, duplicates |
| 3. Feature engineering | Create target, temporal features, route, aggregation features |
| 4. Preprocess | Handle nulls, remove outliers (IQR), convert binary flags to int |
| 5. Encode | Tree models: native `category` dtype. Linear models: one-hot encoding |
| 6. Scale | StandardScaler / MinMaxScaler on numerical columns |
| 7. Split | 80% train / 20% test (stratified by target) |
| 8. Feature selection | SelectKBest with ANOVA F-test, top 20 features |
| 9. Cross-validate | 5-fold stratified CV, metric = ROC-AUC |
| 10. Train + Evaluate | Train on full train set, evaluate on held-out test set |
| Save | Model, scalers, feature names, encoding info, plots, JSON report |

---

## Features

### Engineered features

| Feature | Type | Description |
|---|---|---|
| `dep_hour` | numerical | Scheduled departure hour (0–23) |
| `arr_hour` | numerical | Scheduled arrival hour (0–23) |
| `distance` | numerical | Flight distance in miles |
| `day_of_week` | numerical | Day of week (1=Monday, 7=Sunday) |
| `month` | numerical | Month (1–12) |
| `crs_elapsed_time` | numerical | Scheduled flight duration in minutes |
| `carrier_avg_delay` | numerical | Historical average delay per airline |
| `origin_avg_delay` | numerical | Historical average delay at origin airport |
| `route_avg_delay` | numerical | Historical average delay for this route |
| `op_unique_carrier` | categorical | Airline code |
| `origin` | categorical | Origin airport code |
| `dest` | categorical | Destination airport code |
| `route` | categorical | Origin + destination (e.g. `JFK_LAX`) |
| `dep_hour_bin` | categorical | Time-of-day bucket (morning/afternoon/evening/night) |
| `is_ops_peak` | binary | 1 if departure is during high-traffic hours (6am–9pm) |
| `is_delay_peak` | binary | 1 if departure is during historically high-delay hours (1pm–midnight) |
| `is_weekend` | binary | 1 if Saturday or Sunday |
| `is_holiday_month` | binary | 1 if Jan, May, Jun, Jul, Aug, or Dec |

### Target variable

`is_delayed` — 1 if arrival delay > 15 minutes, 0 otherwise.
Approximately 20% of flights in the 2024 dataset are delayed by this definition.

---

## Model Configuration

All parameters live in `configs/train_config.yaml`. Key settings:

```yaml
model:
  name: "xgboost"           # Options: xgboost, lightgbm, random_forest, logistic
  hyperparameters:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    scale_pos_weight: 4     # Penalise missed delays 4x over false alarms
    tree_method: "hist"     # Histogram algorithm — fast on large datasets

training:
  metric: "roc_auc"
  cross_validation_folds: 5
  prediction_threshold: 0.35  # Lower than 0.5 to catch more delays
```

### Encoding strategy

The pipeline automatically picks the right encoding based on model type:

- **XGBoost / LightGBM** — columns are marked as `category` dtype. No dummy expansion. A dataset with 350+ airports stays at ~19 features instead of exploding to 700+.
- **Logistic Regression** — one-hot encoding via `pd.get_dummies`.

### Threshold tuning

The prediction threshold can be adjusted without retraining. Lower threshold = catches more delays but more false alarms.

| Threshold | Delay capture | False alarm rate |
|---|---|---|
| 0.25 | 97% | 85% |
| 0.35 | 89% | 70% |
| 0.45 | 74% | 46% |
| 0.50 | 66% | 36% |
| 0.60 | 44% | 18% |

---

## Saved Artifacts

After training, four files are saved to `artifacts/models/`:

| File | Purpose |
|---|---|
| `best_model.pkl` | Trained XGBoost model |
| `scalers.pkl` | Fitted scaler objects — used at inference with `.transform()`, never `.fit_transform()` |
| `feature_names.pkl` | Exact ordered list of feature columns the model was trained on |
| `model_info.pkl` | Encoding type and categorical columns — ensures inference uses the same encoding as training |

Plots are saved to `artifacts/plots/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Visualisation | matplotlib, seaborn |
| Configuration | PyYAML, Pydantic |
| Serialisation | joblib |
| Testing | pytest |
| Python | 3.9+ |

---

## License

MIT
