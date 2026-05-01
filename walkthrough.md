# ML Pipeline Project Structure — Walkthrough

Your end-to-end ML pipeline is set up with **35+ files** across **6 layers**. Every single file has detailed comments explaining its purpose — no code imports, just architecture.

## The Big Picture: Data Flow

```mermaid
graph LR
    A["📥 Raw Data"] --> B["✅ Validation"]
    B --> C["🧹 Preprocessing"]
    C --> D["⚙️ Feature Engineering"]
    D --> E["🎯 Feature Selection"]
    E --> F["🤖 Model Training"]
    F --> G["📊 Evaluation"]
    G --> H["💾 Save Artifacts"]
    H --> I["🚀 Inference API"]
```

---

## Layer-by-Layer Breakdown

### 1. `configs/` — Configuration Layer
| File | Purpose |
|------|---------|
| [train_config.yaml](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/configs/train_config.yaml) | All training hyperparameters, paths, feature lists |
| [inference_config.yaml](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/configs/inference_config.yaml) | Model serving configuration |

> [!IMPORTANT]
> **No magic numbers in code.** Every tunable value (learning rate, test split, column names) lives in YAML. Change behavior by editing config, not code.

---

### 2. `data/` — Data Storage (Git-Ignored)
| Folder | Purpose |
|--------|---------|
| `data/raw/` | Original, untouched data files |
| `data/interim/` | Partially processed data (intermediate steps) |
| `data/processed/` | Final, model-ready data |

> [!WARNING]
> Data is **never committed to git**. It's too large and sensitive. Use `.gitkeep` files to preserve the folder structure.

---

### 3. `src/` — Core Source Code (The Heart)

This is structured as **4 sub-packages**, each responsible for one concern:

#### 3a. `src/data/` — Data Layer
| File | Purpose |
|------|---------|
| [ingestion.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/data/ingestion.py) | Load raw data from CSV/DB/API — returns unmodified DataFrame |
| [validation.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/data/validation.py) | Schema checks, null gates, duplicate detection — **fail fast** |
| [preprocessing.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/data/preprocessing.py) | Cleaning, encoding, scaling, imputation, train/test split |

#### 3b. `src/features/` — Feature Layer
| File | Purpose |
|------|---------|
| [build_features.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/features/build_features.py) | Create new features (temporal, interaction, domain-specific) |
| [select_features.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/features/select_features.py) | Pick best features (correlation, RFE, importance-based) |

#### 3c. `src/models/` — Model Layer
| File | Purpose |
|------|---------|
| [trainer.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/models/trainer.py) | Train, cross-validate, tune, save models |
| [evaluator.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/models/evaluator.py) | Metrics, confusion matrix, ROC curve, reports |
| [predictor.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/models/predictor.py) | Load saved model + pipeline → make predictions |

#### 3d. `src/pipelines/` — Orchestration Layer
| File | Purpose |
|------|---------|
| [training_pipeline.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/pipelines/training_pipeline.py) | Ties ALL modules together: data → features → train → evaluate |
| [inference_pipeline.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/pipelines/inference_pipeline.py) | Production prediction flow: load model → preprocess → predict |

#### 3e. `src/utils/` — Shared Utilities
| File | Purpose |
|------|---------|
| [logger.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/utils/logger.py) | Centralized logging (never use `print()` in production) |
| [io_handler.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/utils/io_handler.py) | File I/O: YAML, pickle, CSV, parquet |
| [config.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/utils/config.py) | Config loading and validation |
| [metrics.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/src/utils/metrics.py) | Custom domain-specific metrics |

---

### 4. `scripts/` — CLI Entry Points
| File | Purpose |
|------|---------|
| [train.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/scripts/train.py) | `python scripts/train.py --config configs/train_config.yaml` |
| [predict.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/scripts/predict.py) | `python scripts/predict.py --config ... --input data.csv` |

> [!TIP]
> **Scripts contain ZERO business logic.** They only parse arguments and call pipeline functions. This keeps `src/` importable from notebooks, APIs, and tests without side effects.

---

### 5. `tests/` — Test Suite
| File | Tests For |
|------|-----------|
| [test_ingestion.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_data/test_ingestion.py) | Data loading edge cases |
| [test_preprocessing.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_data/test_preprocessing.py) | Cleaning & transformation correctness |
| [test_validation.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_data/test_validation.py) | Schema and quality gate logic |
| [test_build_features.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_features/test_build_features.py) | Feature engineering correctness |
| [test_trainer.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_models/test_trainer.py) | Model training & serialization |
| [test_training_pipeline.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/tests/test_pipelines/test_training_pipeline.py) | End-to-end integration tests |

---

### 6. `notebooks/` — Exploration (Not Production)
| File | Purpose |
|------|---------|
| [01_eda.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/notebooks/01_eda.py) | Exploratory Data Analysis |
| [02_feature_exploration.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/notebooks/02_feature_exploration.py) | Experiment with feature ideas |
| [03_model_experiments.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/notebooks/03_model_experiments.py) | Compare different models |

---

### 7. Infrastructure Files
| File | Purpose |
|------|---------|
| [Dockerfile](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/docker/Dockerfile) | Reproducible containerized environment |
| [ci.yml](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/.github/workflows/ci.yml) | GitHub Actions: lint → test → typecheck on every push |
| [requirements.txt](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/requirements.txt) | Production dependencies (pinned versions) |
| [requirements-dev.txt](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/requirements-dev.txt) | Dev-only dependencies (pytest, ruff, mypy) |
| [setup.py](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/setup.py) | Makes `src/` installable via `pip install -e .` |
| [.gitignore](file:///Volumes/MainDrive/aviation-data-analytics-end-2-end-pipeline/.gitignore) | Ignores data, artifacts, envs, IDE files |

---

## Key Industry Principles Applied

| Principle | How It's Applied |
|-----------|------------------|
| **Separation of Concerns** | Each file has ONE job. Ingestion ≠ preprocessing ≠ training. |
| **Config-Driven** | All parameters in YAML. Change behavior without changing code. |
| **Reproducibility** | Docker + pinned deps + config = same results anywhere. |
| **Testability** | Test structure mirrors `src/` structure. Unit + integration tests. |
| **No Data Leakage** | Preprocessing pipelines fitted on train, applied to test. |
| **Notebooks ≠ Production** | Notebooks for exploration only. Production code in `src/`. |
| **Fail Fast** | Validation layer catches bad data before it wastes GPU time. |
| **Logging > Printing** | Structured logging with levels, timestamps, and file output. |

## Next Steps

When you're ready, we can start filling in the actual code — I'd suggest this order:
1. **`src/utils/`** — Logger, config, IO (foundation everything else depends on)
2. **`src/data/`** — Ingestion → validation → preprocessing
3. **`notebooks/01_eda.py`** — Explore your actual aviation dataset
4. **`src/features/`** — Feature engineering based on EDA insights
5. **`src/models/`** — Trainer → evaluator → predictor
6. **`src/pipelines/`** — Wire everything together
7. **`tests/`** — Write tests as you go
