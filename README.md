# Aviation Data Analytics — End-to-End ML Pipeline

An end-to-end machine learning pipeline for aviation data analytics, following
industry best practices for reproducibility, modularity, and deployment readiness.

## Project Structure

```
├── configs/              # All configuration (hyperparams, paths, feature lists)
├── data/                 # Raw, interim, and processed data (git-ignored)
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/                  # Core source code (the importable Python package)
│   ├── data/             # Data ingestion, validation, and preprocessing
│   ├── features/         # Feature engineering and selection
│   ├── models/           # Model training, evaluation, and prediction
│   ├── pipelines/        # Orchestration — ties data → features → model
│   └── utils/            # Shared helpers (logging, IO, metrics)
├── tests/                # Unit and integration tests
├── artifacts/            # Saved models, plots, reports (git-ignored)
├── scripts/              # One-off or CLI scripts (train.py, predict.py)
├── docker/               # Dockerfiles for reproducible environments
├── .github/workflows/    # CI/CD pipeline definitions
└── docs/                 # Project documentation
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python scripts/train.py --config configs/train_config.yaml
```

## License

MIT
