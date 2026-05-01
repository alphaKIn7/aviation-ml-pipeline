# ──────────────────────────────────────────────
# training_pipeline.py — Training Orchestration
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Orchestrate the ENTIRE training workflow from raw data to a
#   saved, evaluated model. This is the file that ties everything
#   together — it's the "main()" of training.
#
# FLOW:
#   ┌─────────────┐
#   │  Load Config │
#   └──────┬──────┘
#          ▼
#   ┌─────────────┐
#   │ Ingest Data  │  ← src.data.ingestion
#   └──────┬──────┘
#          ▼
#   ┌─────────────┐
#   │ Validate     │  ← src.data.validation
#   └──────┬──────┘
#          ▼
#   ┌─────────────┐
#   │ Preprocess   │  ← src.data.preprocessing
#   └──────┬──────┘
#          ▼
#   ┌──────────────┐
#   │ Engineer Feat │  ← src.features.build_features
#   └──────┬───────┘
#          ▼
#   ┌──────────────┐
#   │ Select Feat   │  ← src.features.select_features
#   └──────┬───────┘
#          ▼
#   ┌─────────────┐
#   │ Train Model  │  ← src.models.trainer
#   └──────┬──────┘
#          ▼
#   ┌─────────────┐
#   │ Evaluate     │  ← src.models.evaluator
#   └──────┬──────┘
#          ▼
#   ┌──────────────┐
#   │ Save Artifact │  ← src.utils.io_handler
#   └──────────────┘
#
# FUNCTIONS TO BUILD HERE:
#   - run_training_pipeline(config_path) → Executes the full flow above
#
# INDUSTRY PATTERN:
#   The pipeline should be idempotent and config-driven. Running it
#   with the same config + data should produce the same model.
#   Log every step. If a step fails, the error message should tell
#   you EXACTLY what went wrong and where.
# ──────────────────────────────────────────────
