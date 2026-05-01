# ──────────────────────────────────────────────
# src/pipelines/ — Orchestration Layer
# ──────────────────────────────────────────────
# Pipelines are the GLUE. They don't contain business logic —
# they call data, features, and model modules in the right order.
#
# Think of a pipeline as a recipe:
#   1. Load data         (src.data.ingestion)
#   2. Validate data     (src.data.validation)
#   3. Preprocess data   (src.data.preprocessing)
#   4. Engineer features (src.features.build_features)
#   5. Select features   (src.features.select_features)
#   6. Train model       (src.models.trainer)
#   7. Evaluate model    (src.models.evaluator)
#   8. Save artifacts    (src.utils.io_handler)
#
# Files:
#   training_pipeline.py   → Full training workflow
#   inference_pipeline.py  → Full prediction workflow
