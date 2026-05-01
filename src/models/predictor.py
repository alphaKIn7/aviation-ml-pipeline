# ──────────────────────────────────────────────
# predictor.py — Model Inference / Prediction
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Load a trained model and its preprocessing pipeline from disk,
#   accept new raw data, apply the same transformations, and return
#   predictions. This is what runs in production.
#
# WHY SEPARATE FROM TRAINING?
#   Training happens once (or periodically). Inference happens
#   thousands of times per second in production. They have completely
#   different performance requirements and code paths.
#
# FUNCTIONS TO BUILD HERE:
#   - load_model_artifacts(model_path, pipeline_path) → Model + pipeline
#   - preprocess_input(raw_data, pipeline)            → Transform new data
#   - predict(model, processed_data)                  → Return predictions
#   - predict_proba(model, processed_data)            → Return probabilities
#   - predict_batch(model, pipeline, data_path)       → Batch inference
#
# INDUSTRY PATTERN:
#   The preprocessing pipeline saved during training MUST be the same
#   one used during inference. Never refit scalers/encoders on new data.
# ──────────────────────────────────────────────
