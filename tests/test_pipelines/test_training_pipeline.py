# ──────────────────────────────────────────────
# test_training_pipeline.py — Integration Tests
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Integration tests run the ENTIRE pipeline end-to-end
#   with a small synthetic dataset. They verify that all
#   modules work together correctly.
#
# TESTS TO WRITE:
#   - test_full_training_pipeline_runs_without_error
#   - test_pipeline_produces_model_artifact
#   - test_pipeline_produces_evaluation_report
#   - test_pipeline_with_invalid_config_raises_error
#
# INDUSTRY PATTERN:
#   Integration tests are slower than unit tests but catch
#   interface mismatches between modules. Run them in CI/CD.
# ──────────────────────────────────────────────
