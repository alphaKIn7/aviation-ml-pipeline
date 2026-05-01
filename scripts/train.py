# ──────────────────────────────────────────────
# train.py — CLI Entry Point for Training
# ──────────────────────────────────────────────
#
# PURPOSE:
#   This is the file you RUN from the terminal:
#     python scripts/train.py --config configs/train_config.yaml
#
#   It parses command-line arguments and calls the training pipeline.
#   It should contain ZERO business logic — only argument parsing
#   and a single call to run_training_pipeline().
#
# WHY A SEPARATE SCRIPTS/ FOLDER?
#   src/ = importable library (no side effects)
#   scripts/ = executable entry points (has a main() function)
#
#   This separation means your src/ code can be imported by:
#   - Other scripts
#   - Jupyter notebooks
#   - Web APIs
#   - Unit tests
#   ...without accidentally triggering training.
#
# STRUCTURE TO BUILD:
#   if __name__ == "__main__":
#       parser = argparse.ArgumentParser()
#       parser.add_argument("--config", required=True)
#       args = parser.parse_args()
#       run_training_pipeline(args.config)
# ──────────────────────────────────────────────
