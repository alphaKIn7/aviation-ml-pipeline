# ──────────────────────────────────────────────
# logger.py — Centralized Logging
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Set up a consistent logging format across the entire project.
#   Every module should use this logger instead of print().
#
# WHY NOT PRINT()?
#   print() gives you no control over:
#   - Log levels (DEBUG, INFO, WARNING, ERROR)
#   - Output destinations (console, file, or both)
#   - Timestamps and context
#   In production, you NEED structured logs for debugging.
#
# FUNCTIONS TO BUILD HERE:
#   - get_logger(name, log_file, level) → Configured logger instance
#
# USAGE:
#   logger = get_logger(__name__)
#   logger.info("Training started")
#   logger.error("Data validation failed: %s", error)
#
# INDUSTRY PATTERN:
#   Use Python's built-in logging module. In production, teams often
#   add structured logging (JSON format) for log aggregation tools
#   like ELK Stack or Datadog.
# ──────────────────────────────────────────────

import logging
from pathlib import Path

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
