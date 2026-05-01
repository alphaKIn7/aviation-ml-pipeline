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
