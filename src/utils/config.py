# ──────────────────────────────────────────────
# config.py — Configuration Management
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Load, validate, and provide access to configuration values.
#   Acts as the bridge between YAML config files and Python code.
#
# WHY NOT JUST LOAD YAML DIRECTLY?
#   This module adds:
#   - Validation (does the config have all required keys?)
#   - Defaults (what if a key is missing?)
#   - Type checking (is learning_rate actually a float?)
#
# FUNCTIONS TO BUILD HERE:
#   - load_config(config_path)           → Load and validate config
#   - get_data_config(config)            → Extract data section
#   - get_model_config(config)           → Extract model section
#   - get_training_config(config)        → Extract training section
#   - validate_config(config, schema)    → Check required keys exist
#
# INDUSTRY PATTERN:
#   Some teams use Pydantic or dataclasses to define config schemas.
#   This gives you auto-validation and IDE autocomplete for free.
# ──────────────────────────────────────────────
