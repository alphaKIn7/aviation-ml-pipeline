from pathlib import Path
from src.utils.io_handler import load_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_KEYS = {
    "data":     ["raw_path", "processed_path", "target_column"],
    "features": ["numerical", "categorical", "drop_columns"],
    "model":    ["name", "hyperparameters"],
    "training": ["cross_validation_folds", "metric"],
    "artifacts":["model_path"],
}

def validate_config(config: dict) -> None:
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise KeyError(f"Config is missing required section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Config section '{section}' is missing required key: '{key}'")

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from {config_path}")
    config = load_yaml(path)
    validate_config(config)
    logger.info("Config loaded and validated successfully")
    return config

def get_data_config(config: dict) -> dict:
    return config["data"]

def get_model_config(config: dict) -> dict:
    return config["model"]

def get_training_config(config: dict) -> dict:
    return config["training"]

def get_artifacts_config(config: dict) -> dict:
    return config["artifacts"]
