import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(default_path: Path, logging_name: str = 'main') -> logging.Logger:
    path = Path(default_path)
    if path.is_file():
        with open(path, mode='r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        return logging.getLogger(logging_name)
    else:
        raise FileNotFoundError(
            f"Logging configuration file {default_path} not found.")


def get_settings(settings_path: Path, env: str = 'PROD') -> dict:
    """
    Load settings from a YAML file based on the specified environment.

    Args:
        settings_path (Path): Path to the settings YAML file.
        env (str): Environment name to load specific settings.

    Returns:
        dict: Settings for the specified environment.
    """
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    if env in settings:
        return settings[env]
    else:
        raise KeyError(f"Environment '{env}' not found in settings.")
