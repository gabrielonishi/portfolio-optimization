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
