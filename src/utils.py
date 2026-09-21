import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.config import CONFIG


def setup_logger(name: str, log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when modules are imported more than once.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directories() -> None:
    paths = CONFIG["paths"]
    for key, value in paths.items():
        if key.endswith("_dir"):
            Path(value).mkdir(parents=True, exist_ok=True)
