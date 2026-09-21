from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping/object at the top level.")
    return config


CONFIG = load_config()
