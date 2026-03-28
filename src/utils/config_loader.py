import yaml
from pathlib import Path

def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
