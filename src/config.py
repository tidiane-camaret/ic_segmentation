import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / "../config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config