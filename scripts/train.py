
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

config = load_config()
print(config["train"]["module_params"])
