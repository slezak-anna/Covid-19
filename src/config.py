import yaml
from datetime import datetime
from pathlib import Path

# Resolve project root as the folder that contains params.yaml
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"

def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_PARAMS_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

# Dates (keep names from the notebook)
last_train_date = datetime.fromisoformat(cfg["dates"]["last_train_date"])
last_eval_date  = datetime.fromisoformat(cfg["dates"]["last_eval_date"])
last_test_date  = datetime.fromisoformat(cfg["dates"]["last_test_date"])

# Features / modelling params
cat_features     = cfg["features"]["cat_features"]
drop_columns     = cfg["features"]["drop_columns"]
iterations       = int(cfg["modelling"]["iterations"])
days_history_size = int(cfg["modelling"]["days_history_size"])
thresholds       = cfg["modelling"]["thresholds_since"]

# Paths (used as-is relative to project root)
paths = cfg["paths"]
