from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Params:
    random_state: int
    dates: Dict[str, str]
    paths: Dict[str, Any]
    features: Dict[str, Any]
    modelling: Dict[str, Any]

def load_config(path: str = "params.yaml") -> Params:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return Params(**y)