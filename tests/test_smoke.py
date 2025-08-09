from src.config import load_config

def test_config_loads():
    cfg = load_config()
    assert 'iterations' in cfg.modelling
