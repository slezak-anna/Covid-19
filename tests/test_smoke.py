def test_config_loads():
    from src.config import load_config, last_train_date, last_eval_date, last_test_date, paths
    cfg = load_config()
    assert "dates" in cfg and "paths" in cfg
    assert str(last_train_date) and str(last_eval_date) and str(last_test_date)
    assert "kaggle_train" in paths and "kaggle_test" in paths