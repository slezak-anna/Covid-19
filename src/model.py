from typing import Dict
import catboost as cb
import pandas as pd
from .config import load_config

def train_models(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_eval: pd.DataFrame,  y_eval: pd.DataFrame
) -> Dict[str, cb.CatBoostRegressor]:
    cfg = load_config()
    cat_features = cfg.features['cat_features']
    iters = int(cfg.modelling['iterations'])
    models: Dict[str, cb.CatBoostRegressor] = {}

    for target in ['LogNewConfirmedCases','LogNewFatalities']:
        model = cb.CatBoostRegressor(
            has_time=True,
            iterations=iters,
            random_seed=cfg.random_state,
            loss_function='RMSE'
        )
        model.fit(
            X_train, y_train[target],
            eval_set=(X_eval, y_eval[target]),
            cat_features=cat_features,
            verbose=False
        )
        models[target] = model
    return models