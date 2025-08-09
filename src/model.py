import os
import json
import joblib
import catboost as cb

from .config import iterations, cat_features, paths

def train_catboost_models(train_features_df, train_labels, eval_features_df, eval_labels, verbose=100):
    catboost_models = {}
    eval_scores = {}

    for prediction_name in ['LogNewConfirmedCases', 'LogNewFatalities']:
        model = cb.CatBoostRegressor(
            has_time=True,
            iterations=iterations
        )
        model.fit(
            train_features_df,
            train_labels[prediction_name],
            eval_set=(eval_features_df, eval_labels[prediction_name]),
            cat_features=cat_features,
            verbose=verbose
        )
        catboost_models[prediction_name] = model
        rmse = model.evals_result_['validation']['RMSE'][-1]
        print('CatBoost: prediction of %s: RMSLE on test = %s' % (prediction_name, rmse))
        eval_scores[prediction_name] = {"validation_RMSE": float(rmse)}

    return catboost_models, eval_scores

def save_model_bundle(catboost_models, feature_columns, location_columns):
    os.makedirs(paths["model_dir"], exist_ok=True)
    bundle = {
        "catboost_models": catboost_models,
        "cat_features": cat_features,
        "feature_columns": list(feature_columns),
        "location_columns": list(location_columns)
    }
    joblib.dump(bundle, paths["model_bundle"])
    return paths["model_bundle"]

def save_metrics(metrics_dict):
    with open(paths["metrics_file"], "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)