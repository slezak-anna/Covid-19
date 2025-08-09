import json, pathlib, joblib, numpy as np, time
from sklearn.metrics import mean_squared_error
from .data import build_main_df, make_splits, preprocess_df
from .model import train_models
from .config import load_config

import mlflow

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _flatten_for_mlflow(cfg):
    flat = {}
    flat["random_state"] = cfg.random_state
    # dates
    for k, v in cfg.dates.items():
        flat[f"dates.{k}"] = v
    # modelling
    for k, v in cfg.modelling.items():
        flat[f"modelling.{k}"] = json.dumps(v) if isinstance(v, (list, dict)) else v
    # features
    flat["features.cat_features"] = ",".join(cfg.features["cat_features"])
    flat["features.drop_columns"] = ",".join(cfg.features["drop_columns"])
    return flat

def main():
    cfg = load_config()
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("covid19-risk")

    run_name = f"train-catboost-{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten_for_mlflow(cfg))

        main_df = build_main_df()
        splits = make_splits(main_df)
        X_tr, y_tr = preprocess_df(splits['train'])
        X_ev, y_ev = preprocess_df(splits['eval'])

        models = train_models(X_tr, y_tr, X_ev, y_ev)

        metrics = {}
        for target in ['LogNewConfirmedCases','LogNewFatalities']:
            pred = models[target].predict(X_ev)
            metrics[f'rmse_{target}'] = rmse(y_ev[target].values, pred)
        mlflow.log_metrics(metrics)

        pathlib.Path(cfg.paths['model_dir']).mkdir(parents=True, exist_ok=True)
        joblib.dump({'models': models}, cfg.paths['model_bundle'])
        with open(cfg.paths['metrics_file'], 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(cfg.paths["model_bundle"])
        mlflow.log_artifact(cfg.paths["metrics_file"])
        mlflow.log_artifact("params.yaml")
        
        print("Saved:", cfg.paths['model_bundle'])
        print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()