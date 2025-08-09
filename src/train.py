from .data import build_datasets
from .model import train_catboost_models, save_model_bundle, save_metrics
from .config import iterations, days_history_size, thresholds, cat_features, paths

import mlflow
import mlflow.sklearn  
from pathlib import Path

def main():
    # Build datasets
    d = build_datasets()

    # Train models
    catboost_models, eval_scores = train_catboost_models(
        d["train_features_df"], d["train_labels"],
        d["eval_features_df"],  d["eval_labels"],
        verbose=100
    )

    # Save bundle + metrics to disk
    bundle_path = save_model_bundle(
        catboost_models=catboost_models,
        feature_columns=d["train_features_df"].columns,
        location_columns=d["location_columns"]
    )
    save_metrics(eval_scores)

    # ---- MLflow logging (simple & explicit) ----
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("covid19-catboost")

    with mlflow.start_run(run_name="catboost_covid19"):
        # Log params from YAML (keep naming consistent with the notebook/YAML)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("days_history_size", days_history_size)
        mlflow.log_param("thresholds_since", thresholds)
        mlflow.log_param("cat_features", ",".join(cat_features))

        # Log metrics per target
        # Flatten eval_scores: {'LogNewConfirmedCases': {'validation_RMSE': ...}, ...}
        for target, metrics in eval_scores.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{target}_{k}", float(v))

        # Log artifacts (model bundle + metrics file)
        mlflow.log_artifact(bundle_path)
        metrics_file = paths["metrics_file"]
        if Path(metrics_file).exists():
            mlflow.log_artifact(metrics_file)

    print(f"Saved model bundle to: {bundle_path}")
    print("Training complete.")

if __name__ == "__main__":
    main()