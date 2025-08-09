import numpy as np
import pandas as pd
import catboost as cb
import joblib

from .config import last_train_date, last_eval_date, last_test_date
from .data import build_datasets

# Keep original function name/signature
def predict_for_dataset(df, features_df, prev_day_df, first_date, last_date, update_features_data,
                        catboost_models, cat_features, location_columns):
    df['PredictedLogNewConfirmedCases'] = np.nan
    df['PredictedLogNewFatalities'] = np.nan
    df['PredictedConfirmedCases'] = np.nan
    df['PredictedFatalities'] = np.nan

    for day in pd.date_range(first_date, last_date):
        day_df = df[df['Date'] == day]
        day_features_pool = cb.Pool(features_df.loc[day_df.index], cat_features=cat_features)

        for prediction_type in ['LogNewConfirmedCases', 'LogNewFatalities']:
            df.loc[day_df.index, 'Predicted' + prediction_type] = np.maximum(
                catboost_models[prediction_type].predict(day_features_pool),
                0.0
            )

        day_predictions_df = df.loc[day_df.index][
            location_columns + ['PredictedLogNewConfirmedCases', 'PredictedLogNewFatalities']
        ]

        for field in ['ConfirmedCases', 'Fatalities']:
            prev_day_field = field if day == first_date else ('Predicted' + field)
            merged_df = day_predictions_df.merge(
                right=prev_day_df[location_columns + [prev_day_field]],
                how='inner',
                on=location_columns
            )
            df.loc[day_df.index, 'Predicted' + field] = merged_df.apply(
                lambda row: row[prev_day_field] + np.rint(np.expm1(row['PredictedLogNew' + field])),
                axis='columns'
            ).values

        if update_features_data:
            for next_day in pd.date_range(day + pd.Timedelta(days=1), last_date):
                next_day_features_df = features_df[df['Date'] == next_day]
                merged_df = next_day_features_df[location_columns].merge(
                    right=day_predictions_df,
                    how='inner',
                    on=location_columns
                )
                prev_day_idx = (next_day - day).days
                for prediction_type in ['LogNewConfirmedCases', 'LogNewFatalities']:
                    features_df.loc[next_day_features_df.index, prediction_type + '_prev_day_%s' % prev_day_idx] = (
                        merged_df['Predicted' + prediction_type].values
                    )

        prev_day_df = df.loc[day_df.index]

def main():
    d = build_datasets()
    bundle = joblib.load("models/covid19_catboost_bundle.joblib")
    catboost_models = bundle["catboost_models"]
    cat_features = bundle["cat_features"]
    location_columns = bundle["location_columns"]

    # Eval period
    prev_day_df = d["train_df"].loc[d["train_df"]['Date'] == last_train_date]
    first_eval_date = last_train_date + pd.Timedelta(days=1)
    predict_for_dataset(
        d["eval_df"], d["eval_features_df"], prev_day_df,
        first_eval_date, last_eval_date, update_features_data=False,
        catboost_models=catboost_models, cat_features=cat_features, location_columns=location_columns
    )

    # Test period
    prev_day_df = d["eval_df"].loc[d["eval_df"]['Date'] == last_eval_date]
    first_test_date = last_eval_date + pd.Timedelta(days=1)
    predict_for_dataset(
        d["test_df"], d["test_features_df"], prev_day_df,
        first_test_date, last_test_date, update_features_data=True,
        catboost_models=catboost_models, cat_features=cat_features, location_columns=location_columns
    )

    d["eval_df"].to_csv("models/predictions_eval.csv", index=False)
    d["test_df"].to_csv("models/predictions_test.csv", index=False)
    print("Saved predictions to models/predictions_eval.csv and models/predictions_test.csv")

if __name__ == "__main__":
    main()