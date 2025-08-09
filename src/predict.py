import argparse, joblib, numpy as np, pandas as pd
from typing import Dict
import catboost as cb
from .config import load_config
from .data import build_main_df, preprocess_df

LOCATION_COLS = ['Country/Region','Province/State']

def _sequential_predict(df: pd.DataFrame, feats: pd.DataFrame, models: Dict[str, cb.CatBoostRegressor],
                        start: pd.Timestamp, end: pd.Timestamp, cat_features):
    df['PredictedLogNewConfirmedCases'] = np.nan
    df['PredictedLogNewFatalities'] = np.nan
    df['PredictedConfirmedCases'] = np.nan
    df['PredictedFatalities'] = np.nan

    for day in pd.date_range(start, end):
        day_mask = df['Date'] == day
        day_pool = cb.Pool(feats.loc[day_mask], cat_features=cat_features)

        for t in ['LogNewConfirmedCases','LogNewFatalities']:
            df.loc[day_mask, 'Predicted'+t] = np.maximum(models[t].predict(day_pool), 0.0)

        prev_day = day - pd.Timedelta(days=1)
        prev_mask = df['Date'] == prev_day
        merge = df.loc[day_mask, LOCATION_COLS + ['PredictedLogNewConfirmedCases','PredictedLogNewFatalities']].merge(
            df.loc[prev_mask, LOCATION_COLS + ['PredictedConfirmedCases','PredictedFatalities']].rename(
                columns={'PredictedConfirmedCases':'_prev_cc','PredictedFatalities':'_prev_f'}
            ),
            on=LOCATION_COLS, how='left'
        )
        prev_cc = merge['_prev_cc'].fillna(0.0).values
        prev_f  = merge['_prev_f'].fillna(0.0).values
        new_cc = prev_cc + np.rint(np.expm1(merge['PredictedLogNewConfirmedCases'].values))
        new_f  = prev_f  + np.rint(np.expm1(merge['PredictedLogNewFatalities'].values))
        df.loc[day_mask,'PredictedConfirmedCases'] = new_cc
        df.loc[day_mask,'PredictedFatalities']     = new_f

        next_day = day + pd.Timedelta(days=1)
        next_mask = df['Date'] == next_day
        if next_mask.any():
            next_feats = feats.loc[next_mask].copy()
            for field in ['ConfirmedCases','Fatalities']:
                for k in range(30,1,-1):
                    to, fr = f'LogNew{field}_prev_day_{k}', f'LogNew{field}_prev_day_{k-1}'
                    if to in next_feats and fr in next_feats: next_feats[to] = next_feats[fr]
            next_feats['LogNewConfirmedCases_prev_day_1'] = df.loc[day_mask,'PredictedLogNewConfirmedCases'].values
            next_feats['LogNewFatalities_prev_day_1']     = df.loc[day_mask,'PredictedLogNewFatalities'].values
            feats.loc[next_mask] = next_feats

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='YYYY-MM-DD')
    parser.add_argument('--out', required=True, help='output CSV path')
    args = parser.parse_args()

    cfg = load_config()
    bundle = joblib.load(cfg.paths['model_bundle'])
    models = bundle['models']
    cat_features = cfg.features['cat_features']

    main_df = build_main_df()
    start, end = pd.to_datetime(args.start), pd.to_datetime(args.end)
    mask = (main_df['Date'] >= start) & (main_df['Date'] <= end)
    df = main_df.loc[mask].copy()
    feats, _ = preprocess_df(df.copy())

    out = _sequential_predict(df, feats, models, start, end, cat_features)
    out[['Date','Province/State','Country/Region','PredictedConfirmedCases','PredictedFatalities']].to_csv(args.out, index=False)
    print("Saved predictions to", args.out)

if __name__ == "__main__":
    main()
