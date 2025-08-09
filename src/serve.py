from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib, numpy as np, pandas as pd, catboost as cb
from .config import load_config
from .data import build_main_df, preprocess_df
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="COVID-19 Sequential Forecast API", version="0.1.0")
_cfg = load_config()
_bundle = None
_main_df = None

class PredictRangeRequest(BaseModel):
    country: str
    province: Optional[str] = ""
    start_date: str
    end_date: str

def _ensure_loaded():
    global _bundle, _main_df
    if _bundle is None:
        _bundle = joblib.load(_cfg.paths['model_bundle'])
    if _main_df is None:
        _main_df = build_main_df()

def _predict_range(df: pd.DataFrame, feats: pd.DataFrame, start, end, cat_features):
    models = _bundle['models']
    df['PredictedLogNewConfirmedCases'] = np.nan
    df['PredictedLogNewFatalities'] = np.nan
    df['PredictedConfirmedCases'] = np.nan
    df['PredictedFatalities'] = np.nan

    for day in pd.date_range(start, end):
        day_mask = df['Date'] == day
        pool = cb.Pool(feats.loc[day_mask], cat_features=cat_features)
        for t in ['LogNewConfirmedCases','LogNewFatalities']:
            df.loc[day_mask, 'Predicted'+t] = np.maximum(models[t].predict(pool), 0.0)

        prev_day = day - pd.Timedelta(days=1)
        prev_mask = df['Date'] == prev_day
        prev_cc = df.loc[prev_mask,'PredictedConfirmedCases'] if prev_mask.any() else pd.Series([], dtype=float)
        prev_f  = df.loc[prev_mask,'PredictedFatalities'] if prev_mask.any() else pd.Series([], dtype=float)
        prev_cc = prev_cc.reindex(index=df.loc[day_mask].index).fillna(0.0).values
        prev_f  = prev_f.reindex(index=df.loc[day_mask].index).fillna(0.0).values
        df.loc[day_mask,'PredictedConfirmedCases'] = prev_cc + np.rint(np.expm1(df.loc[day_mask,'PredictedLogNewConfirmedCases']))
        df.loc[day_mask,'PredictedFatalities']     = prev_f  + np.rint(np.expm1(df.loc[day_mask,'PredictedLogNewFatalities']))

        # roll lags
        next_day = day + pd.Timedelta(days=1)
        next_mask = df['Date'] == next_day
        if next_mask.any():
            nf = feats.loc[next_mask].copy()
            for field in ['ConfirmedCases','Fatalities']:
                for k in range(30,1,-1):
                    to, fr = f'LogNew{field}_prev_day_{k}', f'LogNew{field}_prev_day_{k-1}'
                    if to in nf and fr in nf: nf[to] = nf[fr]
            nf['LogNewConfirmedCases_prev_day_1'] = df.loc[day_mask,'PredictedLogNewConfirmedCases'].values
            nf['LogNewFatalities_prev_day_1']     = df.loc[day_mask,'PredictedLogNewFatalities'].values
            feats.loc[next_mask] = nf
    return df

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h3>COVID-19 Sequential Forecast API</h3><p>Użyj <a href='/docs'>/docs</a> lub POST /predict-range.</p>"

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.post("/predict-range")
def predict_range(req: PredictRangeRequest):
    _ensure_loaded()
    start = pd.to_datetime(req.start_date); end = pd.to_datetime(req.end_date)
    sub = _main_df[
        (_main_df['Country/Region'] == req.country) &
        (_main_df['Province/State'] == (req.province or "")) &
        (_main_df['Date'] >= start) & (_main_df['Date'] <= end)
    ].copy()
    if sub.empty:
        return {"error":"No rows for given location/dates."}
    feats, _ = preprocess_df(sub.copy())
    out = _predict_range(sub, feats, start, end, _cfg.features['cat_features'])
    return out[['Date','Province/State','Country/Region','PredictedConfirmedCases','PredictedFatalities']].to_dict(orient='records')
