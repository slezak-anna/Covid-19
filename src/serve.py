from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import catboost as cb
import joblib
from datetime import datetime

from .config import paths, last_train_date, last_eval_date
from .data import build_datasets
from .predict import predict_for_dataset  

app = FastAPI(title="COVID19 CatBoost API")

# Load model bundle
bundle = joblib.load(paths["model_bundle"])
catboost_models = bundle["catboost_models"]
cat_features = bundle["cat_features"]
feature_columns = bundle["feature_columns"]
location_columns = bundle["location_columns"]

_d = build_datasets()

# 1) Eval predictions
_prev_day_eval = _d["train_df"].loc[_d["train_df"]["Date"] == last_train_date]
_first_eval_date = last_train_date + pd.Timedelta(days=1)

_eval_df = _d["eval_df"].copy()
_eval_features_df = _d["eval_features_df"].copy()

predict_for_dataset(
    _eval_df, _eval_features_df, _prev_day_eval,
    _first_eval_date, last_eval_date, update_features_data=False,
    catboost_models=catboost_models, cat_features=cat_features, location_columns=location_columns
)

# 2) Test predictions (cascade from eval)
_prev_day_test = _eval_df.loc[_eval_df["Date"] == last_eval_date]
_first_test_date = last_eval_date + pd.Timedelta(days=1)

_test_df = _d["test_df"].copy()
_test_features_df = _d["test_features_df"].copy()

predict_for_dataset(
    _test_df, _test_features_df, _prev_day_test,
    _first_test_date, _d["test_df"]["Date"].max(), update_features_data=True,
    catboost_models=catboost_models, cat_features=cat_features, location_columns=location_columns
)

# Concatenate eval + test with predictions
PRED_DF = pd.concat([_eval_df, _test_df], ignore_index=True)

class PredictRequest(BaseModel):
    rows: list[dict]  # same as before

class PredictRangeRequest(BaseModel):
    country: str
    province: str | None = ""  
    start_date: str  
    end_date: str    

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = pd.DataFrame(req.rows)
    X = X.reindex(columns=feature_columns)
    pool = cb.Pool(X, cat_features=cat_features)
    out = {}
    for prediction_name, model in catboost_models.items():
        preds = model.predict(pool)
        out[prediction_name] = preds.tolist()
    return out

@app.post("/predict-range")
def predict_range(req: PredictRangeRequest):
    # Parse dates
    try:
        start_dt = datetime.fromisoformat(req.start_date)
        end_dt = datetime.fromisoformat(req.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date.")

    # Filter location (province can be empty string)
    prov = (req.province or "")
    mask_loc = (PRED_DF["Country/Region"] == req.country) & (PRED_DF["Province/State"] == prov)

    # Filter date range
    mask_date = (PRED_DF["Date"] >= pd.Timestamp(start_dt)) & (PRED_DF["Date"] <= pd.Timestamp(end_dt))

    out_df = PRED_DF.loc[mask_loc & mask_date, [
        "Date",
        "Country/Region",
        "Province/State",
        "PredictedLogNewConfirmedCases",
        "PredictedLogNewFatalities",
        "PredictedConfirmedCases",
        "PredictedFatalities"
    ]].sort_values("Date")

    if out_df.empty:
        raise HTTPException(status_code=404, detail="No data for given filters (country/province/date range).")

    # JSON-friendly
    out_df = out_df.copy()
    out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
    return {"results": out_df.to_dict(orient="records")}
