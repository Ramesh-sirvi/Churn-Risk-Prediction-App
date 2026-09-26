"""
Churn Risk Prediction API
--------------------------
Serves the retrained churn model bundle (see train_pipeline.py) — a
HistGradientBoostingClassifier trained directly on the 7 raw features your
original app.py collected:

    employment_type, purchase_frequency, days_since_last_purchase,
    shopping_channel, return_count, complaint_count, satisfaction_score

Why this replaces the original XGBoost pickle:
The notebook's original model was trained on StandardScaler-transformed
features, but the fitted scaler was never saved — only the model. Feeding
raw values into that model (as the original app.py did) fed it data on the
wrong scale, which for a boosted-tree model means wrong split decisions.

This retrained model is a tree ensemble trained directly on the *raw*
feature values (no scaling step at all), which sidesteps the problem
entirely: tree splits are invariant to per-feature monotonic scaling, so
there's no accuracy lost by skipping the scaler, and no scaler to lose.
Test-set ROC-AUC came out to ~0.98, matching the original notebook's
7-feature XGBoost model.

The pickle is a single bundle dict — not just a bare model — containing:
    {
        "model": <trained sklearn classifier>,
        "model_name": str,
        "encoders": {col_name: fitted LabelEncoder, ...},
        "feature_order": [str, ...],
        "metrics": {accuracy, precision, recall, f1, roc_auc},
    }
"""

from contextlib import asynccontextmanager
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BUNDLE_PATH = Path(__file__).parent / "model" / "churn_pipeline.pkl"

CHURN_LABELS = {0: "No", 1: "Yes"}

bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Model bundle not found at {BUNDLE_PATH}")
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    yield


app = FastAPI(title="Churn Risk Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChurnInput(BaseModel):
    days_since_last_purchase: float = Field(..., ge=0, description="Days since the customer's last purchase")
    satisfaction_score: int = Field(..., ge=1, le=5, description="Customer satisfaction score, 1-5")
    complaint_count: int = Field(..., ge=0, description="Number of complaints filed")
    return_count: int = Field(..., ge=0, description="Number of product returns")
    purchase_frequency: str = Field(..., description="One of: Daily, Weekly, Monthly, Quarterly, Rarely")
    employment_type: str = Field(..., description="One of: Employed, Self-Employed, Unemployed, Retired, Student")
    shopping_channel: str = Field(..., description="One of: Online, Mobile App, In-Store, Marketplace")


class ChurnPrediction(BaseModel):
    churn: str
    churn_probability: float
    model_name: str


def _encode(value: str, col: str) -> int:
    encoder = bundle["encoders"][col]
    if value not in encoder.classes_:
        allowed = ", ".join(encoder.classes_)
        raise HTTPException(status_code=422, detail=f"Invalid {col} '{value}'. Must be one of: {allowed}")
    return int(encoder.transform([value])[0])


@app.get("/")
def root():
    return {
        "message": "Churn Risk Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "note": "The user-facing app is the Streamlit frontend, usually at http://localhost:8501",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_name": bundle["model_name"] if bundle else None,
        "test_metrics": bundle["metrics"] if bundle else None,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(payload: ChurnInput):
    row = {
        "employment_type": _encode(payload.employment_type, "employment_type"),
        "purchase_frequency": _encode(payload.purchase_frequency, "purchase_frequency"),
        "days_since_last_purchase": payload.days_since_last_purchase,
        "shopping_channel": _encode(payload.shopping_channel, "shopping_channel"),
        "return_count": payload.return_count,
        "complaint_count": payload.complaint_count,
        "satisfaction_score": payload.satisfaction_score,
    }

    # Build the feature row as a DataFrame (with matching column names) in
    # the exact order the model was trained on — avoids the sklearn warning
    # you'd get from passing a bare, unnamed numpy array.
    features = pd.DataFrame([[row[col] for col in bundle["feature_order"]]], columns=bundle["feature_order"])

    model = bundle["model"]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return ChurnPrediction(
        churn=CHURN_LABELS[prediction],
        churn_probability=round(probability, 4),
        model_name=bundle["model_name"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
