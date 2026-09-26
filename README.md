# Churn Risk Predictor — Full-Stack App

A FastAPI backend serving a retrained churn model, with a Streamlit
frontend for interactive predictions.

```
churn-app/
├── backend/
│   ├── main.py                                    # FastAPI app + prediction endpoint
│   ├── train_pipeline.py                           # Retrains the model from the raw CSV
│   ├── requirements.txt
│   ├── E-commerce_Customer_Segmentation_2026.csv    # Training data
│   └── model/
│       └── churn_pipeline.pkl                       # Trained model bundle
├── frontend/
│   ├── streamlit_app.py     # Streamlit UI
│   └── requirements.txt
└── README.md
```

## 1. Run the backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's up: open http://localhost:8000/health — you should see
`{"status": "ok", "model_loaded": true, "model_name": "HistGB", "test_metrics": {...}}`.
Interactive API docs are at http://localhost:8000/docs.

## 2. Run the frontend

In a second terminal:

```bash
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens the UI at http://localhost:8501, which calls the backend at
`http://localhost:8000` by default. To point it at a different backend:

```bash
BACKEND_URL=https://your-api.example.com streamlit run streamlit_app.py
```

## The model was retrained — here's why

Your original notebook trained an XGBoost model on 7 features, but only
after running `StandardScaler` across *all* columns and never saving that
scaler — only the model got pickled. So any app calling `model.predict()`
with raw (unscaled) values, like your original `app.py` did, was feeding
the model numbers on the wrong scale, which shifts every tree's split
decisions.

Rather than guess at scaler statistics I don't have, I retrained on the
same 7 features directly from your CSV, but **skipped scaling entirely**:
tree-ensemble splits are invariant to per-feature monotonic scaling, so a
tree model trained on raw values makes identical decisions to one trained
on scaled values — there's no accuracy cost to dropping the scaler, and no
scaler left to lose. This also means the API can safely accept raw form
inputs with no separate scaling step required, front-to-back.

**Model comparison** (test set, 7 features, no SMOTE — used
`class_weight`/`sample_weight` balancing instead since `imbalanced-learn`
isn't required by this setup):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (balanced) | 0.956 | 0.556 | 0.876 | 0.681 | 0.979 |
| **HistGradientBoosting (selected)** | 0.946 | 0.500 | 0.946 | 0.654 | **0.981** |

For reference, the notebook's original 7-feature XGBoost model (evaluated
on scaled data) scored accuracy 0.971 / ROC-AUC 0.98 — so this retrained,
scale-free model is right in line with it, while being simpler to deploy
(only needs `scikit-learn`, not `xgboost` or `imbalanced-learn`) and
correct to call with raw inputs.

`train_pipeline.py` is included so you can re-run training yourself (e.g.
with more data, different hyperparameters, or to swap back to XGBoost if
you install it) — just run `python train_pipeline.py` from `backend/`.

## Also fixed vs. your original app.py

Tracing the notebook's feature-selection step, the original 7-feature model
was trained on columns in this order: `employment_type, purchase_frequency,
days_since_last_purchase, shopping_channel, return_count, complaint_count,
satisfaction_score`. Your original `app.py` built its input array in a
different order. Since a model only sees array positions, not column names,
that mismatch was silently feeding values into the wrong slots. `main.py`
now builds the array by name, in the order the model actually expects.
