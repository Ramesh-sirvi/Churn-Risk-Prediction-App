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

| Random Forest (balanced) | 0.956 | 0.556 | 0.876 | 0.681 | 0.979 |
| **HistGradientBoosting (selected)** | 0.946 | 0.500 | 0.946 | 0.654 | **0.981** |

## Features (in order)

1. `employment_type`
2. `purchase_frequency`
3. `days_since_last_purchase`
4. `shopping_channel`
5. `return_count`
6. `complaint_count`
7. `satisfaction_score`

## Test metrics

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.956 | 0.556 | 0.876 | 0.681 | 0.979 |


