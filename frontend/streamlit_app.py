import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Corporate theme: navy/slate palette, branded header bar, sidebar for inputs
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --brand-navy: #0f2540;
        --brand-blue: #1f4e8c;
        --brand-accent: #2f7ed8;
        --brand-bg: #f5f7fa;
    }
    .stApp { background-color: var(--brand-bg); }

    /* Top brand bar */
    .brand-bar {
        background: linear-gradient(90deg, var(--brand-navy) 0%, var(--brand-blue) 100%);
        padding: 1.25rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .brand-bar h1 {
        color: white;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 600;
    }
    .brand-bar p {
        color: #cfe0f5;
        margin: 0.25rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Sidebar branding */
    section[data-testid="stSidebar"] {
        background-color: var(--brand-navy);
    }
    section[data-testid="stSidebar"] * {
        color: #e7edf5 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: var(--brand-accent);
        color: white !important;
        border: none;
        font-weight: 600;
    }

    /* Result cards */
    .result-card {
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        border-left: 6px solid;
    }
    .result-high {
        background-color: #fdecec;
        border-left-color: #c0392b;
    }
    .result-low {
        background-color: #eaf6ec;
        border-left-color: #1e8449;
    }
    .result-card h2 { margin: 0 0 0.25rem 0; font-size: 1.4rem; }
    .result-card p { margin: 0; color: #444; }

    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e3e8ef;
    }
    footer {visibility: hidden;}
    </style>

    <div class="brand-bar">
        <h1>📊 Customer Churn Risk Console</h1>
        <p>Internal decision-support tool &middot; Retention Analytics Team</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — customer inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Customer Profile")
    st.caption("Enter account details to assess churn risk")

    days_since_last_purchase = st.number_input(
        "Days since last purchase", min_value=0, value=30, step=1
    )
    satisfaction_score = st.select_slider(
        "Satisfaction score", options=[1, 2, 3, 4, 5], value=3
    )
    complaint_count = st.number_input("Complaint count", min_value=0, value=0, step=1)
    return_count = st.number_input("Return count", min_value=0, value=0, step=1)
    purchase_frequency = st.selectbox(
        "Purchase frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Rarely"]
    )
    employment_type = st.selectbox(
        "Employment type", ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"]
    )
    shopping_channel = st.selectbox(
        "Shopping channel", ["Online", "Mobile App", "In-Store", "Marketplace"]
    )

    st.markdown("---")
    submitted = st.button("Run prediction", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with right:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**System status**")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        if health.get("model_loaded"):
            st.markdown(f"🟢 Backend connected  \nModel: `{health.get('model_name', 'n/a')}`")
        else:
            st.markdown("🟡 Backend reachable, model not loaded")
    except requests.exceptions.RequestException:
        st.markdown(f"🔴 Backend unreachable at `{BACKEND_URL}`")
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    if not submitted:
        st.info("Enter a customer profile in the sidebar and click **Run prediction**.")
    else:
        payload = {
            "days_since_last_purchase": days_since_last_purchase,
            "satisfaction_score": satisfaction_score,
            "complaint_count": complaint_count,
            "return_count": return_count,
            "purchase_frequency": purchase_frequency,
            "employment_type": employment_type,
            "shopping_channel": shopping_channel,
        }

        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            churn = result["churn"]
            probability = result["churn_probability"]

            if churn == "Yes":
                st.markdown(
                    f"""
                    <div class="result-card result-high">
                        <h2>⚠️ High Churn Risk</h2>
                        <p>Estimated churn probability: <b>{probability:.0%}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card result-low">
                        <h2>✅ Low Churn Risk</h2>
                        <p>Estimated churn probability: <b>{probability:.0%}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(probability)

            with st.expander("Submitted profile"):
                st.json(payload)

        except requests.exceptions.ConnectionError:
            st.error(
                f"Couldn't reach the backend at {BACKEND_URL}. "
                "Make sure the FastAPI server is running (see README)."
            )
        except requests.exceptions.HTTPError:
            st.error(f"Backend error: {response.status_code} — {response.text}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption(f"Backend: {BACKEND_URL}")
