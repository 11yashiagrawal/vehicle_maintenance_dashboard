import streamlit as st
import pandas as pd
from utils.model_loader import load_artifacts, load_sample_data
from utils.ui_utils import apply_full_page_background, html_table
from utils.preprocessor import calculate_internal_features
from utils.config import DATA_DIR, FULL_DATA_FILENAME

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")
apply_full_page_background()

st.markdown("""
<h1 style='font-size: 80px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>📈 Model Strategy & Evaluation</h1>
""", unsafe_allow_html=True)
st.caption("Deep dive into the architecture, selection process, and performance metrics.")

st.markdown("### 🧩 Engineered Features & Logic")
st.caption("Advanced features calculated internally to capture complex mechanical interactions.")

engineered_rows = [
    {
        "Feature": "Days Since Last Service",
        "Formula": "today() - last_service_date",
        "Significance": "Time since last service. Indicates maintenance schedule adherence."
    },
    {
        "Feature": "Mileage Per Year",
        "Formula": "mileage_km / vehicle_age_years",
        "Significance": "Captures operational intensity. Two vehicles with identical mileage may experience different wear levels depending on usage rate."
    },
    {
        "Feature": "Thermal Stress",
        "Formula": "oil_temp_avg_celsius * engine_load_percent",
        "Significance": "Mechanical wear increases non-linearly when high load is combined with elevated temperature. Captures interaction effect effectively."
    },
    {
        "Feature": "Engine Hours Per KM",
        "Formula": "engine_hours / mileage_km",
        "Significance": "Indicates excessive idling or mechanical inefficiency, both of which contribute to maintenance risk."
    },
    {
        "Feature": "Fault Density",
        "Formula": "fault_code_count / engine_hours",
        "Significance": "Normalizes fault codes by engine hours to prevent bias toward older vehicles and captures recurrence frequency."
    },
    {
        "Feature": "Load Efficiency",
        "Formula": "engine_load_percent / fuel_efficiency_kmpl",
        "Significance": "Reflects mechanical strain relative to output efficiency, helping detect vehicles operating under disproportionate stress."
    }
]

st.markdown(html_table(["Feature", "Formula", "Significance"], engineered_rows), unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🧠 Model Training and Selection Strategy")
st.write("""
Rather than selecting a single algorithm arbitrarily, a progressive modelling approach was followed. 
The objective was to evaluate increasing levels of model complexity and understand the nature of the relationships within the data.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Baseline: Logistic Regression")
    st.write("""
    - Establishes minimum benchmark performance.
    - Provides interpretability of feature influence.
    - **Limitation**: Lower recall indicated non-linear relationships.
    """)

    st.markdown("#### Decision Tree")
    st.write("""
    - Captures non-linear splits.
    - **Limitation**: Higher variance and overfitting tendency.
    """)

with col2:
    st.markdown("#### Random Forest")
    st.write("""
    - Reduces variance through ensemble averaging.
    - Handles interactions robustly.
    - **Observed**: Strong performance but slightly lower recall for minority class.
    """)

    st.markdown("#### ✅ Final Selection: XGBoost")
    st.write("""
    - **Why?**: Corrects residual errors sequentially.
    - **Results**: Achieved the highest **Recall (0.76)** and **F1-Score (0.78)**.
    - **Justification**: Higher recall is critical in maintenance to avoid costly missing cases.
    """)

st.markdown("### 📊 Performance Comparison")
comparison_rows = [
    {"Model": "Logistic Regression", "Accuracy": "0.84", "Recall": "0.60", "F1-Score": "0.68", "ROC-AUC": "0.92"},
    {"Model": "Decision Tree", "Accuracy": "0.83", "Recall": "0.60", "F1-Score": "0.66", "ROC-AUC": "0.85"},
    {"Model": "Random Forest", "Accuracy": "0.85", "Recall": "0.58", "F1-Score": "0.70", "ROC-AUC": "0.91"},
    {"Model": "XGBoost (Tuned)", "Accuracy": "0.85", "Recall": "0.76", "F1-Score": "0.78", "ROC-AUC": "0.92"},
]

st.markdown(html_table(["Model", "Accuracy", "Recall", "F1-Score", "ROC-AUC"], comparison_rows), unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📋 Sample Data Preview")
sample_df = load_sample_data()
if sample_df is not None:
    st.markdown(html_table(sample_df.columns.tolist(), sample_df.head().to_dict("records")), unsafe_allow_html=True)
else:
    st.info("No sample data found.")
