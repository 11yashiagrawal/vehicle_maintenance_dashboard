import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils.model_loader import load_artifacts, load_sample_data
from utils.ui_utils import apply_full_page_background, html_table
from utils.config import DATA_DIR, FULL_DATA_FILENAME, NUMERIC_FEATURE_META, CAT_FEATURE_META
import os

st.set_page_config(page_title="Data Insights", page_icon="📊", layout="wide")
apply_full_page_background()

@st.cache_data
def load_full_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, FULL_DATA_FILENAME)
    return pd.read_csv(path, parse_dates=["last_service_date"])

st.markdown("""
<h1 style='font-size: 80px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>📊 Data Insights & Feature Exploration</h1>
""", unsafe_allow_html=True)
st.caption("Understand the raw vehicle data and how it maps to the prediction form.")

_, _, _ = load_artifacts()
df = load_full_data()

st.markdown("### Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

st.markdown("### Numerical Features")
st.caption("Every column from the raw dataset, with business meaning and typical ranges.")

rows = []
ignore_cols = ["maintenance_required", "vehicle_id", "time_to_failure_days", "failure_component"]
glossary_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]

for col in glossary_cols:
    series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    meta = NUMERIC_FEATURE_META.get(col, {})
    label = meta.get("label", col.replace("_", " ").title())
    description = meta.get("description", "Numeric feature derived from the vehicle telemetry.")
    unit = meta.get("unit", "")

    if len(series) > 0:
        q05, q95 = series.quantile([0.05, 0.95])
        typical_range = f"{q05:.2f} – {q95:.2f} {unit}".strip()
    else:
        typical_range = "n/a"

    rows.append({
        "Feature name": col,
        "Form label": label,
        "Description": description,
        "Typical range": typical_range
    })

st.markdown(html_table(["Feature name", "Form label", "Description", "Typical range"], rows), unsafe_allow_html=True)

st.markdown("### Categorical Features")
cat_rows = [{"Label": meta["label"], "Description": meta["description"]} for meta in CAT_FEATURE_META.values()]
st.markdown(html_table(["Label", "Description"], cat_rows), unsafe_allow_html=True)

st.markdown("### Data Distribution & Outlier Analysis")
st.caption(
    "Boxplot analysis revealed extreme upper tails in mileage and engine hours. "
    "Instead of removing these records, IQR-based capping was applied because high-usage "
    "vehicles represent realistic operational scenarios."
)

col_outlier_1, col_outlier_2 = st.columns(2)

with col_outlier_1:
    st.markdown("#### Mileage Distribution")
    mileage_box = alt.Chart(df).mark_boxplot(extent='min-max', color='#00ffff').encode(
        x=alt.X('mileage_km:Q', title="Mileage (km)"),
        tooltip=['mileage_km']
    ).properties(height=200)
    st.altair_chart(mileage_box, use_container_width=True)

with col_outlier_2:
    st.markdown("#### Engine Hours Distribution")
    hours_box = alt.Chart(df).mark_boxplot(extent='min-max', color='#00ffff').encode(
        x=alt.X('engine_hours:Q', title="Engine Hours"),
        tooltip=['engine_hours']
    ).properties(height=200)
    st.altair_chart(hours_box, use_container_width=True)

st.markdown("### Target Class Imbalance")
st.caption(
    "The dataset exhibited moderate imbalance (~75:25). Accuracy alone would therefore be misleading. "
    "For example, predicting all vehicles as 'No Maintenance' yields 75% accuracy without predictive power."
)

if "maintenance_required" in df.columns:
    imbalance_data = df["maintenance_required"].value_counts().reset_index()
    imbalance_data.columns = ["Status", "Count"]
    imbalance_data["Status"] = imbalance_data["Status"].map({1: "Maintenance Required", 0: "No Maintenance"})
    
    imbalance_chart = alt.Chart(imbalance_data).mark_bar().encode(
        x=alt.X("Status:N", title="Maintenance Status", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Count:Q", title="Number of Vehicles"),
        color=alt.Color("Status:N", scale=alt.Scale(domain=["No Maintenance", "Maintenance Required"], range=["#00ff00", "#ff0000"]), legend=None),
        tooltip=["Status", "Count"]
    ).properties(height=350)
    
    st.altair_chart(imbalance_chart, use_container_width=True)
else:
    st.info("Target variable `maintenance_required` not found in the dataset for imbalance analysis.")
