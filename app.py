# Fallback Prediction Portal
# This monolithic file provides an emergency backup UI if the main multi-page dashboard fails.
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date
from utils.model_loader import load_artifacts
from utils.ui_utils import apply_full_page_background, inject_custom_css
from utils.preprocessor import build_feature_vector, build_input_form_grid

st.set_page_config(page_title="Vehicle Maintenance Fallback", page_icon="🚗", layout="wide")
apply_full_page_background()
inject_custom_css("""
div.stButton > button[kind="primary"] {
    background-color: #00ffff !important;
    color: #000000 !important;
    font-weight: 800 !important;
    padding: 1.2rem 3rem !important;
    font-size: 1.4rem !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    width: 100% !important;
    height: auto !important;
    min-height: 4rem !important;
}
div.stButton > button[kind="primary"] * {
    color: #000000 !important;
    font-weight: 800 !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 5px rgba(0, 255, 255, 0.8) !important;
    transform: translateY(-1px) !important;
}
""")

st.markdown("""
<h1 style='font-size: 80px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>🚗 Fallback Predictor</h1>
""", unsafe_allow_html=True)
st.markdown("Emergency backup prediction interface.")

model, encoder, features = load_artifacts()

predict_clicked = st.button("🔍 Click to Predict Maintenance", type="primary")
result_container = st.container()

st.markdown("---")
input_data = build_input_form_grid(features, num_cols=2)

if predict_clicked:
    with st.spinner("Running prediction..."):
        X = build_feature_vector(features, input_data)
        prediction = model.predict(X)[0]
        
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0]
            risk_score = float(prob[1])
        else:
            risk_score = float(prediction)

    with result_container:
        col1, col2 = st.columns([1, 3])
        with col1:
            if prediction == 1:
                st.error("🔴 **Maintenance Required**")
            else:
                st.success("🟢 **No Immediate Maintenance Needed**")
        with col2:
            st.metric("Risk Score", f"{risk_score:.1%}")

st.markdown("---")
st.caption("🎯 Production ML dashboard - Backup Mode")
