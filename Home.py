import streamlit as st
import os
from utils.ui_utils import apply_full_page_background, inject_custom_css
from utils.config import ASSETS_DIR

st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="🚗", layout="wide")

apply_full_page_background()

st.markdown("""
<h1 style='font-size: 90px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>🚗 Vehicle Maintenance Predictor</h1>
<div style='display: flex; flex-direction: row; gap: 2rem; width: 100%;'>
    <div class="section-box" style='width: 100%;'>
        <h2>📊 Data Insights</h2>
        <p style='font-style: italic;'><strong>Decode what drives vehicle failure</strong></p>
        <medium>Features • EDA • Correlations</medium>
    </div>
    <div class="section-box" style='width: 100%;'>
        <h2>📈 Model Insights</h2>
        <p style='font-style: italic;'><strong>Trust every prediction with proof</strong></p>
        <medium>Metrics • Importance • Validation</medium>
    </div>
    <div class="section-box" style='width: 100%;'>
        <h2>🔮 Live Prediction</h2>
        <p style='font-style: italic;'><strong>Input vehicle data → Get instant alert</strong></p>
        <medium>Interactive • Real-time • Accurate</medium>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Navigate using sidebar menu")