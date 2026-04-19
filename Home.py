import streamlit as st
from utils.ui_utils import apply_full_page_background, inject_custom_css

st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="🚗", layout="wide")

apply_full_page_background()

inject_custom_css("""
.nav-card {
    background: rgba(8, 22, 28, 0.82);
    border: 1px solid rgba(0, 255, 255, 0.24);
    border-radius: 16px;
    padding: 1.1rem;
    min-height: 160px;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    margin-bottom: 0.65rem;
}

.nav-card h3 {
    margin: 0 0 0.45rem 0;
}

.nav-card p {
    margin: 0;
}
""")

st.markdown("""
<h1 style='font-size: 90px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>🚗 Vehicle Maintenance Predictor</h1>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div class="nav-card">
            <h3>📊 Data Insights</h3>
            <p><strong>Decode what drives vehicle failure</strong></p>
            <p>Features • EDA • Correlations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/data_insights.py", label="Open Data Insights", icon="➡️")

with col2:
    st.markdown(
        """
        <div class="nav-card">
            <h3>📈 Model Insights</h3>
            <p><strong>Trust every prediction with proof</strong></p>
            <p>Metrics • Importance • Validation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/model_insights.py", label="Open Model Insights", icon="➡️")

col3, col4 = st.columns(2)
with col3:
    st.markdown(
        """
        <div class="nav-card">
            <h3>🔮 Live Prediction</h3>
            <p><strong>Input vehicle data and get instant alert</strong></p>
            <p>Interactive • Real-time • Accurate</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/prediction.py", label="Open Live Prediction", icon="➡️")

with col4:
    st.markdown(
        """
        <div class="nav-card">
            <h3>🤖 Agent Assistant</h3>
            <p><strong>Ask maintenance questions and get guided actions</strong></p>
            <p>LangGraph • RAG • Policy Checks</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Agent_Assistant.py", label="Open Agent Assistant", icon="➡️")

st.markdown("---")
st.caption("Navigate using sidebar menu")