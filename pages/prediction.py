import streamlit as st
from utils.model_loader import load_artifacts, load_sample_data
from utils.preprocessor import build_input_form_grid
from utils.model_tool import predict_risk
from utils.ui_utils import apply_full_page_background, inject_custom_css

st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="🚗", layout="wide")

apply_full_page_background()

inject_custom_css("""
div.stButton > button[kind="primary"] {
    background-color: #00ffff !important;
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

/* Force black text for any internal elements (spans, p, etc.) */
div.stButton > button[kind="primary"] * {
    color: #000000 !important;
    font-weight: 800 !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 5px rgba(0, 255, 255, 0.8) !important;
    transform: translateY(-1px) !important;
}

div.stButton > button[kind="primary"]:hover * {
    color: #000000 !important;
    font-weight: 800 !important;
}

div.stButton > button[kind="primary"]:active {
    transform: scale(0.98) !important;
}
""")

st.markdown("""
<h1 style='font-size: 80px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>🔮 Live Maintenance Prediction</h1>
""", unsafe_allow_html=True)
st.caption("Use the controls below to estimate maintenance risk based on live vehicle diagnostics.")

_, _, features = load_artifacts()

predict_clicked = st.button("🔍 Click to Predict Maintenance", type="primary")
result_container = st.container()

st.markdown("---")

input_data = build_input_form_grid(features, num_cols=2, use_sidebar=False)

if predict_clicked:
    with st.spinner("Running prediction..."):
        result = predict_risk(input_data)
        prediction = result["risk_prediction"]
        risk_score = result["risk_probability"]
        risk_label = result["risk_label"]
        normalized_input = result["normalized_input"]

    with result_container:
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            if prediction == 1:
                st.error("🔴 *Maintenance Required*")
            else:
                st.success("🟢 *No Immediate Maintenance Needed*")
        with col2:
            st.metric("Risk", f"{risk_score:.1%}")
        with col3:
            st.metric("Risk Level", risk_label.title())

    if normalized_input.get("fault_code_count_source") == "estimated":
        st.warning(
            "Fault code count was not provided, so the app used a conservative estimate. "
            "Enter the actual fault code count if you have it for a more reliable prediction."
        )

st.markdown("---")

with st.expander("📋 Sample data used during development"):
    df_sample = load_sample_data()
    if df_sample is not None:
        st.dataframe(df_sample.head())
    else:
        st.info("Add data/sample_data.csv to view a sample of the training data.")
