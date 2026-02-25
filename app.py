import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="wide")
st.title("🚗 Vehicle Maintenance Predictor")
st.markdown("Production-ready vehicle maintenance prediction dashboard.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('vehicle_model.joblib')
    try:
        scaler = joblib.load('vehicle_scaler.joblib')
        has_scaler = True
    except:
        scaler = None
        has_scaler = False
    
    # FIX: Skip header row '0', get EXACT 28 features
    features_df = pd.read_csv('features.csv', header=None)
    features = features_df[0].iloc[1:].tolist()  # Skip index 0, get 28 features
    
    return model, scaler, features, has_scaler

model, scaler, features, has_scaler = load_artifacts()

st.success(f"✅ Loaded **{len(features)} features** - Model ready!")

# YOUR EXACT FEATURES FROM CSV (28 total)
numeric_features = [
    'mileage_km', 'engine_hours', 'vehicle_age_years', 'fault_code_count',
    'oil_temp_avg_celsius', 'vibration_level', 'battery_voltage',
    'engine_load_percent', 'fuel_efficiency_kmpl', 'days_since_last_service',
    'mileage_per_year', 'engine_hours_per_km', 'fault_density',
    'thermal_stress', 'load_efficiency'
]

categorical_options = {
    'vehicle_type': ['Sedan', 'Truck', 'Van'],
    'fuel_type': ['Electric', 'Petrol'],
    'region': ['North', 'South', 'West'],
    'road_condition': ['Rural', 'Urban'],
    'weather_condition': ['Hot', 'Normal', 'Rainy']
}

# CLEAN SIDEBAR - EXACTLY YOUR FEATURES
st.sidebar.header("🔧 Vehicle Diagnostics")
input_data = {}

# Numeric inputs (15 features)
for feature in numeric_features:
    default = 50000.0 if 'mileage' in feature else 5.0 if 'age' in feature else 0.0
    input_data[feature] = st.sidebar.number_input(
        feature.replace('_', ' ').title(), 
        value=default, 
        step=1000.0 if 'mileage' in feature else 1.0 if 'age' in feature else 0.1
    )

# Categorical dropdowns (5 original → 13 encoded columns)
st.sidebar.header("📍 Operating Conditions")
for cat_feature, options in categorical_options.items():
    input_data[cat_feature] = st.sidebar.selectbox(cat_feature.replace('_', ' ').title(), options)

# PERFECT PREDICTION PIPELINE (28 FEATURES EXACTLY)
if st.button("🔍 **Predict Maintenance**", type="primary", use_container_width=True):
    with st.spinner("Running prediction..."):
        # CREATE EXACT 28-FEATURE NUMPY ARRAY
        X_predict = np.zeros((1, len(features)))
        
        # MAP EVERY INPUT TO CORRECT COLUMN POSITION
        feature_to_idx = {feature: i for i, feature in enumerate(features)}
        
        # Fill numeric features
        for feature, value in input_data.items():
            if feature in feature_to_idx:
                X_predict[0, feature_to_idx[feature]] = value
        
        # Fill categorical one-hot (EXACT column names from your CSV)
        cat_encodings = {
            'vehicle_type_Sedan': ('vehicle_type', 'Sedan'),
            'vehicle_type_Truck': ('vehicle_type', 'Truck'),
            'vehicle_type_Van': ('vehicle_type', 'Van'),
            'fuel_type_Electric': ('fuel_type', 'Electric'),
            'fuel_type_Petrol': ('fuel_type', 'Petrol'),
            'region_North': ('region', 'North'),
            'region_South': ('region', 'South'),
            'region_West': ('region', 'West'),
            'road_condition_Rural': ('road_condition', 'Rural'),
            'road_condition_Urban': ('road_condition', 'Urban'),
            'weather_condition_Hot': ('weather_condition', 'Hot'),
            'weather_condition_Normal': ('weather_condition', 'Normal'),
            'weather_condition_Rainy': ('weather_condition', 'Rainy')
        }
        
        for encoded_col, (cat_feature, category) in cat_encodings.items():
            if input_data.get(cat_feature) == category and encoded_col in feature_to_idx:
                X_predict[0, feature_to_idx[encoded_col]] = 1.0
        
        # Apply scaler if exists
        if has_scaler:
            X_predict = scaler.transform(X_predict)
        
        # PREDICT
        prediction = model.predict(X_predict)[0]
        prob = model.predict_proba(X_predict)[0]
        
        # RESULTS
        col1, col2 = st.columns([1,3])
        with col1:
            if prediction == 1:
                st.error("🔴 **MAINTENANCE REQUIRED**")
            else:
                st.success("🟢 **No Maintenance Needed**")
        with col2:
            st.metric("Maintenance Risk", f"{prob[1]:.1%}", delta=None)
        
        st.success("✅ **Prediction successful!**")

# SAMPLE DATA
if st.checkbox("📋 Sample Dataset"):
    try:
        st.dataframe(pd.read_csv('sample_data.csv').head())
    except:
        st.info("No sample_data.csv")

st.markdown("---")
st.caption("🎯 Production ML dashboard - Ready for deployment!")
