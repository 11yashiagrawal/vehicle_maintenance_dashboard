from typing import Any, Dict, List
import numpy as np
import streamlit as st
from datetime import date
from utils.config import CATEGORICAL_OPTIONS, CAT_ENCODINGS, CALCULATED_FEATURES, NUMERIC_FEATURE_META

def _get_feature_label(feature: str) -> str:
    meta = NUMERIC_FEATURE_META.get(feature, {})
    return meta.get("label", feature.replace("_", " ").title())

def _default_value(feature_name: str) -> float:
    fname = feature_name.lower()
    if "mileage" in fname: return 50_000.0
    if "age" in fname: return 5.0
    if "hours" in fname: return 1_000.0
    if "percent" in fname: return 50.0
    return 0.0

def _default_step(feature_name: str) -> float:
    fname = feature_name.lower()
    if "mileage" in fname: return 1_000.0
    if "age" in fname: return 1.0
    if "hours" in fname: return 100.0
    if "percent" in fname: return 1.0
    return 0.1

def build_input_form_grid(features: List[str], num_cols: int = 3, use_sidebar: bool = False) -> Dict[str, Any]:
   
    input_data: Dict[str, Any] = {}
    ui = st.sidebar if use_sidebar else st
    
    encoded_cols = set(CAT_ENCODINGS.keys())
    excluded_features = encoded_cols.union(set(CALCULATED_FEATURES))
    
    ui.header("🔧 Vehicle Diagnostics")
    num_cols = max(1, num_cols)
    cols = ui.columns(num_cols)

    for idx, feature in enumerate(features):
        if feature in encoded_cols or feature in CALCULATED_FEATURES:
            
            if feature == "days_since_last_service":
                col = cols[idx % num_cols]
                with col:
                    input_data["last_service_date"] = st.date_input(
                        "Last Service Date",
                        value=date.today(),
                        help="The date of the most recent maintenance service."
                    )
            continue
            
        col = cols[idx % num_cols]
        meta = NUMERIC_FEATURE_META.get(feature, {})
        with col:
            input_data[feature] = col.number_input(
                _get_feature_label(feature),
                value=float(_default_value(feature)),
                step=float(_default_step(feature)),
            )
            ui.caption(meta.get("range_hint", "Enter a realistic value."))

    ui.header("📍 Operating Conditions")
    cat_cols = ui.columns(min(len(CATEGORICAL_OPTIONS), num_cols))

    for idx, (cat_feature, options) in enumerate(CATEGORICAL_OPTIONS.items()):
        col = cat_cols[idx % len(cat_cols)]
        with col:
            input_data[cat_feature] = col.selectbox(_get_feature_label(cat_feature), options)

    return input_data

def calculate_internal_features(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    days_since_last_service = today - last_service_date
    mileage_per_year = mileage_km / vehicle_age_years
    thermal_stress = oil_temp_avg_celsius * engine_load_percent
    engine_hours_per_km = engine_hours / mileage_km
    fault_density = fault_code_count / engine_hours
    load_efficiency = engine_load_percent / fuel_efficiency_kmpl
    """
    data = input_data.copy()

    
    service_date = data.get("last_service_date", date.today())
    if isinstance(service_date, str):
        
        service_date = date.fromisoformat(service_date)
    
    delta = date.today() - service_date
    data["days_since_last_service"] = max(0, delta.days)

    mileage = float(data.get("mileage_km", 0))
    age = float(data.get("vehicle_age_years", 1))
    data["mileage_per_year"] = mileage / max(age, 1e-6)
    
    oil_temp = float(data.get("oil_temp_avg_celsius", 0))
    load = float(data.get("engine_load_percent", 0))
    data["thermal_stress"] = oil_temp * load
    
    hours = float(data.get("engine_hours", 0))
    data["engine_hours_per_km"] = hours / max(mileage, 1e-6)
    
    faults = float(data.get("fault_code_count", 0))
    data["fault_density"] = faults / max(hours, 1e-6)
    
    efficiency = float(data.get("fuel_efficiency_kmpl", 1))
    data["load_efficiency"] = load / max(efficiency, 1e-6)
    
    return data

def build_feature_vector(features: List[str], input_data: Dict[str, Any]) -> np.ndarray:
   
    full_data = calculate_internal_features(input_data)
    
    X = np.zeros((1, len(features)), dtype=float)
    feature_to_idx = {f: i for i, f in enumerate(features)}

    encoded_cols = set(CAT_ENCODINGS.keys())
    
    for feature in features:
        if feature not in encoded_cols:
            X[0, feature_to_idx[feature]] = float(full_data.get(feature, 0.0))

    for encoded_col, (cat_feature, category_value) in CAT_ENCODINGS.items():
        if encoded_col in feature_to_idx:
            selected_value = full_data.get(cat_feature)
            X[0, feature_to_idx[encoded_col]] = 1.0 if selected_value == category_value else 0.0

    return X
