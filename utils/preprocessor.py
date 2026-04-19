from typing import Any, Dict, List
import numpy as np
import streamlit as st
from datetime import date
import pandas as pd

from utils.config import (
    CATEGORICAL_OPTIONS,
    CAT_ENCODINGS,
    CALCULATED_FEATURES,
    INPUT_CATEGORICAL_FEATURES,
    INPUT_NUMERIC_FEATURES,
    NUMERIC_FEATURE_META,
)


def _clamp_numeric(feature: str, value: float) -> float:
    meta = NUMERIC_FEATURE_META.get(feature, {})
    min_value = float(meta.get("min_value", -1_000_000_000.0))
    max_value = float(meta.get("max_value", 1_000_000_000.0))
    return max(min_value, min(max_value, float(value)))

def _get_feature_label(feature: str) -> str:
    meta = NUMERIC_FEATURE_META.get(feature, {})
    return meta.get("label", feature.replace("_", " ").title())

def _default_numeric_fallback(feature: str) -> float:
    safe_defaults = {
        "mileage_km": 0.0,
        "engine_hours": 0.0,
        "vehicle_age_years": 0.0,
        "fault_code_count": 0.0,
        "oil_temp_avg_celsius": 85.0,
        "vibration_level": 0.0,
        "battery_voltage": 12.6,
        "engine_load_percent": 0.0,
        "fuel_efficiency_kmpl": 18.0,
        "days_since_last_service": 0.0,
        "mileage_per_year": 0.0,
        "thermal_stress": 0.0,
        "engine_hours_per_km": 0.0,
        "fault_density": 0.0,
        "load_efficiency": 0.0,
    }
    if feature in safe_defaults:
        return safe_defaults[feature]

    meta = NUMERIC_FEATURE_META.get(feature, {})
    min_value = float(meta.get("min_value", 0.0))
    max_value = float(meta.get("max_value", min_value))
    return (min_value + max_value) / 2.0

def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _estimate_fault_code_count(data: Dict[str, Any]) -> float:
    """Estimate a conservative fault-code count when the user does not know it."""
    score = 0.0

    if float(data.get("mileage_km", 0.0)) >= 100_000:
        score += 1.0
    if float(data.get("vehicle_age_years", 0.0)) >= 15:
        score += 1.0
    if float(data.get("oil_temp_avg_celsius", 0.0)) >= 95:
        score += 1.0
    if float(data.get("vibration_level", 0.0)) >= 4:
        score += 1.0
    if float(data.get("battery_voltage", 15.0)) < 11.5:
        score += 2.0
    if float(data.get("engine_load_percent", 0.0)) >= 75:
        score += 1.0
    if float(data.get("fuel_efficiency_kmpl", 40.0)) <= 15:
        score += 1.0
    if float(data.get("days_since_last_service", 0.0)) >= 90:
        score += 1.0

    return float(max(1.0, min(score, 8.0)))

def build_input_form_grid(features: List[str], num_cols: int = 3, use_sidebar: bool = False) -> Dict[str, Any]:
    """Generates a responsive grid of inputs, replacing days_since_last_service with a Date Picker."""
   
    input_data: Dict[str, Any] = {}
    ui = st.sidebar if use_sidebar else st
    
    encoded_cols = set(CAT_ENCODINGS.keys())
    
    ui.header("🔧 Vehicle Diagnostics")
    num_cols = max(1, num_cols)
    cols = ui.columns(num_cols)
    fault_code_unknown = ui.checkbox(
        "I do not know the fault code count",
        help="If selected, the app will use a conservative estimate instead of treating the value as zero.",
    )

    for idx, feature in enumerate(features):
        # We skip calculated features and one-hot columns in the numeric grid
        if feature in encoded_cols or feature in CALCULATED_FEATURES:
            
            # Special case: The UI uses a Date Picker to capture the last service event
            if feature == "days_since_last_service":
                col = cols[idx % num_cols]
                with col:
                    input_data["last_service_date"] = st.text_input(
                        "Last Service Date",
                        value="",
                        placeholder="YYYY-MM-DD (optional)",
                        help="Optional. Leave blank if unknown. Use YYYY-MM-DD format."
                    )
            continue
            
        col = cols[idx % num_cols]
        meta = NUMERIC_FEATURE_META.get(feature, {})
        with col:
            if feature == "fault_code_count":
                input_data["fault_code_count_unknown"] = fault_code_unknown
            input_data[feature] = col.text_input(
                _get_feature_label(feature),
                value="",
                placeholder=meta.get("range_hint", "Enter a value"),
                disabled=fault_code_unknown if feature == "fault_code_count" else False,
                help=(
                    "Leave this at zero only if there are truly no active codes. "
                    "Use the checkbox above if you do not know the value."
                    if feature == "fault_code_count"
                    else None
                ),
            )
            ui.caption(meta.get("range_hint", "Enter a realistic value."))

    ui.header("📍 Operating Conditions")
    cat_cols = ui.columns(min(len(CATEGORICAL_OPTIONS), num_cols))

    for idx, (cat_feature, options) in enumerate(CATEGORICAL_OPTIONS.items()):
        col = cat_cols[idx % len(cat_cols)]
        with col:
            input_data[cat_feature] = col.selectbox(
                _get_feature_label(cat_feature),
                options,
                index=None,
                placeholder="Select an option",
            )

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

    mileage = float(data.get("mileage_km", 0))
    age = float(data.get("vehicle_age_years", 1))
    hours = float(data.get("engine_hours", 0))
    oil_temp = float(data.get("oil_temp_avg_celsius", 0))
    load = float(data.get("engine_load_percent", 0))
    faults = float(data.get("fault_code_count", 0))
    efficiency = float(data.get("fuel_efficiency_kmpl", 1))

    if "days_since_last_service" in data and data["days_since_last_service"] is not None:
        data["days_since_last_service"] = max(0, int(float(data["days_since_last_service"])))
    else:
        service_date = data.get("last_service_date", date.today())
        if isinstance(service_date, str):
            service_text = service_date.strip()
            if service_text:
                try:
                    service_date = date.fromisoformat(service_text)
                except ValueError:
                    service_date = date.today()
            else:
                service_date = date.today()

        delta = date.today() - service_date
        data["days_since_last_service"] = max(0, delta.days)

    # Keep feature engineering consistent across the Streamlit app and agent pipeline.
    data["mileage_per_year"] = mileage / (age + 1)
    data["thermal_stress"] = oil_temp * load
    data["engine_hours_per_km"] = hours / (mileage + 1)
    data["fault_density"] = faults / (hours + 1)
    data["load_efficiency"] = load / (efficiency + 1)

    return data

def normalize_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw UI or agent input into the schema expected by the model pipeline."""
    normalized: Dict[str, Any] = {}

    for feature in INPUT_NUMERIC_FEATURES:
        parsed_value = _parse_optional_float(input_data.get(feature))
        if parsed_value is None:
            normalized[feature] = _default_numeric_fallback(feature)
        else:
            normalized[feature] = _clamp_numeric(feature, parsed_value)

    for feature in INPUT_CATEGORICAL_FEATURES:
        options = CATEGORICAL_OPTIONS[feature]
        value = input_data.get(feature, options[0])
        normalized[feature] = value if value in options else options[0]

    parsed_days_since_service = _parse_optional_float(input_data.get("days_since_last_service"))
    if parsed_days_since_service is not None:
        normalized["days_since_last_service"] = max(0, int(parsed_days_since_service))
    else:
        service_date = input_data.get("last_service_date", date.today())
        if isinstance(service_date, str):
            service_date = service_date.strip()
            try:
                service_date = date.fromisoformat(service_date)
            except ValueError:
                service_date = date.today()
        normalized["last_service_date"] = service_date

    fault_code_unknown = bool(input_data.get("fault_code_count_unknown", False))
    raw_fault_code_count = _parse_optional_float(input_data.get("fault_code_count"))
    if raw_fault_code_count is None:
        normalized["fault_code_count"] = _estimate_fault_code_count(normalized)
        normalized["fault_code_count_source"] = "estimated"
    elif fault_code_unknown:
        normalized["fault_code_count"] = _estimate_fault_code_count(normalized)
        normalized["fault_code_count_source"] = "estimated"
    else:
        normalized["fault_code_count"] = _clamp_numeric("fault_code_count", float(raw_fault_code_count or 0.0))
        normalized["fault_code_count_source"] = "provided"

    return calculate_internal_features(normalized)

def prepare_model_input_frame(features: List[str], input_data: Dict[str, Any]) -> pd.DataFrame:
    """Build a model-aligned DataFrame from raw inputs."""
    full_data = normalize_input_data(input_data)
    row: Dict[str, float] = {}

    for feature in features:
        if feature in CAT_ENCODINGS:
            cat_feature, category_value = CAT_ENCODINGS[feature]
            row[feature] = 1.0 if full_data.get(cat_feature) == category_value else 0.0
        else:
            row[feature] = float(full_data.get(feature, 0.0))

    return pd.DataFrame([row], columns=features)

def build_feature_vector(features: List[str], input_data: Dict[str, Any]) -> np.ndarray:
    """Transforms raw UI inputs into a model-ready 1x28 feature vector."""
    frame = prepare_model_input_frame(features, input_data)
    return frame.to_numpy(dtype=float)
