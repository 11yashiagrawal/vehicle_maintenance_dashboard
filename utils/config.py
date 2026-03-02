import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_FILENAME = "vehicle_model.joblib"
ENCODER_FILENAME = "vehicle_encoder.joblib"
FEATURES_FILENAME = "features.csv"
SAMPLE_DATA_FILENAME = "sample_data.csv"
FULL_DATA_FILENAME = "Fleet_vehicle.csv"
STYLE_CSS_FILENAME = "style.css"

CATEGORICAL_OPTIONS = {
    "vehicle_type": ["Bus", "Car", "Truck", "Van"],
    "fuel_type": ["Electric", "Petrol", "Diesel"],
    "region": ["North", "South", "West", "East"],
    "road_condition": ["Rural", "Urban"],
    "weather_condition": ["Hot", "Normal", "Rainy"],
}

CAT_ENCODINGS = {
    "vehicle_type_Sedan": ("vehicle_type", "Sedan"),
    "vehicle_type_Truck": ("vehicle_type", "Truck"),
    "vehicle_type_Van": ("vehicle_type", "Van"),
    "fuel_type_Electric": ("fuel_type", "Electric"),
    "fuel_type_Petrol": ("fuel_type", "Petrol"),
    "region_North": ("region", "North"),
    "region_South": ("region", "South"),
    "region_West": ("region", "West"),
    "road_condition_Rural": ("road_condition", "Rural"),
    "road_condition_Urban": ("road_condition", "Urban"),
    "weather_condition_Hot": ("weather_condition", "Hot"),
    "weather_condition_Normal": ("weather_condition", "Normal"),
    "weather_condition_Rainy": ("weather_condition", "Rainy"),
}

NUMERIC_FEATURE_META = {
    "mileage_km": {
        "label": "Mileage (km)",
        "description": "Total distance the vehicle has travelled.",
        "unit": "km",
        "range_hint": "Typical: 0 – 300,000 km",
    },
    "engine_hours": {
        "label": "Engine hours",
        "description": "Total time the engine has been running.",
        "unit": "hours",
        "range_hint": "Typical: 0 – 20,000 hours",
    },
    "vehicle_age_years": {
        "label": "Vehicle age",
        "description": "Age of the vehicle since first registration.",
        "unit": "years",
        "range_hint": "Typical: 0 – 20 years",
    },
    "fault_code_count": {
        "label": "Fault code count",
        "description": "Number of diagnostic fault codes currently active.",
        "unit": "count",
        "range_hint": "Typical: 0 – 100 codes",
    },
    "oil_temp_avg_celsius": {
        "label": "Average oil temperature",
        "description": "Average engine oil temperature during operation.",
        "unit": "°C",
        "range_hint": "Typical: -20 – 150 °C",
    },
    "vibration_level": {
        "label": "Vibration level",
        "description": "Overall vibration severity from engine and drivetrain.",
        "unit": "normalized score",
        "range_hint": "Normalized: 0.0 – 10.0",
    },
    "battery_voltage": {
        "label": "Battery voltage",
        "description": "Electrical system voltage with engine running.",
        "unit": "V",
        "range_hint": "Typical: 10 – 15 V",
    },
    "engine_load_percent": {
        "label": "Engine load",
        "description": "Average percentage of maximum engine load.",
        "unit": "%",
        "range_hint": "0 – 100 %",
    },
    "fuel_efficiency_kmpl": {
        "label": "Fuel efficiency",
        "description": "Average distance covered per unit of fuel.",
        "unit": "km/l",
        "range_hint": "Typical: 0 – 40 km/l",
    },
    "last_service_date": {
        "label": "Last service date",
        "description": "Date of the last recorded service.",
        "unit": "date",
    }
}

CAT_FEATURE_META = {
    "vehicle_type": {
        "label": "Vehicle type",
        "description": "Type of vehicle (Bus, Car, Truck, Van).",
    },
    "fuel_type": {
        "label": "Fuel type",
        "description": "Type of fuel being used (Electric, Petrol, Diesel).",
    },
    "region": {
        "label": "Region",
        "description": "Region vehicle is operating in.",
    },
    "road_condition": {
        "label": "Road condition",
        "description": "Condition of the roads (Rural, Urban).",
    },
    "weather_condition": {
        "label": "Weather condition",
        "description": "Weather condition in the operating region.",
    },
}

CALCULATED_FEATURES = [
    "days_since_last_service",
    "mileage_per_year",
    "thermal_stress",
    "engine_hours_per_km",
    "fault_density",
    "load_efficiency",
]

NUMERIC_FEATURE_META.update({
    "days_since_last_service": {
        "label": "Days since last service",
        "description": "Number of days since the last recorded service.",
        "unit": "days"
    },
    "mileage_per_year": {
        "label": "Mileage per year",
        "description": "Average yearly distance based on current mileage and age.",
        "unit": "km/year",
    },
    "engine_hours_per_km": {
        "label": "Engine hours per km",
        "description": "Engine running time per kilometre travelled.",
        "unit": "hours/km",
    },
    "thermal_stress": {
        "label": "Thermal stress score",
        "description": "Composite score capturing heat and load stress on the engine.",
        "unit": "dimensionless score",
    },
    "fault_density": {
        "label": "Fault density",
        "description": "Number of fault codes per engine hour.",
        "unit": "codes/hour",
    },
    "load_efficiency": {
        "label": "Load efficiency",
        "description": "Average engine load relative to fuel efficiency.",
        "unit": "ratio",
    },
})
