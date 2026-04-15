import joblib
import pandas as pd

# Load model ONLY
model = joblib.load("models/vehicle_model.joblib")


def predict_risk(input_data):
    df = pd.DataFrame([input_data])

    # 🔥 CRITICAL: match training columns
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "risk_probability": float(probability),
        "prediction": int(prediction)
    }