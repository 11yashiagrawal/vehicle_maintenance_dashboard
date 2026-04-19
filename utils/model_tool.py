from functools import lru_cache
from typing import Dict, List
import os

import joblib
import pandas as pd

from utils.config import (
    DATA_DIR,
    FEATURES_FILENAME,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MODELS_DIR,
    MODEL_FILENAME,
)
from utils.preprocessor import normalize_input_data, prepare_model_input_frame


@lru_cache(maxsize=1)
def load_model():
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise ValueError("Loaded model artifact does not expose a predict method.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model artifact does not expose a predict_proba method.")
    return model


@lru_cache(maxsize=1)
def load_feature_names() -> List[str]:
    features_path = os.path.join(DATA_DIR, FEATURES_FILENAME)
    return pd.read_csv(features_path, header=None)[0].tolist()


def predict_risk(input_data: Dict) -> Dict:
    model = load_model()
    features = load_feature_names()
    normalized_input = normalize_input_data(input_data)
    model_input = prepare_model_input_frame(features, normalized_input)

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]

    return {
        "risk_probability": float(probability),
        "risk_prediction": int(prediction),
        "risk_label": (
            "HIGH" if probability > HIGH_RISK_THRESHOLD else
            "MEDIUM" if probability > MEDIUM_RISK_THRESHOLD else
            "LOW"
        ),
        "normalized_input": normalized_input,
        "model_features": features,
    }
